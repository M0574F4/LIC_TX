# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from compressai.ans import BufferedRansEncoder, RansDecoder

# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data
from typing import Dict, List, Optional, Sequence, Tuple
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import torch.optim as optim
import torch.nn.functional as F


from blocks import ImageAnalysis, HyperAnalysis, HyperSynthesis, ImageSynthesis

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    x = torch.exp(torch.linspace(math.log(min), math.log(max), levels))    
    return x

from compressai.models import CompressionModel
class ScaleHyperprior(CompressionModel):
    def __init__(
        self,
        network_channels: Optional[int] = None,
        compression_channels: Optional[int] = None,
        image_analysis: Optional[nn.Module] = None,
        image_synthesis: Optional[nn.Module] = None,
        image_bottleneck: Optional[nn.Module] = None,
        hyper_analysis: Optional[nn.Module] = None,
        hyper_synthesis: Optional[nn.Module] = None,
        hyper_bottleneck: Optional[nn.Module] = None,
        progressiveness_range: Optional[List] = None,
    ):
        super().__init__(entropy_bottleneck_channels=compression_channels)
        self.image_analysis = ImageAnalysis(network_channels, compression_channels)  
        self.hyper_analysis = HyperAnalysis(network_channels, compression_channels) 
        self.hyper_synthesis = HyperSynthesis(network_channels, compression_channels)  
        self.image_synthesis = ImageSynthesis(network_channels, compression_channels)
        
        self.hyper_bottleneck = EntropyBottleneck(channels=network_channels)
        scale_table = get_scale_table()
        self.image_bottleneck = GaussianConditional(scale_table=scale_table)
        self.progressiveness_range = progressiveness_range
        self.p_hyper_latent = None
        self.p_latent = None
        
    def forward(self, images, p=None, return_intermediates=None):
        if p is not None:
            self.p_latent = p
            self.p_hyper_latent = p
        else:
            self.p_latent = None
            self.p_hyper_latent = None
            
        self.latent = self.image_analysis(images)
        self.hyper_latent = self.hyper_analysis(self.latent)
        
        #---***---#
        self.latent = self.rate_less_latent(self.latent)
        self.hyper_latent = self.rate_less_hyper_latent(self.hyper_latent)
        #---***---#

        
        self.noisy_hyper_latent, self.hyper_latent_likelihoods = self.hyper_bottleneck(
            self.hyper_latent
        )

        self.scales = self.hyper_synthesis(self.noisy_hyper_latent)
        self.noisy_latent, self.latent_likelihoods = self.image_bottleneck(self.latent, self.scales)
        
        #---***---#
        self.latent_likelihoods = self.drop_zeros_likelihood(self.latent_likelihoods, self.latent)
        self.hyper_latent_likelihoods = self.drop_zeros_likelihood(self.hyper_latent_likelihoods, self.hyper_latent)
        #---***---#
        

        decoder_output = self.image_synthesis(self.noisy_latent, return_intermediate=return_intermediates)
        if return_intermediates:
            # If intermediates are returned, extract the final reconstruction and store intermediates
            self.reconstruction = decoder_output['sigmoid']  # Assuming 'sigmoid' is the final output
            self.intermediates = decoder_output  # Store the entire dictionary for further use
        else:
            # If no intermediates, assign the output directly
            self.reconstruction = decoder_output        
            
        self.rec_image = self.reconstruction.detach().clone()

        if self.training:
            return self.reconstruction, self.latent_likelihoods, self.hyper_latent_likelihoods
        else:
            return {
                "shape": self.hyper_latent.size()[-2:],
                "latent_likelihoods": self.latent_likelihoods,
                "noisy_hyper_latent": self.noisy_hyper_latent,
                "hyper_latent_likelihoods": self.hyper_latent_likelihoods,
                "x_hat": self.reconstruction,
            }


    
    def compress(self, images: torch.Tensor, p: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if p is not None:
            assert 0 < p <= 1, "Compression ratio p must be in (0, 1]."
            self.p_latent = p
            self.p_hyper_latent = p
        else:
            self.p_latent = None
            self.p_hyper_latent = None
            
        # Image analysis to obtain latent representation y
        latent = self.image_analysis(images)  # Shape: (N, C, H, W)
        
        # Hyper analysis to obtain hyper latent representation z
        hyper_latent = self.hyper_analysis(latent)  # Shape: (N, C', H', W')
        
        # Apply rate reduction by zeroing out less important channels
        latent, latent_num_channels_to_keep = self.rate_less_latent(latent)  # Shape: (N, C, H, W)
        hyper_latent, hyper_latent_num_channels_to_keep = self.rate_less_hyper_latent(hyper_latent)  # Shape: (N, C', H', W')
        
        # Encode the hyper latent z
        noisy_hyper_latent, hyper_latent_likelihoods = self.hyper_bottleneck(hyper_latent)


        # This is same as using hyper_bottleneck forward but it returns the bytes
        hyper_latent_string = self.hyper_bottleneck.compress(hyper_latent) # these are z bytes to be transmitted to the receiver
        hyper_shape = hyper_latent.size()[-2:]
        hyper_latent_hat = self.hyper_bottleneck.decompress(hyper_latent_string, hyper_shape) # same as noisy_hyper_latentbut receovered from bytes/hyper_latent_string

        # print(torch.mean(torch.abs(hyper_latent - hyper_latent_hat)))#/torch.mean(torch.abs(hyper_latent)))
                
        scales = self.hyper_synthesis(noisy_hyper_latent)  # Shape: (N, C, H, W)
        latent_strings = self.image_bottleneck.compress(latent, scales)
        # latent_strings = self.image_bottleneck.compress(latent[:, :latent_num_channels_to_keep, :, :], scales[:, :latent_num_channels_to_keep, :, :])
        latent_shape = latent.size()[-2:]
        
        # Encode the latent y using the scales from hyper synthesis
        noisy_latent, latent_likelihoods = self.image_bottleneck(latent, scales)
        # noisy_latent shape: (N, C, H, W)
        
        # Zero out likelihoods where channels are zeroed
        latent_likelihoods = self.drop_zeros_likelihood(latent_likelihoods, latent)
        hyper_latent_likelihoods = self.drop_zeros_likelihood(hyper_latent_likelihoods, hyper_latent)


        
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        index = self.image_bottleneck.build_indexes(scales)
        indexes_list = index.reshape(-1).tolist()
        latent_q = self.image_bottleneck.quantize(latent, "symbols")
        symbols_list = latent_q.reshape(-1).tolist()
        self.encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = self.encoder.flush()
        




        # encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        # y_string = encoder.flush()

        
        # z_strings = self.hyper_bottleneck.compress(noisy_hyper_latent)
        # encoder = BufferedRansEncoder()
        # cdf = self.image_bottleneck.quantized_cdf.tolist()
        # cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        # offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        
        
        # index = self.image_bottleneck.build_indexes(scales)
        # indexes_list = index.reshape(-1).tolist()

        
        # symbols_list = y_q_slice.reshape(-1).tolist()
        # encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        # y_string = encoder.flush()
        
        return {
                "y": noisy_latent,
                "z": noisy_hyper_latent,
                "y_likelihoods": latent_likelihoods,
                "z_likelihoods": hyper_latent_likelihoods,
                "hyper_latent_string": hyper_latent_string,
                "latent_strings": latent_strings,
                "rans_y_string": y_string,
            }

    def decompress(self, latents: torch.Tensor, hyper_latents: torch.Tensor, rans_y_string, z_string, zshape) -> torch.Tensor:
        """
        Decompress the latents y and z to reconstruct the image.
        
        Args:
            latents (torch.Tensor): Encoded latent tensor y of shape (N, C, H, W).
            hyper_latents (torch.Tensor): Encoded hyper latent tensor z of shape (N, C', H', W').
        
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (N, C, H, W).
        """
        z_hat = self.hyper_bottleneck.decompress(z_string, zshape)
        # print(torch.mean(torch.abs(z_hat-hyper_latents)))
        # Synthesize scales from hyper_latents (z)
        scales = self.hyper_synthesis(hyper_latents)  # Shape: (N, C, H, W)
        decoded_latent, _ = self.image_bottleneck(latents, scales)  # Shape: (N, C, H, W)
        
        # Synthesize the image from the decoded latent
        reconstruction = self.image_synthesis(decoded_latent)  # Shape: (N, C, H, W)




        yshape = [zshape[0] * 4, zshape[1] * 4]
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        self.decoder.set_stream(rans_y_string)

        index = self.image_bottleneck.build_indexes(scales)
        rv = self.decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv = torch.Tensor(rv).reshape(1, -1, yshape[0], yshape[1])
        y_hat = self.image_bottleneck.dequantize(rv)
        analyze_input(decoded_latent, 'decoded_latent')
        analyze_input(y_hat, 'y_hat')
        
        return reconstruction

    def compress2bytes(self, images: torch.Tensor, p: Optional[float] = None, drop_zeros=None) -> Dict[str, torch.Tensor]:
        if p is not None:
            assert 0 < p <= 1, "Compression ratio p must be in (0, 1]."
            self.p_latent = p
            self.p_hyper_latent = p
        else:
            self.p_latent = None
            self.p_hyper_latent = None
            
        # Image analysis to obtain latent representation y
        latent = self.image_analysis(images)  # Shape: (N, C, H, W)
        
        # Hyper analysis to obtain hyper latent representation z
        hyper_latent = self.hyper_analysis(latent)  # Shape: (N, C', H', W')
        
        # Apply rate reduction by zeroing out less important channels
        latent, latent_num_channels_to_keep = self.rate_less_latent(latent)  # Shape: (N, C, H, W)
        hyper_latent, hyper_latent_num_channels_to_keep = self.rate_less_hyper_latent(hyper_latent)  # Shape: (N, C', H', W')
        


        # This is same as using hyper_bottleneck forward but it returns the bytes
        hyper_latent_string = self.hyper_bottleneck.compress(hyper_latent) # these are z bytes to be transmitted to the receiver
        hyper_shape = hyper_latent.size()[-2:]
        hyper_latent_hat = self.hyper_bottleneck.decompress(hyper_latent_string, hyper_shape) # same as noisy_hyper_latent
                
        scales = self.hyper_synthesis(hyper_latent_hat)
        latent_shape = latent.size()[-2:]
        
        # Encode the latent y using the scales from hyper synthesis
        noisy_latent, latent_likelihoods = self.image_bottleneck(latent, scales)



        if drop_zeros is not None:
            scales = scales[:, :latent_num_channels_to_keep, :, :]
            latent = latent[:, :latent_num_channels_to_keep, :, :]
        
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        index = self.image_bottleneck.build_indexes(scales)
        indexes_list = index.reshape(-1).tolist()
        latent_q = self.image_bottleneck.quantize(latent, "symbols")
        symbols_list = latent_q.reshape(-1).tolist()
        self.encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = self.encoder.flush()
        

        return {
                "z_string": hyper_latent_string[0],
                "y_string": y_string,
                "z_shape": hyper_shape,
            }


    def decompress_bytes(self, compressed_img, p_ratio = None):
        y_string, z_string, z_shape = compressed_img['y_string'], [compressed_img['z_string']], compressed_img['z_shape']
        z_hat = self.hyper_bottleneck.decompress(z_string, z_shape)
        scales = self.hyper_synthesis(z_hat)  # Shape: (N, C, H, W)
        if p_ratio is not None:
            scales = scales[:, :int(scales.shape[1]*p_ratio), :, :]

        y_shape = [z_shape[0] * 4, z_shape[1] * 4]
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        self.decoder.set_stream(y_string)

        index = self.image_bottleneck.build_indexes(scales)
        rv = self.decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
        y_hat = self.image_bottleneck.dequantize(rv).cuda()

        decoded_latent, _ = self.image_bottleneck(y_hat, scales)  # Shape: (N, C, H, W)
        if p_ratio is not None:
            decoded_latent = torch.cat((decoded_latent, torch.zeros(decoded_latent.shape[0], int(192 - scales.shape[1]), decoded_latent.shape[2], decoded_latent.shape[3]).cuda()), dim=1)
        # Synthesize the image from the decoded latent
        reconstruction = self.image_synthesis(decoded_latent)  # Shape: (N, C, H, W)
        
        return reconstruction, z_hat

    
    def pre_synth(self, images, p=None, return_intermediates=None):
        if p is not None:
            self.p_latent = p
            self.p_hyper_latent = p
        else:
            self.p_latent = None
            self.p_hyper_latent = None
            
        self.latent = self.image_analysis(images)
        self.hyper_latent = self.hyper_analysis(self.latent)
        
        #---***---#
        self.latent, _ = self.rate_less_latent(self.latent)
        self.hyper_latent, _ = self.rate_less_hyper_latent(self.hyper_latent)
        #---***---#
        
        self.noisy_hyper_latent, self.hyper_latent_likelihoods = self.hyper_bottleneck(
            self.hyper_latent
        )

        self.scales = self.hyper_synthesis(self.noisy_hyper_latent)
        self.noisy_latent, self.latent_likelihoods = self.image_bottleneck(self.latent, self.scales)
        return self.noisy_latent, self.noisy_hyper_latent

    def post_synth(self, latents: torch.Tensor, hyper_latents: torch.Tensor, return_intermediate=None) -> torch.Tensor:

        scales = self.hyper_synthesis(hyper_latents)  # Shape: (N, C, H, W)

        decoded_latent, _ = self.image_bottleneck(latents, scales)  # Shape: (N, C, H, W)
        
        reconstruction = self.image_synthesis(decoded_latent, return_intermediate=return_intermediate)
        
        return reconstruction
        




    def my_compress(self, img, feature_0 = None, N_feature=None):            
        # Image analysis to obtain latent representation y
        latent = self.image_analysis(img)  # Shape: (N, C, H, W)
        # Hyper analysis to obtain hyper latent representation z
        hyper_latent = self.hyper_analysis(latent)  # Shape: (N, C', H', W')
        

        # if N_feature is not None:
            # masked_latent = torch.zeros_like(latent)
            # masked_latent[:,:int(feature_0+N_feature),:,:] = latent[:,:int(feature_0+N_feature),:,:]
            
            # hyper_max_idx = max(int(hyper_latent.shape[1] * (int(feature_0+N_feature))/latent.shape[1]), 1)
            # masked_hyper_latent = torch.zeros_like(hyper_latent)
            # masked_hyper_latent[:, :hyper_max_idx, :, :] = hyper_latent[:, :hyper_max_idx, :, :]
        # else:
            # masked_latent = latent.clone()
            # masked_hyper_latent = hyper_latent.clone()
        
        hyper_latent_string = self.hyper_bottleneck.compress(hyper_latent) # these are z bytes to be transmitted to the receiver
        # hyper_shape = masked_hyper_latent.size()[-2:]
        hyper_shape = hyper_latent.size()[-2:]
        # masked_hyper_latent_hat = self.hyper_bottleneck.decompress(hyper_latent_string, hyper_shape) # same as noisy_hyper_latent
        hyper_latent_hat = self.hyper_bottleneck.decompress(hyper_latent_string, hyper_shape) # same as noisy_hyper_latent
                
        scales = self.hyper_synthesis(hyper_latent_hat)

        
        scales = scales[:, feature_0:int(feature_0+N_feature), :, :]
        latent = latent[:, feature_0:int(feature_0+N_feature), :, :]

            
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        index = self.image_bottleneck.build_indexes(scales)
        indexes_list = index.reshape(-1).tolist()
        latent_q = self.image_bottleneck.quantize(latent, "symbols")
        symbols_list = latent_q.reshape(-1).tolist()
        self.encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = self.encoder.flush()
        
        hyper_latent_string= b'' if feature_0>0 else hyper_latent_string[0]

        return {
                "z_string": hyper_latent_string,
                "y_string": y_string,
                "z_shape": hyper_shape,
            }


    def my_decompress(self, compressed, feature_0, N_feature, y_hat_old=None, scales_old=None):
        y_string, z_string, z_shape = compressed['y_string'], [compressed['z_string']], compressed['z_shape']
        
        if scales_old is None:
            z_hat = self.hyper_bottleneck.decompress(z_string, z_shape)
            scales_all_feature_maps = self.hyper_synthesis(z_hat)
            scales = scales_all_feature_maps[:, feature_0:feature_0+N_feature, :, :]
        else:
            scales_all_feature_maps = scales_old
            scales = scales_old[:, feature_0:feature_0+N_feature, :, :]

        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        self.decoder.set_stream(y_string)
        

        index = self.image_bottleneck.build_indexes(scales)
        y_shape = [z_shape[0] * 4, z_shape[1] * 4]
        rv = self.decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
        fresh_feature_maps = self.image_bottleneck.dequantize(rv).cuda()


        
        past_feature_maps = y_hat_old if y_hat_old is not None else torch.tensor([]).cuda()
        future_feature_maps = torch.zeros(fresh_feature_maps.shape[0], int(192 - (feature_0+N_feature)), fresh_feature_maps.shape[2], fresh_feature_maps.shape[3]).cuda()
        y_hat = torch.cat((past_feature_maps, fresh_feature_maps, future_feature_maps), dim=1)
        # analyze_input(past_feature_maps, 'past_feature_maps')
        # analyze_input(fresh_feature_maps, 'fresh_feature_maps')
        # analyze_input(future_feature_maps, 'future_feature_maps')
        
        # Synthesize the image from the decoded latent
        reconstruction = self.image_synthesis(y_hat)  # Shape: (N, C, H, W)
        
        return reconstruction, y_hat[:, :int(feature_0+N_feature), :, :], scales_all_feature_maps

    
    def rate_progressive_channels(self, tensor: torch.Tensor, p_min: float, p_max: float) -> Tuple[torch.Tensor, int]:
        """
        Retains channels within the p_min to p_max proportion and zeroes out others.
        
        Args:
            tensor (torch.Tensor): The input tensor with shape (N, C, H, W).
            p_min (float): Lower bound proportion.
            p_max (float): Upper bound proportion.
        
        Returns:
            Tuple[torch.Tensor, int]: The masked tensor and the number of channels kept.
        """
        C = tensor.size(1)
        num_channels_min = int(C * p_min)
        num_channels_max = int(C * p_max)
        
        # Create a mask that retains channels from num_channels_min to num_channels_max
        mask = torch.zeros_like(tensor)
        mask[:, num_channels_min:num_channels_max, :, :] = 1
        tensor = tensor * mask
        num_channels_kept = num_channels_max - num_channels_min
        return tensor, num_channels_kept
        
    def compress2bytes_prog(
        self, 
        images, 
        p_min = None, 
        p_max = None, 
        drop_zeros_in_y_string = False
    ):
        
        # Validate and set compression ratios
        if p_min is not None and p_max is not None:
            assert 0 <= p_min < p_max <= 1, "Ensure that 0 <= p_min < p_max <= 1."
            self.p_latent_min = p_min
            self.p_latent_max = p_max
            self.p_hyper_latent_min = p_min
            self.p_hyper_latent_max = p_max
        else:
            self.p_latent_min = None
            self.p_latent_max = None
            self.p_hyper_latent_min = None
            self.p_hyper_latent_max = None
    
        # Image analysis to obtain latent representation y
        latent = self.image_analysis(images)  # Shape: (N, C, H, W)
        
        # Hyper analysis to obtain hyper latent representation z
        hyper_latent = self.hyper_analysis(latent)  # Shape: (N, C', H', W')
        
        # Apply progressive rate reduction by zeroing out channels outside p_min to p_max
        latent, latent_num_channels_kept = self.rate_progressive_channels(
            latent, self.p_latent_min, self.p_latent_max
        )  # Shape: (N, C, H, W)
        
        hyper_latent, hyper_latent_num_channels_kept = self.rate_progressive_channels(
            hyper_latent, self.p_hyper_latent_min, self.p_hyper_latent_max
        )  # Shape: (N, C', H', W')
        
        # Compress hyper_latent to hyper_latent_string
        hyper_latent_string = self.hyper_bottleneck.compress(hyper_latent)  # z bytes to be transmitted
        hyper_shape = hyper_latent.size()[-2:]
        hyper_latent_hat = self.hyper_bottleneck.decompress(hyper_latent_string, hyper_shape)  # Reconstructed hyper_latent
        
        # Synthesize scales from hyper_latent_hat
        scales = self.hyper_synthesis(hyper_latent_hat)
        latent_shape = latent.size()[-2:]
        
        # Encode the latent y using the scales from hyper synthesis
        noisy_latent, latent_likelihoods = self.image_bottleneck(latent, scales)
        if drop_zeros_in_y_string:
            # Retain only the kept channels for encoding
            scales = scales[:, :latent_num_channels_kept, :, :]
            latent = latent[:, :latent_num_channels_kept, :, :]
        
        # Prepare data for encoding
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
        index = self.image_bottleneck.build_indexes(scales)
        indexes_list = index.reshape(-1).tolist()
        latent_q = self.image_bottleneck.quantize(latent, "symbols")
        symbols_list = latent_q.reshape(-1).tolist()
        
        # Encode latent y
        self.encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = self.encoder.flush()
        
        return {
            "z_string": hyper_latent_string[0],
            "y_string": y_string,
            "z_shape": hyper_shape,
        }



    def decompress_bytes_prog(self, compressed_img, p_ratio=None, old_y_hat=None, old_z_hat=None):
        y_string = compressed_img['y_string']
        z_string = [compressed_img['z_string']]
        z_shape = compressed_img['z_shape']
    
        # Step 1 & 2: Use old z_hat if provided, else decode new z_hat
        if old_z_hat is not None:
            z_hat = old_z_hat
        else:
            z_hat = self.hyper_bottleneck.decompress(z_string, z_shape)
    
        scales = self.hyper_synthesis(z_hat)  # Shape: (N, C, H, W)
        total_channels = scales.shape[1]
        current_channels = int(total_channels * p_ratio) if p_ratio is not None else total_channels
    
        # Determine the number of previously decoded channels
        if old_y_hat is not None:
            prev_channels = old_y_hat.shape[1]
        else:
            prev_channels = 0
    
        # Step 3: Decode only the new channels
        new_channels = current_channels - prev_channels
        if new_channels > 0:
            new_scales = scales[:, prev_channels:current_channels, :, :]
            y_shape = [z_shape[0] * 4, z_shape[1] * 4]
    
            cdf = self.image_bottleneck.quantized_cdf.tolist()
            cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
            offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
    
            self.decoder.set_stream(y_string)
    
            # Build indexes and decode the new channels
            new_index = self.image_bottleneck.build_indexes(new_scales)
            rv = self.decoder.decode_stream(new_index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            new_y_hat = self.image_bottleneck.dequantize(rv).cuda()
    
            # Combine old and new y_hat feature maps
            if old_y_hat is not None:
                y_hat = torch.cat((old_y_hat, new_y_hat), dim=1)
            else:
                y_hat = new_y_hat
        else:
            y_hat = old_y_hat
    
        # Step 4: Pad zero feature maps for channels not yet arrived
        if y_hat.shape[1] < total_channels:
            padding_channels = total_channels - y_hat.shape[1]
            y_hat = torch.cat(
                (y_hat, torch.zeros(y_hat.shape[0], padding_channels, y_hat.shape[2], y_hat.shape[3]).cuda()),
                dim=1
            )
    
        # Continue image reconstruction
        decoded_latent, _ = self.image_bottleneck(y_hat, scales)
        reconstruction = self.image_synthesis(decoded_latent)
    
        # Step 5: Return the reconstruction, z_hat, and unpadded y_hat
        return reconstruction, z_hat, y_hat

    
    def compress_additional_bytes(self, images, prev_p, curr_p):
        assert 0 < prev_p < curr_p <= 1, "Compression ratios must satisfy 0 < prev_p < curr_p <= 1."
    
        # Image analysis to obtain latent and hyper-latent representations
        latent = self.image_analysis(images)
        hyper_latent = self.hyper_analysis(latent)
    
        # Determine channels based on previous and current p
        total_latent_channels = latent.size(1)
        total_hyper_channels = hyper_latent.size(1)
    
        prev_latent_channels = int(total_latent_channels * prev_p)
        curr_latent_channels = int(total_latent_channels * curr_p)
    
        prev_hyper_channels = int(total_hyper_channels * prev_p)
        curr_hyper_channels = int(total_hyper_channels * curr_p)
    
        # Create masks for additional channels
        latent_mask = torch.zeros_like(latent)
        latent_mask[:, prev_latent_channels:curr_latent_channels, :, :] = 1
        latent = latent * latent_mask
    
        hyper_latent_mask = torch.zeros_like(hyper_latent)
        hyper_latent_mask[:, prev_hyper_channels:curr_hyper_channels, :, :] = 1
        hyper_latent = hyper_latent * hyper_latent_mask
    
        # Compress the additional hyper-latent channels
        hyper_latent_string = self.hyper_bottleneck.compress(hyper_latent)
    
        # Decompress to get scales for additional channels
        hyper_latent_hat = self.hyper_bottleneck.decompress(hyper_latent_string, hyper_latent.size()[-2:])
        hyper_latent_hat = hyper_latent_hat * hyper_latent_mask  # Apply the mask
    
        scales = self.hyper_synthesis(hyper_latent_hat)
        scales = scales * latent_mask  # Apply the mask
    
        # Quantize and encode the additional latent channels
        latent_q = self.image_bottleneck.quantize(latent, "symbols")
    
        non_zero_positions = latent_mask.bool()
        symbols_list = latent_q[non_zero_positions].tolist()
        indexes = self.image_bottleneck.build_indexes(scales)
        indexes_list = indexes[non_zero_positions].tolist()
    
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
    
        self.encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = self.encoder.flush()
    
        return {
            "z_string": hyper_latent_string[0],
            "y_string": y_string,
            "z_shape": hyper_latent.size()[-2:],
        }



    def decompress_additional_bytes(self, compressed_img):
        y_string = compressed_img['y_string']
        z_string = [compressed_img['z_string']]
        z_shape = compressed_img['z_shape']
        latent_mask = compressed_img['latent_mask'].cuda()
        hyper_latent_mask = compressed_img['hyper_latent_mask'].cuda()
    
        # Decompress hyper latent
        hyper_latent_hat = self.hyper_bottleneck.decompress(z_string, z_shape)
        hyper_latent_hat = hyper_latent_hat * hyper_latent_mask  # Apply the mask
    
        # Generate scales
        scales = self.hyper_synthesis(hyper_latent_hat)
        scales = scales * latent_mask  # Apply the mask
    
        # Prepare for decoding latent
        cdf = self.image_bottleneck.quantized_cdf.tolist()
        cdf_lengths = self.image_bottleneck.cdf_length.reshape(-1).int().tolist()
        offsets = self.image_bottleneck.offset.reshape(-1).int().tolist()
    
        self.decoder.set_stream(y_string)
        indexes = self.image_bottleneck.build_indexes(scales)
    
        # Only decode non-zero positions
        non_zero_positions = latent_mask.bool()
        indexes_list = indexes[non_zero_positions].tolist()
        rv = self.decoder.decode_stream(indexes_list, cdf, cdf_lengths, offsets)
        rv_tensor = torch.zeros_like(latent_mask)
        rv_tensor[non_zero_positions] = torch.Tensor(rv).cuda()
    
        # Dequantize latent
        y_hat = self.image_bottleneck.dequantize(rv_tensor)
    
        # Reconstruct image
        reconstruction = self.image_synthesis(y_hat)
    
        return reconstruction, y_hat, hyper_latent_hat





    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.image_bottleneck.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

        
    # def rate_less_latent(self, data):
    #     self.save_p = []
    #     temp_data = data.clone()
    #     for i in range(data.shape[0]):
    #         if self.p_latent:
    #             # p shows the percentage of keeping
    #             p = self.p_latent
    #         else:
    #             p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1],1)[0]
    #             self.save_p.append(p)

    #         if p == 1.0:
    #             pass            
    #         else:
    #             p = int(p*data.shape[1])
    #             replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

    #             if replace_tensor.shape[0] > 0:
    #                 temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
    #     return temp_data
    
    # def rate_less_hyper_latent(self, data):
    #     temp_data = data.clone()
    #     for i in range(data.shape[0]):
    #         if self.p_hyper_latent:
    #             # p shows the percentage of keeping
    #             p = self.p_hyper_latent
    #         else:
    #             p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1], 1)[0]
    #             p = self.save_p[i]
    #         if p == 1.0:
    #             pass
            
    #         else:
    #             p = int(p*data.shape[1])
    #             replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

    #             if replace_tensor.shape[0] > 0:
    #                 temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
    #     return temp_data

    def rate_less_latent(self, data):
        temp_data = data.clone()
        if self.p_latent is not None:
            # p shows the percentage of keeping
            p = self.p_latent
            num_channels_to_keep = int(p * data.shape[1])
            temp_data[:, num_channels_to_keep:, :, :] = 0.0  # Zero out least important channels
        else:
            # Existing code for training
            for i in range(data.shape[0]):
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1])
                num_channels_to_keep = int(p * data.shape[1])
                temp_data[i, num_channels_to_keep:, :, :] = 0.0
        return temp_data, num_channels_to_keep

    
    def rate_less_hyper_latent(self, data):
        temp_data = data.clone()
        if self.p_hyper_latent is not None:
            # p shows the percentage of keeping
            p = self.p_hyper_latent
            num_channels_to_keep = int(p * data.shape[1])
            temp_data[:, num_channels_to_keep:, :, :] = 0.0  # Zero out least important channels
        else:
            # Existing code for training
            for i in range(data.shape[0]):
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1])
                num_channels_to_keep = int(p * data.shape[1])
                temp_data[i, num_channels_to_keep:, :, :] = 0.0
        return temp_data, num_channels_to_keep



    def drop_zeros_likelihood(self, likelihood, replace):
        temp_data = likelihood.clone()
        temp_data = torch.where(
            replace == 0.0,
            torch.FloatTensor([1.0])[0],
            likelihood,
        )
        return temp_data
    
    

    
class ScaleHyperpriorLightning(pl.LightningModule):
    def __init__(
        self,
        model: ScaleHyperprior,
        distortion_lambda,
    ):
        super().__init__()

        self.model = model
        self.distortion_lambda = distortion_lambda


    def forward(self, images):
        return self.model(images)
        
    def training_step(self, batch, batch_idx):
        
        images = batch

        x_hat, y_likelihoods, z_likelihoods = self.model(images)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, images
        )
        self.log_dict(
            {
                "train_loss": combined_loss.item(),
                "train_distortion_loss": distortion_loss.item(),
                "train_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        return {
            "loss": combined_loss,
           }


    def training_epoch_end(self, outs):
        loss_rec = torch.stack([x["loss"] for x in outs]).mean()
        self.log('train_combined_loss_epoch', loss_rec, on_epoch=True, prog_bar=True, logger=True)

        # normal_imshow(self.model.rec_image[0].to('cpu').detach().numpy())
        # plt.show()

    def validation_step(self, batch, batch_idx):
        
        self.model.p_hyper_latent = .2
        self.model.p_latent = .2
        
        images = batch
        
        x_hat, y_likelihoods, z_likelihoods = self.model(images)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, images
        )
        self.log_dict(
            {
                "val_loss": combined_loss.item(),
                "val_distortion_loss": distortion_loss.item(),
                "val_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        self.model.p_hyper_latent = None
        self.model.p_latent = None

        return {
            "loss": combined_loss,
           }


    def validation_epoch_end(self, outs):
        loss_rec = torch.stack([x["loss"] for x in outs]).mean()
        self.log('val_combined_loss_epoch', loss_rec, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )

        return {
                "optimizer": optimizer,
            }

        
    def rate_distortion_loss(self, reconstruction, latent_likelihoods,
                             hyper_latent_likelihoods, original,):
        
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width

        bits = (
            latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()
        ) / -math.log(2)
        
        bpp_loss = bits / num_pixels

        distortion_loss = F.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss

        return bpp_loss, distortion_loss, combined_loss

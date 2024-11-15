import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(img1, img2, data_range=None):
    """
    Compute SSIM between two images or batches of images, supporting both NumPy arrays 
    and PyTorch tensors, and both channel-first and channel-last formats. 
    Optionally specify the data range.
    
    Args:
    - img1: The first image or batch of images (NumPy array or PyTorch tensor).
    - img2: The second image or batch of images (NumPy array or PyTorch tensor).
    - data_range: The data range for the images (e.g., 1.0 for [0, 1] images, 255 for [0, 255] images).
    
    Returns:
    - Mean SSIM value across all channels (or across all images and channels if batched).
    """
    # Convert PyTorch tensors to NumPy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Handle batch dimension
    if img1.ndim == 3:  # Single image, add batch dimension
        img1 = img1[None, ...]
    if img2.ndim == 3:  # Single image, add batch dimension
        img2 = img2[None, ...]

    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "Input images must have the same dimensions"

    # Determine if images are channel-first or channel-last
    if img1.shape[1] == 3:  # Channel-first
        img1 = np.transpose(img1, (0, 2, 3, 1))  # Convert to channel-last
        img2 = np.transpose(img2, (0, 2, 3, 1))  # Convert to channel-last

    # Determine the default data range if not specified
    if data_range is None:
        if img1.max() > 1 or img2.max() > 1:
            data_range = 255  # Assume the image is in [0, 255] range
        else:
            data_range = 1.0  # Assume the image is in [0, 1] range

    # Check for mismatched data ranges
    if data_range == 1.0 and (img1.max() > 1 or img2.max() > 1):
        print(f'original.min()={img1.min()}, reconstructed_image.min()={img2.min()}')
        print(f'original.max()={img1.max()}, reconstructed_image.max()={img2.max()}')
        raise ValueError("Input images have values larger than 1, but data_range is set to 1.0. Please check your input data.")

    # Compute SSIM per channel and per image
    ssim_values = []
    for i in range(img1.shape[0]):  # Iterate over batch
        channel_ssim_values = []
        for c in range(img1.shape[3]):  # Iterate over channels
            ssim_value = ssim(img1[i, :, :, c], img2[i, :, :, c], data_range=data_range)
            channel_ssim_values.append(ssim_value)
        ssim_values.append(np.mean(channel_ssim_values))  # Mean SSIM for current image

    # Calculate mean SSIM over all images in the batch
    mean_ssim = np.mean(ssim_values)

    return mean_ssim


def calculate_psnr(image1, image2, max_val=255.0):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Parameters:
        image1 (numpy.ndarray or torch.Tensor): The first input image.
        image2 (numpy.ndarray or torch.Tensor): The second input image.
        max_val (float): The maximum possible pixel value of the images (e.g., 255 for uint8 images).

    Returns:
        float: The PSNR value between the two images.
    """
    # Ensure inputs are in torch.Tensor format for consistency
    if isinstance(image1, np.ndarray):
        image1 = torch.tensor(image1)
    if isinstance(image2, np.ndarray):
        image2 = torch.tensor(image2)
    
    # Ensure images are in float32 to prevent overflow in calculations
    image1 = image1.float()
    image2 = image2.float()

    # Handle different channel formats
    if image1.shape != image2.shape:
        if image1.shape[0] == 3 and image1.shape[1:] == image2.shape[-3:]:
            image2 = image2.permute(2, 0, 1)
        elif image2.shape[0] == 3 and image2.shape[1:] == image1.shape[-3:]:
            image1 = image1.permute(2, 0, 1)

    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((image1 - image2) ** 2)

    # Compute PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    return psnr.item()
    
    
    
def compute_metrics(image, reconstructed_image, config=None):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()    
    if isinstance(reconstructed_image, torch.Tensor):
        reconstructed_image = reconstructed_image.detach().cpu().numpy()   
        
    # Compute PSNR, SSIM, and BPP
    psnr = calculate_psnr(image, reconstructed_image, max_val=1)
    ssim = compute_ssim(image, reconstructed_image, data_range=1)
   
    pixel_per_image = image.shape[-1] * image.shape[-2]
    # bpp = bit_per_flattened_patch * flattened_patch_per_image / pixel_per_image
    return {'psnr':psnr, 'ssim':ssim}

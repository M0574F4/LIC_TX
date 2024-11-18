# LIC_TX

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![GitHub Issues](https://img.shields.io/github/issues/M0574F4/LIC_TX)
![GitHub Forks](https://img.shields.io/github/forks/M0574F4/LIC_TX)
![GitHub Stars](https://img.shields.io/github/stars/M0574F4/LIC_TX)

## Podcast

We discussed **LIC_TX** in detail in this podcast episode. [![Listen to the podcast](https://img.icons8.com/ios-filled/50/000000/musical-notes.png)](https://drive.google.com/file/d/1V0q3lVWvcsBc-qkl7RwKw7fnfbZxU2Qf/view?usp=sharing)



## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Introduction
**LIC_TX** is a progressive image transmission pipeline designed for dynamic communication channels. It leverages two accessible image autoencoders to ensure efficient and robust image transmission:

1. **ProgDTD**: A hyperprior-based model ([GitHub Repository](https://github.com/fhennig/ProgDTD))
2. **VQGAN-LC**: A VQGAN-based model ([GitHub Repository](https://github.com/zh460045050/VQGAN-LC))

LIC_TX addresses the challenges of transmitting images over dynamic channels by utilizing these advanced autoencoders, providing a seamless and progressive transmission experience.

## Features
- **Progressive Transmission**: Transmit images in a progressive manner, adapting to channel conditions.
- **Dynamic Channel Adaptation**: Adjusts to varying channel conditions for optimal image quality.
- **Residual Quantization**: Incorporates residual quantization for efficient data compression.
- **Easy Integration**: Built upon accessible and widely-used autoencoder models.

## Dependencies
LIC_TX requires the following dependencies:

- **Software:**
  - Python 3.7+
  - pip 23.3.1

- **Python Libraries:**
  - [ldm-lc](https://github.com/ldm-lc) *(Ensure compatibility)*
  - [ProgDTD](https://github.com/fhennig/ProgDTD)
  - [compress_AI](https://github.com/compress_AI) *(Ensure compatibility)*
  - [PyTorch Lightning](https://www.pytorchlightning.ai/) (1.7.7)
  - [TorchMetrics](https://torchmetrics.readthedocs.io/) (0.11.0)
  - [faiss-cpu](https://github.com/facebookresearch/faiss)

## Installation

### Prerequisites
Ensure you have Python 3.7 or higher installed. It's recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv lic_tx_env

# Activate the virtual environment
# On Windows
lic_tx_env\Scripts\activate
# On Unix or MacOS
source lic_tx_env/bin/activate

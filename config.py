"""
Configuration file for the DCGAN Skin Lesion Data Augmentation project.
This file stores all hyperparameters, paths, and device settings.
"""

import torch
import random

# --- General Configuration ---
# Set random seed for reproducibility
manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# --- Data Paths and Settings ---
# Root directory for the dataset. Adjust this path to your data location.
# Example: "/content/drive/MyDrive/skin_lesion/DCGAN/akiec/gan_input/"
DATA_ROOT = "/content/drive/MyDrive/skin_lesion/DCGAN/akiec/gan_input/"

# Directory to save generated images and trained models
# Example: '/content/drive/MyDrive/skin_lesion/DCGAN/bcc/saved_images/output/'
OUT_DIR = "/content/drive/MyDrive/skin_lesion/DCGAN/bcc/saved_images/output/"

# Number of worker threads for data loading
WORKERS = 2

# Batch size during training
BATCH_SIZE = 257

# Spatial size of training images. All images will be resized to this size.
IMAGE_SIZE = 64

# Number of channels in the training images. For color images this is 3.
NUM_CHANNELS = 3

# --- Model Hyperparameters ---
# Size of z latent vector (i.e., size of generator input)
LATENT_VECTOR_SIZE = 100

# Size of feature maps in the generator
GENERATOR_FEATURE_MAPS = 64

# Size of feature maps in the discriminator
DISCRIMINATOR_FEATURE_MAPS = 64

# Number of training epochs
NUM_EPOCHS = 2000

# Learning rate for optimizers
LEARNING_RATE = 0.0002

# Beta1 hyperparameter for Adam optimizers
BETA1 = 0.5

# Number of GPUs available. Use 0 for CPU mode, >0 for CUDA.
NUM_GPUS = 1

# Number of images to sample during evaluation or saving
IMAGE_SAMPLE_SIZE = 64 # Changed to a common sample size for visualization

# --- Device Configuration ---
# Decide which device we want to run on (CUDA if available, otherwise CPU)
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and NUM_GPUS > 0) else "cpu")

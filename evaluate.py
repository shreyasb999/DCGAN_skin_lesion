"""
Script for evaluating the trained DCGAN model.
This includes generating new images and can be extended for FID score calculation.
"""

import torch
import torchvision.utils as vutils
import os
from models.generator import Generator
from config import (
    DEVICE, NUM_GPUS, LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAPS,
    OUT_DIR, IMAGE_SAMPLE_SIZE
)
import numpy as np
from PIL import Image

# Function to generate and save images
def generate_images(generator_model, num_images, output_path, device):
    """
    Generates new images using the trained generator and saves them.
    Args:
        generator_model (nn.Module): The trained Generator model.
        num_images (int): Number of images to generate.
        output_path (str): Directory to save the generated images.
        device (torch.device): The device to run generation on.
    """
    generator_model.eval() # Set model to evaluation mode
    print(f"Generating {num_images} images...")
    os.makedirs(output_path, exist_ok=True)

    for i in range(num_images):
        noise = torch.randn(1, LATENT_VECTOR_SIZE, 1, 1, device=device)
        with torch.no_grad():
            generated_image = generator_model(noise).detach().cpu()

        # Process and save the image
        # Unnormalize and convert to PIL Image
        img = vutils.make_grid(generated_image, padding=0, normalize=True)
        img_np = np.transpose(img.numpy(), (1, 2, 0)) # C,H,W to H,W,C
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        img_pil.save(os.path.join(output_path, f"generated_image_{i:04d}.png"))
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1} images.")
    print(f"Finished generating {num_images} images to {output_path}")

def load_generator_from_checkpoint(checkpoint_path, device, num_gpus):
    """
    Loads a trained Generator model from a checkpoint.
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file).
        device (torch.device): The device to load the model onto.
        num_gpus (int): Number of GPUs configured for the model.
    Returns:
        Generator: Loaded Generator model.
    """
    netG = Generator(num_gpus).to(device)
    if (device.type == 'cuda') and (num_gpus > 1):
        netG = nn.DataParallel(netG, list(range(num_gpus)))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # Load state_dict from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['modelG_state_dict'])
    print(f"Generator loaded from {checkpoint_path}")
    return netG

if __name__ == "__main__":
    # Specify the path to your trained model checkpoint
    # Example: os.path.join(OUT_DIR, "model_iter_1999.pt") or the latest iteration
    # You might need to manually update the iteration number if you want the very last saved model.
    # The `train.py` saves at iter_0, iter_50, etc.
    LATEST_CHECKPOINT_PATH = os.path.join(OUT_DIR, "model_iter_0.pt") # Placeholder, update this to your latest saved model

    if not os.path.exists(LATEST_CHECKPOINT_PATH):
        print(f"Error: No model checkpoint found at {LATEST_CHECKPOINT_PATH}.")
        print("Please run train.py first or update LATEST_CHECKPOINT_PATH to an existing model file.")
    else:
        # Load the generator model
        generator_model = load_generator_from_checkpoint(LATEST_CHECKPOINT_PATH, DEVICE, NUM_GPUS)

        # Define an output directory for evaluation images
        eval_output_dir = os.path.join(OUT_DIR, "evaluation_images")
        
        # Generate and save new images
        generate_images(generator_model, IMAGE_SAMPLE_SIZE, eval_output_dir, DEVICE)

"""
Main script for training the DCGAN.
Loads data, initializes models, defines loss and optimizers, and runs the training loop.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt # For plotting training images initially

from models.generator import Generator
from models.discriminator import Discriminator
from models.__init__ import weights_init # Contains the weights_init function
from utils.data_loader import get_dataloader
from config import (
    DEVICE, NUM_GPUS, LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAPS,
    DISCRIMINATOR_FEATURE_MAPS, NUM_EPOCHS, LEARNING_RATE, BETA1, OUT_DIR, IMAGE_SAMPLE_SIZE
)

def train_dcgan():
    """
    Orchestrates the training process for the DCGAN.
    """
    print(f"Using device: {DEVICE}")

    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # --- Data Loading ---
    dataloader = get_dataloader()
    print(f"Number of samples in dataset: {len(dataloader.dataset)}")

    # Plot some initial training images (optional, for verification)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(OUT_DIR, "training_images_sample.png"))
    plt.close()

    # --- Model Initialization ---
    netG = Generator(NUM_GPUS).to(DEVICE)
    netD = Discriminator(NUM_GPUS).to(DEVICE)

    # Handle multi-GPU if desired
    if (DEVICE.type == 'cuda') and (NUM_GPUS > 1):
        netG = nn.DataParallel(netG, list(range(NUM_GPUS)))
        netD = nn.DataParallel(netD, list(range(NUM_GPUS)))

    # Apply the weights_init function
    netG.apply(weights_init)
    netD.apply(weights_init)

    print("--- Generator Model ---")
    print(netG)
    print("\n--- Discriminator Model ---")
    print(netD)

    # --- Loss and Optimizers ---
    criterion = nn.BCELoss()

    # Create a batch of latent vectors for visualizing generator progression
    fixed_noise = torch.randn(IMAGE_SAMPLE_SIZE, LATENT_VECTOR_SIZE, 1, 1, device=DEVICE)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # --- Training Loop ---
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("\nStarting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, LATENT_VECTOR_SIZE, 1, 1, device=DEVICE)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D and update
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            print(f"[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}]\t"
                  f"Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t"
                  f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == NUM_EPOCHS - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                # Save generated images
                vutils.save_image(fake, os.path.join(OUT_DIR, f"generated_samples_iter_{iters}.png"), normalize=True)


                # Save model checkpoints
                torch.save({
                    'modelG_state_dict': netG.state_dict(),
                    'modelD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                }, os.path.join(OUT_DIR, f"model_iter_{iters}.pt"))

            iters += 1

    print("Training finished!")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"))
    plt.close()

    # Save final generated images from fixed noise
    vutils.save_image(img_list[-1], os.path.join(OUT_DIR, "final_generated_samples.png"), normalize=True)

if __name__ == '__main__':
    train_dcgan()
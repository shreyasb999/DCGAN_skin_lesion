"""
Defines the Generator model for the DCGAN.
"""

import torch.nn as nn
from config import LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAPS, NUM_CHANNELS
from models.__init__ import weights_init # Import from the __init__.py of the models package

class Generator(nn.Module):
    """
    DCGAN Generator architecture.
    Input: A latent vector `z` (nz x 1 x 1)
    Output: A generated image (nc x image_size x image_size)
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAPS * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAPS * 8, GENERATOR_FEATURE_MAPS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAPS * 4, GENERATOR_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAPS * 2, GENERATOR_FEATURE_MAPS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAPS, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (nc) x 64 x 64 (assuming image_size = 64)
        )

    def forward(self, input):
        """
        Forward pass for the Generator.
        Args:
            input (torch.Tensor): Latent vector.
        Returns:
            torch.Tensor: Generated image.
        """
        return self.main(input)

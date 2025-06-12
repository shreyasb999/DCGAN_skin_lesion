"""
Defines the Discriminator model for the DCGAN.
"""

import torch.nn as nn
from config import NUM_CHANNELS, DISCRIMINATOR_FEATURE_MAPS
from models.__init__ import weights_init # Import from the __init__.py of the models package

class Discriminator(nn.Module):
    """
    DCGAN Discriminator architecture.
    Input: An image (nc x image_size x image_size)
    Output: A scalar probability (real/fake)
    """
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(NUM_CHANNELS, DISCRIMINATOR_FEATURE_MAPS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS, DISCRIMINATOR_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS * 2, DISCRIMINATOR_FEATURE_MAPS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS * 4, DISCRIMINATOR_FEATURE_MAPS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Final output is a single scalar
        )

    def forward(self, input):
        """
        Forward pass for the Discriminator.
        Args:
            input (torch.Tensor): Image tensor.
        Returns:
            torch.Tensor: Scalar probability.
        """
        return self.main(input)

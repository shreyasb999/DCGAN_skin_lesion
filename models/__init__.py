# This file makes the 'models' directory a Python package.

import torch.nn as nn

# Common weight initialization function for Generator and Discriminator
def weights_init(m):
    """
    Custom weights initialization called on netG and netD.
    Initializes convolutional/batch norm layers with Gaussian distribution
    and batch norm biases to 0.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

"""
Utility functions for data loading and preprocessing.
"""

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
from config import IMAGE_SIZE, DATA_ROOT, BATCH_SIZE, WORKERS

def get_dataloader():
    """
    Creates and returns a DataLoader for the image dataset.
    """
    # Define transformations for the images
    dataset_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the dataset using ImageFolder
    dataset = dset.ImageFolder(root=DATA_ROOT, transform=dataset_transforms)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS
    )
    return dataloader
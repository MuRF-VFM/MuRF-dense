import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class ADE20KDataset(Dataset):
    """
    Dataset class for ADE20K semantic segmentation dataset
    """

    def __init__(self, root, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            split (string): 'train', 'val', or 'test' split
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root
        self.split = split

        # ADE20K has 150 object categories (plus background)
        self.num_classes = 150

        # Setup paths for images and annotations
        self.img_dir = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'annotations', split)

        # Get all image filenames
        self.img_files = [f for f in os.listdir(self.img_dir) if
                          f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Get corresponding mask filename (replace extension)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        return {'image': image, 'mask': mask}


def get_ade20k_dataloader(root_dir, batch_size=8, split='train',
                          num_workers=4):
    """
    Create a DataLoader for ADE20K dataset

    Args:
        root_dir (string): Path to the ADE20K dataset
        batch_size (int): Batch size
        split (string): Dataset split ('train', 'val', or 'test')
        num_workers (int): Number of workers for data loading

    Returns:
        dataloader: PyTorch dataloader for the ADE20K dataset
    """
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = ADE20KDataset(
        root_dir=root_dir,
        split=split,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
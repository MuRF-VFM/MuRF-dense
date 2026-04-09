import os

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import cv2
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A


class SUNRGBDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split_file,
                 split='test',
                 depth_range=(1e-3, 10),
                 transform=None):
        """
        work on
        """
        if split not in ['val', 'test']:
            raise ValueError("Split must be in 'test' or 'val'")
        self.root_dir = root_dir
        self.split = split
        self.depth_range = depth_range
        self.transform = transform

        self.image_files = []
        self.depth_files = []

        self.pipeline = self._make_pipeline(flip=False)
        self.flip_pipeline = self._make_pipeline(flip=True)

        self._read_split_file(split_file)

    def _make_pipeline(self, flip=False):
        mean = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
        std = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

        # define augmentations
        if self.split == 'val':
            pipeline = [
                A.Resize(height=480, width=640),
                A.HorizontalFlip(p=0.5),
            ]
        elif self.split == 'test':
            pipeline = [
                A.Resize(height=480, width=640),
            ]
        else:
            raise ValueError("Split must be in 'val' or 'test'")

        if flip:
            pipeline.append(A.HorizontalFlip(p=1.0))
        pipeline.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        return A.Compose(pipeline, additional_targets={'depth': 'mask'})

    def _read_split_file(self, split_file: str):
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, depth_path = line.strip().split()
                self.image_files.append(self.root_dir + '/' + img_path)
                self.depth_files.append(self.root_dir + '/' + depth_path)

        # check first file exists
        assert os.path.exists(
            self.image_files[0]), f"File {self.image_files[0]} does not exist"
        assert os.path.exists(
            self.depth_files[0]), f"File {self.depth_files[0]} does not exist"

    def __getitem__(self, idx):
        # load image and mask and normalize
        pil_image = Image.open(self.image_files[idx]).convert('RGB')
        pil_depth = Image.open(self.depth_files[idx])

        # numpy
        image = np.array(pil_image).astype(np.uint8)
        depth = np.array(pil_depth).astype(np.float32) / 8000.0
        original_depth = torch.from_numpy(depth).unsqueeze(0)


        transformed = self.pipeline(image=image, depth=depth)
        image_tensor = transformed['image']
        depth_result = transformed['depth']

        if self.split == 'test':
            out_flip = self.flip_pipeline(image=image, depth=depth)
            flip_image_tensor = out_flip['image']
            flip_depth_tensor = out_flip['depth']

            if flip_depth_tensor.ndim == 2:
                flip_depth_tensor = flip_depth_tensor.unsqueeze(0)
        else:
            flip_image_tensor = image_tensor
            flip_depth_tensor = depth_tensor
        # ensure depth is tensor
        if isinstance(depth_result, np.ndarray):
            depth_tensor = torch.from_numpy(depth_result)
        else:
            depth_tensor = depth_result

        # ensure depth is [1, H, W]
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        return {
            'image': image_tensor,
            'image_flip': flip_image_tensor,
            'depth': depth_tensor,
            'depth_flip': flip_depth_tensor,
            'img_metas': {
                'image_path': self.image_files[idx],
                'depth_path': self.depth_files[idx]
            },
            'original_depth': original_depth
        }

    def __len__(self):
        return len(self.image_files)


def get_sunrgb_dataloader(root_dir="/nobackup/shared/datasets",
                          split_file="./dataset/SUNRGBD_val_splits.txt",
                          batch_size=8,
                          split='val',
                          num_workers=4,
                          shuffle=True,
                          persistent_workers=True,
                          use_list=False
                          ):
    """
    Create a DataLoader for NYU Depth dataset

    Args:
        root_dir (string): Path to the NYU Depth dataset
        batch_size (int): Batch size
        split (string): Dataset split ('train_mrf' or 'test')
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        persistent_workers (bool): Whether to use persistent workers in DataLoader

    Returns:
        dataloader: PyTorch dataloader for the ADE20K dataset
    """
    dataset = SUNRGBDepthDataset(
        root_dir=root_dir,
        split_file=split_file,
        split=split,
    )

    def collate_fn(inputs):
        batch = dict()
        batch["pixel_values"] = torch.stack([i['image'] for i in inputs], dim=0) if not use_list else [i['image'] for i in inputs]
        batch["pixel_values_flip"] = torch.stack([i['image_flip'] for i in inputs], dim=0) if not use_list else [i['image_flip'] for i in inputs]
        batch["labels"] = torch.stack([i['depth'] for i in inputs], dim=0) if not use_list else [i['depth'] for i in inputs]
        batch["labels_flip"] = torch.stack([i['depth_flip'] for i in inputs], dim=0) if not use_list else [i['depth_flip'] for i in inputs]
        batch["img_metas"] = [i['img_metas'] for i in inputs]
        batch["original_depths"] = torch.stack([i['original_depth'] for i in inputs], dim=0)

        return batch

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers
    )

    return dataloader


if __name__ == "__main__":
    dataset = SUNRGBDepthDataset(root_dir="/nobackup/shared/datasets",
                                 split_file="./SUNRGBD_val_splits.txt",
                                 split='test')

    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample['image'].shape, sample['depth'].shape,
              sample['img_metas'])


import os

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from depth_estimation.dataset.augmentations import (ColorAug,
                                                AlbumentationsColorAug
                                                )

import albumentations as A
import cv2
import numpy as np


class NYUDepthDataset(Dataset):
    """
    Dataset class for NYU Depth V2 dataset.
    """

    def __init__(self,
                 root_dir,
                 split='train',
                 depth_range=(1e-3, 10),
                 ):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            split (string): 'train' or 'test' split
            depth_range (tuple): Min and max depth values range of dataset
            pad_to_divisible (bool): Whether to pad images to be divisible by 14
        """
        self.root_dir = root_dir
        self._validate_root_dir()

        self.split = split
        self.depth_range = depth_range

        # create two pipelines: one for normal, one for flip tta
        self.pipeline = self._make_pipeline()
        self.flip_pipeline = self._make_pipeline(flip=True)

        self.image_files = []
        self.depths_files = []
        self.focal_lengths = []

        if split == 'test' or split == 'val':
            self._read_split_file(self.root_dir + '/nyu_test.txt')
        elif split == 'train':
            self._read_split_file(self.root_dir + '/nyu_train.txt')
        else:
            raise ValueError("Split must be 'train' or 'test'")

    def _validate_root_dir(self):
        if list(os.listdir(self.root_dir)) == ['nyu']:
            self.root_dir = os.path.join(self.root_dir, 'nyu')
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"Root directory {self.root_dir} does not exist. {os.listdir(os.getcwd())}")
        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(
                f"Root directory {self.root_dir} is not a directory.")
        if not os.path.exists(os.path.join(self.root_dir, 'nyu_train.txt')) or \
                not os.path.exists(
                    os.path.join(self.root_dir, 'nyu_test.txt')):
            raise FileNotFoundError(
                f"Split files 'nyu_train.txt' or 'nyu_test.txt' not found in the root directory. Found files: {list(os.listdir(self.root_dir))[:10]}")

    def _read_split_file(self, split_file: str):
        """
        Reads the split file and populates the image and depth file lists.
        """
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, depth_path, focal_length = line.strip().split()
                self.image_files.append(self.root_dir + '/' + img_path)
                self.depths_files.append(self.root_dir + '/' + depth_path)
                self.focal_lengths.append(float(focal_length))

    def _make_pipeline(self, flip=False):
        mean = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
        std = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

        pipeline = []
        if self.split == 'train':
            pipeline = [
                A.Crop(x_min=43, y_min=45, x_max=608, y_max=472),
                # Standard NYU center crop

                A.Rotate(
                    limit=2.5, 
                    p=0.5, 
                    interpolation=cv2.INTER_LINEAR, 
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0, 
                    fill_mask=0 
                ),

                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=416, width=544),

                AlbumentationsColorAug(
                    p=0.5,
                    gamma_range=[0.9, 1.1],
                    brightness_range=[0.75, 1.25],
                    color_range=[0.9, 1.1]),
            ]
        elif self.split in 'val':
            pipeline = [
                A.HorizontalFlip(p=0.5),
            ]
        elif self.split == 'test':
            pipeline = [
                A.Resize(height=480, width=640),
            ]
        if flip:
            pipeline.append(A.HorizontalFlip(p=1.0))
        pipeline.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        return A.Compose(pipeline, additional_targets={'depth': 'mask'})

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load image and mask and normalize
        pil_image = Image.open(self.image_files[idx]).convert('RGB')
        pil_depth = Image.open(self.depths_files[idx])

        # numpy
        image = np.array(pil_image).astype(np.uint8)
        depth = np.array(pil_depth).astype(np.float32) / 1000.0
        original_depth = torch.from_numpy(depth).unsqueeze(0)

        if self.pipeline:
            transformed = self.pipeline(image=image, depth=depth)
            image_tensor = transformed['image']
            depth_tensor = transformed['depth']


            if depth_tensor.ndim == 2:
                depth_tensor = depth_tensor.unsqueeze(0)

        else:
            image_tensor = torch.from_numpy(image).permute(2, 0,
                                                           1).float() / 255.0
            depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        if self.split == 'test':
            out_flip = self.flip_pipeline(image=image, depth=depth)
            flip_image_tensor = out_flip['image']
            flip_depth_tensor = out_flip['depth']

            if flip_depth_tensor.ndim == 2:
                flip_depth_tensor = flip_depth_tensor.unsqueeze(0)
        else:
            flip_image_tensor = image_tensor
            flip_depth_tensor = depth_tensor

        return {
            'image': image_tensor,
            'image_flip': flip_image_tensor,
            'depth': depth_tensor,
            'depth_flip': flip_depth_tensor,
            'focal': self.focal_lengths[idx],
            'img_metas': {
                'image_path': self.image_files[idx],
                'depth_path': self.depths_files[idx],
            },
            'original_depth': original_depth
        }


def get_nyu_dataloader(root_dir,
                       batch_size=8,
                       split='train',
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
        dataloader: PyTorch dataloader for the NYU depth dataset
    """
    dataset = NYUDepthDataset(
        root_dir=root_dir,
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
    # Ensure imports are available for the main block
    import matplotlib.pyplot as plt
    import torch
    import os

    # Configuration
    # ROOT_DIR = "/nobackup/shared/datasets/nyu" # Use your actual path
    ROOT_DIR = "/nobackup/shared/datasets/nyu"
    OUTPUT_DIR = "dataset_check"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Checking Root Directory: {ROOT_DIR} ---")
    if not os.path.exists(ROOT_DIR):
        print(
            f"WARNING: Directory {ROOT_DIR} not found. functionality cannot be fully verified.")
        # Create dummy data for flow testing if path doesn't exist (Optional)
    else:
        print("Root directory found.")

    # Define inverse normalization for visualization
    # Matches values in __init__
    MEAN = torch.tensor(
        [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]).view(3, 1, 1)
    STD = torch.tensor([58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]).view(3,
                                                                             1,
                                                                             1)


    def denormalize(tensor):
        return (tensor.cpu() * STD + MEAN).clamp(0, 1)


    # ==========================================
    # 1. Verify Test Split (TTA Logic)
    # ==========================================
    print("\n--- 1. Checking Test Dataset (TTA Shapes) ---")
    try:
        test_dataset = NYUDepthDataset(root_dir=ROOT_DIR, split='test')
        if len(test_dataset) > 0:
            example = test_dataset[0]
            img_shape = example['image'].shape  # Expect [2, 3, H, W]
            depth_shape = example['depth'].shape  # Expect [2, 1, H, W]

            print(f"[Item 0] Image: {img_shape} | Depth: {depth_shape}")

            if img_shape[0] == 2 and isinstance(example['image'],
                                                torch.Tensor):
                print("[PASS] TTA logic returning stacked tensors.")
            else:
                print(
                    f"[FAIL] Expected dim 0=2 (Flip TTA) and Tensor type. Got {img_shape} {type(example['image'])}")

            print("\n--- 2. Checking Test DataLoader ---")
            test_loader = get_nyu_dataloader(
                root_dir=ROOT_DIR,
                split='test',
                batch_size=2,  # Small batch for debugging
                shuffle=False
            )

            for i, batch in enumerate(test_loader):
                # Shape: [B, TTA_Views, C, H, W] -> [2, 2, 3, 480, 640]
                print(f"Batch {i} Pixel Values: {batch['pixel_values'].shape}")
                print(f"Batch {i} Labels:       {batch['labels'].shape}")
                break
        else:
            print("Test dataset is empty.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Test Split Error: {e}")

    # ==========================================
    # 2. Verify Train Split (Augmentations)
    # ==========================================
    print("\n--- 3. Checking Train DataLoader & Visualization ---")
    try:
        train_loader = get_nyu_dataloader(
            root_dir=ROOT_DIR,
            split='train',
            # Ensure this matches your __init__ logic ('train' vs 'train_mrf')
            batch_size=4,
            shuffle=True,
        )

        for i, batch in enumerate(train_loader):
            images = batch['pixel_values']  # [B, 3, H, W]
            depths = batch['labels']  # [B, 1, H, W]

            print(f"Batch {i}: Image {images.shape} | Depth {depths.shape}")

            # Validate Data Types
            if not isinstance(images, torch.Tensor) or not isinstance(depths,
                                                                      torch.Tensor):
                print(
                    f"[FAIL] Batch contains non-tensors! Img: {type(images)}, Depth: {type(depths)}")
                break

            # Visualize the first image in the batch
            if i == 0:
                viz_img = denormalize(images[0]).permute(1, 2, 0).numpy()
                viz_depth = depths[0].cpu().squeeze(0).numpy()

                # Handle original depths logic if present
                if 'original_depths' in batch and len(
                        batch['original_depths']) > 0:
                    orig_depth = batch['original_depths'][0].cpu().squeeze(
                        0).numpy()
                else:
                    orig_depth = np.zeros_like(viz_depth)

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.title(f"Augmented Image\n{images.shape[-2:]}")
                plt.imshow(viz_img)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title(f"Augmented Depth\n{depths.shape[-2:]}")
                plt.imshow(viz_depth, cmap='inferno')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title(f"Original Depth\n{orig_depth.shape}")
                plt.imshow(orig_depth, cmap='inferno')
                plt.axis('off')

                save_path = os.path.join(OUTPUT_DIR, "train_viz_debug.png")
                plt.savefig(save_path)
                print(f"Saved visualization to {save_path}")

            if i >= 1: break  # Only check 2 batches

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Train Split Error: {e}")

    # ==========================================
    # 3. Verify Val Split
    # ==========================================
    print("\n--- 4. Checking Validation DataLoader ---")
    try:
        val_loader = get_nyu_dataloader(
            root_dir=ROOT_DIR,
            split='val',
            batch_size=4,
            shuffle=False,
        )

        for i, batch in enumerate(val_loader):
            print(
                f"Batch {i}: Image {batch['pixel_values'].shape} | Depth {batch['labels'].shape}")
            if i >= 1: break

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Val Split Error: {e}")

    print("\n--- Checks Complete ---")

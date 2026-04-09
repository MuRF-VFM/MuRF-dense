import os
from pathlib import Path

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .aug import ContentAwareRandomCrop, AlbumentationsPhotoMetricDistortion

import albumentations as A
from albumentations.pytorch import ToTensorV2

class PascalVOCSegmentation(Dataset):
    """
    Dataset class for PASCAL VOC segmentation dataset.
    """

    def __init__(self,
                 root_dir: str | Path,
                 split: str | list[str] ='train',
                 ):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            split (string): 'train' or 'test' split
        """
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        self.split = split
        valid_splits = ['train', 'val', 'trainval', 'aug', 'trainaug']

        self.image_files = None
        self.annotation_files = None

        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)

        self.pipeline = []
        if split in ['train', 'trainaug']:
            self.pipeline = [
                # sopme augmenations to try and recreate the mmcv resize operation
                A.SmallestMaxSize(max_size=512, p=1.0),
                A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, fill=0, fill_mask=255),

                # the random crop. Can be content aware (tries to avoid single class crops)
                # A.RandomCrop(height=512, width=512)
                ContentAwareRandomCrop(512, 512, cat_max_ratio=0.75, ignore_index=255, retry_count=10),

                # final augmentations
                A.HorizontalFlip(p=0.5),
                AlbumentationsPhotoMetricDistortion(), # recreated from mmcv
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=1.0
                ),
                A.PadIfNeeded(512, 512, border_mode=0, fill=0, fill_mask=255),
                ToTensorV2()
            ]
            self.pipeline = A.Compose(self.pipeline, additional_targets={'mask': 'mask'})
        elif split in ['val']:
            # replicates the multiscaleflipaug transforms, without the flip
            self.pipeline = [
                A.Resize(512, 512),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=1.0
                ),
                A.PadIfNeeded(512, 512, border_mode=0, fill=0, fill_mask=255),
                ToTensorV2()
            ]
            self.pipeline = A.Compose(self.pipeline, additional_targets={'mask': 'mask'})
        else:
           raise NotImplementedError("Test split handling not implemented yet.")

        self.image_files = None
        self.annotation_files = None

        self._read_split_file()

    def _read_split_file(self):
        """
        Reads the VOC split file and populates the image and annotation lists.
        """
        self.image_files = []
        self.annotation_files = []
        splits = [self.split] if self.split != 'trainaug' else ['train', 'aug']

        for split in splits:
            split_file = os.path.join(self.root_dir, 'ImageSets', 'Segmentation',
                                      f'{split}.txt')
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with open(split_file, 'r') as f:
                ids = [line.strip() for line in f.readlines() if line.strip()]

            img_dir = os.path.join(self.root_dir, 'JPEGImages')
            ann_dir = os.path.join(self.root_dir, 'SegmentationClassAug' if split == 'aug' else 'SegmentationClass')

            self.image_files.extend([os.path.join(img_dir, f"{img_id}.jpg") for img_id
                                in ids])
            self.annotation_files.extend([os.path.join(ann_dir, f"{img_id}.png") for
                                     img_id in ids])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load image and mask and normalize
        pil_image = Image.open(self.image_files[idx]).convert('RGB')
        pil_depth = Image.open(self.annotation_files[idx])

        # numpy
        image = np.array(pil_image).astype(np.uint8)
        seg_field = np.array(pil_depth).astype(np.float32)

        # augmentations
        if self.pipeline:
            augmented = self.pipeline(image=image, mask=seg_field)
            image = augmented['image']
            seg_field = augmented['mask']

        img_metas = {
            'image_path': self.image_files[idx],
            'depth_path': self.annotation_files[idx],
            'ori_shape': image.shape[:2]
        }

        # return
        return {
            'image': image,
            'gt_seg_map': seg_field.unsqueeze(0),
            'img_metas': img_metas,
        }

def get_voc_dataloader(root_dir,
                       batch_size=8,
                       split='train',
                       num_workers=4,
                       shuffle=True,
                       persistent_workers=True,
                       use_list=False
                       ):
    """
    Create a DataLoader for VOC dataset
    """
    dataset = PascalVOCSegmentation(
        root_dir=root_dir,
        split=split,
    )

    def collate_fn(inputs):
        batch = dict()
        batch["pixel_values"] = torch.stack([i['image'] for i in inputs], dim=0) if not use_list else [i['image'] for i in inputs]
        batch["labels"] = torch.stack([i['gt_seg_map'] for i in inputs], dim=0) if not use_list else [i['depth'] for i in inputs]
        batch["img_metas"] = [i['img_metas'] for i in inputs]

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
    import matplotlib.pyplot as plt
    import os

    path = Path("/home/ubuntu/coding/datasets/VOC2012")
    save_dir = Path("images")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Number of samples to visualize
    i = 10

    # Create dataloader
    train_loader = get_voc_dataloader(root_dir=path, split='val', batch_size=1, shuffle=True)

    for idx, batch in enumerate(train_loader):
        if idx >= i:
            break

        img_tensor = batch["pixel_values"][0]
        seg_tensor = batch["labels"][0][0]
        print(img_tensor.shape)
        print(seg_tensor.shape)

        # Convert to numpy
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        seg = seg_tensor.numpy().astype(np.int32)

        # Normalize seg map to [0, 1] for visualization
        seg_vis = seg / seg.max() if seg.max() > 0 else seg

        # Overlay: blend image and segmentation mask
        overlay = img.copy()
        overlay[..., 0] = 0.5 * overlay[..., 0] + 0.5 * seg_vis  # red channel blend

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[1].imshow(seg_vis, cmap="nipy_spectral")
        axs[1].set_title("Segmentation Map")
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_dir / f"sample_{idx:03d}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {i} visualization samples to {save_dir.resolve()}")


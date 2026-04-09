import random
from typing import Any, Callable, List, Sequence, Optional

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

import albumentations as A

from transformers import AutoImageProcessor, AutoModel

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)

class SegmentationTransform(BaseTransform):
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        crop_size: int = 518,
        mean: Sequence[float] = (123.675 / 255, 116.280 / 255, 103.530 / 255),
        std: Sequence[float] = (58.395 / 255, 57.120 / 255, 57.375 / 255),
        augment: bool = True,
    ):
        # based on DINOv2 config and Perplexity.ai's explanations
        # of the mmseg documentation
        if augment:
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=crop_size),
                    A.RandomCrop(crop_size, crop_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=mean, std=std),
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                    ),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=mean, std=std),
                ]
            )

class SegmentationTransformSiglip(BaseTransform):
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        crop_size: int = 518,
        mean: Sequence[float] = (123.675 / 255, 116.280 / 255, 103.530 / 255),
        std: Sequence[float] = (58.395 / 255, 57.120 / 255, 57.375 / 255),
        augment: bool = True,
        model_name: str = "google/siglip2-base-patch16-naflex",
    ):
        
        self.processor = AutoImageProcessor.from_pretrained(model_name, max_num_patches = crop_size // 16 * crop_size // 16)

        # based on DINOv2 config and Perplexity.ai's explanations
        # of the mmseg documentation
        if augment:
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=crop_size),
                    A.RandomCrop(crop_size, crop_size),
                    A.HorizontalFlip(p=0.5),
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                    ),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(crop_size, crop_size),
                    # A.Normalize(mean=mean, std=std),
                ]
            )

class SegmentationTransformSiglipMRF(BaseTransform):
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        mean: Sequence[float] = (123.675 / 255, 116.280 / 255, 103.530 / 255),
        std: Sequence[float] = (58.395 / 255, 57.120 / 255, 57.375 / 255),
        augment: bool = True,
        model_name: str = "google/siglip2-base-patch16-naflex",
    ):
        crop_size = 768
        self.processor = [AutoImageProcessor.from_pretrained(model_name, max_num_patches = 256 // 16 * 256 // 16)
        ,AutoImageProcessor.from_pretrained(model_name, max_num_patches = 512 // 16 * 512 // 16),
        AutoImageProcessor.from_pretrained(model_name, max_num_patches = 768 // 16 * 768 // 16)]

        # based on DINOv2 config and Perplexity.ai's explanations
        # of the mmseg documentation
        if augment:
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=crop_size),
                    A.RandomCrop(crop_size, crop_size),
                    A.HorizontalFlip(p=0.5),
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                    ),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(crop_size, crop_size),
                    # A.Normalize(mean=mean, std=std),
                ]
            )

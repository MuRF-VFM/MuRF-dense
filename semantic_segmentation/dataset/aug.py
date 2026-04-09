import albumentations
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import collections.abc
from typing import Callable
import random
import numpy as np
import albumentations as A

class ContentAwareRandomCrop(A.RandomCrop):
    """
    Content-Aware Random Crop, attempts to reproduce the behavior of mmcv's random 
    crop by limiting the maximum ratio of any single category in the crop.
    """

    def __init__(
        self, 
        height, 
        width, 
        cat_max_ratio=1.0, 
        ignore_index=255, 
        retry_count=10, 
        p=1.0
    ):
        super(ContentAwareRandomCrop, self).__init__(height=height, width=width, p=p)
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.retry_count = retry_count

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        mask = params['mask']
        h, w = img.shape[:2]
        
        # crop margins
        margin_h = max(h - self.height, 0)
        margin_w = max(w - self.width, 0)

        # iterates through possible crops
        for _ in range(self.retry_count):
            # candidate crop coordinates
            y_min = random.randint(0, margin_h)
            x_min = random.randint(0, margin_w)
            y_max = y_min + self.height
            x_max = x_min + self.width
            
            # default pass if cat_max_ratio is 1.0
            if self.cat_max_ratio >= 1.0:
                return {'crop_coords': (x_min, y_min, x_max, y_max)}

            # check 
            check_y_max = min(y_max, h)
            check_x_max = min(x_max, w)
            
            mask_crop = mask[y_min:check_y_max, x_min:check_x_max]
            labels, cnt = np.unique(mask_crop, return_counts=True)
            
            # ignore index
            valid_mask = labels != self.ignore_index
            cnt = cnt[valid_mask]
            
            # check max ratio
            if len(cnt) > 0:
                max_ratio = np.max(cnt) / np.sum(cnt)
                if max_ratio < self.cat_max_ratio:
                    return {'crop_coords': (x_min, y_min, x_max, y_max)}

        # if we reach here we failed and will just return a random crop
        return super().get_params()

    @property
    def targets_as_params(self):
        return ['image', 'mask']

    def get_transform_init_args_names(self):
        return ('height', 'width', 'cat_max_ratio', 'ignore_index', 'retry_count')

class AlbumentationsPhotoMetricDistortion(A.Compose):
    """
    Apply photometric distortion to image sequentially, replicating the logic of 
    mmcv/mmseg's PhotoMetricDistortion.
    """
    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: tuple[float, float] = (0.5, 1.5),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18,
        p: float = 1.0,
        **kwargs
    ):
        # convert mmcv default paramss to albumentations format
        bright_limit = brightness_delta / 255.0
        contrast_limit = (contrast_range[1] - contrast_range[0]) / 2.0
        sat_limit = (saturation_range[1] - saturation_range[0]) / 2.0
        hue_limit = hue_delta

        # define transforms
        
        t_brightness = A.RandomBrightnessContrast(
            brightness_limit=bright_limit, 
            contrast_limit=0, 
            p=0.5
        )
        
        t_contrast = A.RandomBrightnessContrast(
            brightness_limit=0, 
            contrast_limit=contrast_limit, 
            p=0.5
        )

        t_saturation = A.ColorJitter(
            brightness=0, 
            contrast=0, 
            saturation=sat_limit, 
            hue=0, 
            p=0.5
        )
        
        t_hue = A.HueSaturationValue(
            hue_shift_limit=hue_limit, 
            sat_shift_limit=0, 
            val_shift_limit=0, 
            p=0.5
        )

        # construct the final transforms
        transforms = [
            t_brightness,
            
            # the original logic randomly chooses contrast as being first or last
            A.OneOf([
                A.Compose([
                    t_contrast,
                    t_saturation,
                    t_hue
                ]),
                A.Compose([
                    t_saturation,
                    t_hue,
                    t_contrast
                ])
            ], p=1.0)
        ]

        super(AlbumentationsPhotoMetricDistortion, self).__init__(transforms=transforms, p=p, **kwargs)

    def get_transform_init_args_names(self):
        return ("brightness_delta", "contrast_range", "saturation_range", "hue_delta")


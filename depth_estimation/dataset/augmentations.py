"""
Albumentations based augmentations for depth estimation.

These augmentations are recreated based off of the ones used from
the dinov2 mmcv repo. We ensure that core functionality remains
identical.
"""

import albumentations
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import collections.abc
from typing import Callable
import torch
import numpy as np
import os.path as osp
from numpy.core.fromnumeric import shape

class NYUCrop(object):
    """NYU standard rop when training monocular depth estimation on NYU dataset.

    Args:
        depth (bool): Whether apply NYUCrop on depth map. Default: False.
    """

    def __init__(self, depth=False):
        self.depth = depth

    def __call__(self, results):
        """Call function to apply NYUCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """

        if self.depth:
            depth_cropped = results["depth_gt"][45:472, 43:608]
            results["depth_gt"] = depth_cropped
            results["depth_shape"] = results["depth_gt"].shape

        img_cropped = results["img"][45:472, 43:608, :]
        results["img"] = img_cropped
        results["ori_shape"] = img_cropped.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class KBCrop(object):
    """KB standard krop when training monocular depth estimation on KITTI dataset.

    Args:
        depth (bool): Whether apply KBCrop on depth map. Default: False.
        height (int): Height of input images. Default: 352.
        width (int): Width of input images. Default: 1216.

    """

    def __init__(self, depth=False, height=352, width=1216):
        self.depth = depth
        self.height = height
        self.width = width

    def __call__(self, results):
        """Call function to apply KBCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """
        # print(results)
        height = results["img_shape"][0]
        width = results["img_shape"][1]
        top_margin = int(height - self.height)
        left_margin = int((width - self.width) / 2)

        if self.depth:
            depth_cropped = results["depth_gt"][top_margin:top_margin +
                                                           self.height,
                            left_margin:left_margin +
                                        self.width]
            results["depth_gt"] = depth_cropped
            results["depth_shape"] = results["depth_gt"].shape

        img_cropped = results["img"][top_margin:top_margin + self.height,
                      left_margin:left_margin + self.width, :]
        results["img"] = img_cropped
        results["ori_shape"] = img_cropped.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



class ColorAug(object):
    """Color augmentation used in depth estimation

    Args:
        prob (float, optional): The color augmentation probability. Default: None.
        gamma_range(list[int], optional): Gammar range for augmentation. Default: [0.9, 1.1].
        brightness_range(list[int], optional): Brightness range for augmentation. Default: [0.9, 1.1].
        color_range(list[int], optional): Color range for augmentation. Default: [0.9, 1.1].
    """

    def __init__(self,
                 prob=None,
                 gamma_range=[0.9, 1.1],
                 brightness_range=[0.9, 1.1],
                 color_range=[0.9, 1.1]):
        self.prob = prob
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        """Call function to apply color augmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly colored results.
        """
        aug = True if np.random.rand() < self.prob else False

        if aug:
            image = results['img']

            # gamma augmentation
            gamma = np.random.uniform(min(*self.gamma_range),
                                      max(*self.gamma_range))
            image_aug = image ** gamma

            # brightness augmentation
            brightness = np.random.uniform(min(*self.brightness_range),
                                           max(*self.brightness_range))
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(min(*self.color_range),
                                       max(*self.color_range),
                                       size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)],
                                   axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 255)

            results['img'] = image_aug

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


class AlbumentationsColorAug(A.ImageOnlyTransform):
    """
    Custom Albumentations transform for NYU Depth V2 color augmentation.
    Replicates gamma, brightness, and channel-wise color scaling from old
    mmcv depth estimation code.
    """
    def __init__(
            self,
            gamma_range=[0.9, 1.1],
            brightness_range=[0.75, 1.25],
            color_range=[0.9, 1.1],
            always_apply=False,
            p=0.5,
    ):
        super(AlbumentationsColorAug, self).__init__(always_apply, p)
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.color_range = color_range

    def get_params(self):
        return {
            "gamma": np.random.uniform(self.gamma_range[0], self.gamma_range[1]),
            "brightness": np.random.uniform(self.brightness_range[0], self.brightness_range[1]),
            "colors": np.random.uniform(self.color_range[0], self.color_range[1], size=3)
        }

    def apply(self, img, **params):
        gamma = params["gamma"]
        brightness = params["brightness"]
        colors = params["colors"]

        img_float = img.astype(np.float32)
        
        # transforms
        img_aug = np.power(img_float, gamma)
        img_aug = img_aug * brightness
        img_aug = img_aug * colors

        # clip
        return np.clip(img_aug, 0, 255)

    def get_transform_init_args_names(self):
        return ("gamma_range", "brightness_range", "color_range")
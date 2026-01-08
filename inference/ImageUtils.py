# -*- coding: utf-8 -*-

import numpy as np


class ImageUtils(object):

    @staticmethod
    def fillBlackBorderWithRandomNoise(image=np.ndarray((0, 0, 0)), mean=193):
        # Handle 3D volumes (depth, height, width) or 2D images (height, width)
        if len(image.shape) == 3:
            # 3D volume: process each slice separately
            depth, height, width = image.shape
            # Use the middle slice to create the mask
            middle_slice_idx = depth // 2
            mask_image = np.ones((height, width))
            mask = np.logical_not(np.logical_and(image[middle_slice_idx, :, :], mask_image))
            
            # Apply the same mask to all slices
            for slice_idx in range(depth):
                slice_mask = mask
                noise_array = np.full(image[slice_idx, slice_mask].shape, mean, np.uint8)
                image[slice_idx, slice_mask] = noise_array
        else:
            # 2D image: original behavior
            height, width = image.shape[:2]
            mask_image = np.ones((height, width))
            mask = np.logical_not(np.logical_and(image[:, :], mask_image))
            noise_array = np.full(image[mask].shape, mean, np.uint8)
            image[mask] = noise_array
        return image

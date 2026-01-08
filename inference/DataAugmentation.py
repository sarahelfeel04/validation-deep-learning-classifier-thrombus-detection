# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:56:53 2019

@author: mittmann
"""

import cv2
import numpy as np

import CustomTransforms
import albumentations as albu
import matplotlib.pyplot as plt
import torch
from IndexTracker import IndexTracker
import copy

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

'''Augments given data, which is stored in a dictionary'''


class DataAugmentation(object):
    MAX_SERIES_LENGTH = 62
    MAX_KEYPOINT_LENGTH = 5
    '''====================================================================='''

    def __init__(self, data=dict()):
        self.data = data
        self.transform = None
        self.tracker1 = None
        self.tracker2 = None
        self.tracker = None

    '''====================================================================='''

    def getTransform(self):
        return self.transform

    '''====================================================================='''

    def createTransformTraining(self):
        if not self.data:
            return

        if self.data['frontalAndLateralView']:
            self.transform = albu.Compose(
                [
                    albu.VerticalFlip(p=0.5),
                    albu.ShiftScaleRotate(p=1,
                                          shift_limit=0.15,
                                          scale_limit=(0, 0.1),
                                          rotate_limit=20,
                                          interpolation=cv2.INTER_LINEAR,
                                          border_mode=cv2.BORDER_CONSTANT,
                                          value=(193, 193, 193, 193)),  # previously: 0.757 for float32
                    CustomTransforms.Rotate90(p=1),  # always p=1
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
                    albu.RandomBrightnessContrast(p=0.2,
                                                  brightness_limit=0.2,
                                                  contrast_limit=0.2,
                                                  brightness_by_max=False),
                    albu.MedianBlur(p=0.5, blur_limit=3),
                    albu.OneOf(
                        [
                            albu.MultiplicativeNoise(p=0.2,
                                                     multiplier=(0.9, 1.1),
                                                     per_channel=True,
                                                     elementwise=True),
                            albu.Downscale(p=0.3, scale_min=0.5, scale_max=0.5)
                            # decreases image quality
                        ], p=0.5)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx'),
                additional_targets={'imageOtherView': 'image',
                                    'keypointsOtherView': 'keypoints'}
            )
        else:
            self.transform = albu.Compose(
                [
                    albu.VerticalFlip(p=0.5),
                    albu.ShiftScaleRotate(p=1,
                                          shift_limit=0.15,
                                          scale_limit=(0, 0.1),
                                          rotate_limit=20,
                                          interpolation=cv2.INTER_LINEAR,
                                          border_mode=cv2.BORDER_CONSTANT,
                                          value=(193, 193, 193, 193)),
                    CustomTransforms.Rotate90(p=1),
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
                    albu.RandomBrightnessContrast(p=0.2,
                                                  brightness_limit=0.2,
                                                  contrast_limit=0.2,
                                                  brightness_by_max=False),
                    albu.MedianBlur(p=0.5, blur_limit=3),
                    albu.OneOf(
                        [
                            albu.MultiplicativeNoise(p=0.2,
                                                     multiplier=(0.9, 1.1),
                                                     per_channel=True,
                                                     elementwise=True),
                            albu.Downscale(p=0.3, scale_min=0.5, scale_max=0.5)
                        ], p=0.5)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx')
            )

    '''====================================================================='''

    def createTransformValidation(self):
        if not self.data:
            return

        if self.data['frontalAndLateralView']:
            self.transform = albu.Compose(
                [
                    CustomTransforms.Rotate90(p=1),
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx'),
                additional_targets={'imageOtherView': 'image',
                                    'keypointsOtherView': 'keypoints'}
            )
        else:
            self.transform = albu.Compose(
                [
                    CustomTransforms.Rotate90(p=1),
                    albu.Resize(512, 512, interpolation=cv2.INTER_LINEAR)
                ], p=1,
                keypoint_params=albu.KeypointParams(format='yx')
            )

    '''====================================================================='''

    # @jit
    def applyTransform(self):
        if not self.transform:
            assert "Transform not yet created. Cannot apply transform."

        keypoints = copy.deepcopy(self.data['keypoints'])
        keypointsOtherView = copy.deepcopy(self.data['keypointsOtherView'])

        # Check if images are 3D volumes (depth, height, width)
        image = self.data['image']
        imageOtherView = self.data.get('imageOtherView')
        is_3d = len(image.shape) == 3
        
        if is_3d and imageOtherView is not None:
            # Handle 3D volumes: apply transform to each slice
            depth, height, width = image.shape
            depth_other = imageOtherView.shape[0]
            
            # Note: adjustSequenceLengthBeforeTransform should have already matched depths,
            # but if not, we'll handle it here by using the minimum depth
            min_depth = min(depth, depth_other)
            
            # Apply transform to each slice pair
            transformed_slices = []
            transformed_slices_other = []
            
            for slice_idx in range(min_depth):
                slice_img = image[slice_idx, :, :]
                slice_img_other = imageOtherView[slice_idx, :, :]
                
                # Apply transform to the slice pair
                transform_data = {
                    'image': slice_img,
                    'imageOtherView': slice_img_other,
                    'keypoints': keypoints,
                    'keypointsOtherView': keypointsOtherView
                }
                transformed = self.transform(**transform_data)
                
                transformed_slices.append(transformed['image'])
                transformed_slices_other.append(transformed['imageOtherView'])
            
            # Stack slices back into 3D volumes
            self.data['image'] = np.stack(transformed_slices, axis=0)
            self.data['imageOtherView'] = np.stack(transformed_slices_other, axis=0)
            
            # If depths were different, pad the shorter one (shouldn't happen if adjustSequenceLengthBeforeTransform worked)
            if depth != depth_other:
                if depth > depth_other:
                    # Pad imageOtherView
                    pad_depth = depth - depth_other
                    pad_shape = (pad_depth, transformed_slices_other[0].shape[0], transformed_slices_other[0].shape[1])
                    padding = np.full(pad_shape, 193, dtype=transformed_slices_other[0].dtype)
                    self.data['imageOtherView'] = np.concatenate([self.data['imageOtherView'], padding], axis=0)
                else:
                    # Pad image
                    pad_depth = depth_other - depth
                    pad_shape = (pad_depth, transformed_slices[0].shape[0], transformed_slices[0].shape[1])
                    padding = np.full(pad_shape, 193, dtype=transformed_slices[0].dtype)
                    self.data['image'] = np.concatenate([self.data['image'], padding], axis=0)
            self.data['keypoints'] = transformed.get('keypoints', keypoints)
            self.data['keypointsOtherView'] = transformed.get('keypointsOtherView', keypointsOtherView)
        else:
            # Original 2D image handling
            self.data = self.transform(**self.data)

        # Undo transformation of keypoints, if the keypoints indicated, that
        # there was no thrombus detected by the radiologist [(0,0)] or, that
        # the radiologist did not classify the image [(0,1)]:

        if (keypoints == [(0, 0)]) or (keypoints == [(0, 1)]):
            self.data['keypoints'] = keypoints
        if (keypointsOtherView == [(0, 0)]) or (keypointsOtherView == [(0, 1)]):
            self.data['keypointsOtherView'] = keypointsOtherView

        # Validate that frontal and lateral images have matching spatial dimensions
        if self.data['frontalAndLateralView']:
            if self.data['image'] is not None and self.data['imageOtherView'] is not None:
                img1_shape = self.data['image'].shape
                img2_shape = self.data['imageOtherView'].shape
                if len(img1_shape) >= 2 and len(img2_shape) >= 2:
                    # For 3D volumes, check spatial dimensions (last two dimensions)
                    # For 2D images, check first two dimensions
                    if is_3d:
                        if img1_shape[1] != img2_shape[1] or img1_shape[2] != img2_shape[2]:
                            raise ValueError(
                                f"Height and Width of image and imageOtherView do not match after transform: "
                                f"image shape {img1_shape} vs imageOtherView shape {img2_shape}"
                            )
                    else:
                        if img1_shape[0] != img2_shape[0] or img1_shape[1] != img2_shape[1]:
                            raise ValueError(
                                f"Height and Width of image and imageOtherView do not match after transform: "
                                f"image shape {img1_shape} vs imageOtherView shape {img2_shape}"
                            )

        return

    '''====================================================================='''

    def getTransformedData(self):
        return self.data

    '''====================================================================='''

    # @jit(nopython=True)
    def adjustSequenceLengthBeforeTransform(self):
        if not self.data:
            raise RuntimeError("Data has to be set before \
                               performing adjusting")

        if self.data['image'] is not None and self.data['imageOtherView'] is not None:
            # Handle both 3D volumes (depth, height, width) and old format (height, width, depth)
            img1_shape = self.data['image'].shape
            img2_shape = self.data['imageOtherView'].shape
            
            # Check if images are 3D volumes in (depth, height, width) format
            if len(img1_shape) == 3 and len(img2_shape) == 3:
                # New format: (depth, height, width)
                d1, h1, w1 = img1_shape
                d2, h2, w2 = img2_shape
                
                if d1 != d2:
                    if d1 > d2:
                        slices_to_add = d1 - d2
                        zeros = np.full((slices_to_add, h2, w2), 193, dtype=self.data['imageOtherView'].dtype)
                        self.data['imageOtherView'] = np.concatenate([self.data['imageOtherView'], zeros], axis=0)
                    else:
                        slices_to_add = d2 - d1
                        zeros = np.full((slices_to_add, h1, w1), 193, dtype=self.data['image'].dtype)
                        self.data['image'] = np.concatenate([self.data['image'], zeros], axis=0)
            else:
                # Old format: (height, width, depth) - original behavior
                x1, y1, z1 = img1_shape
                x2, y2, z2 = img2_shape

                if z1 != z2:
                    if z1 > z2:
                        slices_to_add = z1 - z2
                        zeros = np.full((x2, y2, slices_to_add), 193, dtype=np.uint8)
                        self.data['imageOtherView'] = np.append(self.data['imageOtherView'], zeros, axis=2)
                    else:
                        slices_to_add = z2 - z1
                        zeros = np.full((x1, y1, slices_to_add), 193, dtype=np.uint8)
                        self.data['image'] = np.append(self.data['image'], zeros, axis=2)
        return

    '''====================================================================='''

    # @jit(nopython=True)
    def zeroPaddingEqualLength(self):
        if not self.data:
            raise RuntimeError("Data has to be transformed first before \
                               performing zero padding")

        if self.data['image'] is not None:
            self.data['image'] = self.data['image'].astype(np.float32)

        if self.data['imageOtherView'] is not None:
            self.data['imageOtherView'] = self.data['imageOtherView'].astype(np.float32)

        if self.data['keypoints'] is not None:
            keypoints_to_add1 = self.MAX_KEYPOINT_LENGTH - len(self.data['keypoints'])
            for i in range(keypoints_to_add1):
                self.data['keypoints'].append((0, 0))

        if self.data['keypointsOtherView'] is not None:
            keypoints_to_add2 = self.MAX_KEYPOINT_LENGTH - len(self.data['keypointsOtherView'])
            for i in range(keypoints_to_add2):
                self.data['keypointsOtherView'].append((0, 0))

    '''====================================================================='''

    # @jit(nopython=True)
    def normalizeData(self):
        if self.data['image'] is not None:
            self.data['imageMean'] = np.mean(self.data['image'])
            self.data['imageStd'] = np.std(self.data['image'])
            try:
                self.data['image'] -= np.mean(self.data['image'], dtype=np.float32)
                self.data['image'] /= np.std(self.data['image'], dtype=np.float32)
            except ValueError as e:
                raise ValueError(
                    f"Error normalizing image with shape {self.data['image'].shape}: {e}"
                )

        if self.data['imageOtherView'] is not None:
            self.data['imageOtherViewMean'] = np.mean(self.data['imageOtherView'])
            self.data['imageOtherViewStd'] = np.std(self.data['imageOtherView'])
            try:
                self.data['imageOtherView'] -= np.mean(self.data['imageOtherView'], dtype=np.float32)
                self.data['imageOtherView'] /= np.std(self.data['imageOtherView'], dtype=np.float32)
            except ValueError as e:
                raise ValueError(
                    f"Error normalizing imageOtherView with shape {self.data['imageOtherView'].shape}: {e}. "
                    f"Image shape: {self.data['image'].shape if self.data['image'] is not None else 'None'}"
                )

    '''====================================================================='''

    # @jit(nopython=True)
    def convertToTensor(self):
        if not self.data:
            raise RuntimeError("Data has to be transformed first before \
                               converting ToTensor")

        # Check if images are 3D volumes (depth, height, width)
        image = self.data['image']
        is_3d = len(image.shape) == 3
        
        if is_3d:
            # For 3D volumes: convert (depth, height, width) to tensor
            # Model expects (batch, depth, height, width), so we keep depth as first dimension
            # No transpose needed - depth is already the first dimension
            self.data['image'] = torch.from_numpy(image.astype(np.float32))
            if self.data['imageOtherView'] is not None:
                self.data['imageOtherView'] = torch.from_numpy(self.data['imageOtherView'].astype(np.float32))
        else:
            # For 2D images: use original ToTensor transform
            transformToTensor = CustomTransforms.ToTensor()
            self.data['image'] = transformToTensor(self.data['image'])
            if self.data['imageOtherView'] is not None:
                self.data['imageOtherView'] = transformToTensor(self.data['imageOtherView'])
        
        self.data['keypoints'] = torch.tensor(self.data['keypoints'], dtype=torch.int32)
        if self.data['keypointsOtherView'] is not None:
            self.data['keypointsOtherView'] = torch.tensor(self.data['keypointsOtherView'], dtype=torch.int32)

        return self.data

    '''====================================================================='''

    def showAugmentedImages(self):

        if self.data['frontalAndLateralView']:

            fig1, ax1 = plt.subplots(1, 1)
            self.tracker1 = IndexTracker(ax1, self.data['image'], 'Image1', self.data['keypoints'])
            fig1.canvas.mpl_connect('scroll_event', self.tracker1.onscroll)
            plt.show()

            fig2, ax2 = plt.subplots(1, 1)
            self.tracker2 = IndexTracker(ax2, self.data['imageOtherView'], 'ImageOtherView',
                                         self.data['keypointsOtherView'])
            fig2.canvas.mpl_connect('scroll_event', self.tracker2.onscroll)
            plt.show()

        else:
            fig, ax = plt.subplots(1, 1)
            self.tracker = IndexTracker(ax, self.data['image'], 'Image', self.data['keypoints'])
            fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
            plt.show()

    def getImageData(self):
        # Handle both 3D volumes (depth, height, width) and 2D images (height, width, channels)
        image = self.data['image']
        imageOtherView = self.data['imageOtherView']
        
        if len(image.shape) == 3:
            # 3D volume: (depth, height, width) - get middle slice
            seq_length = image.shape[0]
            middle_slice = int(seq_length / 2)
            return np.copy(image[middle_slice, :, :]), np.copy(imageOtherView[middle_slice, :, :])
        else:
            # 2D image: (height, width, channels) - original behavior
            seq_length = image.shape[2]
            return np.copy(image[:, :, int(seq_length / 2)]), np.copy(
                imageOtherView[:, :, int(seq_length / 2)])

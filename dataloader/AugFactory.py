import logging
import importlib

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchio as tio
import cv2
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from torchvision import transforms
from torch.nn.functional import interpolate
from torch.nn.functional import pad
from dataloader.DistanceTransform import DistanceTransform
from dataloader.DistanceTransform import ProximityTransform

class ToPilImage:

    def __init__(self):
        pass

    def __call__(self, data):
        image = TF.to_pil_image(data[0])
        mask = TF.to_pil_image(data[1])
        return [image, mask]


# Already exists in tio: RandomFlip
class RandomHorizontalFlip:

    def __init__(self, execution_probability=0.5):
        self.execution_probability = execution_probability

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.execution_probability:
            image, mask = data
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            return [image, mask]
        return data


# Already exists in tio: RandomFlip
class RandomVerticalFlip:

    def __init__(self, execution_probability=0.5):
        self.execution_probability = execution_probability

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.execution_probability:
            image, mask = data
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
            return [image, mask]
        return data


class ToTensor:

    def __init__(self):
        pass

    def __call__(self, volume):
        volume = torch.from_numpy(volume)
        if volume.ndim == 2:
            volume = torch.unsqueeze(volume, dim=0)
        return volume


class RandomContrast:
    """
    increase the contrast of an image using https://www.sciencedirect.com/science/article/pii/B9780123361561500616
    NOT AFFECTING LABELS!
    Args:
        image (numpy array): 0-1 floating image
    Returns:
        result (numpy array): image with higer contrast
    """

    def __init__(self, alpha=(0.8, 2), execution_probability=0.1, **kwargs):
        assert len(alpha) == 2
        self.alpha = alpha
        self.execution_probability = execution_probability

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.execution_probability:
            image, mask = data

            # assert image.shape == mask.shape
            assert image.ndim == 3
            if image.max() > 1 or image.min() < 0:
                image, _ = Normalize()([image, mask])

            clip_limit = np.random.uniform(self.alpha[0], self.alpha[1])
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

            sharp = image * 255  # [0 - 255]
            sharp = sharp.astype(np.uint8)
            C = image.shape[0]
            for i in range(C):
                sharp[i] = clahe.apply(sharp[i])
            sharp = sharp.astype(np.float32) / 255  # back to [0-1]
            return [np.clip(sharp, 0, 1), mask]
        return data


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, angle_spectrum=8, mode='nearest', order=0, execution_probability=0.5, **kwargs):

        self.angle_spectrum = angle_spectrum
        self.mode = mode
        self.order = order
        self.execution_probability = execution_probability

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.execution_probability:
            angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
            image, mask = data
            # assert image.shape == mask.shape
            assert image.ndim == 3
            image = rotate(image, angle, axes=(0, 2), reshape=False, order=self.order, mode=self.mode)
            mask = rotate(mask, angle, axes=(0, 2), reshape=False, order=0, mode=self.mode)
            return [image, mask]
        return data


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, spline_order=2, alpha=1, sigma=5, execution_probability=0.1,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations (random -> between 0 and alfa)
        :param sigma: smoothing factor for Gaussian filter (random -> between 0 and sigma)
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.spline_order = spline_order
        self.alpha = np.random.uniform() * alpha
        self.sigma = np.random.uniform() * sigma
        self.execution_probability = execution_probability

    def deformate(self, volume, spline_order=0):
        y_dim, x_dim = volume[0].shape
        y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')

        C = volume.shape[0]
        for i in range(C):
            dy, dx = [
                gaussian_filter(
                    np.random.randn(*volume[0].shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]
            indices = y + dy, x + dx
            volume[i] = map_coordinates(volume[i], indices, order=spline_order, mode='reflect')
        return volume

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.execution_probability:
            image, mask = data

            # assert image.shape == mask.shape
            assert image.ndim == 3

            image = self.deformate(image.copy(), self.spline_order)
            mask = self.deformate(mask.copy(), 0)
            return [image, mask]
        return data


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [0, 1].
    DO NOT AFFECT MASKS!
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        image, mask = data
        value_range = image.max() - image.min()
        norm_0_1 = (image - image.min()) / value_range
        return [np.clip(norm_0_1, 0, 1), mask]


class Rescale:
    def __init__(self, scale_factor=None, size=None, interp_fn='trilinear', **kwargs):
        assert (scale_factor is not None) ^ (size is not None), "please specify a size OR a factor"
        self.scale_factor = scale_factor
        self.size = size
        self.interp_fn = interp_fn
        # using a custom function to avoid the align corner warnings/errors
        if self.interp_fn == 'nearest':
            self.scale_fn = lambda img:  interpolate(img, size=self.size, scale_factor=self.scale_factor, mode=self.interp_fn, recompute_scale_factor=False)
        else:
            self.scale_fn = lambda img: interpolate(img, size=self.size, scale_factor=self.scale_factor, mode=self.interp_fn, align_corners=False, recompute_scale_factor=False)

    def __call__(self, data):

        if self.scale_factor == 1:
            return data

        tensor_flag = torch.is_tensor(data)
        image = ToTensor()(data) if not tensor_flag else data

        assert image.ndim == 3

        image = image.unsqueeze(0).unsqueeze(0)
        image = self.scale_fn(image).squeeze()

        if not tensor_flag:
            return image.numpy()
        return image


class Resize:

    def __init__(self, labels, divisor, **kwargs):
        self.labels = labels
        self.divisor = divisor

    def closestDistanceForDivision(self, number):

        q = np.ceil(number / self.divisor).astype(np.int)
        # possible closest numbers
        n1 = self.divisor * q
        return n1

        # if you want to choose between the lower and upper bound use the following
        # n2 = (self.divisor * (q + 1))
        # choices = np.stack((number - n1, number - n2))
        # idx = np.argmin(np.abs(choices), axis=0)
        # return choices[idx, np.indices(idx.shape)[0]]

    def reshape(self, volume, new_shape, pad_val=0):

        target_Z, target_H, target_W = new_shape
        Z, H, W = volume.shape
        # if dest shape is bigger than current shape needs to pad
        H_pad = max(target_H - H, 0) // 2
        W_pad = max(target_W - W, 0) // 2
        Z_pad = max(target_Z - Z, 0) // 2
        # if dest shape is lower than current shape needs to crop
        H_crop = max(H - target_H, 0) // 2
        W_crop = max(W - target_W, 0) // 2
        Z_crop = max(Z - target_Z, 0) // 2

        if isinstance(volume, np.ndarray):
            result = np.full((target_Z, target_H, target_W), fill_value=pad_val, dtype=volume.dtype)
        else:
            result = torch.full((target_Z, target_H, target_W), fill_value=pad_val, dtype=volume.dtype)

        result[Z_pad:Z + Z_pad, H_pad:H + H_pad, W_pad:W + W_pad] = volume[Z_crop:target_Z + Z_crop, H_crop:target_H + H_crop, W_crop:target_W + W_crop]

        return result

    def __call__(self, data):
        image, mask = data

        # compatible with 1 label task
        # ref = self.labels['CONTOUR'] if 'CONTOUR' in self.labels else self.labels['INSIDE']
        #
        # up_b = np.max(np.argwhere(mask == ref), axis=0) + self.divisor // 2
        # low_b = np.min(np.argwhere(mask == ref), axis=0)
        # diff = self.closestDistanceForDivision(up_b - low_b)
        # up_b = up_b - diff
        #
        # image = image[low_b[0]:up_b[0], low_b[1]:up_b[1], low_b[2]:up_b[2]]
        # mask = mask[low_b[0]:up_b[0], low_b[1]:up_b[1], low_b[2]:up_b[2]]

        orig_shape = np.asarray(image.shape)
        bounds = self.closestDistanceForDivision(orig_shape)
        low_bound = np.floor(bounds/2).astype(np.int)
        high_bound = (orig_shape - np.ceil(bounds/2)).astype(np.int)
        return [
            image[low_bound[0]:high_bound[0], low_bound[1]:high_bound[1], low_bound[2]:high_bound[2]],
            mask[low_bound[0]:high_bound[0], low_bound[1]:high_bound[1], low_bound[2]:high_bound[2]]
        ]


class Relabel:
    """
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10,10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        _, unique_labels = np.unique(m, return_inverse=True)
        m = unique_labels.reshape(m.shape)
        return m


class CenterPad:
    def __init__(self, final_shape):
        self.size = final_shape

    def __call__(self, image, pad_val=None):

        if pad_val is None:
            pad_val = image.min()

        tensor_flag = torch.is_tensor(image)
        image = ToTensor()(image) if not tensor_flag else image

        z_offset = self.size[0] - image.shape[-3]
        y_offset = self.size[1] - image.shape[-2]
        x_offset = self.size[2] - image.shape[-1]

        z_offset = int(np.floor(z_offset / 2.)), int(np.ceil(z_offset / 2.))
        y_offset = int(np.floor(y_offset / 2.)), int(np.ceil(y_offset / 2.))
        x_offset = int(np.floor(x_offset / 2.)), int(np.ceil(x_offset / 2.))

        padded = torch.nn.functional.pad(image, [x_offset[0], x_offset[1], y_offset[0], y_offset[1], z_offset[0], z_offset[1]], value=pad_val)

        if not tensor_flag:
            return padded.numpy()
        return padded


class CenterCrop:

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, image, gt=None):

        z_offset = image.shape[-3] - self.target_shape[0]
        y_offset = image.shape[-2] - self.target_shape[1]
        x_offset = image.shape[-1] - self.target_shape[2]
        z_offset = int(np.floor(z_offset / 2.)), image.shape[-3] - int(np.ceil(z_offset / 2.))
        y_offset = int(np.floor(y_offset / 2.)), image.shape[-2] - int(np.ceil(y_offset / 2.))
        x_offset = int(np.floor(x_offset / 2.)), image.shape[-1] - int(np.ceil(x_offset / 2.))

        crop_img = image[..., z_offset[0]:z_offset[1], y_offset[0]:y_offset[1], x_offset[0]:x_offset[1]]
        if gt is not None:
            assert image.shape == gt.shape
            gt = gt[..., z_offset[0]:z_offset[1], y_offset[0]:y_offset[1], x_offset[0]:x_offset[1]]
            return crop_img, gt
        return crop_img

class CropAndPad(tio.Transform):

    def __init__(self, target_shape):
        self.target_shape = target_shape
        super().__init__()

    def apply_transform(self, image):
        batch = len(image.shape) == 4

        if batch and image.shape[0] != 1:
            raise SystemExit

        if batch:
            image.squeeze(0)

        self.pad_val = None
        if isinstance(image, tio.LabelMap):
            # self.pad_val = self.config['labels']['BACKGROUND']
            self.pad_val = 0
        was_numpy = not torch.is_tensor(image)

        # CROPPING
        z_offset = max(image.shape[-3] - self.target_shape[0], 0)
        y_offset = max(image.shape[-2] - self.target_shape[1], 0)
        x_offset = max(image.shape[-1] - self.target_shape[2], 0)
        z_offset = int(np.floor(z_offset / 2.)), image.shape[-3] - int(np.ceil(z_offset / 2.))
        y_offset = int(np.floor(y_offset / 2.)), image.shape[-2] - int(np.ceil(y_offset / 2.))
        x_offset = int(np.floor(x_offset / 2.)), image.shape[-1] - int(np.ceil(x_offset / 2.))

        img = image[..., z_offset[0]:z_offset[1], y_offset[0]:y_offset[1], x_offset[0]:x_offset[1]]

        # PADDING
        z_offset = -min(image.shape[-3] - self.target_shape[0], 0)
        y_offset = -min(image.shape[-2] - self.target_shape[1], 0)
        x_offset = -min(image.shape[-1] - self.target_shape[2], 0)
        z_offset = int(np.floor(z_offset / 2.)), z_offset - int(np.floor(z_offset / 2.))
        y_offset = int(np.floor(y_offset / 2.)), y_offset - int(np.floor(y_offset / 2.))
        x_offset = int(np.floor(x_offset / 2.)), x_offset - int(np.floor(x_offset / 2.))

        img = img if torch.is_tensor(img) else torch.from_numpy(img)
        self.pad_val = self.pad_val if self.pad_val else img.min()
        img = pad(img, (x_offset[0], x_offset[1], y_offset[0], y_offset[1], z_offset[0], z_offset[1]), value=self.pad_val)

        if was_numpy:
            return img.numpy()
        if batch:
            img.unsqueeze(0)
        return img

class LeftSplit(tio.Transform):
    def __init__(self, axes=-1):
        super().__init__()
        self.axes = axes

    def apply_transform(self, image):
        self._crop = tio.Crop((0,0,0,0,0,image.shape[-1]//2))
        self._flip = lambda x: x

        image = self._crop(image)
        image = self._flip(image)
        return image

class RightSplit(tio.Transform):
    def __init__(self, axes=-1):
        super().__init__()
        self.axes = axes

    def apply_transform(self, image):
        self._crop = tio.Crop((0,0,0,0,image.shape[-1]//2,0))
        self._flip = tio.RandomFlip(axes = 2, flip_probability=1)

        image = self._crop(image)
        image = self._flip(image)
        return image

class HalfSplit(tio.Transform):
    def __init__(self, axes=-1):
        super().__init__()
        self.axes = axes
        self.right = RightSplit()
        self.left = LeftSplit()

    def apply_transform(self, image):
        if random.random() > 0.5:
            return self.right(image)
        else:
            return self.left(image)


# TODO: change this pls
class AugFactory:
    def __init__(self, aug_list):
        self.aug_list = aug_list
        self.transforms = self.factory(self.aug_list, [])
        logging.info('Augmentations: {}'.format(self.aug_list))

    def factory(self, auglist, transforms):
        if auglist == None: return []
        for aug in auglist:
            if aug == 'OneOf':
                transforms.append(tio.OneOf(self.factory(auglist[aug], [])))
            else:
                try:
                    kwargs = {}
                    for param, value in auglist[aug].items():
                        kwargs[param] = value
                    if aug == 'DistanceTransform':
                        t = DistanceTransform(**kwargs)
                    elif aug == 'ProximityTransform':
                        t = ProximityTransform(**kwargs)
                    elif aug == 'HalfSplit':
                        t = HalfSplit()
                    else:
                        t = getattr(tio, aug)(**kwargs)
                    transforms.append(t)
                except:
                    raise Exception(f"this transform is not valid: {aug}")
        return transforms

    def get_transform(self):
        """
        return the transform object
        :return:
        """
        transf = tio.Compose(self.transforms)
        return transf

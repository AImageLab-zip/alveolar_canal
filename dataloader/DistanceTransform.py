import torchio as tio
import numpy as np
import torch

from math import sqrt
from scipy.ndimage import distance_transform_edt
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

def max_norm(img):
    max_val = img.max()
    img = img/max_val
    return img

def diag_norm(img):
    shape = img.shape
    diag = sqrt(shape[-1]**2 + shape[-2]**2 + shape[-3]**2)
    img = img / diag
    return img

def exp_norm(img):
    img = img.max()-img
    exp_img = np.exp(img)
    return exp_img / np.sum(exp_img)

class DistanceTransform(LabelTransform):
    def __init__(self, normalization=lambda x: x, *args, **kwargs):
        super(DistanceTransform, self).__init__(*args, **kwargs)
        if normalization == 'max':
            normalization = max_norm
        elif normalization == 'diag':
            normalization = diag_norm
        elif normalization == 'exp':
            normalization = exp_norm

        self.normalization = normalization

    def apply_transform(self, subject):

        images = self.get_images(subject)
        for image in images:
            imgdata = image[tio.DATA]
            if imgdata.sum() == 0:
                image.set_data(torch.ones_like(imgdata))
                continue

            # neg_image = torch.where(imgdata == 1., 0., 1.)
            neg_image = 1 - imgdata
            pos_dist = distance_transform_edt(imgdata)
            neg_dist = distance_transform_edt(neg_image)
            dist = neg_dist - pos_dist

            dist = self.normalization(dist)

            image.set_data(dist)
        return subject

class ProximityTransform(LabelTransform):
    def __init__(self, normalization=lambda x: x, *args, **kwargs):
        super(ProximityTransform, self).__init__(*args, **kwargs)
        if normalization == 'max':
            normalization = max_norm
        elif normalization == 'diag':
            normalization = diag_norm
        elif normalization == 'exp':
            normalization = exp_norm

        self.normalization = normalization

    def proximity(self, x):
        return max(x) - x + 1

    def apply_transform(self, subject):

        images = self.get_images(subject)
        for image in images:
            imgdata = image[tio.DATA]
            if imgdata.sum() == 0:
                image.set_data(torch.ones_like(imgdata))
                continue

            # neg_image = torch.where(imgdata == 1., 0., 1.)
            neg_image = 1 - imgdata
            pos_dist = self.proximity(distance_transform_edt(imgdata))*imgdata.numpy()
            neg_dist = self.proximity(distance_transform_edt(neg_image))*neg_image.numpy()
            dist = neg_dist - pos_dist

            dist = self.normalization(dist)

            image.set_data(dist)
        return subject

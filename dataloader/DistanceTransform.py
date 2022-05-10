import torchio as tio
import numpy as np
import torch

from scipy.ndimage import distance_transform_edt
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

class DistanceTransform(LabelTransform):
    def apply_transform(self, subject):
        for image in self.get_images(subject):
            neg_image = torch.where(image[tio.DATA] == 1., 0., 1.)
            pos_dist = distance_transform_edt(image)
            neg_dist = distance_transform_edt(neg_image)
            dist = neg_dist - pos_dist
            
            # change range to [-1,1]
            dist = dist*2/dist.max()-1

            image.set_data(dist)
        return subject

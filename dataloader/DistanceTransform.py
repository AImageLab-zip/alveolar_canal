import torchio as tio
import numpy as np

from scipy.ndimage import distance_transform_edt
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

class DistanceTransform(LabelTransform):
    def apply_transform(self, subject):
        for image in self.get_images(subject):
            neg_image = torch.where(image[tio.DATA] == 1., 0., 1.)
            dist = distance_transform_edt(neg_image)
            # normalize to [0,1]
            dist = dist / dist.max()
            image.set_data(dist)
        return subject

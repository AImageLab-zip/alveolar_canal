import torch
from torch.nn import functional as F

import torchio as tio

class DiffResample(tio.Transform):
    def __init__(self):
        super().__init__()

    def apply_transform(self, x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor, mode='trilinear')




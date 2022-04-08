import torch
from torch import nn

from .dataset3D import Loader3D

class DatasetFactory(nn.Module):
    def __init__(self, type):
        super(DatasetFactory, self).__init__()
        self.type = type

    def get(self):
        if self.type == '2D':
            raise ValueError('Dataset type not supported yet')
        elif self.type == '3D':
            return Loader3D
        else:
            raise ValueError('Dataset type not found')

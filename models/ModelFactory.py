import torch
from torch import nn

from .PosPadUNet3D import PosPadUNet3D
from .PosPadUNet3DSparse import PosPadUNet3DSparse
from .Competitor import Competitor

class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch, emb_shape=None):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.emb_shape = emb_shape

    def get(self):
        if self.model_name == 'PosPadUNet3D':
            assert self.emb_shape is not None
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'Competitor':
            return Competitor(self.num_classes)
        else:
            raise ValueError('Model name not found')

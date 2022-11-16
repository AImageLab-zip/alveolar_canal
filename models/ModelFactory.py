import torch
from torch import nn

from .PosPadUNet3D import PosPadUNet3D
from .PosPadUNet3DSparse import PosPadUNet3DSparse
from .PosPadUNet3DSparseOnly import PosPadUNet3DSparseOnly
from .Competitor import Competitor
from .DeepLabv3.deeplabv3_3d import DeepLabV3_3D
from .SegNet.segnet_3d import SegNet3D
from .UNETR.UNETR import UNETR
from .DMFNet.DMFNet import DMFNet
from .SwinUNETR.SwinUNETR import SwinUNETR
from .PadUNet3D import PadUNet3D
from .LatePosPadUNet3D import LatePosPadUNet3D
from .TransPosPadUNet3D import TransPosPadUNet3D
# from monai.networks.nets import SwinUNETR

class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch, emb_shape=None, n_layers=2, num_head=8, max_len=10_000):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.emb_shape = emb_shape
        self.n_layers = n_layers
        self.num_head = num_head
        self.max_len = max_len

    def get(self):
        if self.model_name == 'PosPadUNet3D':
            assert self.emb_shape is not None
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'Competitor':
            return Competitor(self.num_classes)
        elif self.model_name == 'DeepLabV3':
            return DeepLabV3_3D(self.num_classes, self.emb_shape, self.in_ch, 'resnet18_os8')
        elif self.model_name == 'SegNet3D':
            return SegNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'UNETR':
            return UNETR(img_shape=(80,80,80), input_dim=self.in_ch, output_dim=self.num_classes, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1)
        elif self.model_name == 'DMFNet':
            return DMFNet(c=self.in_ch, groups=16, norm='sync_bn', num_classes=self.num_classes)
        elif self.model_name == 'PosPadUNet3DSparseOnly':
            return PosPadUNet3DSparseOnly(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'SwinUNETR':
            return SwinUNETR(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'PadUNet3D':
            return PadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'LatePosPadUNet3D':
            return LatePosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'TransPosPadUNet3D':
            return TransPosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch, n_layers=self.n_layers, num_head=self.num_head, max_len=self.max_len)
        else:
            raise ValueError('Model name not found')

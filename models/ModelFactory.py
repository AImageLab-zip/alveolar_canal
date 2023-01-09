import torch
from torch import nn

from .PosPadUNet3DSparse import PosPadUNet3DSparse
# from monai.networks.nets import SwinUNETR

class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch, emb_shape=None):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.emb_shape = emb_shape

    def get(self):
        if self.model_name == 'PosPadUNet3D':
            from .PosPadUNet3D import PosPadUNet3D
            assert self.emb_shape is not None
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'Competitor':
            from .Competitor import Competitor
            return Competitor(self.num_classes)
        elif self.model_name == 'DeepLabV3':
            from .DeepLabv3.deeplabv3_3d import DeepLabV3_3D
            return DeepLabV3_3D(self.num_classes, self.emb_shape, self.in_ch, 'resnet18_os8')
        elif self.model_name == 'SegNet3D':
            from .SegNet.segnet_3d import SegNet3D
            return SegNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'UNETR':
            from .UNETR.UNETR import UNETR
            return UNETR(img_shape=(80,80,80), input_dim=self.in_ch, output_dim=self.num_classes, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1)
        elif self.model_name == 'DMFNet':
            from .DMFNet.DMFNet import DMFNet
            return DMFNet(c=self.in_ch, groups=16, norm='sync_bn', num_classes=self.num_classes)
        elif self.model_name == 'PosPadUNet3DSparseOnly':
            from .PosPadUNet3DSparseOnly import PosPadUNet3DSparseOnly
            return PosPadUNet3DSparseOnly(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'SwinUNETR':
            from .SwinUNETR.SwinUNETR import SwinUNETR
            return SwinUNETR(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'PadUNet3D':
            from .PadUNet3D import PadUNet3D
            return PadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        else:
            raise ValueError(f'Model {self.model_name} not found')

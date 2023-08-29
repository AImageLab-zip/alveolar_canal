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
from .CrossPosPadUNet3D import CrossPosPadUNet3D
from .LatePosPadUNet3D import LatePosPadUNet3D
from .TransPosPadUNet3D import TransPosPadUNet3D
from .SqueezeTransPosPadUNet3D import SqueezeTransPosPadUNet3D
from .MemTransPosPadUNet3D import MemTransPosPadUNet3D
# from monai.networks.nets import SwinUNETR

class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch, 
                        emb_shape=None, n_layers=4, num_head=1, max_len=10**3,
                        config=None):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.emb_shape = emb_shape
        self.max_len = max_len
        self.config = config
        print("CONFIG: ", config)
        if config.model.name in ["TransPosPadUNet3D", "SqueezeTransPosPadUNet3D", "MemTransPosPadUNet3D"]:
            self.n_layers = config.model.n_layers
            self.num_head = config.model.n_head
            self.pos_enc = config.model.pos_enc
        if config.model.name in ["MemTransPosPadUNet3D"]:
            self.mem_len = config.model.mem_len
            self.ABS = config.model.ABS
            dim = config.data_loader.patch_shape[0]//8
            self.max_len = dim*dim*dim


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
        elif self.model_name == 'CrossPosPadUNet3D':
            return CrossPosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'LatePosPadUNet3D':
            return LatePosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'TransPosPadUNet3D':
            return TransPosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch, 
                                        n_layers=self.n_layers, num_head=self.num_head, max_len=self.max_len, pos_enc=self.pos_enc)
        elif self.model_name == 'SqueezeTransPosPadUNet3D':
            return SqueezeTransPosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch, 
                                        n_layers=self.n_layers, num_head=self.num_head, max_len=self.max_len, pos_enc=self.pos_enc)     
        elif self.model_name == 'MemTransPosPadUNet3D':
            return MemTransPosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch, 
                                        n_layers=self.n_layers, num_head=self.num_head, max_len=self.max_len, pos_enc=self.pos_enc, mem_len=self.mem_len, ABS=self.ABS)                                            
        elif self.model_name == 'MultiPosPadUNet3D':
            from .MultiPosPadUNet3D import MultiPosPadUNet3D
            return MultiPosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        else:
            raise ValueError('Model name not found')

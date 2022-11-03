import os
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR as SUNETR

class SwinUNETR(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch):
        super(SwinUNETR, self).__init__()
        self.n_classes = n_classes
        self.emb_shape = emb_shape
        self.in_ch = in_ch
        self._swinunetr = SUNETR(img_size=(96, 96, 96),
                in_channels=1,
                out_channels=1,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0,
                use_checkpoint=True,
                )
        weights = torch.load("/homes/llumetti/alveolar_canal_experimental/models/SwinUNETR/model_swinvit.pt")
        self._swinunetr.load_from(weights=weights)

    def forward(self, x, emb_codes):
        return torch.sigmoid(self._swinunetr(x))

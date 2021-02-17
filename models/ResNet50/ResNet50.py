import torch
import torch.nn as nn
from torchvision import models
from models.ResNet50.Decoder import resnet50_decoder


class ResNet50(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, pretrained=0, dropout=0.3, decoder_version=50):
        super(ResNet50, self).__init__()
        if pretrained:
            self.model = models.resnet50(pretrained=True)
        else:
            self.model = models.resnet50(pretrained=False)
        # set the number of input channels
        if in_channels != 3:
            self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.conv1 = self.conv_1
        # feature extractor definition
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        # set the decoder version
        self.decoder_version = decoder_version
        # decoder definition
        self.decoder = resnet50_decoder(stride=2, out_channels=out_channels)
        # conv to reduce dimension in order to pass from resnet50 encoder to resnet18 decoder
        self.conv1x1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(spectral)
        if self.decoder_version == 18:
            features = self.conv1x1(features)
            # features = self.batch_norm(features)
        recon = self.decoder(features)
        return recon

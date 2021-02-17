import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # main block
        self.res = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.res(x)
        out += self.shortcut(x)
        return out


class ResNetEncoder(nn.Module):

    # define layers
    def __init__(self, n_classes):
        super(ResNetEncoder, self).__init__()
        self.hidden_size = 128
        self.num_classes = n_classes

        # self.encoder = nn.Sequential(
        #
        #     ResidualBlock(2, 64,  stride=2),
        #     nn.ReLU(),
        #     ResidualBlock(64, 128, stride=2),
        #     nn.ReLU(),
        #     ResidualBlock(128, 256, stride=2),
        #     nn.ReLU(),
        #     ResidualBlock(256, 512, stride=2),
        #     nn.ReLU(),
        #
        #     Flatten(),
        #     nn.Linear(8 * 8 * 512, self.hidden_size * 2)
        # )

        # self.mean = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.logvar = nn.Linear(self.hidden_size * 2, self.hidden_size)
        #
        # # AVOID RELU AFTER LAST RESIDUAL!
        self.cond_encoder = nn.Sequential(

            nn.Conv3d(1, 1, kernel_size=7, stride=1, padding=3),

            ResidualBlock(1, 64, stride=2),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            nn.ReLU(),
            ResidualBlock(128, 256, stride=2),
            nn.ReLU(),
            ResidualBlock(256, 512, stride=2),
            nn.ReLU(),
            ResidualBlock(512, self.hidden_size, stride=2),
        )

        self.decoder = nn.Sequential(
            ResidualBlock(self.hidden_size, 64, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),

            ResidualBlock(64, 32, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),

            ResidualBlock(32, 16, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),

            ResidualBlock(16, 8, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),

            nn.Conv3d(8, self.num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, volume, cast=True):
        with autocast(enabled=cast):
            if volume.ndim == 4:
                volume = torch.unsqueeze(volume, dim=1)

            z = self.cond_encoder(volume)
            out = self.decoder(z)
            return out
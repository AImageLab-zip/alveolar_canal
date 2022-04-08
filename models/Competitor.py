import torch
import torch.nn as nn


class Competitor(nn.Module):

    def __init__(self, n_classes):
        super(Competitor, self).__init__()

        self.ec0 = self.conv3Dblock(1, 32)
        self.ec1 = self.conv3Dblock(32, 32)
        self.ec2 = self.conv3Dblock(32, 64, kernel_size=3, stride=2)  # third dimension to even val
        self.ec3 = self.conv3Dblock(64, 64)
        self.ec4 = self.conv3Dblock(64, 64)
        self.ec5 = self.conv3Dblock(64, 128, kernel_size=3, stride=2)
        self.ec6 = self.conv3Dblock(128, 128)
        self.ec7 = self.conv3Dblock(128, 128)
        self.ec8 = self.conv3Dblock(128, 256, kernel_size=3, stride=2)
        self.ec9 = self.conv3Dblock(256, 256)
        self.ec10 = self.conv3Dblock(256, 256)

        self.dc9 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=1)  # we have a concat here
        self.dc8 = self.conv3Dblock(256, 128)
        self.dc7 = self.conv3Dblock(128, 128)
        self.dc6 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=1)  # we have a concat here
        self.dc5 = self.conv3Dblock(128, 64)
        self.dc4 = self.conv3Dblock(64, 64)
        self.dc3 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=1)  # we have a concat here
        self.dc2 = self.conv3Dblock(64, 32)
        self.dc1 = self.conv3Dblock(32, 32)
        self.final = nn.Conv3d(32, n_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
                )

    def forward(self, x, _):

        x = x[:, 0:1]  # keep one channel

        h = self.ec0(x)
        feat_0 = self.ec1(h)
        residual = self.ec2(feat_0 + x)

        h = self.ec3(residual)
        feat_1 = self.ec4(h)
        residual = self.ec5(feat_1 + residual)

        h = self.ec6(residual)
        feat_2 = self.ec7(h)
        residual = self.ec8(feat_2 + residual)

        h = self.ec9(residual)
        h = self.ec10(h)

        # decoder
        residual = self.dc9(h + residual)
        h = self.dc8(torch.cat((residual, feat_2), dim=1))
        h = self.dc7(h)

        residual = self.dc6(h + residual)
        h = self.dc5(torch.cat((residual, feat_1), dim=1))
        h = self.dc4(h)

        residual = self.dc3(h + residual)
        h = self.dc2(torch.cat((residual, feat_0), dim=1))
        h = self.dc1(h)
        return self.final(h + residual)


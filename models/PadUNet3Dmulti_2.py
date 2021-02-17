import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.ec0 = self.conv3Dblock(1, 32)
        self.ec1 = self.conv3Dblock(32, 64)  # third dimension to even val
        self.ec2 = self.conv3Dblock(64, 64)
        self.ec3 = self.conv3Dblock(64, 128)
        self.ec4 = self.conv3Dblock(128, 128)
        self.ec5 = self.conv3Dblock(128, 256)
        self.ec6 = self.conv3Dblock(256, 256)
        self.ec7 = self.conv3Dblock(256, 512)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(256 + 512, 256, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(256, 256, kernel_size=3, stride=1, padding=1)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
        )

    def forward(self, x):
        # print('ENCODER input: device {}, shape {}'.format(x.device, x.shape))
        with autocast():
            h = self.ec0(x)
            feat_0 = self.ec1(h)
            h = self.pool0(feat_0)
            h = self.ec2(h)
            feat_1 = self.ec3(h)

            h = self.pool1(feat_1)
            h = self.ec4(h)
            feat_2 = self.ec5(h)

            h = self.pool2(feat_2)
            h = self.ec6(h)
            h = self.ec7(h)

            h = torch.cat((self.dc9(h), feat_2), dim=1)
            h = self.dc8(h)
            h = self.dc7(h)
            return feat_0, feat_1, h


class Block2(nn.Module):

    def __init__(self):
        super(Block2, self).__init__()
        self.dc6 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(128 + 256, 128, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(128, 128, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(64 + 128, 64, kernel_size=3, stride=1, padding=1)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
        )

    def forward(self, feat_0, feat_1, x):

        # print('DECODER input: device {}, shape {}'.format(x.device, x.shape))
        with autocast():

            h = torch.cat((self.dc6(x), feat_1), dim=1)
            h = self.dc5(h)
            h = self.dc4(h)
            h = torch.cat((self.dc3(h), feat_0), dim=1)
            h = self.dc2(h)
            return h


class Block3(nn.Module):

    def __init__(self, num_classes):
        super(Block3, self).__init__()
        self.num_classes = num_classes
        self.dc1 = self.conv3Dblock(64, 64, kernel_size=3, stride=1, padding=1)
        self.final = nn.ConvTranspose3d(64, self.num_classes, kernel_size=3, padding=1, stride=1)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
        )

    def forward(self, x):

        # print('DECODER input: device {}, shape {}'.format(x.device, x.shape))
        with autocast():
            h = self.dc1(x)
        with autocast(enabled=False):
            return self.final(h.float())


class padUNet3DMulti(nn.Module):
    def __init__(self, n_classes):
        super(padUNet3DMulti, self).__init__()

        self.b1 = Block1().to('cuda:0')
        self.b2 = Block2().to('cuda:1')
        self.b3 = Block3(n_classes).to('cuda:2')

    def forward(self, x):

        if x.ndim == 4:
            x = torch.unsqueeze(x, dim=1)  # add single channel after batchsize

        # print("original input: {}".format(x.shape))
        feat_0, feat_1, embedding = self.b1(x)
        embedding = self.b2(feat_0.to('cuda:1'), feat_1.to('cuda:1'), embedding.to('cuda:1'))
        return self.b3(embedding.to('cuda:2'))
import torch
import torch.nn as nn
from models.Multiscale.ResNet3D import ResNet3D, ResNet2Puls1D, ResNetM3D


class MultiScaleBlock(nn.Module):

    def __init__(self, planes=512):
        super(MultiScaleBlock, self).__init__()
        self.out_planes = planes // 4
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1), dilation=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(4, 4, 4), dilation=(4, 4, 4), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(8, 8, 8), dilation=(8, 8, 8), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(1, 1, 1), bias=False),
        )
        self.after_pool = nn.Sequential(
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_0 = self.conv0(x)
        out_1 = self.conv1(x)
        out_2 = self.conv2(x)
        out_3 = self.pool(x)
        out_3 = nn.functional.interpolate(out_3, out_1.shape[-3:])
        out_3 = self.after_pool(out_3)

        return torch.cat((out_0, out_1, out_2, out_3), dim=1)


class Multiscale3D(nn.Module):
    def __init__(self, backbone='ResNet3D', num_classes=2):
        super(Multiscale3D, self).__init__()
        self.bottle_planes = 128 * 4
        if backbone == 'ResNet3D':
            self.resnet = ResNet3D()
        elif backbone == 'ResNet2Plus1D':
            self.resnet = ResNet2Puls1D()
        elif backbone == 'ResNetM3D':
            self.resnet = ResNetM3D()
        self.resnet.set_output_stride()

        self.mslayer = MultiScaleBlock()

        mid_planes = self.bottle_planes // 4
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels=self.bottle_planes, out_channels=mid_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_planes, out_channels=num_classes, kernel_size=(1, 1, 1), stride=1, bias=False),
        )

    def forward(self, x):
        out, low_level = self.resnet(x)
        out = self.mslayer(out)

        out = nn.functional.interpolate(out, scale_factor=(4, 4, 4), mode='trilinear')
        out = self.decoder(out)
        out = nn.functional.interpolate(out, scale_factor=(1, 2, 2), mode='trilinear')

        return out


class Multiscale2Plus1D(Multiscale3D):
    def __init__(self, num_classes=1):
        super().__init__(backbone='ResNet2Plus1D', num_classes=num_classes)


class MultiscaleM3D(Multiscale3D):
    def __init__(self, num_classes=1):
        super().__init__(backbone='ResNetM3D', num_classes=num_classes)

    def forward(self, x):
        out, low_level = self.resnet(x)
        out = self.mslayer(out)

        out = nn.functional.interpolate(out, scale_factor=(1, 4, 4), mode='trilinear')
        out = self.decoder(out)
        out = nn.functional.interpolate(out, scale_factor=(1, 2, 2), mode='trilinear')
        return out


if __name__ == '__main__':
    batch_size = 4
    x = torch.rand(size=(batch_size, 3, 56, 144, 144), device='cuda')
    y = torch.zeros(size=(batch_size, 1, 56, 144, 144), device='cuda')
    Multiscale = Multiscale3D().to('cuda')
    out = Multiscale(x)
    loss = torch.nn.BCEWithLogitsLoss()(out, y)
    loss.backward()
    pass

import torch
import torch.nn as nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class PosPadUNet3DSparseOnly(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch):
        self.n_classes = n_classes
        self.in_ch = 1
        super(PosPadUNet3DSparseOnly, self).__init__()

        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.ec0 = self.conv3Dblock(self.in_ch, 32)
        self.ec1 = self.conv3Dblock(32, 64, kernel_size=3, padding=1)  # third dimension to even val
        self.ec2 = self.conv3Dblock(64, 64)
        self.ec3 = self.conv3Dblock(64, 128)
        self.ec4 = self.conv3Dblock(128, 128)
        self.ec5 = self.conv3Dblock(128, 256)
        self.ec6 = self.conv3Dblock(256, 256)
        self.ec7 = self.conv3Dblock(256, 512)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = nn.ConvTranspose3d(513, 512, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(256 + 512, 256, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(256, 256, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(128 + 256, 128, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(128, 128, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(64 + 128, 64, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(64, 64, kernel_size=3, stride=1, padding=1)
        self.final = nn.ConvTranspose3d(64, n_classes, kernel_size=3, padding=1, stride=1)
        initialize_weights(self)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
        )

    def forward(self, x, emb_codes):
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

        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        h = torch.cat((h, emb_pos), dim=1)
        h = torch.cat((self.dc9(h), feat_2), dim=1)

        h = self.dc8(h)
        h = self.dc7(h)

        h = torch.cat((self.dc6(h), feat_1), dim=1)
        h = self.dc5(h)
        h = self.dc4(h)

        h = torch.cat((self.dc3(h), feat_0), dim=1)
        h = self.dc2(h)
        h = self.dc1(h)
        h = self.final(h)
        return torch.sigmoid(h)

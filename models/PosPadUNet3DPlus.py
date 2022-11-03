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

class PosPadUNet3DPlus(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch):
        self.n_classes = n_classes
        self.in_ch = in_ch
        super(PosPadUNet3DPlus, self).__init__()

        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.ecx0 = self.conv3Dblock(1, 32)
        self.ecx1 = self.conv3Dblock(32, 64, kernel_size=3, padding=1)  # third dimension to even val
        self.ecx2 = self.conv3Dblock(64, 64)
        self.ecx3 = self.conv3Dblock(64, 128)
        self.ecx4 = self.conv3Dblock(128, 128)
        self.ecx5 = self.conv3Dblock(128, 256)
        self.ecx6 = self.conv3Dblock(256, 256)
        self.ecx7 = self.conv3Dblock(256, 512)

        self.ecs0 = self.conv3Dblock(1, 32)
        self.ecs1 = self.conv3Dblock(32, 64, kernel_size=3, padding=1)  # third dimension to even val
        self.ecs2 = self.conv3Dblock(64, 64)
        self.ecs3 = self.conv3Dblock(64, 128)
        self.ecs4 = self.conv3Dblock(128, 128)
        self.ecs5 = self.conv3Dblock(128, 256)
        self.ecs6 = self.conv3Dblock(256, 256)
        self.ecs7 = self.conv3Dblock(256, 512)

        self.poolx0 = nn.MaxPool3d(2)
        self.poolx1 = nn.MaxPool3d(2)
        self.poolx2 = nn.MaxPool3d(2)

        self.pools0 = nn.MaxPool3d(2)
        self.pools1 = nn.MaxPool3d(2)
        self.pools2 = nn.MaxPool3d(2)

        self.dc9 = nn.ConvTranspose3d(512+512+1, 512, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(512 + 512, 256, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(256, 256, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(256 + 256, 128, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(128, 128, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(128 + 128, 64, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(64, 64, kernel_size=3, stride=1, padding=1)
        self.final = nn.ConvTranspose3d(64, n_classes, kernel_size=3, padding=1, stride=1)
        initialize_weights(self)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
        )

    def forward(self, x, sparse, emb_codes):
        h = self.ecx0(x)
        k = self.ecs0(sparse)
        feat_x_0 = self.ecx1(h)
        feat_s_0 = self.ecs1(k)

        h = self.poolx0(feat_x_0)
        k = self.pools0(feat_s_0)

        h = self.ecx2(h)
        k = self.ecs2(k)
        feat_x_1 = self.ecx3(h)
        feat_s_1 = self.ecs3(k)

        h = self.poolx1(feat_x_1)
        k = self.pools1(feat_s_1)

        h = self.ecx4(h)
        k = self.ecs4(k)
        feat_x_2 = self.ecx5(h)
        feat_s_2 = self.ecs5(k)

        h = self.poolx2(feat_x_2)
        k = self.pools2(feat_s_2)

        h = self.ecx6(h)
        k = self.ecs6(k)
        h = self.ecx7(h)
        k = self.ecs7(k)

        # latent
        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        h = torch.cat((h, k, emb_pos), dim=1)
        h = torch.cat((self.dc9(h), feat_x_2, feat_s_2), dim=1)

        h = self.dc8(h)
        h = self.dc7(h)

        h = torch.cat((self.dc6(h), feat_x_1, feat_s_1), dim=1)
        h = self.dc5(h)
        h = self.dc4(h)

        h = torch.cat((self.dc3(h), feat_x_0, feat_s_0), dim=1)
        h = self.dc2(h)
        h = self.dc1(h)
        h = self.final(h)
        return torch.sigmoid(h)

import torch
import torch.nn as nn
from models.ResNet50.resnet_utils import conv1x1, Bottleneck, BasicBlock


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, stride=2, out_channels=2, inplanes = 512):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.stride = stride
        self.out_channels = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.layer4_dec = self._make_layer(block, 512, layers[3], stride=self.stride,
                                           dilate=replace_stride_with_dilation[2])

        self.layer3_dec = self._make_layer(block, 256, layers[2], stride=self.stride,
                                           dilate=replace_stride_with_dilation[1])

        self.layer2_dec = self._make_layer(block, 128, layers[1], stride=self.stride,
                                           dilate=replace_stride_with_dilation[0])

        self.layer1_dec = self._make_layer(block, 64, layers[0])

        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        self.bn1 = norm_layer(int(inplanes / 8))
        self.relu = nn.ReLU(inplace=True)
        self.conv_final = nn.ConvTranspose2d(int(inplanes / 8), self.out_channels, kernel_size=(3, 3), stride=self.stride,
                                             padding=1, groups=groups, bias=False, output_padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.avgpool(x)
        x = self.layer4_dec(x)
        x = self.layer3_dec(x)
        x = self.layer2_dec(x)
        x = self.layer1_dec(x)
        x = self.upsample(x)
        x = self.bn1(x)
        x = self.conv_final(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, stride, out_channels, inplanes, **kwargs):
    kwargs['stride'] = stride
    kwargs['out_channels'] = out_channels
    kwargs['inplanes'] = inplanes
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18_decoder(stride, out_channels, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inplanes = 512
    return _resnet(BasicBlock, [2, 2, 2, 2], stride, out_channels, inplanes, **kwargs)


def resnet50_decoder(stride, out_channels, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inplanes = 2048
    return _resnet(Bottleneck, [3, 4, 6, 3], stride, out_channels, inplanes, **kwargs)
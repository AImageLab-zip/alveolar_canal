import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from .aspp import ASPP, ASPP_Bottleneck

class DeepLabV3_3D(nn.Module):
    def __init__(self, num_classes, emb_shape, input_channels, resnet, last_activation = 'sigmoid'):
        super(DeepLabV3_3D, self).__init__()
        self.num_classes = num_classes
        self.last_activation = last_activation
        self.emb_shape = torch.as_tensor(emb_shape)

        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())

        if resnet.lower() == 'resnet18_os16':
            self.resnet = ResNet18_OS16(input_channels)
        
        elif resnet.lower() == 'resnet34_os16':
            self.resnet = ResNet34_OS16(input_channels)
        
        elif resnet.lower() == 'resnet50_os16':
            self.resnet = ResNet50_OS16(input_channels)
        
        elif resnet.lower() == 'resnet101_os16':
            self.resnet = ResNet101_OS16(input_channels)
        
        elif resnet.lower() == 'resnet152_os16':
            self.resnet = ResNet152_OS16(input_channels)
        
        elif resnet.lower() == 'resnet18_os8':
            self.resnet = ResNet18_OS8(input_channels)
        
        elif resnet.lower() == 'resnet34_os8':
            self.resnet = ResNet34_OS8(input_channels)

        if resnet.lower() in ['resnet50_os16', 'resnet101_os16', 'resnet152_os16']:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        else:
            self.aspp = ASPP(num_classes=self.num_classes)

    def forward(self, x, emb_codes):

        h = x.size()[2]
        w = x.size()[3]
        c = x.size()[4]

        feature_map = self.resnet(x)
        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        feature_map = torch.cat((feature_map, emb_pos), dim=1)

        output = self.aspp(feature_map)

        output = F.interpolate(output, size=(h, w, c), mode='trilinear', align_corners=True)

        if self.last_activation == None: return output
        
        if self.last_activation.lower() == 'sigmoid':
            output = nn.Sigmoid()(output)
        
        elif self.last_activation.lower() == 'softmax':
            output = nn.Softmax()(output)
        
        return output

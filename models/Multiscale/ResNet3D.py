import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18
from torchvision.models.utils import load_state_dict_from_url

weights_url = 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth'


class ResNet3D(nn.Module):

    def __init__(self):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(ResNet3D, self).__init__()
        self.backbone = r3d_18(pretrained=True)

        # state_dict = load_state_dict_from_url(weights_url, progress=True)
        # self.backbone.load_state_dict(state_dict)

    def set_output_stride(self):
        self.backbone._modules['layer4'][0].conv1[0].stride = (1, 1, 1)
        self.backbone._modules['layer4'][0].downsample[0].stride = (1, 1, 1)
        self.backbone._modules['layer4'][1].conv1[0].dilation = (2, 2, 2)
        self.backbone._modules['layer4'][1].conv1[0].padding = (2, 2, 2)
        self.backbone._modules['layer4'][1].conv2[0].dilation = (2, 2, 2)
        self.backbone._modules['layer4'][1].conv2[0].padding = (2, 2, 2)

    def forward(self, x):
        x = self.backbone.stem(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        low_level_features = x
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x = self.backbone.avgpool(x)
        # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.backbone.fc(x)

        return x, low_level_features


class ResNet2Puls1D(nn.Module):

    def __init__(self):
        super(ResNet2Puls1D, self).__init__()
        self.backbone = r2plus1d_18(pretrained=True)

        # state_dict = load_state_dict_from_url(weights_url, progress=True)
        # self.backbone.load_state_dict(state_dict)

    def set_output_stride(self):
        self.backbone._modules['layer4'][0].conv1[0][0].stride = (1, 1, 1)
        self.backbone._modules['layer4'][0].conv1[0][-1].stride = (1, 1, 1)
        self.backbone._modules['layer4'][0].downsample[0].stride = (1, 1, 1)
        self.backbone._modules['layer4'][1].conv1[0][0].dilation = (1, 2, 2)
        self.backbone._modules['layer4'][1].conv1[0][-1].dilation = (2, 1, 1)
        self.backbone._modules['layer4'][1].conv1[0][0].padding = (0, 2, 2)
        self.backbone._modules['layer4'][1].conv1[0][-1].padding = (2, 0, 0)
        self.backbone._modules['layer4'][1].conv2[0][0].dilation = (1, 2, 2)
        self.backbone._modules['layer4'][1].conv2[0][-1].dilation = (2, 1, 1)
        self.backbone._modules['layer4'][1].conv2[0][0].padding = (0, 2, 2)
        self.backbone._modules['layer4'][1].conv2[0][-1].padding = (2, 0, 0)

    def forward(self, x):
        x = self.backbone.stem(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        low_level_features = x
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x = self.backbone.avgpool(x)
        # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.backbone.fc(x)

        return x, low_level_features


class ResNetM3D(nn.Module):

    def __init__(self):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(ResNetM3D, self).__init__()
        self.backbone = mc3_18(pretrained=True)

        # state_dict = load_state_dict_from_url(weights_url, progress=True)
        # self.backbone.load_state_dict(state_dict)

    def set_output_stride(self):
        self.backbone._modules['layer4'][0].conv1[0].stride = (1, 1, 1)
        self.backbone._modules['layer4'][0].downsample[0].stride = (1, 1, 1)
        self.backbone._modules['layer4'][1].conv1[0].dilation = (1, 2, 2)
        self.backbone._modules['layer4'][1].conv1[0].padding = (0, 2, 2)
        self.backbone._modules['layer4'][1].conv2[0].dilation = (1, 2, 2)
        self.backbone._modules['layer4'][1].conv2[0].padding = (0, 2, 2)

    def forward(self, x):
        x = self.backbone.stem(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        low_level_features = x
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x = self.backbone.avgpool(x)
        # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.backbone.fc(x)

        return x, low_level_features


if __name__ == '__main__':
    x = torch.rand(size=(1, 3, 32, 128, 128), device='cuda')
    network = ResNetM3D().to('cuda')
    # network = ResNet2Puls1D().to('cuda')
    # network = ResNet3D().to('cuda')
    network.set_output_stride()
    out = network(x)
    pass

import torch
import torch.nn as nn
from torch import Tensor


class SqueezeExcitation(nn.Module):
    """
    3D implementation of the SqueezeExcitation layer
    """
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation = torch.nn.ReLU,
        scale_activation = torch.nn.Sigmoid,
        ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

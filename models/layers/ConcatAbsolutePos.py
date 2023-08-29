import torch
import torch.nn as nn


class ConcatAbsolutePos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, abs_pos):
        abs_pos = torch.unsqueeze(abs_pos, 1)
        return torch.cat((h, abs_pos), dim=1)

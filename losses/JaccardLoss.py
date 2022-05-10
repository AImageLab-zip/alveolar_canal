import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, per_volume=False):
        super().__init__()
        self.weight = weight
        self.per_volume = per_volume

    def forward(self, pred, gt):
        assert pred.shape[1] == 1, 'this loss works with a binary prediction'

        batch_size = pred.size()[0]
        eps = 1e-6

        if not self.per_volume:
            batch_size = 1

        dice_gt = gt.contiguous().view(batch_size, -1).float()
        dice_pred = pred.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_pred * dice_gt, dim=1)
        union = torch.sum(dice_pred + dice_gt, dim=1) - intersection
        loss = 1 - (intersection + eps) / (union + eps)
        return loss

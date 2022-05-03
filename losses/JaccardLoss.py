import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_volume=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.weight = weight
        self.per_volume = per_volume
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, pred, gt):
        assert pred.shape[1] == 1, 'this loss works with a binary prediction'
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

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

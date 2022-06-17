import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MJaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, per_volume=True):
        super().__init__()
        self.weight = weight
        self.per_volume = per_volume

    def forward(self, pred, gt):
        assert pred.shape[1] == 1, 'this loss works with a binary prediction'

        batch_size = pred.size()[0]
        eps = 1e-10

        if not self.per_volume:
            batch_size = 1

        dice_gt = gt.view(batch_size, -1).float()
        dice_pred = pred.view(batch_size, -1)

        # over 1s
        one_intersection = torch.sum(dice_pred * dice_gt, dim=-1)
        one_union = torch.sum(dice_pred + dice_gt, dim=-1) - one_intersection
        one_iou = (one_intersection + eps) / (one_union + eps)

        # over 0s
        dice_gt = 1-dice_gt
        dice_pred = 1-dice_pred
        zero_intersection = torch.sum(dice_pred * dice_gt, dim=-1)
        zero_union = torch.sum(dice_pred + dice_gt, dim=-1) - zero_intersection
        zero_iou = (zero_intersection + eps) / (zero_union + eps)

        loss = 1 - ((zero_iou + one_iou)/2)
        print(f'dice_gt: {torch.sum(dice_gt, dim=-1)}')
        print(f'dice_pred: {torch.sum(dice_pred, dim=-1)}')
        return loss

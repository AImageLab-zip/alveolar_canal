import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, pred, gt):
        if pred.shape[1] == 1:
            pred = pred.squeeze()
            gt = gt.float()
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=1/self.weights[0])
        else:
            # one hot encoding for cross entropy with digits. Bx1xHxW -> BxCxHxW
            B, C, Z, H, W = pred.shape
            gt_flat = gt.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W

            gt_onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float)  # 1xB*Z*H*W destination tensor
            gt_onehot.scatter_(1, gt_flat, 1)  # writing the conversion in the destination tensor

            gt = torch.squeeze(gt_onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape
            pred = pred.permute(0, 2, 3, 4, 1)  # for BCE we want classes in the last axis
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.weights)

        return self.loss_fn(pred, gt)


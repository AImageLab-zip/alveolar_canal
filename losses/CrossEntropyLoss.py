import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None, apply_sigmoid=True):
        super().__init__()
        self.weights = weights
        self.apply_sigmoid = apply_sigmoid
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pred, gt):
        pred = self.sigmoid(pred)
        return self.loss_fn(pred, gt)

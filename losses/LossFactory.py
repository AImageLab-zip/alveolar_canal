import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from .DiceLoss import DiceLoss
from .JaccardLoss import JaccardLoss
from .CrossEntropyLoss import CrossEntropyLoss
from .BCEWithLogitsLoss import BCEWithLogitsLoss

class LossFactory:
    def __init__(self, names, classes, weights=None):
        self.names = names
        if not isinstance(self.names, list):
            self.names = [self.names]

        self.classes = classes
        self.weights = weights
        self.losses = []
        for name in self.names:
            loss = self.get_loss(name)
            self.losses.append(loss)

    def get_loss(self, name):
        if name == 'CrossEntropyLoss':
            loss_fn = CrossEntropyLoss(self.weights, True)
        elif name == 'BCEWithLogitsLoss':
            loss_fn = BCEWithLogitsLoss(self.weights)
        elif name == 'Jaccard':
            loss_fn = JaccardLoss(weight=self.weights, apply_sigmoid=True, per_volume=True)
        elif name == 'DiceLoss':
            loss_fn = DiceLoss(self.classes)
        else:
            raise Exception(f"Loss function {name} can't be found.")

        return loss_fn

    def __call__(self, pred, gt, partition_weights):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        assert pred.device == gt.device
        assert gt.device != 'cpu'

        cur_loss = []
        for lossfn in self.losses:
            loss = lossfn(pred, gt)
            if torch.isnan(loss.sum()):
                raise ValueError(f'Loss {lossfn} has some NaN')
            loss = loss * partition_weights
            cur_loss.append(loss.mean())
        return torch.sum(torch.stack(cur_loss))

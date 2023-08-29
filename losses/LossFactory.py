import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from .DiceLoss import DiceLoss
from .JaccardLoss import JaccardLoss
from .CrossEntropyLoss import CrossEntropyLoss
from .BCEWithLogitsLoss import BCEWithLogitsLoss
from .BoundaryLoss import BoundaryLoss
# from .CavityLoss import CavityLoss

class LossFactory:
    def __init__(self, names, classes, weights=None):
        self.names = names
        if not isinstance(self.names, list):
            self.names = [self.names]

        print(f'Losses used: {self.names}')
        self.classes = classes
        self.weights = weights
        self.losses = {}
        for name in self.names:
            loss = self.get_loss(name)
            self.losses[name] = loss

    def get_loss(self, name):
        if name == 'CrossEntropyLoss':
            loss_fn = CrossEntropyLoss(self.weights, True)
        elif name == 'BCEWithLogitsLoss':
            loss_fn = BCEWithLogitsLoss(self.weights)
        elif name == 'Jaccard':
            loss_fn = JaccardLoss(weight=self.weights)
        elif name == 'DiceLoss':
            loss_fn = DiceLoss(self.classes)
        elif name == 'BoundaryLoss':
            loss_fn = BoundaryLoss()
        # elif name=='CavityLoss':
        #     loss_fn = CavityLoss()
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

        # print(f'pred has: {pred.view(2, -1).sum(-1)}')
        # print(f'gt has: {gt.view(2, -1).sum(-1)}')

        cur_loss = []
        for loss_name in self.losses.keys():
            loss = self.losses[loss_name](pred, gt)
            if torch.isnan(loss.sum()):
                raise ValueError(f'Loss {loss_name} has some NaN')
            # print(f'Loss {self.losses[loss_name].__class__.__name__}: {loss}')
            loss = loss * partition_weights
            cur_loss.append(loss.mean())
        return torch.sum(torch.stack(cur_loss))

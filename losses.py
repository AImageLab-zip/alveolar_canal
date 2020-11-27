import numpy as np
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes

    def forward(self, pred, gt):
        c_score = []
        excluded = ['BACKGROUND', 'UNLABELED']
        labels = [v for k, v in self.classes.items() if k not in excluded]
        for c in labels:
            gt_class_idx = np.argwhere(gt.flatten() == c)
            intersection = np.sum(pred.flatten()[gt_class_idx] == c)
            dice_union = np.argwhere(gt.flatten() == c).size + np.argwhere(pred.flatten() == c).size
            c_score.append(1 - (2 * intersection + self.eps) / (dice_union + self.eps))
        return sum(c_score) / len(labels)


class WeightedBinaryCrossEntropy(nn.Module):
    # this class is a test: NOT working
    def __init__(self, class_weights):
        super().__init__()
        self.weights = class_weights

    def forward(self, pred, gt):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (gt * torch.log(pred)) + \
                   self.weights[0] * ((1 - gt) * torch.log(1 - pred))
        else:
            loss = gt * torch.log(pred) + (1 - gt) * torch.log(1 - pred)

        return torch.neg(torch.mean(loss))


class LossFn:
    def __init__(self, loss_config, loader_config, device):

        if not isinstance(loss_config['name'], list):
            self.name = [loss_config['name']]
        else:
            self.name = loss_config['name']
        self.loader_config = loader_config
        self.device = device
        self.classes = loader_config['labels']

    def factory_loss(self, pred, gt, name, warmup, weights):

        if name == 'CrossEntropyLoss':
            # sigmoid here which is included in other losses
            pred = torch.nn.Sigmoid()(pred)
            loss_fn = nn.CrossEntropyLoss(weight=weights).to(self.device)
        elif name == 'BCEWithLogitsLoss':
            # one hot encoding for cross entropy with digits. Bx1xHxW -> BxCxHxW
            B, C, Z, H, W = pred.shape
            gt_flat = gt.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W

            gt_onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float).to(self.device)  # 1xB*Z*H*W destination tensor
            gt_onehot.scatter_(1, gt_flat, 1)  # writing the conversion in the destination tensor

            gt = torch.squeeze(gt_onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape
            pred = pred.permute(0, 2, 3, 4, 1)  # for BCE we want classes in the last axis
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights).to(self.device)
        elif name == 'BCELoss':
            # one hot encoding for cross entropy with digits. Bx1xHxW -> BxCxHxW
            B, C, Z, H, W = pred.shape
            gt_flat = gt.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W

            gt_onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float).to(
                self.device)  # 1xB*Z*H*W destination tensor
            gt_onehot.scatter_(1, gt_flat, 1)  # writing the conversion in the destination tensor

            gt = torch.squeeze(gt_onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape
            pred = pred.permute(0, 2, 3, 4, 1)  # for BCE we want classes in the last axis
            loss_fn = nn.BCELoss().to(self.device)
            # loss_fn = WeightedBinaryCrossEntropy(weights).to(self.device)
        elif name == 'DiceLoss':
            pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
            pred = pred.data.cpu().numpy()
            gt = gt.cpu().numpy()
            return warmup * DiceLoss(self.classes)(pred, gt)
        else:
            raise Exception("specified loss function cant be found.")

        return loss_fn(pred, gt)

    def __call__(self, pred, gt, warmup, weights):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        result = []
        for name in self.name:
            result.append(self.factory_loss(pred, gt, name, warmup, weights))
        return sum(result)

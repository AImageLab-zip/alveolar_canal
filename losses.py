import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images
    return losses.mean()


class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)


class DiceLoss(nn.Module):
    def __init__(self, classes, device):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes
        self.device = device

    def forward(self, pred, gt):
        included = [v for k, v in self.classes.items() if k not in ['UNLABELED']]

        gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=len(self.classes))
        if gt_onehot.ndim == 3:
            gt_onehot = gt_onehot.unsqueeze(0)

        gt_onehot = torch.movedim(gt_onehot, -1, 1)
        input_soft = F.softmax(pred, dim=1)
        dims = (2, 3, 4)

        intersection = torch.sum(input_soft * gt_onehot, dims)
        cardinality = torch.sum(input_soft + gt_onehot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score[:, included])


def one_hot_encode(volume, shape, device):
    B, C, Z, H, W = shape
    flat = volume.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W
    onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float).to(device)  # 1xB*Z*H*W destination tensor
    onehot.scatter_(1, flat, 1)  # writing the conversion in the destination tensor
    return torch.squeeze(onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape


class LossFn:
    def __init__(self, loss_config, loader_config, weights):

        if not isinstance(loss_config['name'], list):
            self.name = [loss_config['name']]
        else:
            self.name = loss_config['name']
        self.loader_config = loader_config
        self.classes = loader_config['labels']
        self.weights = weights

    def factory_loss(self, pred, gt, name):

        if name == 'CrossEntropyLoss':
            pred = torch.nn.Sigmoid()(pred)  # sigmoid here which is already built-in in other losses
            loss_fn = nn.CrossEntropyLoss(weight=self.weights).to(self.device)
        elif name == 'BCEWithLogitsLoss':
            if pred.shape[1] == 1:
                pred = pred.squeeze()
                gt = gt.float()
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=1/self.weights[0]).to(self.device)
            else:
                # one hot encoding for cross entropy with digits. Bx1xHxW -> BxCxHxW
                B, C, Z, H, W = pred.shape
                gt_flat = gt.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W

                gt_onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float).to(self.device)  # 1xB*Z*H*W destination tensor
                gt_onehot.scatter_(1, gt_flat, 1)  # writing the conversion in the destination tensor

                gt = torch.squeeze(gt_onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape
                pred = pred.permute(0, 2, 3, 4, 1)  # for BCE we want classes in the last axis
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.weights).to(self.device)
        elif name == 'Jaccard':
            assert pred.shape[1] == 1, 'this loss works with a binary prediction'
            return JaccardLoss(weight=self.weights, apply_sigmoid=True)(pred, gt)
        elif name == 'DiceLoss':
            # pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
            # pred = pred.data.cpu().numpy()
            # gt = gt.cpu().numpy()
            return DiceLoss(self.classes, self.device)(pred, gt)
        else:
            raise Exception("specified loss function cant be found.")

        return loss_fn(pred, gt)

    def __call__(self, pred, gt):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        assert pred.device == gt.device
        assert gt.device != 'cpu'
        self.device = pred.device

        cur_loss = []
        for name in self.name:
            loss = self.factory_loss(pred, gt, name)
            if torch.isnan(loss):
                raise ValueError('Loss is nan during training...')
            cur_loss.append(loss)
        return torch.sum(torch.stack(cur_loss))

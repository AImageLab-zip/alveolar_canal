from statistics import mean
import torch
import pathlib
import numpy as np
from skimage import metrics
import os
import pandas as pd
import zipfile

class Eval:
    def __init__(self, config, skip_dump=False):
        self.iou_list = []
        self.dice_list = []

    def reset_eval(self):
        self.iou_list.clear()
        self.dice_list.clear()

    def compute_metrics(self, pred, gt):
        pred = pred.detach().to(torch.uint8)
        gt = gt.detach().to(torch.uint8)

        pred = pred.cuda()
        gt = gt.cuda()

        pred = pred[None, ...] if pred.ndim == 3 else pred
        gt = gt[None, ...] if gt.ndim == 3 else gt

        iou, dice = self.iou_and_dice(pred, gt)
        self.iou_list.append(iou)
        self.dice_list.append(dice)

    def iou_and_dice(self, pred, gt):
        eps = 1e-6
        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        iou_union = dice_union - intersection

        iou = (intersection + eps) / (iou_union + eps)
        dice = (2 * intersection + eps) / (dice_union + eps)

        return iou.item(), dice.item()

    def mean_metric(self, phase):
        iou = 0 if len(self.iou_list) == 0 else mean(self.iou_list)
        dice = 0 if len(self.dice_list) == 0 else mean(self.dice_list)

        self.reset_eval()
        return iou, dice

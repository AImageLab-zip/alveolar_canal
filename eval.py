from statistics import mean
import torch
import numpy as np


class Eval:
    def __init__(self, loader_config):
        self.metric_list = []
        self.eps = 1e-06
        self.classes = loader_config['labels']

    def reset_eval(self):
        self.metric_list.clear()

    def mean_metric(self):
        return mean(self.metric_list)

    def iou(self, predition, groundtruth):
        """
        SHAPE MUST BE
        :param image:
        :param gt:
        :return:
        """
        predition = predition[None, ...] if predition.ndim == 3 else predition
        groundtruth = groundtruth[None, ...] if groundtruth.ndim == 3 else groundtruth

        excluded = ['BACKGROUND', 'UNLABELED']
        labels = [v for k, v in self.classes.items() if k not in excluded]  # exclude background from here
        for batch_id in range(predition.shape[0]):
            pred = predition[batch_id]
            gt = groundtruth[batch_id]
            c_score = []
            for c in labels:
                gt_class_idx = np.argwhere(gt.flatten() == c)
                intersection = np.sum(pred.flatten()[gt_class_idx] == c)
                union = np.argwhere(gt.flatten() == c).size + np.argwhere(pred.flatten() == c).size - intersection
                c_score.append((intersection + self.eps) / (union + self.eps))
            self.metric_list.append(sum(c_score) / len(labels))


    def dice_coefficient(self, pred, gt):
        c_score = []
        excluded = ['BACKGROUND', 'UNLABELED']
        labels = [v for k, v in self.classes.items() if k not in excluded]  # exclude background from here
        for c in labels:
            gt_class_idx = np.argwhere(gt.flatten() == c)
            intersection = np.sum(pred.flatten()[gt_class_idx] == c)
            dice_union = np.argwhere(gt.flatten() == c).size + np.argwhere(pred.flatten() == c).size
            c_score.append((2 * intersection + self.eps) / (dice_union + self.eps))
        self.metric_list.append(sum(c_score) / len(labels))


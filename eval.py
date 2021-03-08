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

    def single_class(self, label_pred, label_gt, num_classes=1, threshold=0.5, ignore_label=255, dims_to_keep=(1,)):
        assert label_pred.shape == label_gt.shape
        sum_axis = tuple([i for i in tuple(range(len(label_gt.shape))) if i not in dims_to_keep])
        threshold_mask = torch.where(label_pred > threshold, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))

        if num_classes != 1:
            classes_dummy = torch.arange(num_classes, device='cuda')[None, :, None, None]
            label_gt = torch.unsqueeze(label_gt, dim=1)
            argmax_mask = (torch.unsqueeze(label_pred.argmax(axis=1), dim=1) == classes_dummy).int()
            c_lbpred = argmax_mask * threshold_mask
            c_lbgt = (label_gt == classes_dummy).int()
            ignore_mask = torch.where(label_gt != ignore_label, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
            c_lbpred = c_lbpred * ignore_mask
            c_lbgt = c_lbgt * ignore_mask
        else:
            c_lbpred = threshold_mask
            c_lbgt = label_gt

        product = c_lbpred * c_lbgt
        eps = 1e-6
        num = torch.sum(product, dim=sum_axis)
        den = (torch.sum(c_lbgt, dim=sum_axis) + torch.sum(c_lbpred, dim=sum_axis))
        self.metric_list.append((num + eps) / (den - num + eps))
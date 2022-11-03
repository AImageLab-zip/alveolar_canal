import torch
import numpy as np

from scipy.ndimage import distance_transform_edt

class BoundaryLoss():
    def __init__(self, weight=1, **kwargs):
        self.weight = weight
        # self.distance_transform = DistanceTransform()
        pass

    def __call__(self, preds, gt_dist):
        batch_size = preds.shape[0]

        preds = preds.view(batch_size, -1)
        gt_dist = gt_dist.view(batch_size, -1)

        product = preds*gt_dist
        loss = torch.mean(product, dim=-1)

        return loss*self.weight


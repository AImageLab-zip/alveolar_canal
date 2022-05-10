from dataloader.DistanceTransform import DistanceTransform

class BoundaryLoss():
    def __init__(self, **kwargs):
        self.distance_transform = DistanceTransform()
        pass

    def __call__(self, preds, gt):
        gt = gt.clone()
        for b in range(gt.shape[0]):
            gt[b] = self.distance_transform(gt[b])
        return (preds*gt).mean()

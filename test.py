import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop
import numpy as np
import torchio as tio


def test(model, test_loader, epoch, evaluator, dumper=None, final_mean=True):

    model.eval()

    with torch.no_grad():
        evaluator.reset_eval()
        for i, (subject, loader) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):
            aggr = tio.inference.GridAggregator(subject)
            for subvolume in loader:
                images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                emb_codes = subvolume[tio.LOCATION].float().cuda()

                output = model(images, emb_codes)  # BS, Classes, Z, H, W

                aggr.add_batch(output, subvolume[tio.LOCATION])

            output = aggr.get_output_tensor()
            labels = np.load(subject[0]['gt_path'])  # original labels from storage
            images = np.load(subject[0]['data_path'])  # high resolution image from storage

            D, H, W = labels.shape[-3:]
            rD, rH, rW = output.shape[-3:]
            tmp_ratio = np.array((D / W, H / W, 1))
            pad_factor = tmp_ratio / np.array((rD / rW, rH / rW, 1))
            pad_factor /= np.max(pad_factor)
            reshape_size = np.array((D, H, W)) / pad_factor
            reshape_size = np.round(reshape_size).astype(np.int)

            output = interpolate(output.unsqueeze(0), size=tuple(reshape_size), mode='trilinear', align_corners=False).squeeze()  # (classes, Z, H, W) or (Z, H, W) if classes = 1
            output = CenterCrop((D, H, W))(output)

            # final predictions
            if output.ndim > 3:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
            else:
                output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                output[output > .5] = 1
                output[output != 1] = 0
                output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W

            evaluator.iou(output, labels)

            if dumper is not None:
                dumper.dump(labels, output, images, subject[0]['folder'], score=evaluator.metric_list[-1])

    if final_mean:
        epoch_val_metric = evaluator.mean_metric()
        return epoch_val_metric
    else:
        return evaluator.metric_list

import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop
import numpy as np
import torchio as tio
import logging


def test(model, test_loader, epoch, writer, evaluator, type):

    model.eval()

    with torch.no_grad():
        evaluator.reset_eval()
        for i, (subject, loader) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):
            aggr = tio.inference.GridAggregator(subject, overlap_mode='average')
            for subvolume in loader:
                # batchsize with torchio affects the number of grids we extract from a patient.
                # when we aggragate the patient the volume is just one.
                images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                emb_codes = subvolume[tio.LOCATION].float().cuda()

                output = model(images, emb_codes)  # BS, Classes, Z, H, W

                aggr.add_batch(output, subvolume[tio.LOCATION])

            output = aggr.get_output_tensor()  # C, Z, H, W
            labels = np.load(subject[0]['gt_path'])  # original labels from storage
            images = np.load(subject[0]['data_path'])  # high resolution image from storage

            D, H, W = labels.shape[-3:]
            rD, rH, rW = output.shape[-3:]
            tmp_ratio = np.array((D / W, H / W, 1))
            pad_factor = tmp_ratio / np.array((rD / rW, rH / rW, 1))
            pad_factor /= np.max(pad_factor)
            reshape_size = np.array((D, H, W)) / pad_factor
            reshape_size = np.round(reshape_size).astype(np.int)

            # BS=1, after interpolate we can have (classes, Z, H, W) or (Z, H, W) if classes = 1
            output = interpolate(output.unsqueeze(0), size=tuple(reshape_size), mode='trilinear', align_corners=False).squeeze()
            output = CenterCrop((D, H, W))(output)

            # final predictions
            if output.ndim > 3:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
            else:
                output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                output[output > .5] = 1
                output[output != 1] = 0
                output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W

            evaluator.compute_metrics(output, labels, images, subject[0]['folder'], type)

            # TB DUMP FOR BINARY CASE!
            # images = np.clip(images, 0, None)
            # images = (images.astype(np.float))/images.max()
            # if writer is not None:
            #     unempty_idx = np.argwhere(np.sum(labels != config['labels']['BACKGROUND'], axis=(0, 2)) > 0)
            #     randidx = np.random.randint(0, unempty_idx.size - 1, 5)
            #     rand_unempty_idx = unempty_idx[randidx].squeeze()  # random slices from unempty ones
            #
            #     dump_img = np.concatenate(np.moveaxis(images[:, rand_unempty_idx], 0, 1))
            #
            #     dump_gt = np.concatenate(np.moveaxis(labels[:, rand_unempty_idx], 0, 1))
            #     dump_pred = np.concatenate(np.moveaxis(output[:, rand_unempty_idx], 0, 1))
            #
            #     dump_img = np.stack((dump_img, dump_img, dump_img), axis=-1)
            #     a = dump_img.copy()
            #     a[dump_pred == config['labels']['INSIDE']] = (0, 0, 1)
            #     b = dump_img.copy()
            #     b[dump_gt == config['labels']['INSIDE']] = (0, 0, 1)
            #     dump_img = np.concatenate((a, b), axis=-2)
            #     writer.add_image(
            #         "3D_results",
            #         dump_img,
            #         len(test_loader) * epoch + i,
            #         dataformats='HWC'
            #     )
            # END OF THE DUMP

    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=type)
    if writer is not None and type != "Final":
        writer.add_scalar(f'Metric/{type}/IoU', epoch_iou, epoch)
        writer.add_scalar(f'Metric/{type}/Dice', epoch_dice, epoch)
        writer.add_scalar(f'Metric/{type}/Hauss', epoch_haus, epoch)

    logging.info(
        f'{type} Epoch [{epoch}], '
        f'{type} Mean Metric (IoU): {epoch_iou}'
        f'{type} Mean Metric (Dice): {epoch_dice}'
        f'{type} Mean Metric (haus): {epoch_haus}'
    )

    return epoch_iou, epoch_dice, epoch_haus

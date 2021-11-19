import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop
import numpy as np
import torchio as tio
import logging
from augmentations import CropAndPad


def test2D(model, test_loader, epoch, writer, evaluator, phase, splitter):

    num_splits = splitter.get_batch()
    whole_image, whole_output, whole_names = [], [], []
    patient_count = 0
    model.eval()
    with torch.no_grad():
        evaluator.reset_eval()
        for i, (images, labels, names, partition_weights, gt_paths) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):

            images = images.cuda()  # BS, 3, H, W

            output = model(images)  # BS, Classes, H, W

            if isinstance(output, tuple):
                output, _, _ = output

            whole_image += list(images.unsqueeze(-2).cpu())
            whole_output += list(output.unsqueeze(-2).cpu())
            whole_names += names

            while len(whole_image) >= num_splits:  # volume completed. let's evaluate it

                patient_count += 1

                images = splitter.merge(whole_image[:num_splits])
                output = splitter.merge(whole_output[:num_splits])  # Classes, Z, H, W
                name = set(whole_names[:num_splits])  # keep unique names from this sub-volume
                assert len(name) == 1, "mixed patients!"  # must be just one!
                name = name.pop()

                gt_path = set(gt_paths[:num_splits])  # keep unique names from this sub-volume
                assert len(gt_path) == 1, "mixed patients!"  # must be just one!
                labels = np.load(gt_path.pop())

                D, H, W = labels.shape[-3:]
                rD, rH, rW = output.shape[-3:]
                tmp_ratio = np.array((D / W, H / W, 1))
                pad_factor = tmp_ratio / np.array((rD / rW, rH / rW, 1))
                pad_factor /= np.max(pad_factor)
                reshape_size = np.array((D, H, W)) / pad_factor
                reshape_size = np.round(reshape_size).astype(np.int)

                output = interpolate(output.unsqueeze(0), size=tuple(reshape_size), mode='trilinear', align_corners=False).squeeze()  # (classes, Z, H, W) or (Z, H, W) if classes = 1
                output = CenterCrop((D, H, W))(output)

                images = interpolate(images.view(1, *images.shape), size=tuple(reshape_size), mode='trilinear', align_corners=False).squeeze()
                images = CenterCrop((D, H, W))(images)[0]  # Z, H W

                # final predictions
                if output.ndim > 3:
                    output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
                else:
                    output = nn.Sigmoid()(output)  # Z, H, W
                    output[output > .5] = 1
                    output[output != 1] = 0
                    output = output.squeeze().cpu().detach().numpy()  # Z, H, W
                labels = labels.squeeze()
                images = images.numpy()

                evaluator.compute_metrics(output, labels, images, name, phase)

                # TB DUMP
                # if writer is not None:
                #     unempty_idx = np.argwhere(np.sum(labels != 2, axis=(0, 2)) > 0)
                #     randidx = np.random.randint(0, unempty_idx.size - 1, 5)
                #     rand_unempty_idx = unempty_idx[randidx].squeeze()  # random slices from unempty ones
                #
                #     dump_img = np.concatenate(np.moveaxis(images[:, rand_unempty_idx], 0, 1))
                #     dump_img = dump_img * config['std'] + config['mean']
                #
                #     dump_gt = np.concatenate(np.moveaxis(labels[:, rand_unempty_idx], 0, 1))
                #     dump_pred = np.concatenate(np.moveaxis(output[:, rand_unempty_idx], 0, 1))
                #
                #     dump_img = np.stack((dump_img, dump_img, dump_img), axis=-1)
                #     a = dump_img.copy()
                #     a[dump_pred == config['labels']['INSIDE']] = (1, 0, 0)
                #     # a[dump_pred == config['labels']['CONTOUR']] = (0, 0, 1)
                #     b = dump_img.copy()
                #     b[dump_gt == config['labels']['INSIDE']] = (1, 0, 0)
                #     # b[dump_gt == config['labels']['CONTOUR']] = (0, 0, 1)
                #     dump_img = np.concatenate((a, b), axis=-2)
                #     writer.add_image(
                #         "2D_results",
                #         dump_img,
                #         epoch * len(test_loader) * config['batch_size'] / num_splits + patient_count,
                #         dataformats='HWC'
                #     )
                # END OF THE DUMP

                # update the container by removing the processed patient.
                whole_image = whole_image[num_splits:]
                whole_output = whole_output[num_splits:]
                whole_names = whole_names[num_splits:]
                gt_paths = gt_paths[num_splits:]

    assert len(whole_output) == 0, "something wrong here"
    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=phase)
    if writer is not None and phase != "Final":
        writer.add_scalar(f'{phase}/IoU', epoch_iou, epoch)
        writer.add_scalar(f'{phase}/Dice', epoch_dice, epoch)
        writer.add_scalar(f'{phase}/Hauss', epoch_haus, epoch)

    if phase in ['Test', 'Final']:
        logging.info(
            f'{phase} Epoch [{epoch}], '
            f'{phase} Mean Metric (IoU): {epoch_iou}'
            f'{phase} Mean Metric (Dice): {epoch_dice}'
            f'{phase} Mean Metric (haus): {epoch_haus}'
        )

    return epoch_iou, epoch_dice, epoch_haus


def test3D(model, test_loader, epoch, writer, evaluator, phase):

    model.eval()

    with torch.no_grad():
        evaluator.reset_eval()
        for i, (subject, loader) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):
            aggr = tio.inference.GridAggregator(subject, overlap_mode='average')
            import time
            start_time = time.time()
            for subvolume in loader:
                # batchsize with torchio affects the number of grids we extract from a patient.
                # when we aggragate the patient the volume is just one.

                images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                emb_codes = subvolume[tio.LOCATION].float().cuda()

                output = model(images, emb_codes)  # BS, Classes, Z, H, W

                aggr.add_batch(output, subvolume[tio.LOCATION])

            output = aggr.get_output_tensor()  # C, Z, H, W
            print("--- %s seconds ---" % (time.time() - start_time))
            labels = np.load(subject[0]['gt_path'])  # original labels from storage
            images = np.load(subject[0]['data_path'])  # high resolution image from storage

            orig_shape = labels.shape[-3:]
            output = CropAndPad(orig_shape)(output).squeeze()  # keep pad_val = min(output) since we are dealing with probabilities

            # final predictions
            if output.ndim > 3:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
            else:
                output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                output = torch.where(output > .5, 1, 0)
                output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W

            evaluator.compute_metrics(output, labels, images, subject[0]['folder'], phase)

            # TB DUMP FOR BINARY CASE!
            # images = np.clip(images, 0, None)
            # images = (images.asphase(np.float))/images.max()
            # if writer is not None:
            #     unempty_idx = np.argwhere(np.sum(labels != config['labels']['BACKGROUND'], axis=(0, 2)) > 0)
            #     randidx = np.random.randint(0, unempty_idx.size - 1, 5)
            #     rand_unempty_idx = unempty_idx[randidx].squeeze()  # random slices from unempty ones
            #ok
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

    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=phase)
    if writer is not None and phase != "Final":
        writer.add_scalar(f'{phase}/IoU', epoch_iou, epoch)
        writer.add_scalar(f'{phase}/Dice', epoch_dice, epoch)
        writer.add_scalar(f'{phase}/Hauss', epoch_haus, epoch)

    if phase in ['Test', 'Final']:
        logging.info(
            f'{phase} Epoch [{epoch}], '
            f'{phase} Mean Metric (IoU): {epoch_iou}'
            f'{phase} Mean Metric (Dice): {epoch_dice}'
            f'{phase} Mean Metric (haus): {epoch_haus}'
        )

    return epoch_iou, epoch_dice, epoch_haus

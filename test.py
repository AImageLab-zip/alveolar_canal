import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop
import numpy as np


def test(model, test_loader, splitter, epoch, evaluator, dumper=None, final_mean=True):

    num_splits = splitter.get_batch()
    whole_image, whole_labels, whole_output, whole_names = [], [], [], []
    patient_count = 0
    model.eval()
    with torch.no_grad():
        evaluator.reset_eval()
        for i, (images, labels, names) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):

            images = images.cuda()  # BS, 3, Z, H, W
            output = model(images)  # BS, Classes, Z, H, W

            whole_image += list(images.cpu())
            whole_labels += labels
            whole_output += list(output.cpu())
            whole_names += names

            while len(whole_labels) >= num_splits:  # volume completed. let's evaluate it

                images = splitter.merge(whole_image[:num_splits])
                output = splitter.merge(whole_output[:num_splits])  # Classes, Z, H, W
                labels = splitter.merge(whole_labels[:num_splits])  # Z, H, W
                name = set(whole_names[:num_splits])  # keep unique names from this sub-volume
                assert len(name) == 1, "mixed patients!"  # must be just one!

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
                    output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                    output[output > .5] = 1
                    output[output != 1] = 0
                    output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W
                labels = labels.numpy()
                images = images.numpy()

                evaluator.iou(output, labels)

                if dumper is not None:
                    dumper.dump(labels, output, images, name.pop(), score=evaluator.metric_list[-1])
                    patient_count += 1

                whole_labels = whole_labels[num_splits:]
                whole_image = whole_image[num_splits:]
                whole_output = whole_output[num_splits:]
                whole_names = whole_names[num_splits:]

    assert len(whole_output) == 0, "something wrong here"
    if final_mean:
        epoch_val_metric = evaluator.mean_metric()
        return epoch_val_metric
    else:
        return evaluator.metric_list

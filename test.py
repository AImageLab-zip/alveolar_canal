import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import Resize
from torch.cuda.amp import autocast


def test(model, test_loader,  loss_fn, epoch, writer, evaluator, warmup, dumper=None):

    model.eval()
    with torch.no_grad():
        evaluator.reset_eval()
        for i, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):

            images = images.to('cuda:0')
            labels = [label.to('cuda:0') for label in labels]

            output = []

            with autocast():
                outputs = model(images)
                for batch_id in range(outputs.shape[0]):
                    output.append(
                        interpolate(
                            outputs[batch_id].unsqueeze(dim=0),
                            size=(labels[batch_id].shape[-3:]),
                            mode='trilinear',
                            align_corners=False
                        )
                    )
                del outputs
                losses = loss_fn(output, labels, warmup)

            if np.any(np.isnan([loss.item() for loss in losses])):
                raise ValueError('Loss is nan during training...')

            # final predictions
            for batch_id in range(len(output)):
                output[batch_id] = torch.argmax(torch.nn.Softmax(dim=0)(output[batch_id].squeeze()), dim=0).cpu().numpy()
                labels[batch_id] = labels[batch_id].squeeze().cpu().numpy()

            evaluator.iou(output, labels)

            # if dumper is not None:
            #     dumper.dump(labels, output, i)

    epoch_val_loss = sum([loss.item() for loss in losses]) / len(losses)
    epoch_val_metric = evaluator.mean_metric()
    writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
    writer.add_scalar('Metric/validation', epoch_val_metric, epoch)
    logging.info(
        f'Validation Epoch [{epoch}], '
        f'Validation Mean Loss: {epoch_val_loss}, '
        f'Validation Mean Metric: {epoch_val_metric}'
    )

    return epoch_val_metric

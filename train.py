import numpy as np
import torch
import logging
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.nn.functional import interpolate
from torch.cuda.amp import autocast


def train(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, warmup):

    model.train()
    loss_list = []
    evaluator.reset_eval()
    scaler = GradScaler()

    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

        images = images.to('cuda:0')

        optimizer.zero_grad()
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
            loss_device = outputs.device
            del outputs
            labels = [label.to(loss_device) for label in labels]
            losses = loss_fn(output, labels, warmup)

        if np.any(np.isnan([loss.item() for loss in losses])):
            raise ValueError('Loss is nan during training...')

        # cur_loss.backward()
        for loss_n, loss in enumerate(losses):
            if loss_n != (len(losses) - 1):
                scaler.scale(loss).backward(retain_graph=True)
            else:
                scaler.scale(loss).backward()

        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        # final predictions
        for batch_id in range(len(output)):
            output[batch_id] = torch.argmax(torch.nn.Softmax(dim=0)(output[batch_id].squeeze()), dim=0).cpu().numpy()
            labels[batch_id] = labels[batch_id].squeeze().cpu().numpy()

        evaluator.iou(output, labels)

    epoch_train_loss = sum([loss.item() for loss in losses]) / len(losses)
    epoch_train_metric = evaluator.mean_metric()
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Metric/train', epoch_train_metric, epoch)

    logging.info(
        f'Train Epoch [{epoch}], '
        f'Train Mean Loss: {epoch_train_loss}, '
        f'Train Mean Metric: {epoch_train_metric}'
    )
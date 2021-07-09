import torch
import logging
from tqdm import tqdm
from torch import nn
import torchio as tio
import torch.distributed as dist


def train2D(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, phase='Train'):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, (images, labels, names, partition_weights, _) in tqdm(enumerate(train_loader), total=len(train_loader),
                                       desc='train epoch {}'.format(str(epoch))):

        images = images.cuda()
        labels = labels.cuda()
        partition_weights = partition_weights.cuda()

        optimizer.zero_grad()

        outputs = model(images)  # BS, Classes, H, W
        loss = loss_fn(outputs, labels, partition_weights)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # final predictions
        if outputs.shape[1] > 1:
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        else:
            outputs = nn.Sigmoid()(outputs)  # BS, 1, H, W
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0
            outputs = outputs.squeeze().cpu().detach().numpy()  # BS, H, W

        labels = labels.squeeze().cpu().numpy()  # BS, Z, H, W
        evaluator.compute_metrics(outputs, labels, images, names, phase)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=phase)
    if writer is not None:
        writer.add_scalar(f'Loss/{phase}', epoch_train_loss, epoch)
        writer.add_scalar(f'{phase}', epoch_iou, epoch)

    logging.info(
        f'{phase} Epoch [{epoch}], '
        f'{phase} Mean Loss: {epoch_train_loss}, '
        f'{phase} Mean Metric (IoU): {epoch_iou}'
        f'{phase} Mean Metric (Dice): {epoch_dice}'
        f'{phase} Mean Metric (haus): {epoch_haus}'
    )
    return epoch_train_loss, epoch_iou


def train3D(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, phase='Train', competitor=False):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, d in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{phase} epoch {str(epoch)}'):
        images = d['data'][tio.DATA].float().cuda()

        if competitor:
            labels = d['sparse'][tio.DATA].cuda()[:, 0:1]  # sparse gt - temporary option for the competitor
        else:
            labels = d['label'][tio.DATA].cuda()

        emb_codes = torch.cat((
            d['index_ini'],
            d['index_ini'] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()
        partition_weights = d['weight'].cuda()

        optimizer.zero_grad()
        outputs = model(images, emb_codes)  # output -> B, C, Z, H, W
        assert outputs.ndim == labels.ndim, f"Gt and output dimensions are not the same before loss. {outputs.ndim} vs {labels.ndim}"

        loss = loss_fn(outputs, labels, partition_weights)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # final predictions
        # shape B, C, xyz -> softmax -> B, xyz
        # shape 1, C, xyz -> softmax -> 1, xyz
        # shape B, 1, xyz -> sigmoid + sqz -> B, xyz
        # shape B, 1, xyz -> sigmoid + sqz -> xyz
        if outputs.shape[1] > 1:
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        else:
            outputs = nn.Sigmoid()(outputs)  # BS, 1, Z, H, W
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0
            outputs = outputs.squeeze().cpu().detach().numpy()  # BS, Z, H, W

        labels = labels.squeeze().cpu().numpy()  # BS, Z, H, W
        evaluator.compute_metrics(outputs, labels, images, d['folder'], phase)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=phase)
    if writer is not None:
        writer.add_scalar(f'Loss/{phase}', epoch_train_loss, epoch)
        writer.add_scalar(f'{phase}', epoch_iou, epoch)

    logging.info(
        f'{phase} Epoch [{epoch}], '
        f'{phase} Mean Loss: {epoch_train_loss}, '
        f'{phase} Mean Metric (IoU): {epoch_iou}'
        f'{phase} Mean Metric (Dice): {epoch_dice}'
        f'{phase} Mean Metric (haus): {epoch_haus}'
    )

    return epoch_train_loss, epoch_iou
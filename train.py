import numpy as np
import torch
import logging
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


def train(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, warmup):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels, warmup)

        losses.append(loss)
        loss.backward()
        optimizer.step()

        # final predictions
        outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        evaluator.iou(outputs, labels)

    epoch_train_loss = sum([loss.item() for loss in losses]) / len(losses)
    epoch_train_metric = evaluator.mean_metric()
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Metric/train', epoch_train_metric, epoch)

    logging.info(
        f'Train Epoch [{epoch}], '
        f'Train Mean Loss: {epoch_train_loss}, '
        f'Train Mean Metric: {epoch_train_metric}'
    )
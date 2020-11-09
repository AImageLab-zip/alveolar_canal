import numpy as np
import torch
import logging
from tqdm import tqdm


def train(model, train_loader, loss_fn, optimizer, device, epoch, writer, evaluator, warmup):

    model.train()
    loss_list = []
    evaluator.reset_eval()

    for i, (images, labels, weights) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        cur_loss = loss_fn(outputs, labels, warmup, weights)
        if np.isnan(cur_loss.item()):
            raise ValueError('Loss is nan during training...')
        loss_list.append(cur_loss.item())

        cur_loss.backward()
        optimizer.step()

        # final predictions
        outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1)

        # Track the metric
        outputs = outputs.data.cpu().numpy()
        labels = labels.cpu().numpy()
        evaluator.iou(outputs, labels)

    epoch_train_loss = sum(loss_list) / len(loss_list)
    epoch_train_metric = evaluator.mean_metric()
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Metric/train', epoch_train_metric, epoch)

    logging.info(
        f'Train Epoch [{epoch}], '
        f'Train Mean Loss: {epoch_train_loss}, '
        f'Train Mean Metric: {epoch_train_metric}'
    )
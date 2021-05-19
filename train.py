import torch
import logging
from tqdm import tqdm
from torch import nn
import torchio as tio

def train(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, (d) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

        images = d['data'][tio.DATA].float().cuda()
        labels = d['label'][tio.DATA].cuda()
        emb_codes = torch.cat((
            d['index_ini'],
            d['index_ini'] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()

        optimizer.zero_grad()

        outputs = model(images, emb_codes)  # BS, Classes, Z, H, W
        loss = loss_fn(outputs, labels)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # final predictions
        if outputs.shape[1] > 1:
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        else:
            outputs = nn.Sigmoid()(outputs)  # BS, 1, Z, H, W
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0
            outputs = outputs.squeeze().cpu().detach().numpy()  # BS, Z, H, W

        labels = labels.cpu().numpy()  # BS, Z, H, W
        evaluator.iou(outputs, labels)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_train_metric = evaluator.mean_metric()
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Metric/train', epoch_train_metric, epoch)

    logging.info(
        f'Train Epoch [{epoch}], '
        f'Train Mean Loss: {epoch_train_loss}, '
        f'Train Mean Metric: {epoch_train_metric}'
    )

    return epoch_train_loss, epoch_train_metric
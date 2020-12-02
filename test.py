import torch
import logging
import numpy as np
from tqdm import tqdm


def test(model, test_loader, loss_fn, device, epoch, writer, evaluator, warmup, dumper=None):

    model.eval()
    with torch.no_grad():
        loss_list = []
        evaluator.reset_eval()
        for i, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            cur_loss = loss_fn(outputs, labels, warmup)
            loss_list.append(cur_loss.item())

            # final predictions
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1)

            # Track the metric
            outputs = outputs.data.cpu().numpy()
            images = images.data.cpu().numpy()
            labels = labels.cpu().numpy()
            evaluator.iou(outputs, labels)

            if dumper is not None:
                dumper.dump(images, labels, outputs, i)


    epoch_val_loss = sum(loss_list) / len(loss_list)
    epoch_val_metric = evaluator.mean_metric()
    # writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
    writer.add_scalar('Metric/validation', epoch_val_metric, epoch)
    logging.info(
        f'Validation Epoch [{epoch}], '
        f'Validation Mean Loss: {epoch_val_loss}, '
        f'Validation Mean Metric: {epoch_val_metric}'
    )

    return epoch_val_metric

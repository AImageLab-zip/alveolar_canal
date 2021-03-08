import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop


def test(model, test_loader,  loss_fn, epoch, evaluator, warmup, dumper=None):

    model.eval()
    with torch.no_grad():
        evaluator.reset_eval()
        for i, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):

            images = images.cuda()

            output = model(images)

            images = torch.cat((images[0], images[1]), dim=-1)
            output = torch.cat((output[0], output[1]), dim=-1)
            labels = torch.cat((labels[0], labels[1]), dim=-1)

            Z, H, W = labels.shape[-3:]
            reshape_size = output.shape[-3:]

            ratio = reshape_size[1] / reshape_size[2]
            new_shape = (Z, H, int(H // ratio)) if H / W > ratio else (Z, int(W * ratio), W)

            output = interpolate(output.unsqueeze(0), size=(new_shape), mode='trilinear', align_corners=False).squeeze()
            output = CenterCrop((Z, H, W))(output)

            images = interpolate(images.view(1, *images.shape), size=(new_shape), mode='trilinear', align_corners=False).squeeze()
            images = CenterCrop((Z, H, W))(images)

            # final predictions
            if output.ndim == 4 and output.shape[0] > 1:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).cpu().numpy()
            else:
                output = nn.Sigmoid()(output)
                output[output > .5] = 1
                output[output != 1] = 0
                output = output.squeeze().cpu().detach().numpy()

            labels = labels.cpu().numpy()
            images = images.cpu().numpy()

            evaluator.iou(output, labels)

            if dumper is not None:
                dumper.dump(labels, output, images, i)

    print(evaluator.metric_list)
    epoch_val_metric = evaluator.mean_metric()

    return epoch_val_metric

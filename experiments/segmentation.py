import sys
import os
import argparse
import logging
import logging.config
import yaml
import pathlib
import builtins
import socket
import time
import random
import numpy as np
import torch
import logging
from tqdm import tqdm
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
from models.PosPadUNet3D import PosPadUNet3D
from dataloader.dataset3D import Loader3D
from dataloader.augmentations import *
from train import train
from test import test
from eval import Eval as Evaluator
from experiments.experiment import Experiment

class Segmentation(Experiment):
    def __init__(self, model, loss, optimizer,
            scheduler, evaluator, train_loader,
            test_loader, val_loader, splitter,
            writer, start_epoch=0, device=None):
        self.super().__init__(model, loss, optimizer, scheduler,
            evaluator, train_loader, test_loader, val_loader, splitter,
            writer, start_epoch, device)

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        losses = []
        for i, d in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Train epoch {str(self.epoch)}'):
            images = d['data'][tio.DATA].float().cuda()
            gt = d['label'][tio.DATA].cuda()

            emb_codes = torch.cat((
                d[tio.LOCATION][:,:3],
                d[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
            ), dim=1).float().cuda()

            # TODO: also remove split weights from dataloader3D.py?
            # these will be overwritter at line 30, useless
            # partition_weights = d['weight'].cuda()
            gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))

            # Skip if all the gt volumes are empty
            if torch.sum(gt_count) == 0:
                continue

            eps = 1e-10
            partition_weights = (eps + gt_count) / torch.max(gt_count)  # TODO: set this only when it is not competitor and we are on grid
            # partition_weights = (eps + gt_count) / torch.sum(gt_count)  # over max tecnique is better

            self.optimizer.zero_grad()
            preds = self.model(images, emb_codes)  # output -> B, C, Z, H, W
            assert preds.ndim == gt.ndim, f"Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}"

            loss = self.loss(preds, gt, partition_weights)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            # final predictions
            # shape B, C, xyz -> softmax -> B, xyz
            # shape 1, C, xyz -> softmax -> 1, xyz
            # shape B, 1, xyz -> sigmoid + sqz -> B, xyz
            # shape B, 1, xyz -> sigmoid + sqz -> xyz
            if preds.shape[1] > 1:
                preds = torch.argmax(torch.nn.Softmax(dim=1)(preds), dim=1)
            else:
                preds = nn.Sigmoid()(preds)  # BS, 1, Z, H, W
                preds[preds > .5] = 1
                preds[preds != 1] = 0
                preds = preds.squeeze().detach()  # BS, Z, H, W

            gt = gt.squeeze()  # BS, Z, H, W
            self.evaluator.compute_metrics(preds, gt, images, d['folder'], 'Train')

        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice, epoch_haus = self.evaluator.mean_metric(phase='Train')
        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
            'haus': epoch_haus
        }
        if self.writer is not None:
            self.writer.add_scalar(f'Train/Loss', epoch_train_loss, self.epoch)
            self.writer.add_scalar(f'Train/Dice', epoch_dice, self.epoch)
            self.writer.add_scalar(f'Train/Hausdorff', epoch_haus, self.epoch)
            self.writer.add_scalar(f'Train/IoU', epoch_iou, self.epoch)

        return epoch_train_loss, epoch_iou

    def test(self, phase):

        if phase == 'Test' or phase == 'Final':
            data_loader = self.test_loader
        elif phase == 'Validation':
            data_loader = self.val_loader
        elif phase == 'Train':
            data_loader = self.train_loader
        else:
            raise Exception(f'this phase is not valid {phase}')

        self.model.eval()

        with torch.inference_mode():
            self.evaluator.reset_eval()
            for i, (subject, loader) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'{phase} epoch {self.epoch}'):
                aggr = tio.inference.GridAggregator(subject, overlap_mode='average')
                for subvolume in loader:
                    # batchsize with torchio affects the number of grids we extract from a patient.
                    # when we aggragate the patient the volume is just one.

                    images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                    emb_codes = subvolume[tio.LOCATION].float().cuda()

                    output = self.model(images, emb_codes)  # BS, Classes, Z, H, W

                    aggr.add_batch(output, subvolume[tio.LOCATION])

                output = aggr.get_output_tensor()  # C, Z, H, W

                # TODO: Aren't this already on memory?
                gt = np.load(subject[0]['gt_path'])  # original gt from storage
                images = np.load(subject[0]['data_path'])  # high resolution image from storage

                orig_shape = gt.shape[-3:]
                output = CropAndPad(orig_shape)(output).squeeze()  # keep pad_val = min(output) since we are dealing with probabilities

                # final predictions
                if output.ndim > 3:
                    output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
                else:
                    output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                    output = torch.where(output > .5, 1, 0)
                    output = output.squeeze()  # BS, Z, H, W

                # TODO: this is quite slow... can be moved to GPU?
                self.evaluator.compute_metrics(output, gt, images, subject[0]['folder'], phase)

        epoch_iou, epoch_dice, epoch_haus = self.evaluator.mean_metric(phase=phase)
        self.metrics[phase] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
            'haus': epoch_haus
        }
        if self.writer is not None and phase != "Final":
            self.writer.add_scalar(f'{phase}/IoU', epoch_iou, self.epoch)
            self.writer.add_scalar(f'{phase}/Dice', epoch_dice, self.epoch)
            self.writer.add_scalar(f'{phase}/Hauss', epoch_haus, self.epoch)

        if phase in ['Test', 'Final']:
            logging.info(
                f'{phase} Epoch [{self.epoch}], '
                f'{phase} Mean Metric (IoU): {epoch_iou}'
                f'{phase} Mean Metric (Dice): {epoch_dice}'
                f'{phase} Mean Metric (haus): {epoch_haus}'
            )

        return epoch_iou, epoch_dice, epoch_haus

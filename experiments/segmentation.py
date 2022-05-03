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
from experiments.experiment import Experiment
from eval import Eval as Evaluator
from dataloader.AugFactory import *

class Segmentation(Experiment):
    def __init__(self, config):
        super().__init__(config)

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()
        
        data_loader = self.train_loader
        if self.config.data_loader.training_set == 'generated':
            print('using the generated set')
            data_loader = self.synthetic_loader

        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images = d['data'][tio.DATA].float().cuda()
            gt = d['dense'][tio.DATA].cuda()

            emb_codes = torch.cat((
                d[tio.LOCATION][:,:3],
                d[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
            ), dim=1).float().cuda()

            # TODO: also remove split weights from dataloader3D.py?
            # these will be overwritter at line 30, useless
            # partition_weights = d['weight'].cuda()

            eps = 1e-10
            partition_weights = 1
            if self.model.__class__.__name__ != 'Competitor':
                gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                # Skip if all the gt volumes are empty
                if torch.sum(gt_count) == 0:
                    continue
                # weight the batch based on the number of canal's voxels
                partition_weights = (eps + gt_count) / torch.max(gt_count)

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
                preds = (preds > 0).int()
                preds = preds.squeeze().detach()  # BS, Z, H, W

            gt = gt.squeeze()  # BS, Z, H, W
            self.evaluator.compute_metrics(preds, gt)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase='Train')
        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }
        if self.writer is not None:
            self.writer.add_scalar(f'Train/Loss', epoch_train_loss, self.epoch)
            self.writer.add_scalar(f'Train/Dice', epoch_dice, self.epoch)
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
            losses = []
            self.evaluator.reset_eval()
            for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'{phase} epoch {str(self.epoch)}'):
                images = d['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                gt = d['dense'][tio.DATA].float().cuda()
                emb_codes = torch.cat((
                    d[tio.LOCATION][:,:3],
                    d[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
                ), dim=1).float().cuda()

                gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))

                eps = 1e-10
                partition_weights = 1
                if self.model.__class__.__name__ != 'Competitor':
                    if torch.sum(gt_count) == 0: continue
                    partition_weights = (eps + gt_count) / torch.max(gt_count)

                output = self.model(images, emb_codes)  # BS, Classes, Z, H, W

                loss = self.loss(output, gt, partition_weights)
                losses.append(loss.item())

                if output.ndim > 3 and False:
                    assert False, 'This part looks wrong, dim=0 or dim=1?'
                    output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
                else:
                    output = (output > 0).int()
                    output = output.squeeze().detach()

                self.evaluator.compute_metrics(output, gt)

        epoch_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase=phase)
        self.metrics[phase] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }
        if self.writer is not None and phase != "Final":
            self.writer.add_scalar(f'{phase}/Loss', epoch_loss, self.epoch)
            self.writer.add_scalar(f'{phase}/IoU', epoch_iou, self.epoch)
            self.writer.add_scalar(f'{phase}/Dice', epoch_dice, self.epoch)

        # TODO: write also the metrics when Final? Can be written on Test/Metrics

        if phase in ['Test', 'Final']:
            logging.info(
                f'{phase} Epoch [{self.epoch}], '
                f'{phase} Mean Metric (IoU): {epoch_iou}'
                f'{phase} Mean Metric (Dice): {epoch_dice}'
            )

        return epoch_iou, epoch_dice

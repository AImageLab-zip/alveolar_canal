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
from tqdm import tqdm
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data
import wandb

from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from models.PosPadUNet3D import PosPadUNet3D
from eval import Eval as Evaluator
from experiments.experiment import Experiment
from dataloader.AugFactory import *
from dataloader.Maxillo import Maxillo

class Generation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        losses = []
        for i, d in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Train epoch {str(self.epoch)}'):

            images = d['data'][tio.DATA].float().cuda()
            sparse = d['sparse'][tio.DATA].float().cuda()
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
                # Skip if all the gt volumes are empty
                gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                if torch.sum(gt_count) == 0:
                    continue
                partition_weights = (eps + gt_count) / torch.max(gt_count)

            self.optimizer.zero_grad()
            img_sparse = torch.cat([images, sparse], dim=1)
            preds = self.model(img_sparse, emb_codes)  # output -> B, C, Z, H, W

            assert preds.ndim == gt.ndim, f"Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}"

            # in original code, partition_weights is hardcoded to 1
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
                # TODO: Can be optimized? Is Softmax here useless?
                preds = torch.argmax(torch.nn.Softmax(dim=1)(preds), dim=1)
            else:
                preds = (preds > 0.5).int()
                preds = preds.squeeze().detach()  # BS, Z, H, W

            self.evaluator.compute_metrics(preds, gt)


        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase='Train')
        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }

        wandb.log({
            f'Epoch': self.epoch,
            f'Train/Loss': epoch_train_loss,
            f'Train/Dice': epoch_dice,
            f'Train/IoU': epoch_iou,
            f'Train/Lr': self.optimizer.param_groups[0]['lr']
        })

        return epoch_train_loss, epoch_iou

    def test(self, phase):

        self.model.eval()

        with torch.no_grad():
            self.evaluator.reset_eval()
            losses = []
            if phase == 'Test':
                dataset = self.test_dataset
            elif phase == 'Validation':
                dataset = self.val_dataset

            for i, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'{phase} epoch {str(self.epoch)}'):
                sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        0
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler)
                gt_aggregator = tio.inference.GridAggregator(sampler)

                for j, patch in enumerate(loader):
                    images = patch['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                    sparse = patch['sparse'][tio.DATA].float().cuda()
                    gt = patch['dense'][tio.DATA].cuda()
                    emb_codes = patch[tio.LOCATION].float().cuda()
                    # join sparse + data
                    x = torch.cat([images, sparse], dim=1)
                    output = self.model(x, emb_codes)  # BS, Classes, Z, H, W

                    aggregator.add_batch(output, patch[tio.LOCATION])
                    gt_aggregator.add_batch(gt, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                gt = gt_aggregator.get_output_tensor()

                eps = 1e-6
                partition_weights = 1
                if self.model.__class__.__name__ != 'Competitor':
                    # Skip if all the gt volumes are empty
                    gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                    if torch.sum(gt_count) != 0:
                        partition_weights = (eps + gt_count) / (eps + torch.max(gt_count))
                    else:
                        partition_weights = 1

                loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
                losses.append(loss.item())

                output = output.squeeze(0)
                output = (output > 0.5).int()

                self.evaluator.compute_metrics(output, gt)

            epoch_loss = sum(losses) / len(losses)
            epoch_iou, epoch_dice = self.evaluator.mean_metric(phase=phase)

            wandb.log({
                f'Epoch': self.epoch,
                f'{phase}/Loss': epoch_loss,
                f'{phase}/Dice': epoch_dice,
                f'{phase}/IoU': epoch_iou
            })

            return epoch_iou, epoch_dice


    def inference(self, output_path):

        self.model.eval()

        with torch.no_grad():
            dataset = Maxillo(
                    self.config.data_loader.dataset,
                    splits='test',
                    transform=self.config.data_loader.preprocessing,
                    dist_map=['sparse', 'dense']
            )
            crop_or_pad_transform = tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0)
            for i, subject in tqdm(enumerate(dataset), total=len(dataset)):
                directory = os.path.join(output_path, f'{subject.patient}')
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, 'generated.npy')

                if os.path.exists(file_path):
                    logging.info(f'skipping {subject.patient}...')
                    continue

                sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        patch_overlap=self.config.data_loader.grid_overlap,
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler, overlap_mode='average')

                logging.info(f'patient {subject.patient}...')
                for j, patch in enumerate(loader):
                    images = patch['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                    sparse = patch['sparse'][tio.DATA].float().cuda()
                    emb_codes = patch[tio.LOCATION].float().cuda()

                    # join sparse + data
                    images = torch.cat([images, sparse], dim=1)
                    output = self.model(images, emb_codes)  # BS, Classes, Z, H, W
                    aggregator.add_batch(output, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                # output = tio.CropOrPad(original_shape, padding_mode=0)(output)
                output = output.squeeze(0)
                # output = (output > 0.5).int()
                output = output.detach().cpu().numpy()  # BS, Z, H, W

                np.save(file_path, output)
                logging.info(f'patient {subject.patient} completed, {file_path}.')

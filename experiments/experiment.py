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
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data
import wandb

from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.Maxillo import Maxillo
from dataloader.ToothFairy import ToothFairy
from dataloader.AugFactory import AugFactory
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from eval import Eval as Evaluator

eps = 1e-10
class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        filename = 'splits.json'
        if self.debug:
            filename += '.small'

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss.name or num_classes == 2:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1
        emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape, config=self.config).get().cuda()
        self.model = nn.DataParallel(self.model)
        wandb.watch(self.model, log_freq=100)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate
        weight_decay = float(self.config.optimizer.weight_decay)
        momentum = self.config.optimizer.momentum

        self.optimizer = OptimizerFactory(optim_name, train_params, lr, weight_decay, momentum).get()

        # load scheduler
        sched_name = self.config.lr_scheduler.name
        sched_milestones = self.config.lr_scheduler.get('milestones', None)
        sched_gamma = self.config.lr_scheduler.get('factor', None)

        self.scheduler = SchedulerFactory(
                sched_name,
                self.optimizer,
                LR = self.config.optimizer.learning_rate,
                milestones=sched_milestones,
                gamma=sched_gamma,
                mode='max',
                verbose=True,
                patience=15,
                epochs=self.config.trainer.epochs
            ).get()

        # load loss
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        self.train_dataset = ToothFairy(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='train',
                transform=tio.Compose([
                    tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                    self.config.data_loader.preprocessing,
                    self.config.data_loader.augmentations,
                    ]),
                # dist_map=['sparse','dense']
        )
        self.val_dataset = ToothFairy(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='val',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense']
        )
        self.test_dataset = ToothFairy(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='test',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense']
        )
        self.synthetic_dataset = ToothFairy(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='synthetic',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense'],
        ) 
        self.mixed_dataset = ToothFairy(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits=['synthetic', 'train'],
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense'],
        ) 

        # self.test_aggregator = self.train_dataset.get_aggregator(self.config.data_loader)
        # self.synthetic_aggregator = self.synthetic_dataset.get_aggregator(self.config.data_loader)

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)
        self.synthetic_loader = self.synthetic_dataset.get_loader(self.config.data_loader)
        self.mixed_loader = self.mixed_dataset.get_loader(self.config.data_loader)

        if self.config.trainer.reload:
            self.load()

    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
        logging.info(f'Saving checkpoint at {path}')
        state = {
            'title': self.config.title,
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, path)

    def load(self, name=None, set_epoch=False):
        if name is None:
            path = self.config.trainer.checkpoint
        else:
            if '.pth' not in name:
                name = name + '.pth'
            path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
        logging.info(f'Loading checkpoint from {path}')
        state = torch.load(path)

        if 'title' in state.keys():
            # check that the title headers (without the hash) is the same
            self_title_header = self.config.title[:-11]
            load_title_header = state['title'][:-11]
            if self_title_header == load_title_header:
                self.config.title = state['title']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['state_dict'])
        if set_epoch or True:
            self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def extract_images_from_patch(self, patch):
        raise Exception('Not implemented')

    def extract_data_from_patch(self, patch, *, sparse_labels):
        volume = patch['data'][tio.DATA].float().cuda()
        gt = patch['dense'][tio.DATA].float().cuda()

        if sparse_labels:
            sparse = patch['sparse'][tio.DATA].float().cuda()
            images = torch.cat([volume, sparse], dim=1)
        else:
            images = volume

        emb_codes = torch.cat((
            patch[tio.LOCATION][:,:3],
            patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()

        return images, gt, emb_codes

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader
        if self.config.data_loader.training_set == 'generated':
            logging.info('using the generated dataset')
            data_loader = self.synthetic_loader
        elif self.config.data_loader.training_set == 'mixed':
            logging.info('using both the real and the generated dataset')
            data_loader = self.mixed_loader


        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt, emb_codes = self.extract_data_from_patch(d)

            partition_weights = 1
            # TODO: Do only if not Competitor
            # removed Competitor code because i don't know if we still need
            # something that will never be executed again (maybe to compare datasets?)
            gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
            if torch.sum(gt_count) == 0: continue
            partition_weights = (eps + gt_count) / torch.max(gt_count)

            self.optimizer.zero_grad()
            preds = self.model(images, emb_codes)

            assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
            loss = self.loss(preds, gt, partition_weights)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.)
            self.optimizer.step()

            preds = (preds > 0.5).squeeze().detach()

            gt = gt.squeeze()
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

        # with torch.no_grad():
        with torch.inference_mode():
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
                        self.config.data_loader.grid_overlap # 0 crop 60 hann
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler, overlap_mode="hann")     # crop # hann
                gt_aggregator = tio.inference.GridAggregator(sampler, overlap_mode="hann")  # crop # hann

                for j, patch in enumerate(loader):
                    images, gt, emb_codes = self.extract_data_from_patch(patch)

                    preds = self.model(images, emb_codes)
                    aggregator.add_batch(preds, patch[tio.LOCATION])
                    gt_aggregator.add_batch(gt, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                gt = gt_aggregator.get_output_tensor()
                partition_weights = 1

                gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                if torch.sum(gt_count) != 0:
                    partition_weights = (eps + gt_count) / (eps + torch.max(gt_count))

                loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
                losses.append(loss.item())

                output = output.squeeze(0)
                output = (output > 0.5)

                if phase == 'Test':
                    wandb.log({
                        f'{phase}/Prediction-{i}': wandb.Object3D(
                            np.stack(np.where(output==1)).T
                        )
                    })
                    
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
                                

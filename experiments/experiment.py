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
from tqdm import tqdm

from dataloader.Maxillo import Maxillo
from dataloader.AugFactory import AugFactory
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from eval import Eval as Evaluator

class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        filename = 'splits.json.small'
        if self.debug:
            filename = 'splits.json.small'

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss.name or num_classes == 2:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1
        emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
        self.model = nn.DataParallel(self.model)
        wandb.watch(self.model, log_freq=10)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate
        weight_decay =  self.config.optimizer.weight_decay
        momentum =  self.config.optimizer.momentum

        self.optimizer = OptimizerFactory(optim_name, train_params, lr, weight_decay, momentum).get()

        # load scheduler
        sched_name = self.config.lr_scheduler.name
        sched_milestones = self.config.lr_scheduler.get('milestones', None)
        sched_gamma = self.config.lr_scheduler.get('factor', None)

        self.scheduler = SchedulerFactory(
                sched_name,
                self.optimizer,
                milestones=sched_milestones,
                gamma=sched_gamma,
                mode='max',
                verbose=True,
                patience=15
            ).get()

        # load loss
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        self.train_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='train',
                transform=tio.Compose([
                    tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                    self.config.data_loader.preprocessing,
                    self.config.data_loader.augmentations,
                    ]),
                dist_map=['sparse','dense']
        )
        self.val_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='val',
                transform=self.config.data_loader.preprocessing,
                dist_map=['sparse', 'dense']
        )
        self.test_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='test',
                transform=self.config.data_loader.preprocessing,
                dist_map=['sparse', 'dense']
        )
        self.synthetic_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='synthetic',
                transform=self.config.data_loader.preprocessing,
                dist_map=['sparse', 'dense'],
        ) 

        # self.test_aggregator = self.train_dataset.get_aggregator(self.config.data_loader)
        # self.synthetic_aggregator = self.synthetic_dataset.get_aggregator(self.config.data_loader)

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)
        self.synthetic_loader = self.synthetic_dataset.get_loader(self.config.data_loader)

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

    def load(self):
        path = self.config.trainer.checkpoint
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
        self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def train(self):
        raise NotImplementedError

    def test(self, phase):
        raise NotImplementedError

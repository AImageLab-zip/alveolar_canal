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

from dataloader.Maxillo import Maxillo
from dataloader.AugFactory import AugFactory
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from eval import Eval as Evaluator

class Experiment:
    def __init__(self, config):
        self.title = config.title
        self.config = config
        self.epoch = 0
        self.metrics = {}

        num_classes = len(config.data_loader.labels)
        if 'Jaccard' in config.loss.name:
            num_classes = 1

        # load model
        model_name = config.model.name
        in_ch = 2 if config.experiment.name == 'Generation' else 1
        emb_shape = [dim // 8 for dim in config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
        self.model = nn.DataParallel(self.model)

        # load optimizer
        optim_name = config.optimizer.name
        train_params = self.model.parameters()
        lr = config.optimizer.learning_rate

        self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

        # load scheduler
        sched_name = config.lr_scheduler.name
        sched_milestones = config.lr_scheduler.get('milestones', None)
        sched_gamma = config.lr_scheduler.get('factor', None)

        self.scheduler = SchedulerFactory(
                sched_name,
                self.optimizer,
                milestones=sched_milestones,
                gamma=sched_gamma,
                mode='max',
                verbose=True,
                patience=7
            ).get()

        # load loss
        self.loss = LossFactory(config.loss.name, config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(config, skip_dump=True)

        # TODO: load everything in a single line?
        # load datasets
        config.data_loader.augmentations = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 2100)),
            config.data_loader.augmentations
            ])
        self.train_dataset = Maxillo(config.data_loader.dataset, 'train', config.data_loader.augmentations)
        self.val_dataset = Maxillo(config.data_loader.dataset, 'val')
        self.test_dataset = Maxillo(config.data_loader.dataset, 'test')
        self.synthetic_dataset = Maxillo(config.data_loader.dataset, 'synthetic')

        # self.test_aggregator = self.train_dataset.get_aggregator(self.config.data_loader)
        # self.synthetic_aggregator = self.synthetic_dataset.get_aggregator(self.config.data_loader)

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)
        self.synthetic_loader = self.synthetic_dataset.get_loader(self.config.data_loader)

        self.writer = None
        if config.tensorboard_dir is not None and os.path.exists(config.tensorboard_dir):
            config.tensorboard_dir = os.path.join(config.tensorboard_dir, config.title)
            self.writer = SummaryWriter(log_dir=config.tensorboard_dir)

    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, 'checkpoints', name)
        logging.info(f'Saving checkpoint at {path}')
        state = {
            'title': self.title,
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
            self.title = state['title']
            self.config.project_dir = os.path.join(self.config.project_dir, self.title)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch'] + 1
        if 'metrics' in state.keys():
            self.metrics = state['metrics']
        # logging.info(f'Restarting training from epoch: {self.epoch}')

    def train(self):
        raise NotImplementedError

    def test(self, phase):
        raise NotImplementedError

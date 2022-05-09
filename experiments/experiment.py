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
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.metrics = {}

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss.name:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1
        emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
        self.model = nn.DataParallel(self.model)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate

        self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

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
                patience=7
            ).get()

        # load loss
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        # TODO: load everything in a single line?
        # load datasets
        self.base_augmentations = tio.Compose([
            tio.Clamp(
                out_min=self.config.data_loader.volumes_min,
                out_max=self.config.data_loader.volumes_max
                ),
            tio.RescaleIntensity(
                out_min_max=(0,1)
            )
        ])

        self.config.data_loader.augmentations = tio.Compose([
            self.base_augmentations,
            self.config.data_loader.augmentations
        ])

        self.train_dataset = Maxillo(self.config.data_loader.dataset, 'train', self.config.data_loader.augmentations)
        self.val_dataset = Maxillo(self.config.data_loader.dataset, 'val', self.base_augmentations)
        self.test_dataset = Maxillo(self.config.data_loader.dataset, 'test', self.base_augmentations)
        self.synthetic_dataset = Maxillo(self.config.data_loader.dataset, 'synthetic', self.config.data_loader.augmentations)

        # self.test_aggregator = self.train_dataset.get_aggregator(self.config.data_loader)
        # self.synthetic_aggregator = self.synthetic_dataset.get_aggregator(self.config.data_loader)

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)
        self.synthetic_loader = self.synthetic_dataset.get_loader(self.config.data_loader)

        if self.config.trainer.reload:
            self.load()

        self.writer = None
        if self.config.tensorboard_dir is not None and os.path.exists(self.config.tensorboard_dir):
            self.config.tensorboard_dir = os.path.join(self.config.tensorboard_dir, self.config.title)
            self.writer = SummaryWriter(log_dir=self.config.tensorboard_dir)
            logging.info(f'tensorboard_dir @ {self.config.tensorboard_dir}')

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
            if 'pretrain' not in state['title'] or 'finetuning' not in self.config.title:
                self.config.title = state['title']
                # brand new optimizer if we are doing finetuning, to avoid a too low learning rate
                self.optimizer.load_state_dict(state['optimizer'])
            # TODO: fix the following, as i've changed the hash function
            # if self_title_header == load_title_header:
            #    self.config.title = state['title']
            # elif self_title_header != 'alveolar_canal_finetuning' or load_title_header != 'alveolar_canal_pretrain':
            #    logging.warn(f'Different titles while loading and not pretrain->finetuning')
            #    logging.warn(f'self_title_header: {self_title_header}')
            #    logging.warn(f'self_title: {self.config.title}')
            #    logging.warn(f'load_title_header: {load_title_header}')
            #    logging.warn(f'load_title: {state["title"]}')

        self.model.load_state_dict(state['state_dict'])
        self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def train(self):
        raise NotImplementedError

    def test(self, phase):
        raise NotImplementedError

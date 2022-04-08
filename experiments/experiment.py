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

class Experiment:
    def __init__(self, model, loss, optimizer,
            scheduler, evaluator, train_loader,
            test_loader, val_loader, splitter,
            writer, start_epoch=0, device=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.splitter = splitter
        self.epoch = start_epoch
        self.writer = writer
        self.metrics = {}

    def save(self, path):
        print("Saving checkpoint '{}'".format(path))
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, path)

    def load(self, path):
        print("Loading checkpoint '{}'".format(path))
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch'] + 1
        self.metrics = state['metrics']

    def train(self):
        raise NotImplementedError

    def test(self, phase):
        raise NotImplementedError

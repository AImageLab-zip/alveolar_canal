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
from experiments.experiment import Experiment
from eval import Eval as Evaluator
from dataloader.AugFactory import *

class Segmentation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)

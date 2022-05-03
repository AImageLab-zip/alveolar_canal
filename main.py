import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import pathlib
import builtins
import socket
import random
import time
import numpy as np
import torch
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data

from munch import Munch, munchify
from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler

from experiments.ExperimentFactory import ExperimentFactory
from dataloader.AugFactory import AugFactory

# used to generate random names that will be append to the
# experiment name
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def timehash():
    t = time.time()
    h = hash(t)
    return ''.join([alphabet[int(c)] for c in str(h)])

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="config.yaml", help="the config file to be used to run the experiment", required=True)
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        logging.info("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # make title unique to avoid overriding
    config.title = f'{config.title}_{timehash()}'
    config.project_dir = os.path.join(config.project_dir, config.title)
    os.makedirs(config.project_dir, exist_ok=True)

    # check if augmentations is set and file exists
    if config.data_loader.augmentations is None:
        aug = []
    elif not os.path.exists(config.data_loader.augmentations):
        logging.info(f'Augmentations file does not exist: {config.augmentations}')
        aug = []
    else:
        with open(config.data_loader.augmentations) as aug_file:
            aug = yaml.load(aug_file, yaml.FullLoader)
    config.data_loader.augmentations = AugFactory(aug).get_transform()

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(config.project_dir, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(config.project_dir, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    if not os.path.exists(config.data_loader.dataset):
        logging.info("Dataset path does not exist: {}".format(config.data_loader.dataset))
        raise SystemExit

    # pre-calculate the checkpoints path
    checkpoints_path = path.join(config.project_dir, 'checkpoints')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if config.trainer.reload and not os.path.exists(config.trainer.checkpoint):
        logging.info(f'Checkpoint file does not exist: {config.trainer.checkpoint}')
        raise SystemExit

    # Setup to be deterministic
    setup(config.seed)

    experiment = ExperimentFactory(config).get()

    best_val = float('-inf')
    best_test = {
            'value': float('-inf'),
            'epoch': -1
            }


    # Train the model
    if config.trainer.do_train:
        if config.trainer.reload:
            experiment.load()

        assert experiment.epoch < config.trainer.epochs
        for epoch in range(experiment.epoch, config.trainer.epochs+1):
            experiment.train()

            val_iou, val_dice = experiment.test(phase="Validation")
            logging.info(f'Val results at epoch {epoch}\nIoU: {val_iou}\nDice: {val_dice}')

            if val_iou < 1e-05 and experiment.epoch > 15:
                logging.info('WARNING: drop in performances detected.')

            optim_name = experiment.optimizer.name
            sched_name = experiment.scheduler.name

            if experiment.scheduler is not None:
                if optim_name == 'SGD' and sched_name == 'Plateau':
                    experiment.scheduler.step(val_iou)
                else:
                    experiment.scheduler.step(epoch)

            if epoch % 5 == 0 and epoch != 0:
                test_iou, test_dice = experiment.test(phase="Test")
                logging.info(f'Test results at epoch {epoch}\nIoU: {test_iou}\nDice: {test_dice}')
                if best_test['value'] > test_iou:
                    best_test['value'] = test_iou
                    best_test['epoch'] = epoch

            experiment.save('last.pth')

            if val_iou > best_val:
                best_val = val_iou
                experiment.save('best.pth')

            experiment.epoch += 1

        logging.info(f'''
                Best test IoU found: {best_test['value']} at epoch: {best_test['epoch']}
                ''')

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        experiment.load()
        test_iou, test_dice = experiment.test(phase="Test")
        logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}')

    # Do the inference
    if config.trainer.do_inference:
        logging.info('Doing inference...')
        experiment.load()
        experiment.inference(config.dataset / 'SPARSE')
        print('end')

# TODO: add a Final test metric

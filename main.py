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
import json

import numpy as np
import torch
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data

from hashlib import shake_256
from munch import Munch, munchify, unmunchify
from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
import wandb

from experiments.ExperimentFactory import ExperimentFactory
from dataloader.AugFactory import AugFactory

# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5) # output len: 2*5=10
    return h.upper()

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="./configs/gen-training-unet-trans_train_1.yaml", help="the config file to be used to run the experiment")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # start wandb
    wandb.init(
        project="alveolar_canal",
        name=f"TFF_{config.model.name}_{config.seed}_M{config.model.mem_len}", # f"TFF_{config.model.name}_L{config.model.n_layers}H{config.model.n_head}_{config.seed}", # f"TFF_{config.model.name}_{config.seed}"
        entity="maxillo",
        config=unmunchify(config),
        mode=config.wandb.mode,
    )

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # check if preprocessing is set and file exists
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preproc = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preproc = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preproc_file:
            preproc = yaml.load(preproc_file, yaml.FullLoader)
    config.data_loader.preprocessing = AugFactory(preproc).get_transform()

    # check if augmentations is set and file exists
    logging.info(f'loading augmentations')
    if config.data_loader.augmentations is None:
        aug = []
    elif not os.path.exists(config.data_loader.augmentations):
        logging.warning(f'Augmentations file does not exist: {config.augmentations}')
        aug = []
    else:
        with open(config.data_loader.augmentations) as aug_file:
            aug = yaml.load(aug_file, yaml.FullLoader)
    config.data_loader.augmentations = AugFactory(aug).get_transform()

    # make title unique to avoid overriding
    config.title = f'{config.title}_{timehash()}'

    logging.info(f'Instantiation of the experiment')
    experiment = ExperimentFactory(config, args.debug).get()
    logging.info(f'experiment title: {experiment.config.title}')

    project_dir_title = os.path.join(experiment.config.project_dir, experiment.config.title)
    os.makedirs(project_dir_title, exist_ok=True)
    logging.info(f'project directory: {project_dir_title}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(project_dir_title, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(project_dir_title, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    if not os.path.exists(experiment.config.data_loader.dataset):
        logging.error("Dataset path does not exist: {}".format(experiment.config.data_loader.dataset))
        raise SystemExit

    # pre-calculate the checkpoints path
    checkpoints_path = path.join(project_dir_title, 'checkpoints')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if experiment.config.trainer.reload and not os.path.exists(experiment.config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {experiment.config.trainer.checkpoint}')
        raise SystemExit

    best_val = float('-inf')
    best_test = {
            'value': float('-inf'),
            'epoch': -1
            }

    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        assert experiment.epoch < config.trainer.epochs
        for epoch in range(experiment.epoch, config.trainer.epochs+1):
            experiment.train()

            val_iou, val_dice = experiment.test(phase="Validation")
            logging.info(f'Epoch {epoch} Val IoU: {val_iou}')
            logging.info(f'Epoch {epoch} Val Dice: {val_dice}')

            if val_iou < 1e-05 and experiment.epoch > 15:
                logging.warning('WARNING: drop in performances detected.')

            optim_name = experiment.optimizer.name
            sched_name = experiment.scheduler.name

            if experiment.scheduler is not None:
                if optim_name == 'SGD' and sched_name == 'Plateau':
                    experiment.scheduler.step(val_iou)
                else:
                    experiment.scheduler.step(epoch)

            if epoch % 5 == 0:
                test_iou, test_dice = experiment.test(phase="Test")
                logging.info(f'Epoch {epoch} Test IoU: {test_iou}')
                logging.info(f'Epoch {epoch} Test Dice: {test_dice}')

                if test_iou > best_test['value']:
                    best_test['value'] = test_iou
                    best_test['epoch'] = epoch

            experiment.save('last.pth')

            if val_iou > best_val:
                best_val = val_iou
                wandb.run.summary["Highest_Validation_IOU/Epoch"] = experiment.epoch
                wandb.run.summary["Highest_Validation_IOU/Valdiation_IOU"] = val_iou
                wandb.run.summary["Highest_Validation_IOU/Valdiation_Dice"] = val_dice
  
                experiment.save('best.pth')

            experiment.epoch += 1

        logging.info(f'''
                Best test IoU found: {best_test['value']} at epoch: {best_test['epoch']}
                ''')
        logging.info('Testing the model...')
        experiment.load(name="best", set_epoch=True)
        test_iou, test_dice = experiment.test(phase="Test")
        logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}')
        wandb.run.summary["Highest_Validation_IOU/Test_IOU"] = test_iou
        wandb.run.summary["Highest_Validation_IOU/Test_Dice"] = test_dice

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        experiment.load()
        test_iou, test_dice = experiment.test(phase="Test")
        logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}')
        wandb.run.summary["Highest_Validation_IOU/Test_IOU"] = test_iou
        wandb.run.summary["Highest_Validation_IOU/Test_Dice"] = test_dice

    # Do the inference
    if config.trainer.do_inference:
        logging.info('Doing inference...')
        experiment.load()
        experiment.inference(os.path.join(config.data_loader.dataset, 'Dataset'))
        # experiment.inference('/homes/llumetti/out')

# TODO: add a Final test metric

import sys
import os
import argparse
import logging
import logging.config
import yaml
import pathlib
import builtins
import socket
import random
import numpy as np
import torch
import logging
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data

from munch import Munch, munchify
from collections import namedtuple
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
from dataloader.DatasetFactory import DatasetFactory
from train import train
from test import test
from eval import Eval as Evaluator
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from experiments import Segmentation

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-c", "--config", default="config.yaml", help="the config file to be used to run the experiment")

    args = arg_parser.parse_args()
    config_path = args.config

    # check if the config files exists
    if not os.path.exists(config_path):
        print("Config file does not exist: {}".format(config_path))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    config = yaml.load(open(config_path, "r"), yaml.FullLoader)
    config = munchify(config)

    if not os.path.exists(config.data_loader.dataset):
        print("Dataset config file does not exist: {}".format(config.data_loader.dataset))
        raise SystemExit

    config.data_loader.dataset = yaml.load(open(config.data_loader.dataset, "r"), yaml.FullLoader)
    config.data_loader.dataset = munchify(config.data_loader.dataset)

    cp_best = os.path.join(config.trainer.checkpoint_dir, 'best.pth')
    cp_last = os.path.join(config.trainer.checkpoint_dir, 'last.pth')

    if not os.path.exists(config.trainer.checkpoint_dir):
        print(f'Checkpoint path doesn\'t exist: {config.trainer.checkpoint_dir}, i\'ll create it for you')
        os.makedirs(config.trainer.checkpoint_dir)

    if config.trainer.reload and not os.path.exists(cp_last):
        print(f'Checkpoint file does not exist: {cp_last}')
        print('Can\'t reload, will start from scratch')
        config.trainer.reload = False


    # Setup to be deterministic
    setup(config.seed)

    # rank and world size manually setted, must be changed
    rank = 0
    world_size = 1

    labels = config.data_loader.labels

    num_classes = len(labels)
    if 'Jaccard' in config.loss.name:
        num_classes = 1

    # load model
    model_name = config.model.name
    emb_shape = [dim // 8 for dim in config.data_loader.patch_shape]

    model = ModelFactory(model_name, num_classes, emb_shape).get().cuda()
    model = nn.DataParallel(model)

    # load optimizer
    optim_name = config.optimizer.name
    train_params = model.parameters()
    lr = config.optimizer.learning_rate

    optimizer = OptimizerFactory(optim_name, train_params, lr).get()

    # load scheduler
    sched_name = config.lr_scheduler.name
    sched_milestones = config.lr_scheduler.get('milestones', None)
    sched_gamma = config.lr_scheduler.get('factor', None)

    scheduler = SchedulerFactory(
            sched_name,
            optimizer,
            milestones=sched_milestones,
            gamma=sched_gamma,
            mode='max',
            verbose=True,
            patience=7
        ).get()

    # load loss
    loss = LossFactory(config.loss.name, config.data_loader.labels)
    # loss = LossFn(config, weights=None)  # TODO: fix this, weights are disabled now

    # load evaluator
    evaluator = Evaluator(config, skip_dump=True)

    # load data
    loader = DatasetFactory('3D')
    dataloader = loader(config.data_loader, config.trainer.do_train, None, False)
    train_loader, test_loader, val_loader, splitter = dataloader.load(rank, world_size, False)

    # Writer has to have epoch_start, but i get it from experiment.load...
    # purge_step looks like it's used to recover a crashed training, and
    # overriding from step {purge_step} ahead
    writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.title), purge_step=0)

    experiment = Segmentation(
            model, loss, optimizer,
            scheduler, evaluator,
            train_loader, test_loader, val_loader,
            splitter, writer, 0
            )

    if config.trainer.reload:
        print(f'Reloading from checkpoint: {cp_last}')
        experiment.load(cp_last)

    best_val = float('-inf')
    best_test = float('-inf')

    for epoch in range(experiment.epoch, config.trainer.epochs):
        # TODO: old distributed code, to be re-implemented
        # if world_size > 1:
        #     train_loader.sampler.set_epoch(np.random.seed(np.random.randint(0, 10000)))
        #     dist.barrier()

        # if rank == 0:
        #     writer = SummaryWriter(log_dir=os.path.join(config['tensorboard_dir'], config['title']), purge_step=start_epoch)
        # else:
        #     writer = None

        experiment.train()

        if not rank == 0:
            continue

        val_model = model.module
        val_iou, val_dice, val_haus = experiment.test(phase="Validation")

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
            test_iou, test_dice, test_haus = experiment.test(phase="Test")
            best_test = best_test if best_test > test_iou else test_iou

        cp_best = os.path.join(config.trainer.checkpoint_dir, 'best.pth')
        cp_last = os.path.join(config.trainer.checkpoint_dir, 'last.pth')
        experiment.save(cp_last)

        if val_iou > best_val:
            best_val = val_iou
            experiment.save(cp_best)

        experiment.epoch += 1

    logging.info('BEST TEST METRIC IS {}'.format(best_test))

#if rank == 0:
#    val_model = model.module
#    test(val_model, test_loader, epoch="Final", writer=None, evaluator=evaluator, phase="Final")


import argparse
import os
import pathlib
import torch
import torch.utils.data as data
import builtins
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import utils
from dataset import NewLoader
from eval import Eval as Evaluator
from losses import LossFn
from test import test
import sys
import logging
import torch.nn as nn
import yaml
import numpy as np
from os import path
import socket
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader, DistributedSampler
import torch
import logging
from train import train
from torch import nn
import torchio as tio
import torch.distributed as dist


def save_weights(epoch, model, optim, score, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'metric': score
    }
    torch.save(state, path)


def main(experiment_name, args):

    assert torch.cuda.is_available()
    logging.info(f"This model will run on {torch.cuda.get_device_name(torch.cuda.current_device())}")

    ## DETERMINISTIC SET-UP
    seed = config.get('seed', 47)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # END OF DETERMINISTIC SET-UP

    loader_config = config.get('data-loader', None)
    train_config = config.get('trainer', None)

    model = utils.load_model(config)
    # DDP setting
    world_size = 1
    rank = 0
    if "WORLD_SIZE" in os.environ:
        logging.info('using DISTRIBUTED data parallel')
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = None
        if args.local_rank != -1:  # for torch.distributed.launch
            rank = args.local_rank
            gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            rank = int(os.environ['SLURM_PROCID'])
            gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=int(os.environ["WORLD_SIZE"]), rank=rank)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        assert gpu is not None
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    else:
        logging.info('using data parallel')
        model = nn.DataParallel(model).cuda()
    is_distributed = world_size > 1

    train_params = model.parameters()

    optim_config = config.get('optimizer')
    optim_name = optim_config.get('name', None)
    if not optim_name or optim_name == 'Adam':
        optimizer = torch.optim.Adam(params=train_params, lr=optim_config['learning_rate'])
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(params=train_params, lr=optim_config['learning_rate'])
    else:
        raise Exception("optimizer not recognized")

    sched_config = config.get('lr_scheduler')
    scheduler_name = sched_config.get('name', None)
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config['milestones'],
            gamma=sched_config.get('factor', 0.1),
        )
    elif scheduler_name == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=7)
    else:
        scheduler = None

    evaluator = Evaluator(loader_config)

    data_utils = NewLoader(loader_config, train_config.get("do_train", True), train_config.get("do_pre_train", True))
    train_d, test_d, val_d, pretrain_d = data_utils.split_dataset(rank=rank, world_size=world_size)

    # TODO: more a warning. samples per volume by params is disabled now and automatically computed.
    #  we need to further investigate damages caused by a wrong number here
    # samples_per_volume = loader_config.get('samples_per_volume', 'auto')
    # if samples_per_volume == 'auto':
    #     samples_per_volume = int(np.prod([np.round(i / j) for i, j in zip(loader_config['resize_shape'], loader_config['patch_shape'])]))
    # samples_per_volume = int(samples_per_volume)
    samples_per_volume = int(np.prod([np.round(i / j) for i, j in zip(loader_config['resize_shape'], loader_config['patch_shape'])]))

    train_queue = tio.Queue(
        train_d,
        max_length=samples_per_volume * 4,  # queue len
        samples_per_volume=samples_per_volume,
        sampler=data_utils.get_sampler(loader_config.get('sampler_type', 'grid'), loader_config.get('grid_overlap', 0)),
        num_workers=loader_config['num_workers'],
    )
    sampler = DistributedSampler(train_queue, shuffle=False) if is_distributed else None
    train_loader = data.DataLoader(train_queue, loader_config['batch_size'], num_workers=0, sampler=sampler)

    if rank == 0:
        test_loader = [(test_p, data.DataLoader(test_p, loader_config['batch_size'] * world_size, num_workers=loader_config['num_workers'])) for test_p in test_d]
        val_loader = [(val_p, data.DataLoader(val_p, loader_config['batch_size'] * world_size, num_workers=loader_config['num_workers'])) for val_p in val_d]
        vol_writer = utils.SimpleDumper(loader_config, experiment_name, project_dir) if args.dump_results else None

    loss = LossFn(config.get('loss'), loader_config, weights=None)  # TODO: fix this, weights are disabled now

    start_epoch = 0
    if train_config['checkpoint_path'] is not None:
        try:
            checkpoint = torch.load(train_config['checkpoint_path'])
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Checkpoint loaded successfully at epoch {start_epoch}, score:{checkpoint.get('metric', 'unavailable')})")
        except OSError as e:
            logging.info("No checkpoint exists from '{}'. Skipping...".format(train_config['checkpoint_path']))

    if train_config['do_train']:

        if rank == 0:
            writer = SummaryWriter(log_dir=os.path.join(config['tb_dir'], experiment_name), purge_step=start_epoch)
        else:
            writer = None

        if train_config['do_pre_train']:
            logging.info("starting pre-training on syntetic data")
            pre_train_queue = tio.Queue(
                pretrain_d,
                max_length=samples_per_volume * 4,  # queue len
                samples_per_volume=samples_per_volume,
                sampler=data_utils.get_sampler(loader_config.get('sampler_type', 'grid'), loader_config.get('grid_overlap', 0)),
                num_workers=loader_config['num_workers'],
            )
            sampler = DistributedSampler(pre_train_queue, shuffle=False) if is_distributed else None
            pre_train_loader = data.DataLoader(pre_train_queue, loader_config['batch_size'], num_workers=0, sampler=sampler)

            for epoch in range(0, 35):
                # fix sampling seed such that each gpu gets different part of dataset
                if is_distributed:
                    pre_train_loader.sampler.set_epoch(np.random.seed(np.random.randint(0, 10000)))
                train(model, pre_train_loader, loss, optimizer, epoch, writer, evaluator, type='Pretrain')

            if rank == 0:
                test_iou, _ = test(model, test_loader, train_config['epochs'] + 1, evaluator, loader_config, writer=None)
                save_weights(epoch, model, optimizer, test_iou, os.path.join(project_dir, 'checkpoints', 'pretraining.pth'))
                logging.info("pretraining is over. we start training with a IoU on test of {test_iou}")

        best_metric = 0
        # warm_up = np.ones(shape=train_config['epochs'])
        # warm_up[0:int(train_config['epochs'] * train_config.get('warm_up_length', 0.35))] = np.linspace(
        #     0, 1, num=int(train_config['epochs'] * train_config.get('warm_up_length', 0.35))
        # )

        for epoch in range(start_epoch, train_config['epochs']):

            if is_distributed:
                train_loader.sampler.set_epoch(np.random.seed(np.random.randint(0, 10000)))
                dist.barrier()
                
            train(model, train_loader, loss, optimizer, epoch, writer, evaluator, type="train")

            if rank == 0:
                val_model = model.module
                val_iou, val_dice = test(val_model, val_loader, epoch, evaluator, loader_config, writer=writer)
                logging.info(f'VALIDATION Epoch [{epoch}] - Mean Metric (iou): {val_iou} - (dice) {val_dice}')
                writer.add_scalar('Metric/validation', val_iou, epoch)

                if scheduler is not None:
                    if optim_name == 'SGD' and scheduler_name == 'Plateau':
                        scheduler.step(val_iou)
                    else:
                        scheduler.step(epoch)

                if val_iou > best_metric and not args.test:
                    best_metric = val_iou
                    save_weights(epoch, model, optimizer, best_metric, os.path.join(project_dir, 'best.pth'))

                if val_iou < 1e-05 and epoch > 10:
                    logging.info('drop in performances detected. aborting the experiment')
                    return 0
                elif not args.test:  # save current weights for debug, overwrite the same file
                    save_weights(epoch, model, optimizer, val_iou, os.path.join(project_dir, 'checkpoints', 'last.pth'))

                if epoch % 5 == 0 and epoch != 0:
                    test_iou, test_dice = test(val_model, test_loader, train_config['epochs'] + 1, evaluator, loader_config, writer=None)
                    logging.info(f'TEST Epoch [{epoch}] - Mean Metric (iou): {test_iou} - (dice) {test_dice}')
                    writer.add_scalar('Metric/Test', test_iou, epoch)

        logging.info('BEST METRIC IS {}'.format(best_metric))

    # final test
    if rank == 0:
        final_iou_list = test(model, test_loader, train_config['epochs'] + 1, evaluator, loader_config, writer=None, dumper=vol_writer, skip_mean=True)
        logging.info(f'FINAL METRIC (iou): {np.mean(final_iou_list)}')
        logging.info(f'debug: final metric list: {final_iou_list}')

        if vol_writer is not None:
            logging.info("going to create zip archive. wait the end of the run pls")
            vol_writer.save_zip()


if __name__ == '__main__':
    # execute the following line if there are new data in the dataset to be fixed
    # utils.fix_dataset_folder(r'Y:\work\datasets\maxillo\VOLUMES')

    RESULTS_DIR = r'Y:\work\results' if socket.gethostname() == 'DESKTOP-I67J6KK' else r'/nas/softechict-nas-2/mcipriano/results/maxillo/3D'
    BASE_YAML_PATH = os.path.join('configs', 'config.yaml') if socket.gethostname() == 'DESKTOP-I67J6KK' else os.path.join('configs', 'remote_config.yaml')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--base_config', default="config.yaml", help='path to the yaml config file')
    arg_parser.add_argument('--verbose', action='store_true', help="if true sdout is not redirected, default: false")
    arg_parser.add_argument('--dump_results', action='store_true', help="dump test data, default: false")
    arg_parser.add_argument('--test', action='store_true', help="set up test params, default: false")
    arg_parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    arg_parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    arg_parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

    args = arg_parser.parse_args()
    yaml_path = args.base_config

    if path.exists(yaml_path):
        print(f"loading config file in {yaml_path}")
        config = utils.load_config_yaml(yaml_path)
        experiment_name = config.get('title')
        project_dir = os.path.join(RESULTS_DIR, experiment_name)
    else:
        config = utils.load_config_yaml(BASE_YAML_PATH)  # load base config (remote or local)
        experiment_name = config.get('title', 'test')
        print('this experiment is on debug. no folders are going to be created.')
        project_dir = os.path.join(RESULTS_DIR, 'test')

    log_dir = pathlib.Path(os.path.join(project_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    ##############################
    #   DISTRIBUTED DATA PARALLEL
    if "WORLD_SIZE" in os.environ:
        rank = 0
        if args.local_rank != -1:  # for torch.distributed.launch
            rank = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            rank = int(os.environ['SLURM_PROCID'])

        # suppress printing if not on master gpu, and not debugging on 00
        if rank != 0 and os.environ['SLURM_NODELIST'] != 'aimagelab-srv-00':
            def print_pass(*args, end=None):
                pass
            builtins.print = print_pass

        print(f'cuda visible divices: {torch.cuda.device_count()}')
        print(f'world size: {os.environ["WORLD_SIZE"]}')
        print(f'master address: {os.environ["MASTER_ADDR"]}')
        print(f'master port: {os.environ["MASTER_PORT"]}')
        print(f'dist backend: {args.dist_backend}')
        print(f'dist url: {args.dist_url}')
    # END OF DISTRIBUTED BOOTSTRAP
    #####

    if not args.verbose:
        # redirect streams to project dir
        sys.stdout = open(os.path.join(log_dir, 'std.log'), 'a+')
        sys.stderr = sys.stdout
        utils.set_logger(os.path.join(log_dir, 'logging.log'))
    else:
        # not create folder here, just log to console
        utils.set_logger()

    if args.test:
        config['trainer']['do_train'] = False
        config['trainer']['do_pre_train'] = False
        config['data-loader']['num_workers'] = False
        config['trainer']['checkpoint_path'] = os.path.join(project_dir, 'best.pth')

    main(experiment_name, args)

import argparse
import os
import pathlib
import torch
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import utils
from dataset import AlveolarDataloader
from eval import Eval as Evaluator
from losses import LossFn
from test import test
from train import train
import sys
import logging
import torch.nn as nn
from shutil import copyfile
import numpy as np
from os import path

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment_path', default="test", help="path to the experiment folder")
arg_parser.add_argument('--base_config', default="config.yaml", help='path to the yaml config file')
arg_parser.add_argument('--verbose', action='store_true', help="if true sdout is not redirected")
# arg_parser.add_argument('--save_on_disk', action='store_true', help="if true sdout is not redirected")

args = arg_parser.parse_args()
project_dir = args.experiment_path
log_dir = pathlib.Path(os.path.join(project_dir, 'logs'))

if project_dir != 'test' and path.exists(project_dir):
    config = utils.load_config_yaml(path.join(project_dir, 'logs', 'config.yaml'))
else:
    print('creating a new config file for this experiment.')
    config = utils.load_config_yaml(path.join('configs', args.base_config))  # load base config

    # creating main directory for this experiment, cp subdir, result subdir, logsubdir
    pathlib.Path(os.path.join(project_dir)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(project_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(project_dir, 'files')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(project_dir, 'numpy')).mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    copyfile(os.path.join('configs', args.base_config), path.join(log_dir, 'config.yaml'))
    print("local folders and config files creating. going to execute.")

# redirect streams to project dir
if not args.verbose:
    sys.stdout = open(os.path.join(log_dir, 'std.log'), 'a+')
    sys.stderr = sys.stdout
utils.set_logger(os.path.join(log_dir, 'logging.log'))


def main():

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        logging.info(f"This model will run on {torch.cuda.get_device_name(torch.cuda.current_device())}")

    loader_config = config.get('data-loader', None)
    train_config = config.get('trainer', None)

    model = utils.load_model(
        config.get('model'),
        num_classes=len(loader_config['labels']),
    )

    if torch.cuda.device_count() > 1:
        logging.info("Let's use {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

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
    if not sched_config.get('name', None):
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config['milestones'],
            gamma=sched_config['factor'],
        )

    evaluator = Evaluator(loader_config)

    alveolar_data = AlveolarDataloader(config=loader_config)
    train_id, test_id = alveolar_data.split_dataset()

    train_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=SubsetRandomSampler(train_id),
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    test_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=SubsetRandomSampler(test_id),
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    loss = LossFn(config.get('loss'), loader_config, device, weights=alveolar_data.get_weights())

    current_epoch = 0
    if train_config['checkpoint_path'] is not None:
        try:
            checkpoint = torch.load(train_config['checkpoint_path'])
            model.load_state_dict(checkpoint['state_dict'])
            current_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Checkpoint loaded successfully at epoch {current_epoch}, score:{checkpoint.get('metric', 'unavailable')})")
        except OSError as e:
            logging.info("No checkpoint exists from '{}'. Skipping...".format(train_config['checkpoint_path']))

    tb_name = path.basename(path.normpath(args.experiment_path))
    writer = SummaryWriter(log_dir=os.path.join(config['tb_dir'], tb_name), purge_step=current_epoch)
    vol_writer = utils.SimpleDumper(loader_config, tb_name, project_dir)

    if train_config['do_train']:
        best_metric = 0

        warm_up = np.ones(shape=train_config['epochs'])
        warm_up[0:int(train_config['epochs'] * train_config.get('warm_up_length', 0.35))] = np.linspace(0, 1, num=int(train_config['epochs'] * 0.35))

        for epoch in range(current_epoch, train_config['epochs']):

            train(model, train_loader, loss, optimizer, device, epoch, writer, evaluator, warm_up[epoch])

            if scheduler is not None:
                scheduler.step(current_epoch)

            if epoch % train_config.get('validate_after_iters', 2) == 0:
                val_metric = test(model, test_loader, loss, device, epoch, writer, evaluator, warm_up[epoch])
                if val_metric > best_metric:
                    best_metric = val_metric
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'metric': best_metric
                    }
                    torch.save(state, os.path.join(project_dir, 'best.pth'))

            if epoch % 10 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                torch.save(
                    state,
                    os.path.join(project_dir, 'checkpoints', 'cp_epoch_' + str(epoch) + '.pth')
                )
        logging.info('BEST METRIC IS {}'.format(best_metric))

    # final test
    test_score = test(model, test_loader, loss, device, train_config['epochs'] + 1, writer, evaluator, 1, dumper=vol_writer)
    logging.info('TEST METRIC IS {}'.format(test_score))


if __name__ == '__main__':
    main()

import logging
import os
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import cv2
import yaml
import torch.nn as nn
import torch
from models.PadUNet3D import padUNet3D


def convert_to_two_labels(volume):
    """
    WARNING: HARD CODED LABELS: USE THIS TOOL AT YOUR OWN RISK!
    :param volume: ground truth volume to convert
    :return: converted ground truth volume
    """
    volume[volume == 0] = 1
    volume[volume == 3] = 2
    volume[volume == 2] = 0
    return volume


def npy_maker(path):
    """
    creating a npy file from a set of mask images.
    obsolete with tool version 1.1
    :param path: path to the image folder
    """
    idxs = [
            int(re.search('(\d{1,5})_mask.png', filename).group(1))
            for filename in listdir(join(path, 'mask'))
            if isfile(join(path, 'mask', filename))
        ]
    idxs.sort()
    np.save(
        join(path, 'gt.npy'),
        np.stack([cv2.imread(join(path, 'mask', '{}_mask.png'.format(idx))) for idx in idxs])
    )
    np.save(
        join(path, 'data.npy'),
        np.stack([cv2.imread(join(path, 'img', '{}_img.png'.format(idx))) for idx in idxs])
    )


def set_logger(log_path=None):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(os.path.join(log_path))
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # # Logging to console
        # stream_handler = logging.StreamHandler(sys.stdout)
        # stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(stream_handler)


def load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def load_model(model_config, num_classes):
    if model_config['name'] == 'UNet3D':
        return padUNet3D(n_classes=num_classes)
    else:
        raise Exception("Model not found, check the config.yaml")


class SimpleDumper:
    def __init__(self, loader_config, exp_name, project_dir):
        self.config = loader_config
        self.title = exp_name
        self.project_dir = project_dir

    def dump(self, volume, gt_volume, prediction, iteration):
        np.save(os.path.join(self.project_dir, 'files', '{}_patient{}_data.npy'.format(self.title, iteration)), volume)
        np.save(os.path.join(self.project_dir, 'files', '{}_patient{}_gt.npy'.format(self.title, iteration)), gt_volume)
        np.save(os.path.join(self.project_dir, 'files', '{}_patient{}_pred.npy'.format(self.title, iteration)), prediction)

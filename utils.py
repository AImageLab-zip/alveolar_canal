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


def crop_spatial_dims(input, ref):
    """
    Crops the spatial dimensions of the input tensor to stick to the spatial dimensions
    of the reference tensor. The input is cropped in order to extract the center part.
    Both arguments must have the spatial dimensions as the last 3 axes (3D data) with same value,
    and the dimensionality must be even.

    Args:
        input (torch.Tensor): tensor to crop on spatial dimensions
        ref (torch.Tensor): tensor reference for dimensions.
    Returns:
        (torch.Tensor): tensor with as many dimensions as the input,
            but with reduced spatial dimensions, as ref spatial dimensions.
    """
    if input.shape[-3:] != ref.shape[-3:]:
        assert (input.shape[-1] % 2 == 0 and ref.shape[-1] % 2 == 0)
        start = (input.shape[-1] - ref.shape[-1]) // 2
        end = input.shape[-1] - start
        return input[..., start:end, start:end, start:end]
    return input


# Since this functions is imported in KwakUNet3D,
# it must be defined before its import in this module
from models.KwakUNet3D import KwakUNet3D
from models.PadUNet3D import padUNet3D


def npy_maker(path):
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
    if model_config['name'] == 'KwakUNet3D':
        return KwakUNet3D(n_classes=num_classes)
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

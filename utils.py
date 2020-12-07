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
from models.ResidualEncoder import ResNetEncoder


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
    elif model_config['name'] == 'RESNET18':
        return ResNetEncoder(n_classes=num_classes)
    else:
        raise Exception("Model not found, check the config.yaml")


def compute_skeleton(img):
    """
    create the skeleton using morphology
    Args:
        img (numpy array): source image

    Returns:
        (numpy array), b&w image: 0 background, 255 skeleton elements
    """

    img = img.astype(np.uint8)
    size = img.size
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = np.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            return skel


def arch_detection(slice):
    """
    compute a polynomial spline of the dental arch from a DICOM file
    Args:
        slice (numpy array): source image. Must be float with values in range [0,1]

    Returns:
        (poly1d object): polynomial function approximation
        (float) starting value for the X axis
        (float) ending value for the X axis
    """

    def score_func(arch, th):
        tmp = cv2.threshold(arch, th, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        score = tmp[tmp == 1].size / tmp.size
        return score

    # initial closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    arch = cv2.morphologyEx(slice, cv2.MORPH_CLOSE, kernel)

    th = 0.50

    max_it = 20
    step_l = 0.01

    # varying threshold until we white area is about 12%
    h_th = 0.17
    l_th = 0.11

    poly_x = [th / 20 for th in range(0, 20, 1)]
    poly_y = [score_func(arch, th) for th in poly_x]
    th2score = np.poly1d(np.polyfit(poly_x, poly_y, 12))

    for _ in range(max_it):
        score = th2score(th)
        if h_th > score > l_th:
            arch = cv2.threshold(arch, th, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
            break
        d = -np.polyder(th2score)(th)
        if score > h_th:
            th = th + step_l * d  # higher threshold, lower score
        elif score < l_th:
            th = th - step_l * d  # lower threshold, higher score

    # major filtering with labelling: first remove all the little white parts
    ret, labels = cv2.connectedComponents(arch)
    sizes = np.asarray([labels[labels == label].size for label in range(1, ret)])
    labels[labels != (sizes.argmax() + 1)] = 0  # set all not maximum components to background
    labels[labels == (sizes.argmax() + 1)] = 1  # set the biggest components as foreground

    # let's now fill the rest of the holes if any
    labels = 1 - labels
    ret, labels = cv2.connectedComponents(labels.astype(np.uint8))
    sizes = np.asarray([labels[labels == label].size for label in range(1, ret)])
    labels[labels != (sizes.argmax() + 1)] = 0
    labels[labels == (sizes.argmax() + 1)] = 1
    labels = 1 - labels

    # for label in range(1, ret):
    #     if labels[labels == label].size < 10000:
    #         labels[labels == label] = 0


    # compute skeleton
    skel = compute_skeleton(labels)


    # regression polynomial function
    coords = np.argwhere(skel > 0)
    y = [y for y, x in coords]
    x = [x for y, x in coords]
    pol = np.polyfit(x, y, 12)
    p = np.poly1d(pol)

    return p, min(x), max(x)


def arch_lines(func, start, end, offset=50):
    """
    this functions uses the first order derivative of the function func to track the proper points (x,y) from start to end.
    Args:
        func (poly1d object): polynomial function approximation
        end (float) starting value for the X axis
        start (float) ending value for the X axis
        offset (Int): offset for generating two more curves

    Returns:
        low_offset (numpy array): set of sets of xy coordinates (lower offset)
        coords (numpy array): set of sets of xy coordinates
        high_offset (numpy array): set of sets of xy coordinates (higer offset)
        derivative: set of derivates foreach point of coords
    """

    d = 1
    delta = 0.3
    coords = []
    x = start + 1
    # we start from the range of X values on the X axis,
    # we create a new list X of x coords along the curve
    # we exploit the first order derivative to place values in X
    # so that f(X) is equally distant for each point in X
    while x < end:
        coords.append((x, func(x)))
        alfa = (func(x + delta / 2) - func(x - delta / 2)) / delta
        x = x + d * np.sqrt(1 / (alfa ** 2 + 1))

    # creating lines parallel to the spline
    high_offset = []
    low_offset = []
    derivative = []
    for x, y in coords:
        alfa = (func(x + delta / 2) - func(x - delta / 2)) / delta  # first derivative
        alfa = -1 / alfa  # perpendicular coeff
        cos = np.sqrt(1 / (alfa ** 2 + 1))
        sin = np.sqrt(alfa ** 2 / (alfa ** 2 + 1))
        if alfa > 0:
            low_offset.append((x + offset * cos, y + offset * sin))
            high_offset.append((x - offset * cos, y - offset * sin))
        else:
            low_offset.append((x - offset * cos, y + offset * sin))
            high_offset.append((x + offset * cos, y - offset * sin))
        derivative.append(alfa)

    return low_offset, coords, high_offset, derivative


def background_suppression(data):
    """
    detect the best spline from a set of 80 central slices of the volume,
    draw the parallel splines to select the most relevant zone of the volume
    suppress all the data out of this zone
    :param data:
    :return:
    """

    Z = data.shape[0]
    best = 100
    setup = []
    for i in range(Z // 2 - 20, Z // 2 + 20):
        section = data[i]
        p, start, end = arch_detection(section)
        mid = (start + end) // 2
        new_start = start + np.argmax([p(i) for i in range(start, mid)])
        new_end = mid + np.argmax([p(i) for i in range(mid, end)])
        score = abs(p(new_start) - p(new_end))
        if score < best:
            best = score
            setup = [p, new_start, new_end]

    low, _, high, _ = arch_lines(*setup, offset=70)
    assert (len(low) == len(high))

    for i in range(len(low)):
        # dealing with aliasing
        lx1, ly1 = np.floor(low[i]).astype(int)
        lx2, ly2 = np.ceil(low[i]).astype(int)
        hx1, hy1 = np.floor(high[i]).astype(int)
        hx2, hy2 = np.ceil(high[i]).astype(int)

        # avoid coordinates overflow
        hy1 = max(hy1, 0)
        hy2 = max(hy2, 0)
        ly1 = min(ly1, data.shape[1] - 1)
        ly2 = min(ly2, data.shape[1] - 1)
        if hx2 < 0 or hx1 < 0:
            hx1 = hx2 = 0
        elif hx2 >= data.shape[2] or hx1 >= data.shape[2]:
            hx1 = hx2 = data.shape[2] - 1

        # suppressing raw data
        data[:, ly1:, lx1] = 0
        data[:, ly2:, lx2] = 0
        data[:, :hy1, hx1] = 0
        data[:, :hy2, hx2] = 0

    # suppress areas out of the spline on the left using the first coord for the rest of the volume
    data[:, min(int(low[-1][1]), data.shape[1] - 1):, min(int(low[-1][0]), data.shape[2] - 1):] = 0
    data[:, :min(int(high[-1][1]), data.shape[1] - 1), min(int(high[-1][0]), data.shape[2] - 1):] = 0
    # suppress areas out of the spline on the right using the coord coord for the rest of the volume
    data[:, min(int(low[0][1]), data.shape[1] - 1):, :max(int(low[0][0]), 0)] = 0
    data[:, :min(int(high[0][1]), data.shape[1] - 1), :max(int(high[0][0]), 0)] = 0
    return data


class SimpleDumper:
    def __init__(self, loader_config, exp_name, project_dir):
        self.config = loader_config
        self.title = exp_name
        self.project_dir = project_dir

    def dump(self, volume, gt_volume, prediction, iteration):
        np.save(os.path.join(self.project_dir, 'files', '{}_patient{}_data.npy'.format(self.title, iteration)), volume)
        np.save(os.path.join(self.project_dir, 'files', '{}_patient{}_gt.npy'.format(self.title, iteration)), gt_volume)
        np.save(os.path.join(self.project_dir, 'files', '{}_patient{}_pred.npy'.format(self.title, iteration)), prediction)

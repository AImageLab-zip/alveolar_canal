from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from augmentations import RandomRotate, RandomContrast, ElasticDeformation, Normalize, ToTensor, CenterPad, RandomHorizontalFlip, Resize, Rescale
import torch
import json
from tqdm import tqdm
from Jaw import Jaw
import logging
from utils import Splitter


class AlveolarDataloader(Dataset):

    def __init__(self, config, do_train=True, additional_dataset=None):
        """
        data loader
        :param config: yaml file
        :param tuning: use validation as a training set
        """

        self.config = config

        self.patients = {
            'data': [],
            'gt': [],
            'gt_path': [],
            'folder': [],
            'weights': []
        }

        self.split_weights = {'train': 1, 'syntetic': 0.1,'test': 0,'val': 0}

        self.indices = {
            'test': [],
            'train': [],
            'val': [],
            'syntetic': []
        }
        self.additional_dataset = additional_dataset

        self.splitter = Splitter(config.get('split_volumes', (1, 128, 2)))

        self.dicom_max = config.get('volumes_max', 2100)
        self.dicom_min = config.get('volumes_min', 0)

        self.augmentation = transforms.Compose([
            # RandomRotate(execution_probability=0.5, order=4),
            RandomHorizontalFlip(execution_probability=0.7),
            # RandomContrast(execution_probability=0.5),
            # ElasticDeformation(execution_probability=0.2),
        ])

        reshape_size = self.config.get('resize_shape', (152, 224, 256))
        self.reshape_size = tuple(reshape_size) if type(reshape_size) == list else reshape_size

        self.mean = self.config.get('mean', None)
        self.std = self.config.get('std', None)
        self.means = []
        self.stds = []

        split_filepath = config.get('split_filepath')
        logging.info(f"split filepath is {split_filepath}")
        with open(split_filepath) as f:
            folder_splits = json.load(f)

        if not do_train:
            folder_splits['train'] = []
            folder_splits['syntetic'] = []
            logging.info("training is going to be skipped")
        else:
            if self.additional_dataset:
                train_len, syntetic_len = len(folder_splits['train']), len(folder_splits['syntetic'])
                self.split_weights['train'] = 1 - train_len / (train_len + syntetic_len)
                self.split_weights['syntetic'] = 1 - syntetic_len / (train_len + syntetic_len)
                logging.info(f"using syntetic dataset -> {self.additional_dataset}")
            else:
                folder_splits['syntetic'] = []

        index = 0
        for partition, folders in folder_splits.items():
            logging.info(f"loading data for {partition} - tot: {len(folders)}.")
            for patient_num, folder in tqdm(enumerate(folders), total=len(folders)):

                if partition == "syntetic":
                    data_path = os.path.join(config['sparse_path'], folder, 'data.npy')
                    gt_filename = 'syntetic.npy' if additional_dataset == 'Naive' else 'generated.npy'
                    assert gt_filename != 'generated', "NOT READY YET!"
                    gt_path = os.path.join(config['sparse_path'], folder, gt_filename)
                else:
                    data_path = os.path.join(config['file_path'], folder, 'data.npy')
                    gt_filename = 'gt_alpha_multi.npy' if 'CONTOUR' in self.config['labels'] else 'gt_alpha.npy'
                    gt_path = os.path.join(config['file_path'], folder, gt_filename)

                data = np.load(data_path)
                gt = np.load(gt_path)
                assert np.max(data) > 1  # data should not be normalized by default
                assert np.unique(gt).size <= len(self.config['labels'])

                data, gt = self.preprocessing(data, gt, folder, partition=partition)
                self.patients['data'] += data
                self.patients['gt'] += gt
                self.patients['gt_path'] += [gt_path for i in data]  # replicating the name of the folder N times
                self.patients['folder'] += [folder for i in data]  # replicating the name of the folder N times
                self.patients['weights'] += [self.split_weights[partition] for i in range(len(data))]  # replicating weights N times
                self.indices[partition] += list(range(index, index + len(data)))
                index = index + len(data)

        if self.mean is None or self.std is None:
            self.mean = np.mean(self.means)
            self.std = np.mean(self.stds)

        logging.info(f'mean for the dataset: {self.mean}, std: {self.std}')

        self.indices['train'] = np.asarray(self.indices['train'] + self.indices['syntetic'])
        self.indices['test'] = np.asarray(self.indices['test'])
        self.indices['val'] = np.asarray(self.indices['val'])

        self.weights = self.config.get('weights', None)
        if self.weights is None:
            logging.info('going to compute weights')
            self.weights = self.median_frequency_balancing()
        else:
            self.weights = torch.Tensor(self.weights)

        logging.info(f'weights for this dataset: {self.weights}')
        logging.info('folders in validation set: {}'.format(folder_splits.get('val', 'None')))
        logging.info('folders in test set: {}'.format(folder_splits['test']))

    def __len__(self):
        return self.indices['train'].size + self.indices['test'].size + self.indices['val'].size

    def get_weights(self):
        return self.weights

    def get_config(self):
        return self.config

    def preprocessing(self, data, gt, folder, partition='train'):

        # rescale
        data = np.clip(data, self.dicom_min, self.dicom_max)
        data = (data.astype(np.float) + self.dicom_min) / (self.dicom_max + self.dicom_min)   # [0-1] with shifting

        if self.mean is None or self.std is None:
            self.means.append(np.mean(data))
            self.stds.append(np.std(data))

        D, H, W = data.shape[-3:]
        rD, rH, rW = self.reshape_size
        tmp_ratio = np.array((D/W, H/W, 1))
        pad_factor = tmp_ratio / np.array((rD/rW, rH/rW, 1))
        pad_factor /= np.max(pad_factor)
        new_shape = np.array((D, H, W)) / pad_factor
        new_shape = np.round(new_shape).astype(np.int)

        data = CenterPad(new_shape)(data)
        data = Rescale(size=self.reshape_size)(data)

        if partition == 'train':
            gt = CenterPad(new_shape)(gt, pad_val=self.config['labels']['BACKGROUND'])
            gt = Rescale(size=self.reshape_size, interp_fn='nearest')(gt)
        else:
            gt = np.zeros_like(data)  # this is because in test and train we load gt at runtime

        data = [np.squeeze(d) for d in self.splitter.split(data)]
        gt = [np.squeeze(g) for g in self.splitter.split(gt)]
        return data, gt

    def __getitem__(self, index):
        vol, gt = self.patients['data'][index].astype(np.float32), self.patients['gt'][index].astype(np.int64)
        weights = self.patients['weights'][index]
        folders = self.patients['folder'][index]
        gt_paths = self.patients['gt_path'][index]

        if index in self.indices['train']:
            vol, gt = self.augmentation([vol, gt])
            assert np.array_equal(gt, gt.astype(bool)), 'something wrong with augmentations here'

        vol = transforms.Normalize(self.mean, self.std)(ToTensor()(vol.copy()))
        gt = ToTensor()(gt.copy())
        ToTensor()(np.asarray(self.patients['weights'][index]).astype(np.float32))
        vol = vol.repeat(3, 1, 1)  # creating the channel axis and making it RGB
        return vol, gt, folders, weights, gt_paths

    def get_splitter(self):
        return self.splitter

    def split_dataset(self):

        np.random.shuffle(self.indices['train'])
        return self.indices['train'], self.indices['test'], self.indices['val']

    def class_freq(self):
        """
        Computes class frequencies for each label.
        Returns the number of pixels of class c (in all images) divided by the total number of pixels (in images where c is present).
        Returns:
            (torch.Tensor): tensor with shape n_labels, with class frequencies for each label.
        """
        num_labels = len(self.config['labels'])
        class_pixel_count = torch.zeros(num_labels)
        total_pixel_count = torch.zeros(num_labels)

        for gt in self.patients['gt']:
            gt_ = torch.from_numpy(gt)
            counts = torch.bincount(gt_.flatten())
            class_pixel_count += counts
            n_pixels = gt_.numel()
            total_pixel_count = torch.where(counts > 0, total_pixel_count + n_pixels, total_pixel_count)

        return class_pixel_count / total_pixel_count

    def class_freq_2(self, valid_labels):

        num_labels = len(self.config.get('labels'))
        class_pixel_count = num_labels * [0]

        for idx in self.indices['train']:
            gt = self.patients['gt'][idx]
            for l in valid_labels:
                class_pixel_count[l] += np.sum(gt == l) / np.sum(np.in1d(gt, valid_labels))

        return [c / len(self.patients['gt']) for c in class_pixel_count]

    def median_frequency_balancing(self):
        """
        Computes class weights using Median Frequency Balancing.
        Source paper: https://arxiv.org/pdf/1411.4734.pdf (par. 6.3.2)
        Returns:
            (torch.Tensor): class weights
        """
        excluded = ['UNLABELED']
        valid_labels = [v for k, v in self.config.get('labels').items() if k not in excluded]
        freq = self.class_freq_2(valid_labels)
        # freq = self.class_freq()
        # sorted, _ = torch.sort(freq)
        median = torch.median(torch.Tensor([f for (i, f) in enumerate(freq) if i in valid_labels]))
        # median = torch.median(freq)
        weights = torch.Tensor([median / f if f != 0 else 0 for f in freq])
        weights /= weights.sum()  # normalizing
        return weights

    def pytorch_weight_sys(self):
        excluded = ['UNLABELED']
        valid_labels = [v for k, v in self.config.get('labels').items() if k not in excluded]

        class_pixel_count = torch.zeros(len(self.config.get('labels')))
        not_class_pixel_count = torch.zeros(len(self.config.get('labels')))
        for gt in self.patients['gt']:
            for l in valid_labels:
                class_pixel_count[l] += np.sum(gt == l)
                not_class_pixel_count[l] += np.sum(np.in1d(gt, [v for v in valid_labels if v != l]))

        return not_class_pixel_count / (class_pixel_count + 1e-06)

    def custom_collate(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        folders = [item[2] for item in batch]
        weights = [item[3] for item in batch]
        gt_paths = [item[4] for item in batch]
        return images, labels, folders, weights, gt_paths

import numpy as np
import os
import torch
import json
import torch.utils.data as data

from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm

import logging
import torchio as tio

# from augmentations import *
from jaw.Jaw import Jaw
from dataloader.AugFactory import *


class Loader3D():
    def __init__(self, config, do_train=True, additional_dataset=None, is_competitor=False):

        self.config = config

        self.subjects = {
            'training': [],
            'synthetic': [],
            'validation': [],
            'test': [],
        }

        #  default weights to be overridden
        self.split_weights = {'training': 1, 'synthetic': 1}

        self.do_train = do_train
        self.additional_dataset = additional_dataset

        self.dicom_max = config.get('volumes_max', 2100)
        self.dicom_min = config.get('volumes_min', 0)

        aug_filepath = config.get("augmentations_file", None)
        aug_list = [] if aug_filepath is None else yaml.load(open(aug_filepath, 'r'), yaml.FullLoader)

        # Is a class really necessary here?
        self.transforms = AugFactory(aug_list).get_transform()

        reshape_size = self.config.get('resize_shape', (152, 224, 256))
        self.reshape_size = tuple(reshape_size) if type(reshape_size) == list else reshape_size

        # split_filepath = config.get('split_filepath')
        # logging.info(f"split filepath is {split_filepath}")
        # with open(split_filepath) as f:
        #     folder_splits = json.load(f)
        folder_splits = config.dataset.split

        if not do_train:
            folder_splits['training'] = []
            folder_splits['synthetic'] = []
            logging.info("training is going to be skipped")

        for partition, folders in folder_splits.items():
            print(f"loading data for {partition} - tot: {len(folders)}.")
            logging.info(f"loading data for {partition} - tot: {len(folders)}.")
            for patient_num, folder in tqdm(enumerate(folders), total=len(folders)):

                data_path = os.path.join(config.dataset.sparse_path, folder, 'data.npy')

                if partition == 'synthetic':
                    assert additional_dataset in ['Naive', 'Generated']
                    filename = 'synthetic.npy' if additional_dataset == 'Naive' else 'generated.npy'
                    gt_path = os.path.join(config.dataset.sparse_path, folder, filename)
                elif partition in ['training', 'validation'] and is_competitor:
                    gt_path = os.path.join(config.dataset.sparse_path, folder, 'synthetic.npy')
                else:
                    gt_filename = 'gt_alpha_multi.npy' if 'CONTOUR' in self.config.labels else 'gt_alpha.npy'
                    gt_path = os.path.join(config.dataset.file_path, folder, gt_filename)

                data = np.load(data_path)
                gt = np.load(gt_path)

                assert np.max(data) > 1  # data should NOT be normalized by default
                assert np.unique(gt).size <= len(self.config.labels)

                self.subjects[partition].append(
                    self.preprocessing(data, gt, infos=(data_path, gt_path, folder, partition))
                )

        self.weights = self.config.get('weights', None)
        if self.weights is None:
            logging.info('going to compute weights')
            self.weights = self.median_frequency_balancing()
        else:
            self.weights = torch.Tensor(self.weights)
        logging.info(f'weights for this dataset: {self.weights}')

        logging.info('folders in validation set: {}'.format(folder_splits.validation))
        logging.info('folders in test set: {}'.format(folder_splits.test))

    def get_weights(self):
        return self.weights

    def preprocessing(self, data, gt, infos):

        data_path, gt_path, folder, partition = infos

        # rescale
        data = np.clip(data, self.dicom_min, self.dicom_max)
        data = (data.astype(np.float) + self.dicom_min) / (self.dicom_max + self.dicom_min)   # [0-1] with shifting

        safe_gt_check = np.sum(gt)
        data = CropAndPad(self.reshape_size)(data)
        gt = CropAndPad(self.reshape_size, pad_val=self.config.labels.BACKGROUND)(gt)
        if safe_gt_check != np.sum(gt):
            logging.info(f"BIG WARNING: we are missing some GT voxel with this crop! {folder}, {partition}")

        gt = gt.astype(np.uint8)

        # adding the 4th dim and making it RGB
        if gt.ndim == 3: gt = gt.reshape(1, *gt.shape)
        if data.ndim == 3: data = np.tile(data.reshape(1, *data.shape), (3, 1, 1, 1))

        if partition in ['training', 'synthetic']:
            return tio.Subject(
                data=tio.ScalarImage(tensor=data),
                gt_path=gt_path,
                data_path=data_path,
                folder=folder,
                weight=self.split_weights[partition],
                partition=partition,
                label=tio.LabelMap(tensor=gt),
            )

        return tio.Subject(
            data=tio.ScalarImage(tensor=data),
            gt_path=gt_path,
            data_path=data_path,
            folder=folder,
            partition=partition
        )

    def get_aggregator(self):
        sampler = self.get_sampler()
        return tio.inference.GridAggregator(sampler)

    def get_sampler(self, type, overlap=0):
        patch_shape = self.config.patch_shape
        return tio.GridSampler(patch_size=patch_shape, patch_overlap=overlap)

    def split_dataset(self, rank=0, world_size=1):
        training_set = self.subjects['training'] + self.subjects['synthetic']
        train = tio.SubjectsDataset(training_set[rank::world_size], transform=self.transforms) if self.do_train else None
        # logging.info("using the following augmentations: ", train[0].history)

        if rank != 0: return train, None, None

        patch_shape = self.config.patch_shape
        test = [tio.GridSampler(subject, patch_size=patch_shape, patch_overlap=0) for subject in self.subjects['test']]
        val = [tio.GridSampler(subject, patch_size=patch_shape, patch_overlap=0) for subject in self.subjects['validation']]
        return train, test, val


    ##################################
    #   WEIGHTS COMPUTATION ALGORITHMS

    def class_freq(self):
        """
        Computes class frequencies for each label.
        Returns the number of pixels of class c (in all images) divided by the total number of pixels (in images where c is present).
        Returns:
            (torch.Tensor): tensor with shape n_labels, with class frequencies for each label.
        """
        num_labels = len(self.config.labels)
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

        num_labels = len(self.config.labels)
        class_pixel_count = num_labels * [0]

        for p in self.subjects['training']:
            gt = p['label'][tio.DATA].cpu().numpy()
            for l in valid_labels:
                class_pixel_count[l] += np.sum(gt == l) / np.sum(np.in1d(gt, valid_labels))

        return [c / len(self.subjects['training']) for c in class_pixel_count]

    def median_frequency_balancing(self):
        """
        Computes class weights using Median Frequency Balancing.
        Source paper: https://arxiv.org/pdf/1411.4734.pdf (par. 6.3.2)
        Returns:
            (torch.Tensor): class weights
        """
        excluded = ['UNLABELED']
        valid_labels = [v for k, v in self.config.labels.items() if k not in excluded]
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

    def load(self, rank, world_size, is_distributed, is_competitor=False):

        train_loader, test_loader, val_loader = None, None, None

        # data_utils = Loader3D(loader_config, train_config.get("do_train", None), None, is_competitor)
        train_d, test_d, val_d = self.split_dataset()
        splitter = None

        if self.do_train:
            samples_per_volume = int(np.prod([np.round(i / j) for i, j in zip(self.config.resize_shape, self.config.patch_shape)]))
            train_queue = tio.Queue(
                train_d,
                max_length=samples_per_volume * 4,  # queue len
                samples_per_volume=samples_per_volume,
                sampler=self.get_sampler(self.config.sampler_type, self.config.grid_overlap),
                num_workers = self.config.num_workers,
            )
            sampler = DistributedSampler(train_queue, shuffle=False) if is_distributed else None
            train_loader = data.DataLoader(train_queue, self.config.batch_size // world_size, num_workers=0, sampler=sampler)

        if rank == 0:
            test_loader = [(test_p, data.DataLoader(test_p, self.config.batch_size, num_workers=self.config.num_workers)) for test_p in test_d]
            val_loader = [(val_p, data.DataLoader(val_p, self.config.batch_size, num_workers=self.config.num_workers)) for val_p in val_d]

        return train_loader, test_loader, val_loader, splitter


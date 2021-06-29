from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from augmentations import RandomRotate, RandomContrast, ElasticDeformation, Normalize, ToTensor, CenterPad, RandomVerticalFlip, Resize, Rescale
import torch
import json
from tqdm import tqdm
from Jaw import Jaw
import logging
import torchio as tio
import utils

class NewLoader():

    def __init__(self, config, do_train=True, use_syntetic=True):

        self.config = config

        self.subjects = {
            'train': [],
            'syntetic': [],
            'test': [],
            'val': []
        }

        self.split_weights = {
            'train': 1,
            'syntetic': 0.1,
        }

        self.DEBUG = []
        self.do_train = do_train
        self.use_syntetic = use_syntetic

        self.dicom_max = config.get('volumes_max', 2100)
        self.dicom_min = config.get('volumes_min', 0)

        aug_filepath = config.get("augmentations_file", None)
        auglist = [] if aug_filepath is None else utils.load_config_yaml(aug_filepath)
        augment = AugFactory(auglist)
        augment.log()  # write what we are using to logfile
        self.transforms = augment.get_transform()

        reshape_size = self.config.get('resize_shape', (152, 224, 256))
        self.reshape_size = tuple(reshape_size) if type(reshape_size) == list else reshape_size

        with open(config.get('split_filepath', '/homes/mcipriano/projects/alveolar_canal_3Dtraining/configs/splits.json')) as f:
            folder_splits = json.load(f)

        if not do_train:
            folder_splits['train'] = []
        else:
            if self.use_syntetic:
                train_len, syntetic_len = len(folder_splits['train']), len(folder_splits['syntetic'])
                self.split_weights['train'] = 1 - train_len / (train_len + syntetic_len)
                self.split_weights['syntetic'] = 1 - syntetic_len / (train_len + syntetic_len)
                logging.info("using syntetic data too")

        for partition, folders in folder_splits.items():
            logging.info(f"loading data for {partition} - tot: ({len(folders)}")
            for patient_num, folder in tqdm(enumerate(folders), total=len(folders)):

                sparse_path = os.path.join(config['sparse_path'], folder, 'gt_sparse.npy')
                if partition == "syntetic":
                    data_path = os.path.join(config['sparse_path'], folder, 'data.npy')
                    gt_path = os.path.join(config['sparse_path'], folder, 'syntetic.npy')
                else:
                    data_path = os.path.join(config['file_path'], folder, 'data.npy')
                    gt_filename = 'gt_alpha_multi.npy' if 'CONTOUR' in self.config['labels'] else 'gt_alpha.npy'
                    gt_path = os.path.join(config['file_path'], folder, gt_filename)

                data = np.load(data_path)
                gt = np.load(gt_path)
                sparse = np.load(sparse_path)

                assert np.max(data) > 1  # data should not be normalized by default
                assert np.unique(gt).size <= len(self.config['labels'])
                assert np.max(sparse) <= 1

                self.subjects[partition].append(
                    self.preprocessing(data, gt, sparse, infos=(data_path, gt_path, sparse_path, folder, partition))
                )

        self.weights = self.config.get('weights', None)
        if self.weights is None:
            logging.info('going to compute weights')
            self.weights = self.median_frequency_balancing()
        else:
            self.weights = torch.Tensor(self.weights)
        logging.info(f'weights for this dataset: {self.weights}')

        logging.info('folders in validation set: {}'.format(folder_splits['val']))
        logging.info('folders in test set: {}'.format(folder_splits['test']))

    def get_weights(self):
        return self.weights

    def preprocessing(self, data, gt, sparse, infos):

        data_path, gt_path, sparse_path, folder, partition = infos

        # rescale
        data = np.clip(data, self.dicom_min, self.dicom_max)
        data = (data.astype(np.float) + self.dicom_min) / (self.dicom_max + self.dicom_min)   # [0-1] with shifting

        D, H, W = data.shape[-3:]
        rD, rH, rW = self.reshape_size
        tmp_ratio = np.array((D/W, H/W, 1))
        pad_factor = tmp_ratio / np.array((rD/rW, rH/rW, 1))
        pad_factor /= np.max(pad_factor)
        new_shape = np.array((D, H, W)) / pad_factor
        new_shape = np.round(new_shape).astype(np.int)

        data = CenterPad(new_shape)(data)
        data = Rescale(size=self.reshape_size)(data)

        sparse = CenterPad(new_shape)(sparse, pad_val=self.config['labels']['BACKGROUND'])
        sparse = Rescale(size=self.reshape_size, interp_fn='nearest')(sparse)

        # adding the 4th dim and making it RGB
        if data.ndim == 3: data = np.tile(data.reshape(1, *data.shape), (3, 1, 1, 1))
        if sparse.ndim == 3: sparse = np.tile(sparse.reshape(1, *sparse.shape), (3, 1, 1, 1))

        if partition in ['train', 'syntetic']:
            gt = CenterPad(new_shape)(gt)
            gt = Rescale(size=self.reshape_size, interp_fn='nearest')(gt)
            if gt.ndim == 3: gt = gt.reshape(1, *gt.shape)
            return tio.Subject(
                data=tio.ScalarImage(tensor=data),
                label=tio.LabelMap(tensor=gt),
                sparse=tio.LabelMap(tensor=sparse),
                gt_path=gt_path,
                data_path=data_path,
                folder=folder,
                weight=self.split_weights[partition],
                partition=partition,
            )

        return tio.Subject(
            data=tio.ScalarImage(tensor=data),
            sparse=tio.LabelMap(tensor=sparse),
            gt_path=gt_path,
            data_path=data_path,
            folder=folder,
            partition=partition
        )

    def get_aggregator(self):
        sampler = self.get_sampler()
        return tio.inference.GridAggregator(sampler)

    def get_sampler(self, type, overlap=0):
        patch_shape = self.config['patch_shape']
        if type == 'grid':
            return tio.GridSampler(patch_size=patch_shape, patch_overlap=overlap)
        elif type == 'by_label':
            return tio.LabelSampler(
                patch_size=patch_shape,
                label_name='label',
                label_probabilities={self.config['labels']['BACKGROUND']: 0.1, self.config['labels']['INSIDE']: 0.9}
            )
        else:
            raise Exception('no valid sampling type provided')

    def split_dataset(self, rank=0, world_size=1):
        training_set = self.subjects['train'] + self.subjects['syntetic']
        train = tio.SubjectsDataset(training_set[rank::world_size], transform=self.transforms) if self.do_train else None
        # logging.info("using the following augmentations: ", train[0].history)

        if rank == 0:
            patch_shape = self.config['patch_shape']
            test = [tio.GridSampler(subject, patch_size=patch_shape, patch_overlap=0) for subject in self.subjects['test']]
            val = [tio.GridSampler(subject, patch_size=patch_shape, patch_overlap=0) for subject in self.subjects['val']]
        else:
            test = val = None
        # TODO: grid sampling: might be interesting to make some test with overlapping!
        # TODO: check if grid or weight sampling is selected for the training data. assuming grid right now.

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

        for gt in self.patients['gt']:
            for l in valid_labels:
                class_pixel_count[l] += np.sum(gt == l) / np.sum(np.in1d(gt, valid_labels))

        return [c / len(self.patients['gt']) for c in class_pixel_count]

    def  median_frequency_balancing(self):
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


class AugFactory:
    def __init__(self, aug_list):
        self.aug_list = aug_list
        self.transforms = self.factory(self.aug_list, [])

    def log(self):
        """
        save the list of aug for this experiment to the default log file
        :param path:
        :return:
        """
        logging.info('going to use the following augmentations:: %s', self.aug_list)

    def factory(self, auglist, transforms):
        for aug in auglist:
            if aug == 'OneOf':
                transforms.append(tio.OneOf(self.factory(auglist[aug], [])))
            else:
                try:
                    kwargs = {}
                    for param, value in auglist[aug].items():
                        kwargs[param] = value
                    transforms.append(getattr(tio, aug)(**kwargs))
                except:
                    raise Exception(f"this transform is not valid: {aug}")
        return transforms

    def get_transform(self):
        """
        return the transform object
        :return:
        """
        return tio.Compose(self.transforms)
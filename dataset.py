from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from augmentations import RandomRotate, RandomContrast, ElasticDeformation, Normalize, ToTensor, CenterPad, RandomVerticalFlip, Resize, Rescale
import torch
import utils
from tqdm import tqdm
from Jaw import Jaw
import logging

MAXILLO_DICOM_MAX = 64535
DATASET_MU = 0.453
DATASET_STD = 0.162


class AlveolarDataloader(Dataset):

    def __init__(self, config):

        self.config = config
        self.patients = {
            'data': [],
            'gt': []
        }
        self.indices = {
            'test': [],
            'train': [],
            'val': []
        }

        self.LCM = self.config.get('LCM_cut_factor', 1)
        self.cut_idx = config.get('cut_index', 0.5)

        # test_ids = self.config.get('test_patients_id', [0, 1])
        # if not isinstance(test_ids, list):
        #     test_ids = [test_ids]

        gt_filename = 'gt_4labels.npy' if len(self.config['labels']) > 2 else 'gt_2labels.npy'
        index = 0

        patients = os.listdir(config['file_paths'])
        tot_patients = len(patients)
        patients_ids = np.arange(tot_patients)
        np.random.shuffle(patients_ids)
        test_ids = patients_ids[:int(tot_patients * 0.2)]
        val_ids = patients_ids[int(tot_patients * 0.2):int(tot_patients * 0.3)]

        for patient_num, folder in tqdm(enumerate(patients), total=len(os.listdir(config['file_paths']))):
            # data = np.load(os.path.join(config['file_paths'], folder, 'data.npy'))
            data = Jaw(os.path.join(config['file_paths'], folder, 'DICOM', 'DICOMDIR')).get_volume()
            gt = np.load(os.path.join(config['file_paths'], folder, gt_filename))

            assert data.max() <= MAXILLO_DICOM_MAX  # maximum values of those dicom
            assert data.max() > 1  # data should not be normalized by default

            partition = 'train'
            if patient_num in test_ids:
                partition = 'test'
                logging.info(f'INFO: this patient is in the test-set: {patient_num}, {folder}')
            elif patient_num in val_ids:
                partition = 'val'
                logging.info(f'INFO: this patient is in the val-set: {patient_num}, {folder}')

            right, gright, left, gleft = self.preprocessing(data, gt, folder, partition=partition)
            self.patients['data'] += [right, left]
            self.patients['gt'] += [gright, gleft]

            self.indices[partition] += [index, index + 1]
            index = index + 2

        self.indices['train'] = np.asarray(self.indices['train'])
        self.indices['test'] = np.asarray(self.indices['test'])
        self.indices['val'] = np.asarray(self.indices['val'])

        self.weights = self.median_frequency_balancing()

    def __len__(self):
        return self.indices['train'].size + self.indices['test'].size + self.indices['val'].size

    def get_weights(self):
        return self.weights

    def preprocessing(self, data, gt, folder, partition='train'):

        # rescale
        data = data.astype(np.float) / MAXILLO_DICOM_MAX  # maximum became 1

        Z, H, W = data.shape[-3:]

        reshape_size = self.config.get('resize_shape', (152, 224, 256))
        reshape_size = tuple(reshape_size) if type(reshape_size) == list else reshape_size

        ratio = reshape_size[1]/reshape_size[2]
        new_shape = (Z, H, H // ratio) if H / W > ratio else (Z, W * ratio, W)
        data = CenterPad(new_shape)(data)

        # suppress areas out of the splines
        # if self.config.get('background_suppression', True):
        #     data = utils.background_suppression(data, folder)
        #
        #     # cut the overflowing null areas -> extract cube with extreme limits of where are the values != 0
        #     xs, ys, zs = np.where(data != 0)
        #     data = data[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
        #     if partition == 'train':
        #         gt = gt[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]

        # rescaling
        data = Rescale(size=reshape_size)(data)

        if partition == 'train':
            gt = CenterPad(new_shape)(gt)
            gt = Rescale(size=reshape_size, interp_fn='nearest')(gt)

        # split the jawbone in two pieces
        left, right = np.split(data, [int(0.5 * data.shape[2])], axis=2)
        gleft, gright = np.split(gt, [int(0.5 * gt.shape[2])], axis=2)
        gright = gright[..., :-1] if gright.shape[-1] != gleft.shape[-1] else gright

        return left, gleft, right, gright,

    def augment_dataset(self):
        augmentation = transforms.Compose([
            RandomRotate(execution_probability=0.8),
            RandomVerticalFlip(execution_probability=0.7),
            RandomContrast(execution_probability=0.7),
            ElasticDeformation(execution_probability=0.8),
        ])

        augment_rate = self.config.get('augment_rate', 0)
        for idx in self.indices['train']:
            vol, gt = self.patients['data'][idx], self.patients['gt'][idx]
            for i in range(augment_rate):
                aug_vol, aug_gt = augmentation([vol, gt])
                self.patients['data'].append(aug_vol)
                self.patients['gt'].append(aug_gt)

        return augment_rate * len(self.indices['train'])

    def __getitem__(self, index):
        vol, gt = self.patients['data'][index].astype(np.float32), self.patients['gt'][index].astype(np.int64)
        vol = transforms.Normalize(DATASET_MU, DATASET_STD)(ToTensor()(vol))
        gt = ToTensor()(gt)
        return vol, gt

    def split_dataset(self):

        num_patient = len(self.patients['data'])
        if num_patient < 2:
            raise Exception('less available patients than the ones required for training validate and test!')

        tot_new = self.augment_dataset()
        augmented_idx = np.arange(num_patient, num_patient + tot_new)
        self.indices['train'] = np.concatenate((self.indices['train'], augmented_idx))

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

        for gt in self.patients['gt']:
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

        images = [item[0] for item in batch]
        # finding the best common shape for this batch and pad the data
        batch_shape = np.max(np.array([p.shape for p in images]), axis=0)
        resizer = Resize(self.config.get('labels'), self.LCM)
        batch_shape = resizer.closestDistanceForDivision(np.asarray(batch_shape))

        # resizing the entire batch at the final shape
        images = torch.stack([Rescale(size=tuple(batch_shape))(image) for image in images])
        # images = torch.stack([resizer.reshape(image, new_shape=batch_shape, pad_val=image.min()) for image in images])

        labels = [item[1].unsqueeze(0) if item[1].ndim == 3 else item[1] for item in batch]

        return images, labels
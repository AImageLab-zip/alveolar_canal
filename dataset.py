from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from augmentations import RandomRotate, RandomContrast, ElasticDeformation, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, Resize, Rescale
import torch
import utils

GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class AlveolarDataloader(Dataset):

    def __init__(self, config):

        self.config = config
        self.patients = {
            'data': [],
            'gt': []
        }
        self.indices = {
            'test': [],
            'train': []
        }

        self.cut_idx = config.get('cut_index', 0.5)

        test_ids = self.config.get('test_patients_id', [0, 1])
        if not isinstance(test_ids, list):
            raise Exception('provide test patients ids as a list')

        gt_filename = 'gt_4labels.npy' if len(self.config['labels']) > 2 else 'gt_2labels.npy'

        index = 0
        for patient_num, folder in enumerate(os.listdir(config['file_paths'])):

            data = np.load(os.path.join(config['file_paths'], folder, 'data.npy'))
            gt = np.load(os.path.join(config['file_paths'], folder, gt_filename))

            # fixing channels if needed
            if data.ndim == 4 or gt.ndim == 4:  # (B, H, W, 3) becomes  (B, H, W)
                raise Exception('weird number of dimension for patient {}'.format(folder))

            # fixing values if needed
            if data.max() > 1:
                data = data.astype(np.float) / data.max()

            if patient_num in test_ids:
                # this patient is for testing, won't change anything here
                self.indices['test'].append(index)
                self.patients['data'].append(data)
                self.patients['gt'].append(gt)
                index = index + 1
            else:
                # train data -> split jawbone in two pieces
                right, gright, left, gleft = self.preprocessing(data, gt)
                self.patients['data'] += [right, left]
                self.patients['gt'] += [gright, gleft]
                self.indices['train'] += [index, index + 1]
                index = index + 2

        # pad all the volumes in the training set to the same shape
        LCM = self.config.get('LCM_cut_factor', 1)
        resizer = Resize(self.config.get('labels'), LCM)
        final_shape = max([self.patients['data'][i].shape for i in self.indices['train']])
        final_shape = resizer.closestDistanceForDivision(np.asarray(final_shape))
        for i in range(len(self.patients['data'])):
            self.patients['data'][i], self.patients['gt'][i] = resizer.reshape(
                [self.patients['data'][i], self.patients['gt'][i]],
                new_shape=final_shape
            )

        self.indices['train'] = np.asarray(self.indices['train'])
        self.indices['test'] = np.asarray(self.indices['test'])

        self.seed = GLOBAL_RANDOM_STATE.randint(41)
        self.random_state = np.random.RandomState(self.seed)
        self.weights = self.median_frequency_balancing()

    def __len__(self):
        return self.indices['train'].size + self.indices['test'].size

    def get_weights(self):
        return self.weights

    def preprocessing(self, data, gt):

        # suppress areas out of the splines
        if self.config.get('background_suppression', True):
            data = utils.background_suppression(data)

        # rescale
        if self.config.get('scale_factor', 1) != 1:
            data, gt = Rescale(self.config.get('scale_factor', 0.5), self.config.get('labels'))([data, gt])

        # cut the overflowing null areas -> extract cube with extreme limits of where are the values != 0
        xs, ys, zs = np.where(data != 0)
        data = data[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
        gt = gt[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]

        # split the jawbone in two pieces
        right, left = np.split(data, [int(self.cut_idx * data.shape[2])], axis=2)
        gright, gleft = np.split(gt, [int(self.cut_idx * data.shape[2])], axis=2)
        return right, gright, left, gleft

    def augment_dataset(self):
        augmentation = transforms.Compose([
            RandomRotate(self.random_state, execution_probability=0.5),
            # RandomHorizontalFlip(self.random_state, execution_probability=0.5),
            RandomVerticalFlip(self.random_state, execution_probability=0.7),
            RandomContrast(self.random_state, execution_probability=0.90),
            ElasticDeformation(self.random_state, execution_probability=0.80),
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

        vol, gt = self.patients['data'][index], self.patients['gt'][index]
        vol, gt = ToTensor()([vol, gt])
        vol, gt = Normalize()([vol, gt])

        return vol, gt

    def split_dataset(self):

        num_patient = len(self.patients['data'])
        if num_patient < 2:
            raise Exception('less available patients than the ones required for training validate and test!')

        tot_new = self.augment_dataset()
        augmented_idx = np.arange(num_patient, num_patient + tot_new)
        self.indices['train'] = np.concatenate((self.indices['train'], augmented_idx))

        np.random.seed(41)
        np.random.shuffle(self.indices['train'])

        return self.indices['train'], self.indices['test']

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

    def median_frequency_balancing(self):
        """
        Computes class weights using Median Frequency Balancing.
        Source paper: https://arxiv.org/pdf/1411.4734.pdf (par. 6.3.2)
        Returns:
            (torch.Tensor): class weights
        """
        freq = self.class_freq()
        sorted, _ = torch.sort(freq)
        median = torch.median(freq)
        weights = median / sorted
        weights /= weights.sum()  # normalizing
        return weights
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
import torch

GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class ToTensor:

    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        # Transform to tensor
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))
        if image.ndim == 2:
            image = torch.unsqueeze(image, dim=0)
            mask = torch.unsqueeze(mask, dim=0)
        return [image, mask]


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [0, 1].
    DO NOT AFFECT MASKS!
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        image, mask = data
        value_range = image.max() - image.min()
        norm_0_1 = (image - image.min()) / value_range
        return [np.clip(norm_0_1, 0, 1), mask]


class KwakDataloader(Dataset):
    def __init__(self, config):

        self.config = config
        self.patients = {
            'pos': [],
            'data': [],
            'gt': [],
        }
        self.size = 132
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)
        self.random_state = np.random.RandomState(self.seed)
        self.indices = {}
        self.num_samples = 64

        test_patient_id = config.get('patient_id', 0)
        test_ids = []
        train_ids = []

        gt_filename = 'gt_4labels.npy' if len(self.config['labels']) > 2 else 'gt_2labels.npy'

        for i, folder in enumerate(os.listdir(config['file_paths'])):
            data = np.load(os.path.join(config['file_paths'], folder, 'data.npy'))
            gt = np.load(os.path.join(config['file_paths'], folder, gt_filename))

            # if there are unlabelled voxels, if not specified, we wash them to background
            if config.get('suppress_unlabelled', True) and config['labels'].get('UNLABELED', False):
                gt[gt == config['labels']['UNLABELED']] = config['labels']['BACKGROUND']

            # pre-processing on the channels
            if data.ndim == 4:  # (B, H, W, 3) becomes  (B, H, W)
                data = np.take(data, 0, axis=-1)
            if gt.ndim == 4:  # (B, H, W, 3) becomes  (B, H, W,)
                gt = np.take(gt, 0, axis=-1)

            # continous sub-volumes (test)
            if test_patient_id == i:
                step = self.size // 3
                pad_shape = tuple(map(lambda k: k + 2 * step, data.shape))
                pad_data = np.zeros(pad_shape)
                pad_gt = np.full(pad_shape, self.config['labels']['BACKGROUND'])
                pad_data[step:-step, step:-step, step:-step] = data
                pad_gt[step:-step, step:-step, step:-step] = gt
                padZ, padY, padX = pad_shape

                # fix shape multiple of step
                fix_pad_shape = tuple(map(lambda k: (k // step + 1) * step, pad_shape))
                fix_pad_data = np.zeros(fix_pad_shape)
                fix_pad_gt = np.full(fix_pad_shape, self.config['labels']['BACKGROUND'])
                fix_pad_data[0:padZ, 0:padY, 0:padX] = pad_data
                fix_pad_gt[0:padZ, 0:padY, 0:padX] = pad_gt

                Z, Y, X = tuple(map(lambda k: k - self.size + 1, fix_pad_data.shape))
                for z in range(0, Z, step):
                    for y in range(0, Y, step):
                        for x in range(0, X, step):
                            gcube = fix_pad_gt[
                                    z:z + self.size,
                                    y:y + self.size,
                                    x:x + self.size]
                            if np.equal(gcube, np.full_like(gcube, self.config['labels']['BACKGROUND'])).all():
                                continue
                            cube = fix_pad_data[
                                   z:z + self.size,
                                   y:y + self.size,
                                   x:x + self.size]
                            self.patients['data'].append(cube)
                            self.patients['gt'].append(gcube)
                            test_ids.append(len(self.patients['data']) - 1)
            else:
                # random sampled sub-volumes (train)
                Z, Y, X = tuple(map(lambda k: k - self.size - 1, data.shape))
                for _ in range(self.num_samples):
                    while True:
                        z, y, x = tuple(map(lambda k: self.random_state.randint(0, k), (Z, Y, X)))
                        cube = data[
                               z:z + self.size,
                               y:y + self.size,
                               x:x + self.size]
                        gcube = gt[
                                z:z + self.size,
                                y:y + self.size,
                                x:x + self.size]
                        if np.sum(gcube != self.config['labels']['BACKGROUND']) != 0:
                            break

                    self.patients['data'].append(cube)
                    self.patients['gt'].append(gcube)
                    train_ids.append(len(self.patients['data']) - 1)

        self.indices = {
            'test': test_ids,
            'train': train_ids
        }
        self.weights = self.median_frequency_balancing()

    def _prepare_data(self, index):

        vol, gt = self.patients['data'][index], self.patients['gt'][index]

        vol, gt = ToTensor()([vol, gt])

        vol, gt = Normalize()([vol, gt])
        return vol, gt

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

    def __getitem__(self, index):
        vol, gt = self._prepare_data(index)
        return vol, gt, self.weights

    def __len__(self):
        return self.indices['train'].size + self.indices['test'].size

    def split_dataset(self):
        return self.indices['train'], self.indices['test']

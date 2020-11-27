from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from augmentations import RandomRotate, RandomContrast, ElasticDeformation, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, Resize, Rescale
import torch

GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class AlveolarDataloader(Dataset):

    def __init__(self, config):

        self.config = config
        self.patients = {
            'data': [],
            'gt': []
        }
        self.cut_idx = config.get('cut_index', 0.5)

        for folder in os.listdir(config['file_paths']):

            data = np.load(os.path.join(config['file_paths'], folder, 'data.npy'))
            gt = np.load(os.path.join(config['file_paths'], folder, 'gt.npy'))

            # pre-processing on the channels
            if data.ndim == 4:  # (B, H, W, 3) becomes  (B, H, W)
                data = np.take(data, 0, axis=-1)
            if gt.ndim == 4:  # (B, H, W, 3) becomes  (B, H, W,)
                gt = np.take(gt, 0, axis=-1)

            # from visualize_results import delaunay_attempt
            # delaunay_attempt(gt)

            right, left = np.split(data, [int(self.cut_idx * data.shape[2])], axis=2)
            gright, gleft = np.split(gt, [int(self.cut_idx * data.shape[2])], axis=2)
            if self.config.get('do_smart_cut', False):
                right, gright = Resize(self.config.get('labels', 1), self.config.get('smart_divisor', 8))(
                    [right, gright])
                left, gleft = Resize(self.config.get('labels', 1), self.config.get('smart_divisor', 8))([left, gleft])

            # debug
            def checkth(th, right):
                debug = np.zeros_like(right)
                val = 255
                axis = 0
                while right[axis, :, :].max() < th:
                    axis = axis + 1
                axis = 0
                while right[:, axis, :].max() < th:
                    debug[:, axis, :] = val
                    axis = axis + 1
                axis = 0
                while right[:, :, axis].max() < th:
                    debug[:, : axis] = val
                    axis = axis + 1
                axis = -1
                while right[axis, :, :].max() < th:
                    debug[axis, :, :] = val
                    axis = axis - 1
                axis = -1
                while right[:, axis, :].max() < th:
                    debug[:, axis, :] = val
                    axis = axis - 1
                axis = -1
                while right[:, :, axis].max() < th:
                    debug[:, :, axis] = val
                    axis = axis - 1
                return debug

            # from visualize_results import MultiView
            debug = checkth(0.78, right)
            # MultiView([(right, .5), gright], [(right, .5), gright, debug]).show()

            #end debug
            self.patients['data'] += [right, left]
            self.patients['gt'] += [gright, gleft]

        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)
        self.random_state = np.random.RandomState(self.seed)
        self.indices = {}

    def __len__(self):
        return self.indices['train'].size + self.indices['test'].size

    def augment_dataset(self):
        augmentation = transforms.Compose([
            RandomRotate(self.random_state, execution_probability=0.5),
            # RandomHorizontalFlip(self.random_state, execution_probability=0.5),
            RandomVerticalFlip(self.random_state, execution_probability=0.7),
            RandomContrast(self.random_state, execution_probability=0.90),
            ElasticDeformation(self.random_state, execution_probability=0.80),
        ])

        augment_rate = self.config.get('augment_rate', 1)
        for idx in self.indices['train']:
            vol, gt = self.patients['data'][idx], self.patients['gt'][idx]
            for i in range(augment_rate):
                aug_vol, aug_gt = augmentation([vol, gt])
                self.patients['data'].append(aug_vol)
                self.patients['gt'].append(aug_gt)

        return augment_rate * len(self.indices['train'])

    def _prepare_data(self, index):
        vol, gt = self.patients['data'][index], self.patients['gt'][index]

        vol, gt = ToTensor()([vol, gt])

        if self.config.get('do_rescale', False):
            vol, gt = Rescale(self.config.get('scale_factor', 0.5), self.config.get('labels'))([vol, gt])

        vol, gt = Normalize()([vol, gt])
        return vol, gt

    def __compute_class_weight(self, gt):
        weights = torch.zeros(len(self.config['labels']), dtype=torch.float32)  # init vector
        for label in self.config['labels']:
            weights[self.config['labels'][label]] = (gt.numel() / (gt == self.config['labels'][label]).sum())

        if 'UNLABELED' in self.config['labels']:  # set unlabelled weight to 0 if not binary task
            weights[self.config['labels']['UNLABELED']] = 0
        weights = weights / weights.sum()  # normalising
        return weights

    def __getitem__(self, index):
        vol, gt = self._prepare_data(index)
        weights = self.__compute_class_weight(gt)
        return vol, gt, weights

    def split_dataset(self, test_patient_idx=None, augmentation=True):

        num_patient = len(self.patients['data'])
        if num_patient < 2:
            raise Exception('less available patients than the ones required for training validate and test!')

        test_id = 0 if not test_patient_idx else test_patient_idx

        self.indices = {
            'test': np.asarray([test_id]),
            'train': np.asarray([i for i in range(num_patient) if i != test_id])
        }

        if augmentation:
            tot_new = self.augment_dataset()
            augmented_idx = np.arange(num_patient, num_patient + tot_new)
            self.indices['train'] = np.concatenate((self.indices['train'], augmented_idx))

        np.random.seed(40)
        np.random.shuffle(self.indices['train'])

        return self.indices['train'], self.indices['test']


class KwakDataloader(AlveolarDataloader):
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

        for i, folder in enumerate(os.listdir(config['file_paths'])):
            data = np.load(os.path.join(config['file_paths'], folder, 'data.npy'))
            gt = np.load(os.path.join(config['file_paths'], folder, 'gt_volume.npy'))

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

        if self.config.get('do_rescale', False):
            vol, gt = Rescale(self.config.get('scale_factor', 0.5), self.config.get('labels'))([vol, gt])

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


    def split_dataset(self):
        return self.indices['train'], self.indices['test']

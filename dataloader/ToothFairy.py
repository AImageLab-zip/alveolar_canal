import json
import logging
import logging.config
from pathlib import Path
import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader
class ToothFairy(tio.SubjectsDataset):
    """
    ToothFairy dataset
    TODO: Add more information about the dataset
    """
    def __init__(self, root, filename, splits, transform=None, dist_map=None, **kwargs):
        if type(dist_map) == str:
            dist_map = [dist_map]
        root = Path(root)
        if not isinstance(splits, list):
            splits = [splits]
        subjects_list = self._get_subjects_list(root, filename, splits, dist_map)
        super().__init__(subjects_list, transform, **kwargs)

    def _numpy_reader(self, path):
        # print(f'Path: {path}')
        data = torch.from_numpy(np.load(path)).float()
        affine = torch.eye(4, requires_grad=False)
        return data, affine

    def _get_subjects_list(self, root, filename, splits, dist_map=None):
        patients_path = root / 'Dataset'
        splits_path = root / filename
        with open(splits_path) as splits_file:
            json_splits = json.load(splits_file)
        if dist_map is None:
            dist_map = []
        subjects = []
        for split in splits:
            for patient in json_splits[split]:
                data_path = patients_path / patient / 'data.npy'
                sparse_path = patients_path / patient / 'gt_sparse.npy'
                dense_path = patients_path / patient / 'gt_alpha.npy'
                # TODO: add naive volume
                if split == 'synthetic':
                    dense_path = patients_path / patient / 'generated.npy'
                    if not dense_path.is_file():
                        print(f'Couldn\'t find generated.npy file for {patient}!')
                        dense_path = patients_path / patient / 'data.npy'
                if not data_path.is_file():
                    raise ValueError(f'Missing data file for patient {patient} ({data_path})')
                if not sparse_path.is_file():
                    raise ValueError(f'Missing sparse file for patient {patient} ({sparse_path})')
                if not dense_path.is_file():
                    raise ValueError(f'Missing dense file for patient {patient} ({dense_path})')
                subject_dict = {
                        'partition': split,
                        'patient': patient,
                        'data': tio.ScalarImage(data_path, reader=self._numpy_reader),
                        'sparse': tio.LabelMap(sparse_path, reader=self._numpy_reader),
                        'dense': tio.LabelMap(dense_path, reader=self._numpy_reader),
                        }
                if 'dense' in dist_map:
                    subject_dict['dense-dist'] = tio.LabelMap(dense_path, reader=self._numpy_reader)
                if 'sparse' in dist_map:
                    subject_dict['sparse-dist'] = tio.LabelMap(sparse_path, reader=self._numpy_reader)
                subjects.append(tio.Subject(**subject_dict))
            print(f"Loaded {len(subjects)} patients for split {split}")
        return subjects
    def get_loader(self, config, aggr=None):
        print(config)
        samples_per_volume = [np.round(i / (j-config.grid_overlap)) for i, j in zip(config.resize_shape,
config.patch_shape)]
        samples_per_volume = int(np.prod(samples_per_volume))
        #sampler = tio.GridSampler(patch_size=config.patch_shape, patch_overlap=config.grid_overlap)
        sampler = tio.UniformSampler(patch_size=config.patch_shape)
        queue = tio.Queue(
                subjects_dataset=self,
                max_length=100,
                samples_per_volume=10,
                sampler=sampler,
                num_workers=config.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,
                start_background=False,
        )
        loader = DataLoader(queue, batch_size=config.batch_size, num_workers=0, pin_memory=True)
        return loader
if __name__ == "__main__":
    print('Running')
    toothfairy = ToothFairy(
        root='/work/publicfiles/aimagelab-zip/ToothFairy_Dataset/',
        filename='splits.json',
        splits=['train', 'test', 'val'],
    )
    class Config(object):
        patch_shape =  (80, 80, 80)
        grid_overlap = 0
        resize_shape = (168, 280, 360)
        num_workers = 2
        batch_size = 2
    loader = toothfairy.get_loader(Config())
    print('loader')
    for data in loader:
        print(data)

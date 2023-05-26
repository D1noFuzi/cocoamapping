from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py


class H5Dataset(Dataset):

    def __init__(self, mode, config):
        super(H5Dataset, self).__init__()
        self.config = config
        self.data = h5py.File(join(self.config.data.data_dir, f'{mode}.hdf5'), 'r', swmr=True)
        stats = np.load(join(self.config.data.data_dir, 'statistics.npz'))
        # We assume that the mean and std include the height mean and std (concatenated along last axis)
        # Shape should be (1, #sentinel_channels + 1)
        self.mean, self.std = torch.from_numpy(stats['mean']).float(), torch.from_numpy(stats['std']).float()

    def __len__(self):
        return len(self.data['patches'])

    def __getitem__(self, idx):
        """
        This getitem method assumes a h5py file with an input dataset (key: patches) that is a big numpy array of size
        (#patches, #sentinel_channels + mean_height, patch_width, patch_height)
        a corresponding ground truth dataset of size (#patches, 1, patch_width, patch_height) where
        0 is background class, 1 is cocoa and 3 is cloud (i.e. you need to mask it when generating the h5py file)

        :param idx: row index of patch
        :return: input patch of shape (#sentinel_channels + mean_height, patch_width, patch_height)
                 ground truth of shape (1, patch_width, patch_height)
        """
        input_patch = torch.from_numpy(self.data['patches'][idx]).float()
        groundtruth_patch = torch.from_numpy(self.data['gt'][idx]).long()
        # Normalize input
        input_patch = (input_patch - self.mean[:, None, None]) / self.std[:, None, None]
        return input_patch, groundtruth_patch
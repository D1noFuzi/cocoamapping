import torch
from typing import Iterable

from dataset.dataset import H5Dataset


class DataLoader:

    def __init__(self, rank: int, config):
        self.rank = rank
        self.config = config
        self.train_dataset, self.val_dataset, self.sampler = self._init_dataset()

    def train_loader(self) -> Iterable:
        data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                  batch_size=self.config.training.batch_size,
                                                  num_workers=self.config.distributed.num_workers,
                                                  sampler=self.sampler)
        return data_loader

    def train_epoch_start(self, epoch: int):
        if self.config.distributed:
            self.sampler.set_epoch(epoch)
        else:
            pass

    def val_loader(self) -> Iterable:
        data_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                  batch_size=self.config.training.batch_size,
                                                  num_workers=self.config.distributed.num_workers,
                                                  shuffle=False)
        return data_loader

    def _init_dataset(self):
        train_dataset = H5Dataset(config=self.config, mode="train")
        val_dataset = H5Dataset(config=self.config, mode="val")

        if self.config.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.config.distributed.world_size,
                rank=self.rank,
                drop_last=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(train_dataset)
        return train_dataset, val_dataset, sampler

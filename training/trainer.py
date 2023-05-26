import sys
import os

from pathlib import Path
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from typing import Tuple, Iterable

from dataset.dataloader import DataLoader
from algorithm.algorithm import Algorithm


class Trainer:

    def __init__(self,
                 rank: int,
                 config) -> None:
        self.rank = rank
        self.config = config
        if self.config.distributed.distributed:
            self.setup_distributed(self.config.distributed.address, self.config.distributed.port, self.config.distributed.world_size)
        self.model = self.init_model()
        self.dataloader, self.train_loader, self.val_loader = self.setup_datasets()
        if self.rank == 0:
            self.save_path = Path(self.config.training.save_dir) / f'run-{wandb.run.id}'
            self.save_path.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        for epoch in tqdm(range(self.config.training.epochs),
                          disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                          position=0,
                          file=sys.stderr,
                          leave=True,
                          desc='Epochs'):
            self.train_loop(epoch)
            if epoch % self.config.validation.interval == 0:
                self.val_loop(epoch)
            sys.stderr.flush()
        self.model.save_model(epoch, str(self.save_path))

    def train_loop(self, epoch: int):
        self.model.train()
        self.dataloader.train_epoch_start(epoch)  # E.g. set_epoch for distributed training
        iterator = iter(self.train_loader)
        for idx, (x, y) in tqdm(enumerate(iterator),
                                disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                                total=len(self.train_loader),
                                file=sys.stderr,
                                position=1,
                                leave=False,
                                desc='Train loop'):
            self.model.update(x.to(self.rank), y.to(self.rank))
            sys.stderr.flush()
        self.model.train_epoch_end()
        self.model.log_train(step=epoch)

    def val_loop(self, epoch: int):
        self.model.eval()
        iterator = iter(self.val_loader)
        for idx, (x, y) in tqdm(enumerate(iterator),
                                disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                                total=len(self.val_loader),
                                file=sys.stderr,
                                position=1,
                                leave=False,
                                desc='Val loop'):
            self.model.val(x.to(self.rank), y.to(self.rank))
            sys.stderr.flush()
        self.model.log_val(step=epoch)

    def setup_datasets(self) -> Tuple[Iterable, Iterable, Iterable]:
        dataloader = DataLoader(rank=self.rank, config=self.config)
        train_loader = dataloader.train_loader()
        val_loader = dataloader.val_loader()

        return dataloader, train_loader, val_loader

    def init_model(self) -> Algorithm:
        model = Algorithm(rank=self.rank, config=self.config)
        model = model.to(self.rank)
        if self.config.distributed.distributed:
            model.model = DDP(model.model,
                              device_ids=[self.rank])
        return model

    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.rank, world_size=world_size)
        torch.cuda.set_device(self.rank)

    @staticmethod
    def cleanup_distributed():
        dist.destroy_process_group()

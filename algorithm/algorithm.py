import os

import wandb

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from torchmetrics import Accuracy, MeanMetric, Precision, Recall

from algorithm.network import Network

from algorithm.loss import DiceLoss


class Algorithm(torch.nn.Module):

    def __init__(self, rank: int, config):
        super(Algorithm, self).__init__()
        self.rank = rank
        self.config = config
        self.model = self._init_model()
        self.scaler = GradScaler(enabled=self.config.distributed.distributed)
        self.optimizer, self.scheduler = self._init_optimizer()
        self.train_metrics, self.val_metrics = self._init_metrics()
        self.loss = DiceLoss(ignore_index=3)

    def update(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.config.training.mixed_prec):
            logits = self.model(x)
            loss = self.loss(logits, y)

        # Backward loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.train_metrics['acc'](preds.flatten(), y.squeeze(1).flatten())
        self.train_metrics['prec'](preds.flatten(), y.squeeze(1).flatten())
        self.train_metrics['rec'](preds.flatten(), y.squeeze(1).flatten())
        self.train_metrics['loss'](loss)

    def train_epoch_end(self):
        self.scheduler.step()

    def val(self, x, y):
        with torch.no_grad():
            with autocast(enabled=self.config.training.mixed_prec):
                logits = self.model(x)
                loss = self.loss(logits, y)
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics['acc'](preds.flatten(), y.squeeze(1).flatten())
        self.val_metrics['prec'](preds.flatten(), y.squeeze(1).flatten())
        self.val_metrics['rec'](preds.flatten(), y.squeeze(1).flatten())
        self.val_metrics['loss'](loss)

    def log_train(self, step: int):
        acc = self.train_metrics['acc'].compute()
        prec = self.train_metrics['prec'].compute()
        rec = self.train_metrics['rec'].compute()
        loss = self.train_metrics['loss'].compute()
        if self.rank == 0:
            wandb.log({'train/acc': acc, 'train/loss': loss, 'train/precision': prec, 'train/recall': rec}, step=step)
        self.train_metrics['acc'].reset()
        self.train_metrics['prec'].reset()
        self.train_metrics['rec'].reset()
        self.train_metrics['loss'].reset()

    def log_val(self, step: int):
        acc = self.val_metrics['acc'].compute()
        prec = self.val_metrics['prec'].compute()
        rec = self.val_metrics['rec'].compute()
        loss = self.val_metrics['loss'].compute()
        if self.rank == 0:
            wandb.log({'val/acc': acc, 'val/loss': loss, 'val/precision': prec, 'val/recall': rec}, step=step)
        self.val_metrics['acc'].reset()
        self.val_metrics['prec'].reset()
        self.val_metrics['rec'].reset()
        self.val_metrics['loss'].reset()

    def save_model(self, step: int, path: str):
        torch.save({
            'epoch': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(path, f'epoch_{step}.pth'))

    def _init_optimizer(self):
        optimizer = Adam(params=self.model.parameters(),
                         lr=self.config.training.learning_rate,
                         weight_decay=self.config.training.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=self.config.training.milestones, gamma=self.config.training.gamma)
        return optimizer, scheduler

    def _init_model(self):
        net = Network()
        return net

    def _init_metrics(self):
        train_metrics = {
            'acc': Accuracy(ignore_index=3).to(self.rank),
            'prec': Precision(ignore_index=3).to(self.rank),
            'rec': Recall(ignore_index=3).to(self.rank),
            'loss': MeanMetric().to(self.rank)
        }
        if self.rank == 0:
            wandb.define_metric("train/acc", summary="max")
        val_metrics = {
            'acc': Accuracy(ignore_index=3).to(self.rank),
            'prec': Precision(ignore_index=3).to(self.rank),
            'rec': Recall(ignore_index=3).to(self.rank),
            'loss': MeanMetric().to(self.rank)
        }
        if self.rank == 0:
            wandb.define_metric("val/acc", summary="max")
        return train_metrics, val_metrics
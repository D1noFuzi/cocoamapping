import torch
import wandb

from pathlib import Path

from omegaconf import OmegaConf

from training.trainer import Trainer


class Training:

    def __init__(self, config):
        self.config = config

    def launch_training(self):
        if self.config.distributed.distributed:
            torch.multiprocessing.spawn(self.run, nprocs=self.config.distributed.world_size, join=True)
        else:
            self.run(gpu=0)

    def run(self, gpu):
        if gpu == 0:
            wcfg = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
            log_path = Path(self.config.wandb.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            wandb.init(project=self.config.wandb.project,
                       entity=self.config.wandb.entity,
                       dir=str(log_path),
                       config=wcfg,
                       resume='allow',
                       mode=self.config.wandb.mode,
                       settings=wandb.Settings(start_method="fork"))
        trainer = Trainer(rank=gpu,
                          config=self.config)
        trainer.train()

        if self.config.distributed.distributed:
            trainer.cleanup_distributed()
        if gpu == 0:
            wandb.finish()

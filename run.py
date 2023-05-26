import os
import argparse
import wandb

from omegaconf import OmegaConf
from training.train import Training
from training.launcher import basic_launcher, slurm_launcher

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['train', 'launch'], help='Train X models on a cluster or locally.')
parser.add_argument('--data_dir', type=str, help='Root directory of datasets.')
args = parser.parse_args()

# Load config file
cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'config.yaml'))
if args.data_dir is not None:
    OmegaConf.update(cfg, 'data.data_dir', args.data_dir)


def launch_training():
    print(f'Start training.')
    train = Training(config=cfg)
    train.launch_training()


def main():
    if args.command == 'train':
        print(f'Launching {cfg.model.count} training runs.')
        for i in range(cfg.model.count):
            # Pass sweep id to agents running on cluster node
            # slurm_launcher(cmd='python run.py launch --data_dir {cfg.data.data_dir}')
            launch_training()
    elif args.command == 'launch':
        launch_training()


if __name__ == '__main__':
    main()

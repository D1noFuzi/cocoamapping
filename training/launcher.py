import subprocess


def slurm_launcher(cmd):
    """
    Launch commands on slurm
    """
    print("Now launching...")
    bsub = f"""sbatch <<-EOF
#!/bin/bash
#SBATCH -n 8
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=20000
#SBATCH --gpus=8
#SBATCH --job-name=cocoamapping


{cmd}
EOF"""
    subprocess.run(bsub, shell=True)


def basic_launcher(cmd):
    subprocess.run(cmd, shell=True)

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --job-name=val3
#SBATCH --output=slurm/val3.out
#SBATCH --error=slurm/val3.out



cd ..

python attacks.py dicl --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py dip --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py fastflownet --pretrained_ckpt kitti --val_dataset kitti-2015
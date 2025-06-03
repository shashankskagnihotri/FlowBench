#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --job-name=val7
#SBATCH --output=slurm/val7.out
#SBATCH --error=slurm/val7.out



cd ..

python attacks.py scopeflow --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py skflow --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py starflow --pretrained_ckpt kitti --val_dataset kitti-2015
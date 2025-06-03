#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --job-name=val2
#SBATCH --output=slurm/val2.out
#SBATCH --error=slurm/val2.out



cd ..

python attacks.py ccmr --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py craft --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py csflow --pretrained_ckpt kitti --val_dataset kitti-2015
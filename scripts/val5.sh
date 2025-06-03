#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --job-name=val5
#SBATCH --output=slurm/val5.out
#SBATCH --error=slurm/val5.out



cd ..

python attacks.py flowformer++ --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py gmflow --pretrained_ckpt kitti --val_dataset kitti-2015
python attacks.py gmflownet --pretrained_ckpt kitti --val_dataset kitti-2015
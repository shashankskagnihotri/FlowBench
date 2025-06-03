#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --job-name=TestAttacks
#SBATCH --output=slurm/TestAttacks.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/TestAttacks.out

echo TestAttacks

python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015+sintel-clean+sintel-final --attack bim fgsm cospgd --attack_iterations 3 5 10 --attack_targeted False True





#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=47:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --output=slurm/pwcnet_training_kitti15.out
#SBATCH --error=slurm/pwcnet_training_kitti15.out.err

cd ../../../ptlflow_attacked

python train.py pwcnet --train_dataset kitti-15-train --val_dataset kitti-15-val --pretrained_ckpt things --train_crop_size 896 320 --lr 0.00000001 --train_batch_size 4 --max_epochs 10 --clear_train_state
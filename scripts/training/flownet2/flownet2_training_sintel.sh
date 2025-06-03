#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=47:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --output=slurm/flownet2_training_sintel.out
#SBATCH --error=slurm/flownet2_training_sintel.out.err

cd ../../../ptlflow_attacked

python train.py flownet2 --train_dataset sintel-train --pretrained_ckpt things --train_crop_size 768 384 --lr 0.0000001 --train_batch_size 8 --max_epochs 50 --gpus 1 --clear_train_state
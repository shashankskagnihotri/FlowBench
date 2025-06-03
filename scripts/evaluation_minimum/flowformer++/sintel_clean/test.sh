#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --array=0-14%4
#SBATCH --job-name=flowformer_kitti-2015_test
#SBATCH --output=slurm/flowformer_kitti-2015_test_%A_%a.out
#SBATCH --error=slurm/flowformer_kitti-2015_test_err_%A_%a.out

model="flowformer"
dataset="kitti-2015"
checkpoint="kitti"
targeteds="True False"
targets="negative zero"
norms="inf two"
attacks="test"
jobnum=0
#SLURM_ARRAY_TASK_ID=0

cd ../../../../

python attacks_flowformer.py flowformer --val_dataset kitti-2015 --pretrained_ckpt kitti
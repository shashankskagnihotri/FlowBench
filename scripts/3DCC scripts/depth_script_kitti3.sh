#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:00
#SBATCH --job-name=TestAttacks
#SBATCH --output=slurm/TestAttacks.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/TestAttacks.out

# The parent directory you want to scan
cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/MiDaS
python run.py --model_type dpt_beit_large_512 --input_path /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets/kitti2012/training/colored_0 --output_path /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_depth/kitti2012/training/colored_0 --grayscale
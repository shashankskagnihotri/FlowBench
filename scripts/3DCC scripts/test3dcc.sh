#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --job-name=TestAttacks
#SBATCH --output=slurm/TestAttacks.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/TestAttacks.out

cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/3DCommonCorruptions/create_3dcc
# python create_3dcc.py --path_rgb /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets/kitti2015/training/image_2 --path_depth /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_depth/kitti2015/training/image_2 --path_target /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_corrupted/kitti2015/training/image_2 --batch_size 1
python create_3dcc.py --path_rgb /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/MiDaS/input --path_depth /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/MiDaS/output --path_target /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test --batch_size 1

#python create_3dcc.py --path_rgb /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test/rgb --path_depth /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test/depth --path_target /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test --batch_size 1

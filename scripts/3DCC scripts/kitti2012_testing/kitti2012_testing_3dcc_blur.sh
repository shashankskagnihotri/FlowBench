#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=02:59:00
#SBATCH --job-name=kitti2012_testing_3DCCblur
#SBATCH --output=slurm/kitti2012_testing_3DCCblur.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/kitti2012_testing3DCCblur.out

rgb_path=/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets/kitti2012/testing/colored_0
depth_path=/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_depth/kitti2012/testing/colored_0
output_path=/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_corrupted/kitti2012/testing/colored_0
cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/3DCommonCorruptions/create_3dcc
# python create_3dcc.py --path_rgb /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets/kitti2015/training/image_2 --path_depth /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_depth/kitti2015/training/image_2 --path_target /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_corrupted/kitti2015/training/image_2 --batch_size 1
python create_3dcc_blur.py --path_rgb $rgb_path --path_depth $depth_path --path_target $output_path --batch_size 1

#python create_3dcc.py --path_rgb /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test/rgb --path_depth /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test/depth --path_target /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/test --batch_size 1

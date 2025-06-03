#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=05:59:00
#SBATCH --job-name=sintel_training_clean_3DCCblur
#SBATCH --output=slurm/sintel_training_clean_3DCCblur.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/sintel_training_clean_3DCCblur.out

# The parent directory you want to scan
RGB_DIR="/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets/Sintel/training/clean/"
DEPTH_DIR="/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_depth/Sintel/training/clean/"
OUTPUT_DIR="/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_corrupted/Sintel/training/clean/"
# The Python script you want to execute with each directory as an argument
# PYTHON_SCRIPT="/path/to/your/script.py"
cd "$RGB_DIR"
# Find directories in the parent directory and iterate over them
find . -mindepth 1 -maxdepth 1 -type d | while read dir; do
    # Remove the leading './' from the directory path
    relative_dir=${dir#./}

    echo "Executing Python script on directory: $relative_dir"
    rgb_path=$RGB_DIR$relative_dir
    depth_path=$DEPTH_DIR$relative_dir
    output_path=$OUTPUT_DIR$relative_dir
    echo $rgb_path
    echo $output_path
    echo $depth_path
    cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/3DCommonCorruptions/create_3dcc
    python create_3dcc_blur.py --path_rgb $rgb_path --path_depth $depth_path --path_target $output_path --batch_size 1

done
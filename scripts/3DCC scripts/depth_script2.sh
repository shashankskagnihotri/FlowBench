#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:00
#SBATCH --job-name=Sintel2
#SBATCH --output=slurm/Sintel2.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/Sintel2.out

# The parent directory you want to scan
PARENT_DIR="/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets/Sintel/test/final/"
OUTPUT_DIR="/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/datasets_depth/Sintel/test/final/"
# The Python script you want to execute with each directory as an argument
# PYTHON_SCRIPT="/path/to/your/script.py"
cd "$PARENT_DIR"
# Find directories in the parent directory and iterate over them
find . -mindepth 1 -maxdepth 1 -type d | while read dir; do
    # Remove the leading './' from the directory path
    relative_dir=${dir#./}

    echo "Executing Python script on directory: $relative_dir"
    source_path=$PARENT_DIR$relative_dir
    output_path=$OUTPUT_DIR$relative_dir
    echo $source_path
    echo $output_path
    #cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/Depth-Anything
    #python run.py --encoder vitl --img-path $source_path --outdir $output_path --pred-only
    cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/MiDaS
    python run.py --model_type dpt_beit_large_512 --input_path $source_path --output_path $output_path --grayscale
done
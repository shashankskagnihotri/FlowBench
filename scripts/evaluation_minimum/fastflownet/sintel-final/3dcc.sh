#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_8
#SBATCH --array=0-7%4
#SBATCH --job-name=fastflownet_sintel-final_3dcc
#SBATCH --output=slurm/fastflownet_sintel-final_3dcc_%A_%a.out
#SBATCH --error=slurm/fastflownet_sintel-final_3dcc_err_%A_%a.out

model="fastflownet"
dataset="sintel-final"
checkpoint="sintel"
attack="3dcc"
tdcc_corruptions="far_focus near_focus fog_3d color_quant iso_noise low_light xy_motion_blur z_motion_blur"
tdcc_intensities="3"
jobnum=0

#SLURM_ARRAY_TASK_ID=0

cd ../../../../
for tdcc_corruption in $tdcc_corruptions
do
    for tdcc_intensity in $tdcc_intensities
    do
        if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
        then
            echo "Running job $model $checkpoint $dataset $attack $tdcc_corruption $tdcc_intensity $jobnum"
            python attacks.py \
                $model \
                --pretrained_ckpt $checkpoint \
                --val_dataset $dataset \
                --attack $attack \
                --3dcc_corruption $tdcc_corruption \
                --3dcc_intensity $tdcc_intensity \
                --write_outputs                                       
            #SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
        fi
        jobnum=$((jobnum + 1))
    done
done
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --array=0-1%2
#SBATCH --job-name=dicl_kitti-2015_pgd_cospgd_i20_inf_init
#SBATCH --output=slurm/dicl_kitti-2015_pgd_cospgd_i20_inf_init_%A_%a.out
#SBATCH --error=slurm/dicl_kitti-2015_pgd_cospgd_i20_inf_init_err_%A_%a.out

model="dicl"
dataset="kitti-2015"
checkpoint="kitti"
targeted="False"
norm="inf"
attacks="pgd cospgd"
iterations="20"
jobnum=0
#SLURM_ARRAY_TASK_ID=0

cd ../../../../

epsilons="8"
alphas="0.01"

for epsilon in $epsilons
do
    epsilon=$(echo "scale=10; $epsilon/255" | bc)
    for alpha in $alphas
    do
        for attack in $attacks
        do
            for iteration in $iterations
            do                         
                if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
                then
                    echo "Running job $model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $jobnum"
                    python attacks.py \
                        $model \
                        --pretrained_ckpt $checkpoint \
                        --val_dataset $dataset \
                        --attack $attack \
                        --attack_iterations $iteration \
                        --attack_norm $norm \
                        --attack_alpha $alpha \
                        --attack_epsilon $epsilon \
                        --attack_targeted "$targeted" \
                        --attack_optim_target "initial_flow" \
                        --write_outputs 
                    #SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                fi
                jobnum=$((jobnum + 1))
            done
        done
    done
done
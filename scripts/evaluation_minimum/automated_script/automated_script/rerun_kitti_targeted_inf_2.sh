#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --array=0-39%4
#SBATCH --job-name=rerun_kitti_targeted_inf_2
#SBATCH --output=slurm/rerun_kitti_targeted_inf_2_%A_%a.out
#SBATCH --error=slurm/rerun_kitti_targeted_inf_2_err_%A_%a.out

models="liteflownet2 liteflownet3_psudored llaflow matchflow rapidflow"
dataset="kitti-2015"
checkpoint="kitti"
targeted="True"
targets="negative zero"
norm="inf"
attacks="bim pgd cospgd fgsm"
iterations="20"
epsilon="8"
alpha="0.01"
jobnum=0
#SLURM_ARRAY_TASK_ID=0

cd ../../../../


for model in $models
do
    epsilon="8"
    epsilon=$(echo "scale=10; $epsilon/255" | bc)
    for attack in $attacks
    do
        for iteration in $iterations
        do
            if [[ $targeted = "True" ]]
            then
                for target in $targets
                do      
                    if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
                    then
                        echo "Running job $model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $target $jobnum"
                        python attacks.py \
                            $model \
                            --pretrained_ckpt $checkpoint \
                            --val_dataset $dataset \
                            --attack $attack \
                            --attack_iterations $iteration \
                            --attack_norm $norm \
                            --attack_alpha $alpha \
                            --attack_epsilon $epsilon \
                            --attack_targeted $targeted \
                            --attack_target $target \
                            --write_outputs                                       
                        #SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                    fi
                    jobnum=$((jobnum + 1))
                done
            else                           
                if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
                then
                    echo "Running job $model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $target $jobnum"
                    python attacks.py \
                        $model \
                        --pretrained_ckpt $checkpoint \
                        --val_dataset $dataset \
                        --attack $attack \
                        --attack_iterations $iteration \
                        --attack_norm $norm \
                        --attack_alpha $alpha \
                        --attack_epsilon $epsilon \
                        --attack_targeted $targeted \
                        --attack_target "zero" \
                        --write_outputs 
                    #SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                fi
                jobnum=$((jobnum + 1))
            fi
        done
    done
done
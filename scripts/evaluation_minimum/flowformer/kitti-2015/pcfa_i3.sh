#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=04:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --array=0-1%2
#SBATCH --job-name=flowformer_kitti-2015_pcfa_i3
#SBATCH --output=slurm/flowformer_kitti-2015_pcfa_i3_%A_%a.out
#SBATCH --error=slurm/flowformer_kitti-2015_pcfa_i3_err_%A_%a.out

model="flowformer"
dataset="kitti-2015"
checkpoint="kitti"
targeteds="True"
targets="negative zero"
norms="two"
attacks="pcfa"
iterations="3"
jobnum=0
#SLURM_ARRAY_TASK_ID=0

cd ../../../../

for norm in $norms
do
    epsilons="0.005"
    alphas="0.0000001"
    for epsilon in $epsilons
    do
        for alpha in $alphas
        do
            for attack in $attacks
            do
                for iteration in $iterations
                do
                    for targeted in $targeteds
                    do
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
                    done
                done
            done
        done
    done
done
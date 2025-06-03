#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --array=0-99%4
#SBATCH --output=slurm/raft_pgd_i3_kitti15.out

models="raft"
datasets="kitti-2015"
checkpoints="kitti"
targeteds="False"
targets="zero"
norms="two"
attacks="bim pgd cospgd"
iterations="3"
# jobnum=0
# SLURM_ARRAY_TASK_ID=0

cd ../../../../../../

for model in $models
do
    for dataset in $datasets
    do
        for norm in $norms
        do
            if [[ $norm = "inf" ]]
            then
                epsilons="0.1 0.5 1 2 3 4 5 6 7 8 10 12 25"
                alphas="0.01"
                for epsilon in $epsilons; do
                    epsilon=$(echo "scale=10; $epsilon/255" | bc)
                done
            else
                epsilons="0.251 0.502 0.0005 0.005 0.001 0.05 0.01"
                alphas="0.1 0.2 0.0000001"
            fi
            for attack in $attacks
            do
                for iteration in $iterations
                do
                    for targeted in $targeteds
                    do
                        if [[ targeted = "True" ]]
                        then
                            for target in $targets
                            do
                                for alpha in $alphas
                                do
                                    for epsilon in $epsilons
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
                                                --attack_target $target                                       
                                            jobnum=$((jobnum + 1))
                                            export jobnum
                                            # SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                                        fi
                                    done
                                done
                            done
                        else
                            for alpha in $alphas
                            do
                                for epsilon in $epsilons
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
                                           --attack_target "zero"
                                        jobnum=$((jobnum + 1))
                                        export jobnum
                                        # SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                                    fi
                                done
                            done
                        fi
                    done
                done
            done
        done
    done
done
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=04:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --array=0-20%4
#SBATCH --job-name=scopeflow_sintel-clean_bim_pgd_cospgd_i20
#SBATCH --output=slurm/scopeflow_sintel-clean_bim_pgd_cospgd_i20_%A_%a.out
#SBATCH --error=slurm/scopeflow_sintel-clean_bim_pgd_cospgd_i20_err_%A_%a.out

model="scopeflow"
dataset="sintel-clean"
checkpoint="sintel"
targeteds="True False"
targets="negative zero"
norms="inf two"
attacks="bim pgd cospgd"
iterations="20"
jobnum=0
#SLURM_ARRAY_TASK_ID=0

cd ../../../../

for norm in $norms
do
    for targeted in $targeteds
    do
        if [[ $norm = "inf" && $targeted = "False" ]]
        then
            epsilons="4 8"
            alphas="0.01"          
        elif [[ $norm = "inf" && $targeted = "True" ]]
        then
            epsilons="12.75"
            alphas="0.01"
        elif [[ $norm = "two" ]]
        then
            epsilons="12.75"
            alphas="0.0001"
        fi
        for epsilon in $epsilons
        do
            epsilon=$(echo "scale=10; $epsilon/255" | bc)
            for alpha in $alphas
            do
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
        done
    done
done
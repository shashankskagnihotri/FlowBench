#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

cd ..
python attacks.py \
    $1 \
    --pretrained_ckpt $2 \
    --val_dataset $3 \
    --attack $4 \
    --attack_iterations $5 \
    --attack_norm $6 \
    --attack_alpha $7 \
    --attack_epsilon $8 \
    --attack_targeted $9 \
    --attack_target $10

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
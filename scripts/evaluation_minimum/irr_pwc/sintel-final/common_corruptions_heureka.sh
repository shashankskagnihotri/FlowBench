#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=09:29:59
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --job-name=irr_pwc_sintel-final_cc
#SBATCH --output=slurm/irr_pwc_sintel-final_cc_%A_%a.out
#SBATCH --error=slurm/irr_pwc_sintel-final_cc_%A_%a.out
#SBATCH --array=0-3%4

model="irr_pwc"
dataset="sintel-final"
checkpoint="sintel"
attack="common_corruptions"
cc_names="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
cc_severitys="3"
combinations=()

# Change to the appropriate directory
cd ../../../../

# Step 1: Create a list of all variable combinations
for cc_name in $cc_names
do
    for cc_severity in $cc_severitys
    do
        # Append each combination as a string to the combinations list
        combinations+=("$model $checkpoint $dataset $attack $cc_name $cc_severity")
    done
done

# Step 2: Echo the content of the combinations array
echo "Generated combinations:"
for comb in "${combinations[@]}"; do
    echo "$comb"
done

# Step 3: Calculate the number of SLURM array tasks
total_jobs=${#combinations[@]}
echo "Total number of jobs: $total_jobs"
jobs_per_task=4  # Number of jobs to run per SLURM task

# Compute the total number of array tasks (rounding up)
array_length=$(( (total_jobs + jobs_per_task - 1) / jobs_per_task ))

# If running as part of a SLURM array job, determine start job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

start_job=$(( SLURM_ARRAY_TASK_ID * jobs_per_task ))

# Run up to 4 combinations on separate GPUs
for i in $(seq 0 3)  # Loop over 4 GPUs (0 to 3)
do
    jobnum=$(( start_job + i ))
    
    if [[ $jobnum -lt $total_jobs ]]
    then
        # Get the combination for the current task
        combination="${combinations[$jobnum]}"
        
        # Extract the individual variables from the combination string
        IFS=' ' read -r model checkpoint dataset attack cc_name cc_severity <<< "$combination"
        
        # Set the GPU for this job
        export CUDA_VISIBLE_DEVICES=$i
        
        echo "Running job $model $checkpoint $dataset $attack $cc_name $cc_severity on GPU $i (job $jobnum)"
        
        # Run the Python script with the corresponding parameters in the background (&)
        python attacks.py \
            $model \
            --pretrained_ckpt $checkpoint \
            --val_dataset $dataset \
            --attack $attack \
            --cc_name $cc_name \
            --cc_severity $cc_severity \
            --write_outputs &
    fi
done

# Wait for all background jobs to complete before exiting
wait

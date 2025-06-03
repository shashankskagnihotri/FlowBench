#!/bin/bash



# User whose jobs you want to monitor
USER="ma_jcaspary"
LOGFILE="/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/memflow_splatflow_neuflow/kitti_heureka_fgsm_bim.log"
cd /hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/memflow_splatflow_neuflow
# List of shell script names and their corresponding job amounts
declare -A scripts_and_amounts=(
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/memflow/kitti-2015/fgsm_inf_heureka.sh"]=1
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/splatflow/kitti-2015/fgsm_inf_heureka.sh"]=1
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/memflow/kitti-2015/fgsm_two_heureka.sh"]=1
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/splatflow/kitti-2015/fgsm_two_heureka.sh"]=1
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/memflow/kitti-2015/bim_pgd_cospgd_i20_inf_heureka.sh"]=3
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/splatflow/kitti-2015/bim_pgd_cospgd_i20_inf_heureka.sh"]=3
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/memflow/kitti-2015/bim_pgd_cospgd_i20_two_heureka.sh"]=3
  ["/hkfs/work/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/splatflow/kitti-2015/bim_pgd_cospgd_i20_two_heureka.sh"]=3
)

# Function to check the number of running and pending jobs
get_running_jobs_count() {
  squeue -u "$USER" -h -t pending,running -r | wc -l
}

# Function to submit a job
submit_job() {
  local script_name=$1
  log "Submitting job: $script_name"
  DIR=$(dirname "$script_name")
  cd $DIR
  sbatch $script_name
}

# Function to log messages with timestamp
log() {
  local message=$1
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" >> "$LOGFILE"
}

# Main script logic
while (( ${#scripts_and_amounts[@]} > 0 )); do
  running_jobs_count=$(get_running_jobs_count)
  available_slots=$((95 - running_jobs_count))
  log "Available slots: $available_slots"

  if (( available_slots > 0 )); then
    for script_name in "${!scripts_and_amounts[@]}"; do
      job_amount=${scripts_and_amounts[$script_name]}
      
      if (( available_slots >= job_amount )); then
        submit_job "$script_name"
        unset scripts_and_amounts["$script_name"]
        break
      fi
    done
  else
    log "No available slots to submit new jobs."
  fi

  # Wait for 2 seconds before the next iteration
  sleep 120
done

log "All jobs have been submitted."

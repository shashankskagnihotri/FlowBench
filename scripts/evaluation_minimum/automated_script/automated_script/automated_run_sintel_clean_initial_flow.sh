#!/bin/bash

# User whose jobs you want to monitor
USER="ma_lucschwa"
LOGFILE="/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/automated_script/automated_run_sintel_clean_initial_flow.log"
cd /pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/automated_script
# List of shell script names and their corresponding job amounts
declare -A scripts_and_amounts=(
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet3/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/raft/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/ms_raft+/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/maskflownet/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/flowformer++/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/flow1d/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/craft/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/rpknet/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/scopeflow/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/gma/sintel-clean/pgd_cospgd_i20_inf_initial_flow.sh"]=2
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

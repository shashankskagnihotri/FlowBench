#!/bin/bash



#gma sintel final
#rkp clean+final
#flowformer clean+final
#flowformer++ clean+final


# User whose jobs you want to monitor
USER="ma_lucschwa"
LOGFILE="/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/automated_script/automated_run_kitti_1.log"
cd /pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/automated_script
# List of shell script names and their corresponding job amounts
declare -A scripts_and_amounts=(
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/raft/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/gma/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/rpknet/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/craft/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/csflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/dicl/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/fastflownet/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/maskflownet/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/flow1d/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/gmflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/gmflownet/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/hd3/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/irr_pwc/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet2/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet3_pseudoreg/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/llaflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/matchflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/rapidflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/scopeflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/scv4/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/separableflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/skflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/starflow/kitti-2015/pcfa_i20.sh"]=2
  ["/pfs/work7/workspace/scratch/ma_lucschwa-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/videoflow_bof/kitti-2015/pcfa_i20.sh"]=2
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
  #echo $DIR
  cd $DIR
  sbatch $script_name
  #echo $script_name
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

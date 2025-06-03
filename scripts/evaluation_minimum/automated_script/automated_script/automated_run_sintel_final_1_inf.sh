#!/bin/bash



#gma sintel final
#rkp clean+final
#flowformer clean+final
#flowformer++ clean+final


# User whose jobs you want to monitor
USER="ma_jcaspary"
LOGFILE="/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/automated_script/automated_run_sintel_final_1.log"
cd /pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/automated_script/automated_script
# List of shell script names and their corresponding job amounts
declare -A scripts_and_amounts=(
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/raft/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/raft/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/pwcnet/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/pwcnet/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/gma/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/gma/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/rpknet/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/rpknet/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/ccmr/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/ccmr/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/craft/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/craft/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/dicl/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/dicl/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/dip/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/dip/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/fastflownet/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/fastflownet/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/maskflownet/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/maskflownet/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/flow1d/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/flow1d/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet2/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet2/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet3_pseudoreg/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/liteflownet3_pseudoreg/sintel-final/fgsm_inf.sh"]=4
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/llaflow/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/llaflow/sintel-final/fgsm_inf.sh"]=4  
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/matchflow/sintel-final/bim_pgd_cospgd_i20_inf.sh"]=12
  ["/pfs/work7/workspace/scratch/ma_jcaspary-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/scripts/evaluation_minimum/matchflow/sintel-final/fgsm_inf.sh"]=4  
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

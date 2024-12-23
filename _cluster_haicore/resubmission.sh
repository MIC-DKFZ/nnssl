#!/bin/bash -x
#SBATCH --account=cthrp
#SBATCH --job-name=%J
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/out-%x.%j
#SBATCH --error=logs/err-%x.%j
#SBATCH --time=23:30:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --signal=B:USR1@6000

# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

# Without this, srun does not inherit cpus-per-task from sbatch.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010
export GPUS_PER_NODE=4

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# load conda env
source /p/home/jusers/stock2/juwels/env/env_nnssl/activate.sh

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# nnssl env vars
export nnssl_preprocessed="/p/data1/thrp/datasets/development_data"
export nnssl_raw=""
export nnssl_results="/p/data1/thrp/experiments/MAE"
export rocket_preprocessed="/p/data1/thrp/datasets/development_data"

# Define JOB_ID based on first argument
export WANDB_RUN_ID="attempt_1"

# resubmission logic
resubmit_job() {
    echo "USR1 signal received. Saving state and resubmitting job..."
    # Save the current state (modify this as needed for your application)
    echo "Sending stop signal to running job"
    kill -USR1 ${PID}
    wait ${PID}
    echo "Job successfully terminated"
    # Check if '--c' is in the arguments
    args=("$@")
    if [[ ! " ${args[@]} " =~ " --c " ]]; then
        args+=("--c")
        echo "'--c' was not in the arguments. Adding it."
    else
        echo "'--c' is already in the arguments."
    fi
    # Resubmit the job with the modified arguments
    echo sbatch "$0" "${args[@]}"
    sbatch "$0" "${args[@]}"
    echo "Job resubmitted. Exiting current job."
    exit 0
}
trap 'resubmit_job "$@"' USR1
# Start script as background job
nnssl_train_wandb "$@" &
PID=$!
echo "PID of nnssl_train: ${PID}"
PGID=$(ps -o pgid= -p ${PID} | grep -o '[0-9]*') # get the process group ID
echo "PGID of nnssl_train: ${PGID}"
# Wait for the background job to finish
wait ${PID}

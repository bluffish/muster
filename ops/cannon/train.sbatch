#!/usr/bin/env bash
#SBATCH --job-name=muster-v21
#SBATCH --account=avanderburg_lab
#SBATCH --partition=gpu_requeue
#SBATCH --gpus=nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --output=/n/holylabs/avanderburg_lab/Lab/bowen/muster-runs/slurm-v21.log

# Preemptible H100 training for muster. Preemption requeues automatically;
# ten minutes before walltime Slurm sends USR1 and the script requeues
# itself, so the run survives both. Training resumes from latest.pt.

set -uo pipefail

lab=/n/holylabs/avanderburg_lab/Lab/bowen
project="$lab/muster"
venv="$lab/muster-venv"
config_name=${MUSTER_CONFIG:-v21-assault}

source /etc/profile.d/lmod.sh
module load python/3.12.11-fasrc02
source "$venv/bin/activate"

cd "$project"
source "configs/$config_name.env"
run_dir="$lab/muster-runs/$RUN_NAME"
mkdir -p "$run_dir"
ln -sfn "$lab/muster-runs" runs

if [[ ! -f runs/distilled_charger.pt ]]; then
    echo "== distilling the assault charger =="
    python ops/cannon/distill_assault.py || exit 1
fi

requeue_self() {
    echo "== walltime approaching: requeueing job $SLURM_JOB_ID =="
    kill "$train_pid" 2>/dev/null
    wait "$train_pid" 2>/dev/null
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_self USR1

args=(
    --run "$run_dir"
    --wandb-project muster
    --wandb-entity bluffish
    --wandb-name "$WANDB_NAME"
    --wandb-id "$WANDB_NAME"
    "${TRAIN_ARGS[@]}"
)
if [[ -f "$run_dir/latest.pt" ]]; then
    args+=(--resume "$run_dir/latest.pt")
fi

echo "== starting training ($(date -Is)) on $(hostname) =="
python train.py "${args[@]}" >> "$run_dir/train.log" 2>&1 &
train_pid=$!
wait "$train_pid"
status=$?
echo "== training exited with status $status ($(date -Is)) =="
exit "$status"

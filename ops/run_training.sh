#!/usr/bin/env bash
set -euo pipefail

name=${1:?usage: run_training.sh <config-name>}
project_dir=/home/ubuntu/muster-local-hex
python=/home/ubuntu/muster-viewer/.venv/bin/python

cd "$project_dir"
source "configs/$name.env"
run_dir="runs/$RUN_NAME"
mkdir -p "$run_dir"
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

exec env PYTHONPATH=/dev/shm/torch "$python" train.py "${args[@]}" \
    >> "$run_dir/train.log" 2>&1

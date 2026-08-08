#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/ubuntu/muster-local-hex
run_dir=runs/local-hex-v11-maven-v1
python=/home/ubuntu/muster-viewer/.venv/bin/python

cd "$project_dir"
mkdir -p "$run_dir"
args=(
    --device cuda
    --run "$run_dir"
    --opponent nearest
    --soldiers 256
    --envs 256
    --rollout 150
    --action-repeat 3
    --updates 10000
    --epochs 4
    --minibatches 32
    --hidden-size 256
    --entity-size 16
    --tile-size 32
    --local-radius 2
    --mode-count 16
    --mode-size 16
    --mode-mi-beta 0.005
    --mode-mi-anneal-updates 4000
    --anchor-every 25
    --anchor-episodes 4
    --dtype bf16
    --compile
    --compile-mode default
    --learning-rate 0.0003
    --gamma 1.0
    --gae-lambda 0.941192
    --clip 0.2
    --value-coefficient 0.5
    --entropy-coefficient 0.003
    --maximum-gradient-norm 0.5
    --target-kl 0.02
    --save-every 10
    --replay-every 1
    --wandb-project muster
    --wandb-entity bluffish
    --wandb-name bowen3-v11-maven-v1
    --wandb-id bowen3-v11-maven-v1
    --seed 0
)

if [[ -f "$run_dir/latest.pt" ]]; then
    args+=(--resume "$run_dir/latest.pt")
fi

exec env PYTHONPATH=/dev/shm/torch "$python" train.py "${args[@]}" \
    >> "$run_dir/train.log" 2>&1

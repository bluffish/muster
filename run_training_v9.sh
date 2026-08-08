#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/ubuntu/muster-local-hex
run_dir=runs/local-hex-v9-side-equivariant-v1
python=/home/ubuntu/muster-viewer/.venv/bin/python

cd "$project_dir"
args=(
    --device cuda
    --run "$run_dir"
    --opponent pool
    --opponent-pool-size 8
    --opponent-snapshot-every 1
    --opponent-latest-probability 0.5
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
    --wandb-name bowen3-v9-side-equivariant-v1
    --wandb-id bowen3-v9-side-equivariant-v1
    --seed 0
)

if [[ -f "$run_dir/latest.pt" ]]; then
    args+=(--resume "$run_dir/latest.pt")
fi

exec env PYTHONPATH=/dev/shm/torch "$python" train.py "${args[@]}"

# Muster handoff

Muster is a GPU-batched 256-vs-256 hex battle simulator used for MAPPO
training. The source of truth is this directory (`/root/muster`, a git
repo). CUDA training runs on the AWS host below; `muster.bowen.sh` is
served from this machine by Caddy.

```sh
remote=ubuntu@52.90.113.210
identity=/root/.ssh/id_ed25519
```

Game rules live in `SIMULATOR_RULES.md` (version 0.8: health-weighted soft influence
control with presence requirement, time-integral scoring, score
observability). Version history and the reasoning behind each change:
`CHANGELOG.md`.

## Layout

```
muster/sim/      constants, Config, geometry, CPU reference sim, warp GPU sim
muster/rl/       env (observations, kNN, opponents), policy, pool, rollout,
                 ppo (flat + recurrent BPTT), modes, evaluation, metrics, train
muster/viewer/   replay packing/recording, template.html (browser viewer),
                 scripted benchmark policies, optional pygame viewer
ops/             launcher, systemd units, publication pipeline, retemplate tool
configs/         one small env file per experiment (flags live here)
tests/           pytest suite (GPU-optional; runs on CPU warp where possible)
```

Root-level `simulator.py`, `rl_env.py`, `policy.py`, `train.py`, `viewer.py`,
`simulator_gpu.py` are thin compatibility facades; new code should import
from `muster.*`.

## Current run

- Config: `configs/v19-score.env` → run `local-hex-v19-score-v1`,
  W&B `bowen3-v19-score-v1`
- Service on AWS: `muster-training@v19-score.service`
- Rules v0.7 with score observability (features 11/12); architecture: entity
  attention (v14), 64-unit GRU memory with BPTT-15 (v15), 16 episode modes,
  dormant 8-wide message slot; mixed opponents (75% pool / 25% charger);
  LR 1e-3.

## Start, stop, and inspect training

One template unit serves every experiment; the instance name selects the
config file:

```sh
ssh -i "$identity" "$remote" 'sudo systemctl enable --now muster-training@v19-score'
ssh -i "$identity" "$remote" 'sudo systemctl stop muster-training@v19-score'      # pause
ssh -i "$identity" "$remote" 'sudo systemctl disable --now muster-training@v19-score'
ssh -i "$identity" "$remote" 'tail -5 /home/ubuntu/muster-local-hex/runs/local-hex-v19-score-v1/metrics.jsonl'
```

The launcher resumes `latest.pt` when it exists. **Starting a new
experiment:** write `configs/<name>.env` (RUN_NAME, WANDB_NAME,
TRAIN_ARGS array), deploy it, `sudo systemctl enable --now
muster-training@<name>`. Only one training instance at a time — they share
the GPU.

## Deploy training code

Stop training before deploying code the Python process imports:

```sh
scp -r -i "$identity" muster train.py simulator.py rl_env.py policy.py viewer.py simulator_gpu.py configs "$remote":/home/ubuntu/muster-local-hex/
scp -r -i "$identity" tests conftest.py pyproject.toml "$remote":/home/ubuntu/muster-local-hex/
scp -i "$identity" ops/run_training.sh "$remote":/home/ubuntu/muster-local-hex/ops/
```

If the unit changes: `scp ops/muster-training@.service` to `/tmp` on the
remote, `sudo install -m 0644` it into `/etc/systemd/system/`, and
`sudo systemctl daemon-reload`.

## Testing

This server has no CUDA; only syntax checks run locally. Validate on AWS
(stop training first for reliable GPU tests):

```sh
ssh -i "$identity" "$remote" 'cd /home/ubuntu/muster-local-hex && env PYTHONPATH=/dev/shm/torch /home/ubuntu/muster-viewer/.venv/bin/python -m pytest'
```

## Viewer and replay publication

Public viewer: <https://muster.bowen.sh/>. The pipeline runs on this server
about every 10 seconds:

1. Training writes `update-N.html` on AWS.
2. `muster-replay-archive.timer` copies it to
   `/srv/muster-archive/<active-run>/replays` and deletes the remote copy.
3. `muster-replay-sync.timer` applies the current
   `muster/viewer/template.html` and publishes to `/srv/muster`.

**The active run is a single line in `/etc/default/muster`**
(`MUSTER_ACTIVE_RUN=...`), read by both pipeline scripts. Switching the
public run = edit that line. If the scripts themselves change, reinstall:

```sh
sudo install -m 0755 ops/archive_replays.sh /usr/local/bin/muster-archive-replays
sudo install -m 0755 ops/sync_replay.sh /usr/local/bin/muster-sync-replay
sudo install -m 0755 ops/retemplate_replay.py /usr/local/bin/muster-retemplate-replay
sudo install -m 0644 ops/muster-replay-archive.{service,timer} ops/muster-replay-sync.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload
```

Force a refresh: `sudo systemctl start muster-replay-archive.service
muster-replay-sync.service`. Published history for earlier runs is
preserved under `/srv/muster/runs/<run>/replays`. Replays alternate
matchups by update (even = scripted charger, odd = pool self-play); blue is
always the learner, and each page records `opponent_mode`, `learner_team`,
`learner_mode`, and continuous control shares (`control_u8`).

## Metrics quick reference

`territory_advantage` is the objective (time-averaged weighted control
advantage; per-episode its sign is the winner). Absolute skill lives in the
anchor (`anchor_win_rate`, per-mode best/worst, every 25 updates) and the
`scripted_*` slice — pool win rates sit near 50% by symmetry. Combat facts:
`episode_damage_dealt/taken`, `*_final_alive_fraction`. Mode system:
`mode_probe_accuracy` (chance = 1/16), `mode_action_delta` (~0 = dead
latent), `mode_intrinsic_fraction` (>~0.5 sustained means the MI bonus is
overpowering the objective — lower `--mode-mi-beta`). Recurrent-PPO health:
`approximate_kl` vs the 0.02 target, `early_stopped`, `gradient_norm`.

Avoid committing credentials; W&B auth is configured for the `ubuntu` user
on AWS.

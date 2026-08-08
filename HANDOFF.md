# Muster handoff

Muster is a GPU-batched 256-vs-256 hex battle simulator used for MAPPO training. The source of truth is this directory (`/root/muster`). CUDA training runs on the AWS host below; `muster.bowen.sh` is served from this machine by Caddy.

```sh
remote=ubuntu@52.90.113.210
identity=/root/.ssh/id_ed25519
```

## Current run

- Run: `local-hex-v15-memory-v1`
- Service on AWS: `muster-training-v15.service`
- W&B name/ID: `bowen3-v15-memory-v1`
- New in v15: per-soldier GRU memory (64 units, BPTT window 15) and the
  reserved dormant message slot; fresh start
- Opponent: **mixed** — 75% of environments play the self-play snapshot pool,
  25% play the scripted charger (side-balanced)
- Fresh start (v13 checkpoints are architecture-incompatible with the new
  perception); raised noise floor kept (`--log-std-floor -0.8`)
- MAVEN episode modes: 16 per team, MI bonus `beta=0.005` annealed by update
  4000, scripted-charger anchor every 25 updates
- The scripted charger now **occupies the nearest strongpoint when no enemies
  remain** instead of halting (see the v13 section)
- Scale: 256 environments, 256 soldiers per team
- Rollout: 150 policy actions, each held for 3 simulator steps (0.3 seconds)
- Training: BF16 MAPPO, `torch.compile`, 4 epochs, 32 minibatches, up to 10,000 updates
- Checkpoint every 10 updates; training replay every update

Public replays alternate matchups by update: even updates capture a
scripted-opponent environment, odd updates a pool self-play environment. Blue
is always the learner, and the viewer's matchup line says which opponent the
episode featured. Across the training batch, learner colors alternate evenly
to avoid side bias.

The only reward is zero-sum weighted territory change. With `gamma=1`, its episode sum equals final territory advantage. Normal cells have weight 1; the 21 strongpoint cells have weight 30.

## GRU memory + sequence PPO (v15)

v15 gives each soldier a persistent 64-unit GRU state (`--memory-size 64`)
between the backbone and the heads, trained with truncated backprop through
time over 15-decision windows (`--bptt-window 15`; recurrent-branch PPO
minibatches are (window, env) chunks instead of shuffled timesteps, and the
rollout stores hidden states at window boundaries). Memory zeroes on death
and episode reset; the pool, anchor, and evaluation paths all carry it. The
per-entity message slot is reserved but dormant (`--message-size 8` widens
entity tokens with a zero-fed channel) so the v16 communication run can
switch messaging on without an architecture break.

All memory kwargs default to zero, so v14 checkpoints remain loadable and
`CHECKPOINT_VERSION` stays 11. Launched 2026-08-09 by explicit decision,
cutting v14's baseline short at ~update 390 (still in its avoidance valley;
resumable via `muster-training-v15.service`). The v14 partial baseline plus
the v13 full trajectory serve as the memoryless comparison points.

## Entity-attention perception (v14)

v14 replaces the pooled entity sense with egocentric per-entity attention.
Previously entities were summed into territory cells and the 19-cell
neighborhood summed again before attention, destroying individual identity
and direction: soldiers perceived kernel-weighted means, quantized at
~1.5-unit tile granularity against a 0.84-unit contact range, anchored to
the cell grid (asymmetric, discontinuous at cell boundaries). Melee verbs -
face *that* attacker, strike *the wounded one* - were unrepresentable, and
v10-v13 tactics were correspondingly crowd-scale only.

Now each soldier attends over its k=16 nearest living entities within the
same 5-unit radius (`entity_neighbors` in `rl_env.py`, recomputed from exact
simulator positions at every observation; `LocalState.neighbors`,
`(envs, 2, soldiers, k)` int16, -1 = empty, ally indices < S, enemies >= S
in each team's related space). Tokens bind each entity's embedding to its
exact egocentric offset and side; masked multi-head cross-attention replaces
the old linear-attention pooling (`Policy._entity_context`). Sensing *range*
is unchanged by design - information stays scarce so communication (v15+)
keeps a niche; this upgrade is fidelity within the horizon. The token
interface is also where recurrence and received messages will plug in later.

`CHECKPOINT_VERSION` 10 -> 11; v13 code is preserved on AWS as
`*.pre-v14-20260809` copies, and the v13 run remains resumable with it.

## Mixed self-play (v13)

v13 switches the training distribution after v11/v12 showed that a fixed
scripted opponent gets farmed for exploits instead of teaching competence:
v11 learned kiting (equal-speed pursuit can never close), v12 learned
martyrdom-painting (the charger halted when every enemy died, freezing
painted territory). Both capped at -0.32 without contesting strongpoints.

Three changes:

1. **Honest bot.** `nearest-charge` (GPU kernel and the `viewer.py` scripted
   policy) now marches to the nearest strongpoint and holds when no living
   enemy exists. This removes the freeze exploit and makes the anchor
   ungameable by dying. **Anchor numbers before/after this change are not
   comparable.**
2. **Mixed opponents.** `--opponent mixed --mixed-scripted-fraction 0.25`:
   a deterministic, side-balanced quarter of environments face the charger
   (in-gradient anti-charger pressure), the rest face the snapshot pool
   (auto-curriculum for combat: the opponent is exactly as weak as the
   learner). Per-slice results are logged as `scripted_wins/...` and
   `pool_wins/...`; pool win rates sit near 50% by construction, so absolute
   progress lives in `scripted_*` and the anchor.
3. **Exploration repair.** `--log-std-floor -0.8` keeps per-step noise from
   collapsing (v10 and v12 both collapsed), and `--warm-start` +
   `--reset-log-std` initializes model+optimizer from another run's
   checkpoint while restoring the initial noise level. Warm start is ignored
   whenever the run's own `latest.pt` exists (normal resume wins).

Expected signals: martyrdom unravels (a live opponent's survivors repaint an
undefended map), `learner_final_alive_fraction` lifts off zero,
`episode_damage_dealt` climbs in pool play, and the anchor becomes the only
scoreboard that matters.

## Episode clock (v12)

v12 adds `time_remaining` (normalized, counting 1 -> 0) as the eleventh local
observation feature, visible to both actor and critic. Rationale: with
`gamma=1` and a fixed 45-second horizon, only the final territory snapshot
scores, so the optimal anti-charger strategy is phased — cede ground early,
win attrition, reconvert territory in the closing seconds. A time-blind
policy cannot represent that phase switch, and a time-blind critic cannot
distinguish "territory locked in at t=44s" from "anything can happen at
t=5s" (Pardo et al. 2018, "Time Limits in RL"). v11 plateaued at exactly this
boundary: kiting and near-even attrition trades, but no endgame push, flat at
-0.32 advantage.

This changed `LOCAL_FEATURE_SIZE` 10 -> 11 and `CHECKPOINT_VERSION` 9 -> 10:
v10/v11 checkpoints cannot be resumed under current code. The code they need
is preserved on AWS as `*.pre-v12-20260808` copies next to the live files.

## MAVEN episode modes (v11)

The training stack now supports MAVEN-style per-episode exploration: each team
draws a discrete strategy mode at every episode start, all 256 soldiers share
it, and the policy conditions on it (backbone input, a fixed-gain additive bias
on the pre-tanh action mean, and the centralized critic). The goal is
coordinated team-level exploration instead of per-soldier white noise, which
averages out across 256 soldiers.

Modes are opt-in: `--mode-count 0` (the default) builds a model with no mode
parameters. (The original v10-compatibility guarantee ended with v12's
checkpoint version bump — see the episode-clock section.)

New `train.py` flags:

- `--mode-count 16 --mode-size 16`: enable modes (fresh runs only; resumes
  take the model shape from the checkpoint).
- `--mode-mi-beta`: DIAYN/MAVEN-style diversity bonus
  `beta * (log q(mode | team summary) - log(1/K))` added to the learner's
  reward; at 0 the discriminator still trains passively as a probe. v11 runs
  `0.05` annealed to zero by update 4000 because v10 showed deep entropy
  collapse into a losing local optimum. If `mode_intrinsic_fraction` stays
  above ~0.5 for hundreds of updates, the bonus is overpowering the territory
  objective: lower beta (edit the launcher and restart; the run resumes).
- `--mode-bandit` (+ `--mode-bandit-temperature`): sample modes from recent
  per-mode territory returns (25% uniform floor) instead of uniformly. Leave
  off until modes visibly differentiate.
- `--anchor-every 25 --anchor-episodes 4`: periodic deterministic benchmark
  against the scripted nearest-charge opponent, per mode and per side. This is
  the absolute-skill measure — pool self-play win rates sit at 50% by symmetry
  and cannot show progress.

New metrics: `anchor_win_rate` / `anchor_territory_advantage` (and per-mode
best/worst), `mode_probe_accuracy` (mode identifiability from behavior;
chance is `1/K`), `mode_action_delta` (action change when modes are rerolled;
~0 means the latent died), `mode_win_best/worst` and `mode_advantage_spread`
(training-time per-mode outcomes), and combat facts
(`episode_damage_dealt/taken`, `*_final_alive_fraction`) that the territory
reward never surfaces. At deployment of this feature, the v10 policy at update
~1400 lost 100% of anchor episodes at −0.55 territory advantage.

The v11 run (`local-hex-v11-maven-v1`, W&B `bowen3-v11-maven-v1`, stopped at
~130 updates) was v10 plus 16 episode modes, the MI bonus (0.05, cut to 0.005
at update ~80), and the anchor. It escaped v10's passivity into kiting with
near-even attrition trades but plateaued at −0.32 without an endgame push —
the finding that motivated v12's clock. All training services share the GPU —
run one at a time. Replays record `learner_mode`.

## Start, stop, and inspect training

Start or resume the current run:

```sh
ssh -i "$identity" "$remote" 'sudo systemctl enable --now muster-training-v15.service'
```

The launcher resumes `latest.pt` when it exists. To pause without changing boot behavior:

```sh
ssh -i "$identity" "$remote" 'sudo systemctl stop muster-training-v15.service'
```

To stop it persistently, including after reboots:

```sh
ssh -i "$identity" "$remote" 'sudo systemctl disable --now muster-training-v15.service'
```

Status and recent metrics:

```sh
ssh -i "$identity" "$remote" 'systemctl --no-pager --full status muster-training-v15.service'
ssh -i "$identity" "$remote" 'tail -5 /home/ubuntu/muster-local-hex/runs/local-hex-v15-memory-v1/metrics.jsonl'
```

For a genuinely fresh experiment, use a new run directory, W&B name/ID, launcher, and service rather than deleting or overwriting an old run. `run_training_v15.sh` is the current launcher template.

## Deploy training code

Stop training before deploying code that the Python process imports:

```sh
scp -i "$identity" rl_env.py train.py viewer.py "$remote":/home/ubuntu/muster-local-hex/
scp -i "$identity" run_training_v15.sh "$remote":/home/ubuntu/muster-local-hex/
ssh -i "$identity" "$remote" 'chmod 0755 /home/ubuntu/muster-local-hex/run_training_v15.sh'
```

If the unit changes, install it and reload systemd:

```sh
scp -i "$identity" muster-training-v15.service "$remote":/tmp/
ssh -i "$identity" "$remote" 'sudo install -m 0644 /tmp/muster-training-v15.service /etc/systemd/system/ && sudo systemctl daemon-reload'
```

Then start the service with the command above.

## Viewer and replay publication

Public viewer: <https://muster.bowen.sh/>

The pipeline runs automatically about every 10 seconds:

1. Training writes `update-N.html` on AWS.
2. `muster-v15-replay-archive.timer` copies it to `/srv/muster-archive/local-hex-v15-memory-v1/replays` on this server.
3. `muster-replay-sync.timer` applies the current `viewer.py` template and publishes it.
4. `/srv/muster/replays` points at the active run's history, while `/srv/muster/index.html` is the latest replay.

Force an immediate refresh:

```sh
sudo systemctl start muster-v15-replay-archive.service
sudo systemctl start muster-replay-sync.service
```

Check the pipeline and public manifest:

```sh
systemctl is-active muster-v15-replay-archive.timer muster-replay-sync.timer caddy
readlink /srv/muster/replays
curl -fsS https://muster.bowen.sh/replays/manifest.json
```

After editing the browser UI in `viewer.py`, existing archived replays receive the new template when first published. To republish an already-published replay, remove only its corresponding file from the active `/srv/muster/runs/<run>/replays` directory and run `muster-replay-sync.service` again. Do not remove the archive copy.

When switching to a new run, update `run` in `sync_replay.sh` and the source/archive paths in the archive script. Install them with:

```sh
sudo install -m 0755 sync_replay.sh /usr/local/bin/muster-sync-replay
sudo install -m 0755 archive_v15_replays.sh /usr/local/bin/muster-v15-archive-replays
sudo systemctl daemon-reload
```

Published history for earlier runs is preserved under `/srv/muster/runs/<run>/replays` (v9 side-equivariant; v10 nearest, ~1450 updates, zero anchor wins; v11 maven, kiting plateau at -0.32; v12 clock, martyrdom-painting plateau at -0.32 with the first 11 episode wins ever recorded). Resuming v10/v11 requires the `*.pre-v12-20260808` code copies on AWS; v12 code cannot load their checkpoints.

## Main files

- `simulator.py`: NumPy reference simulator and shared constants
- `simulator_gpu.py`: batched Warp simulator
- `rl_env.py`: observations, bookkeeping, and fixed nearest-enemy opponent
- `policy.py`: actor and centralized MAPPO critic
- `train.py`: rollouts, reward, GAE/PPO, checkpoints, W&B, and replay capture
- `viewer.py`: replay packing and the self-contained browser viewer
- `run_training_v10.sh`: current AWS launch arguments
- `sync_replay.sh`: active-run publication

## Testing

This server has no CUDA. Syntax and CPU-only viewer tests can run locally; validate GPU/torch changes on AWS. For a reliable GPU test, stop training first.

```sh
ssh -i "$identity" "$remote" 'cd /home/ubuntu/muster-local-hex && env PYTHONPATH=/dev/shm/torch /home/ubuntu/muster-viewer/.venv/bin/python -m pytest -q'
```

Avoid putting credentials in the repository. W&B authentication is already configured for the `ubuntu` user on AWS.

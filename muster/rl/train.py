"""Training entrypoint: CLI, checkpointing, and the update loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

from muster.rl.env import SUMMARY_SIZE, NearestEnemyOpponent, RLEnv
from muster.rl.evaluation import AnchorEvaluator
from muster.rl.metrics import combat_metrics, episode_metrics, territory_metrics
from muster.rl.modes import (
    ModeDiscriminator,
    add_mode_intrinsic_reward,
    mode_action_sensitivity,
    mode_outcome_metrics,
    mode_probe_update,
)
from muster.rl.policy import CHECKPOINT_VERSION, Policy
from muster.rl.pool import OpponentPool
from muster.rl.ppo import compute_gae, ppo_update
from muster.rl.rollout import (
    RolloutReplay,
    collect_rollout,
    make_rollout,
    scripted_environment_mask,
    synchronize,
    write_rollout_replay,
)
from muster.sim import Config

def compile_policies(policy: Policy, pool: OpponentPool | None, mode: str) -> None:
    """Compile hot neural callables without wrapping checkpointed modules."""
    policy.act = torch.compile(policy.act, mode=mode)
    policy.value = torch.compile(policy.value, mode=mode)
    policy.evaluate_actions = torch.compile(policy.evaluate_actions, mode=mode)
    if pool is not None:
        for model in pool.models:
            model.actor_step = torch.compile(model.actor_step, mode=mode, dynamic=True)

def save_checkpoint(
    path: Path,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    update: int,
    args: argparse.Namespace,
    config: Config,
    pool: OpponentPool | None = None,
    discriminator: ModeDiscriminator | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    bandit_returns: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "model_kwargs": policy.model_kwargs,
        "sim_config": asdict(config),
        "train_args": vars(args),
        "torch_rng": torch.get_rng_state(),
    }
    if next(policy.parameters()).is_cuda:
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state(next(policy.parameters()).device)
    if pool is not None:
        checkpoint["opponent_pool"] = pool.checkpoint()
    if discriminator is not None:
        checkpoint["mode_discriminator"] = discriminator.state_dict()
        checkpoint["mode_discriminator_optimizer"] = discriminator_optimizer.state_dict()
    if bandit_returns is not None:
        checkpoint["mode_bandit_returns"] = bandit_returns.detach().cpu()
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)

def start_wandb(args: argparse.Namespace, config: Config, run_dir: Path):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("install the RL extra: pip install -e '.[rl]'") from error
    tracking = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        dir=str(run_dir),
        config={"simulator": asdict(config), "training": vars(args)},
    )
    tracking.define_metric("update")
    tracking.define_metric("*", step_metric="update")
    return tracking

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--resume")
    parser.add_argument(
        "--warm-start",
        help="initialize model and optimizer from another run's checkpoint; "
        "ignored when --resume applies",
    )
    parser.add_argument(
        "--reset-log-std",
        action="store_true",
        help="reset exploration noise to its initial level after a warm start",
    )
    parser.add_argument(
        "--log-std-floor",
        type=float,
        help="override the policy's minimum log standard deviation",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=0,
        help="per-soldier GRU hidden units; zero keeps the feedforward policy",
    )
    parser.add_argument(
        "--bptt-window",
        type=int,
        default=15,
        help="decisions per truncated-backprop window; must divide --rollout",
    )
    parser.add_argument(
        "--message-size",
        type=int,
        default=0,
        help="reserved per-entity message embedding width (dormant channel)",
    )
    parser.add_argument(
        "--opponent", choices=("self", "pool", "nearest", "mixed"), default="pool"
    )
    parser.add_argument(
        "--mixed-scripted-fraction",
        type=float,
        default=0.25,
        help="fraction of environments facing the scripted charger when --opponent mixed",
    )
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--opponent-snapshot-every", type=int, default=1)
    parser.add_argument("--opponent-latest-probability", type=float, default=0.5)
    parser.add_argument(
        "--reset-opponent-pool",
        action="store_true",
        help="initialize the pool from the resumed learner instead of loading it",
    )
    parser.add_argument("--soldiers", type=int, default=256, help="soldiers per team")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rollout", type=int, default=450)
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="hold each sampled policy action for this many 0.1-second simulator steps",
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--entity-size", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--local-radius", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--mode-count",
        type=int,
        default=0,
        help="per-team episode strategy modes (MAVEN-style latent); zero disables",
    )
    parser.add_argument("--mode-size", type=int, default=16, help="mode embedding width")
    parser.add_argument(
        "--mode-mi-beta",
        type=float,
        default=0.0,
        help="diversity bonus scale on discriminator log-probability; zero keeps the probe passive",
    )
    parser.add_argument(
        "--mode-mi-anneal-updates",
        type=int,
        default=0,
        help="linearly anneal the diversity bonus to zero by this update; zero keeps it constant",
    )
    parser.add_argument(
        "--mode-bandit",
        action="store_true",
        help="sample episode modes from recent territory returns instead of uniformly",
    )
    parser.add_argument("--mode-bandit-temperature", type=float, default=0.05)
    parser.add_argument(
        "--anchor-every",
        type=int,
        default=25,
        help="updates between scripted-charger benchmark episodes; zero disables",
    )
    parser.add_argument(
        "--anchor-episodes",
        type=int,
        default=4,
        help="even number of anchor episodes per mode",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode", choices=("default", "max-autotune-no-cudagraphs"), default="default"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.003)
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--replay-every", type=int, default=1, help="zero disables replay export")
    parser.add_argument("--wandb-project", help="enable W&B logging")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id", help="stable ID used to resume the same W&B run")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA device does not support BF16")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if checkpoint is not None and checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("this MAPPO model is checkpoint-incompatible; start a fresh run")
    warm = None
    if checkpoint is None and args.warm_start:
        warm = torch.load(args.warm_start, map_location=device, weights_only=False)
        if warm.get("version") != CHECKPOINT_VERSION:
            raise ValueError("the warm-start checkpoint is incompatible with this code")
    source = checkpoint or warm
    config = Config(**source["sim_config"]) if source else Config(soldiers_per_team=args.soldiers)
    if args.action_repeat < 1:
        raise ValueError("--action-repeat must be positive")
    if config.maximum_decision_steps % args.action_repeat:
        raise ValueError("episode length must be divisible by --action-repeat")
    if args.mode_count < 0 or args.mode_count == 1:
        raise ValueError("--mode-count must be zero or at least two")
    model_kwargs = dict(source["model_kwargs"]) if source else {
        "hidden_size": args.hidden_size,
        "entity_size": args.entity_size,
        "tile_size": args.tile_size,
        "local_radius": args.local_radius,
        "mode_count": args.mode_count,
        "mode_size": args.mode_size,
        "memory_size": args.memory_size,
        "message_size": args.message_size,
    }
    if args.log_std_floor is not None:
        model_kwargs["log_std_floor"] = args.log_std_floor
    mode_count = int(model_kwargs.get("mode_count", 0) or 0)
    memory_size = int(model_kwargs.get("memory_size", 0) or 0)
    if memory_size and args.rollout % args.bptt_window:
        raise ValueError("--rollout must be divisible by --bptt-window")
    env = RLEnv(config, args.envs, args.device, mode_count=max(1, mode_count))
    policy = Policy(**model_kwargs).to(device)
    policy.use_bf16 = args.dtype == "bf16"
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    def _restore_optimizer(state: dict[str, object]) -> None:
        # load_state_dict restores the checkpointed hyperparameters, which
        # would silently override a changed --learning-rate on resume.
        optimizer.load_state_dict(state)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate

    first_update = 1
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["model"])
        _restore_optimizer(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
        first_update = int(checkpoint["update"]) + 1
    elif warm is not None:
        policy.load_state_dict(warm["model"])
        _restore_optimizer(warm["optimizer"])
        if args.reset_log_std:
            with torch.no_grad():
                policy.log_std.fill_(-0.5)

    pool = None
    opponent_slots = None
    if args.opponent in ("pool", "mixed"):
        if args.opponent_snapshot_every < 1:
            raise ValueError("--opponent-snapshot-every must be positive")
        pool = OpponentPool(
            policy,
            args.opponent_pool_size,
            latest_probability=args.opponent_latest_probability,
            checkpoint=(
                checkpoint.get("opponent_pool")
                if checkpoint and not args.reset_opponent_pool
                else None
            ),
        )
        opponent_slots = pool.initial_slots(args.envs)
    discriminator = None
    discriminator_optimizer = None
    if mode_count > 1:
        discriminator = ModeDiscriminator(SUMMARY_SIZE, mode_count).to(device)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        if source is not None and "mode_discriminator" in source:
            discriminator.load_state_dict(source["mode_discriminator"])
            discriminator_optimizer.load_state_dict(
                source["mode_discriminator_optimizer"]
            )
    bandit_returns = None
    if args.mode_bandit and mode_count > 1:
        bandit_returns = torch.zeros(mode_count, device=device)
        if checkpoint is not None and "mode_bandit_returns" in checkpoint:
            bandit_returns.copy_(checkpoint["mode_bandit_returns"].to(device))
    anchor = (
        AnchorEvaluator(
            config, args.device, mode_count, args.anchor_episodes, args.action_repeat
        )
        if args.anchor_every > 0
        else None
    )
    if args.compile:
        compile_policies(policy, pool, args.compile_mode)

    state = env.reset()
    rollout = make_rollout(args.rollout, state, memory_size, args.bptt_window)
    learner_memory = policy.initial_memory(state)
    opponent_memory = (
        policy.initial_memory(state) if memory_size and pool is not None else None
    )
    learner_teams = torch.zeros((args.envs, 2), dtype=torch.bool, device=device)
    indices = torch.arange(args.envs, device=device)
    learner_teams[indices, indices.remainder(2)] = True
    nearest_opponent = (
        NearestEnemyOpponent(env, learner_teams)
        if args.opponent in ("nearest", "mixed")
        else None
    )
    scripted_envs = (
        scripted_environment_mask(args.envs, args.mixed_scripted_fraction, device)
        if args.opponent == "mixed"
        else None
    )
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    tracking = start_wandb(args, config, run_dir)
    replay_capture = (
        RolloutReplay(
            env,
            args.rollout,
            args.action_repeat,
            opponent_mode=args.opponent,
            learner_team=0,
        )
        if args.replay_every > 0
        else None
    )
    replay_writer = ThreadPoolExecutor(max_workers=1) if replay_capture is not None else None
    replay_future = None

    pool_replay_env = None
    if scripted_envs is not None:
        eligible = ~scripted_envs & (torch.arange(args.envs, device=device) % 2 == 0)
        if bool(eligible.any()):
            pool_replay_env = int(eligible.long().argmax())
    for update in range(first_update, args.updates + 1):
        replay_due = args.replay_every > 0 and (
            update % args.replay_every == 0 or update == args.updates
        )
        if replay_due and replay_capture is not None and scripted_envs is not None:
            if pool_replay_env is not None and update % 2:
                replay_capture.retarget(pool_replay_env, "pool", 0)
            else:
                replay_capture.retarget(0, "nearest", 0)
        state, bootstrap, collection_seconds = collect_rollout(
            env,
            policy,
            rollout,
            state,
            learner_teams,
            pool,
            opponent_slots,
            args.action_repeat,
            replay_capture if replay_due else None,
            nearest_opponent=nearest_opponent,
            scripted_envs=scripted_envs,
            learner_memory=learner_memory,
            opponent_memory=opponent_memory,
            bptt_window=args.bptt_window,
        )
        if replay_due:
            replay = replay_capture.replay(rollout["done"], rollout["winner"], update)
            if replay_future is not None:
                replay_future.result()
            replay_future = replay_writer.submit(write_rollout_replay, run_dir, replay)
        mode_metrics = {}
        if discriminator is not None and args.mode_mi_beta > 0:
            beta = args.mode_mi_beta
            if args.mode_mi_anneal_updates > 0:
                beta *= max(0.0, 1.0 - update / args.mode_mi_anneal_updates)
            if beta > 0:
                mode_metrics.update(
                    add_mode_intrinsic_reward(
                        rollout, discriminator, learner_teams, beta, mode_count
                    )
                )
        compute_gae(rollout, bootstrap, args.gamma, args.gae_lambda)
        synchronize(device)
        started = time.perf_counter()
        losses = ppo_update(
            policy,
            optimizer,
            rollout,
            learner_teams,
            args.epochs,
            args.minibatches,
            args.clip,
            args.value_coefficient,
            args.entropy_coefficient,
            args.maximum_gradient_norm,
            args.target_kl,
            args.bptt_window,
        )
        synchronize(device)
        learning_seconds = time.perf_counter() - started
        policy_decisions = args.envs * args.rollout
        decisions = policy_decisions * args.action_repeat
        pool_metrics = (
            pool.advance(policy, opponent_slots, update % args.opponent_snapshot_every == 0)
            if pool is not None
            else {}
        )
        if pool is not None:
            pool.resample_finished(opponent_slots, rollout["done"][-1])
        if discriminator is not None:
            mode_metrics.update(
                mode_probe_update(
                    discriminator, discriminator_optimizer, rollout, learner_teams
                )
            )
        if mode_count > 1:
            outcome_metrics, advantage_by_mode, present = mode_outcome_metrics(
                rollout, learner_teams, mode_count
            )
            mode_metrics.update(outcome_metrics)
            mode_metrics.update(
                mode_action_sensitivity(policy, rollout, learner_teams, mode_count)
            )
            if bandit_returns is not None and advantage_by_mode is not None:
                bandit_returns[present] = (
                    0.9 * bandit_returns[present] + 0.1 * advantage_by_mode[present]
                )
                env.set_mode_distribution(
                    0.25 / mode_count
                    + 0.75
                    * torch.softmax(
                        bandit_returns / args.mode_bandit_temperature, dim=0
                    )
                )
        if anchor is not None and (
            update % args.anchor_every == 0 or update == args.updates
        ):
            mode_metrics.update(anchor.run(policy))
        metrics = {
            "update": update,
            "environment_decisions_per_second": decisions / collection_seconds,
            "soldier_decisions_per_second": decisions * config.soldier_count / collection_seconds,
            "training_soldier_decisions_per_second": decisions * config.soldier_count / (collection_seconds + learning_seconds),
            "policy_environment_decisions_per_second": policy_decisions / collection_seconds,
            "policy_soldier_decisions_per_second": policy_decisions * config.soldier_count / (collection_seconds + learning_seconds),
            "collection_seconds": collection_seconds,
            "learning_seconds": learning_seconds,
            **episode_metrics(rollout, learner_teams),
            **(
                {
                    **episode_metrics(
                        rollout, learner_teams, env_mask=scripted_envs, prefix="scripted_"
                    ),
                    **episode_metrics(
                        rollout, learner_teams, env_mask=~scripted_envs, prefix="pool_"
                    ),
                }
                if scripted_envs is not None
                else {}
            ),
            **territory_metrics(rollout, learner_teams),
            **combat_metrics(rollout, learner_teams),
            **losses,
            **pool_metrics,
            **mode_metrics,
        }
        line = json.dumps(metrics, separators=(",", ":"))
        print(line, flush=True)
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
        if tracking is not None:
            tracking.log(metrics)
        if update % args.save_every == 0 or update == args.updates:
            save_checkpoint(
                run_dir / "latest.pt",
                policy,
                optimizer,
                update,
                args,
                config,
                pool,
                discriminator,
                discriminator_optimizer,
                bandit_returns,
            )
    if replay_future is not None:
        replay_future.result()
    if replay_writer is not None:
        replay_writer.shutdown()
    if tracking is not None:
        tracking.finish()

if __name__ == "__main__":
    main()

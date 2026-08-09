"""Single-GPU MAPPO training for the hex battle simulator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import warp as wp
from torch import nn
from torch.nn import functional as F

from policy import CHECKPOINT_VERSION, Policy
from rl_env import SUMMARY_SIZE, LocalState, NearestEnemyOpponent, RLEnv
from simulator import (
    STRONGPOINT_CELLS,
    STRONGPOINT_WEIGHT,
    TERRITORY_CELLS,
    TERRITORY_RADIUS,
    Config,
)

STATE_KEYS = ("features", "cells", "alive", "owners", "mirror_y", "mode", "neighbors")


def select_state(state: LocalState, index) -> LocalState:
    return LocalState(*(tensor[index] for tensor in state))


class OpponentPool:
    """GPU-resident policy snapshots that never change during an episode."""

    def __init__(
        self,
        policy: Policy,
        size: int,
        latest_probability: float = 0.5,
        checkpoint: dict[str, object] | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("opponent pool needs at least two snapshots")
        if not 0 <= latest_probability <= 1:
            raise ValueError("latest snapshot probability must be between zero and one")
        self.models = [copy.deepcopy(policy).requires_grad_(False).eval() for _ in range(size)]
        self.device = next(policy.parameters()).device
        self.latest_probability = latest_probability
        self.available = list(range(size))
        self.retiring: int | None = None
        self.pending: dict[str, torch.Tensor] | None = None
        self.next_slot = 0
        self.latest_slot = 0
        self.snapshots = 0
        if checkpoint is not None:
            states = checkpoint["models"]
            if len(states) != size:
                raise ValueError("checkpoint pool size does not match --opponent-pool-size")
            for model, state in zip(self.models, states, strict=True):
                model.load_state_dict(state)
            self.next_slot = int(checkpoint.get("next_slot", 0)) % size
            self.latest_slot = int(checkpoint.get("latest_slot", 0)) % size
            self.snapshots = int(checkpoint.get("snapshots", 0))
        self._refresh_available()

    def initial_slots(self, num_envs: int) -> torch.Tensor:
        return self._sample_slots((num_envs,))

    @torch.no_grad()
    def actions(
        self,
        state: LocalState,
        slots: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty((*state.features.shape[:-1], 4), device=state.features.device)
        for slot, model in enumerate(self.models):
            selected = slots == slot
            slot_memory = memory[selected] if memory is not None else None
            actions, new_memory = model.actor_step(
                select_state(state, selected), slot_memory
            )
            output[selected] = actions
            if memory is not None:
                memory[selected] = new_memory
        return output

    def resample_finished(self, slots: torch.Tensor, finished: torch.Tensor) -> None:
        slots.copy_(torch.where(finished, self._sample_slots(slots.shape), slots))

    def _refresh_available(self) -> None:
        self.available_tensor = torch.tensor(self.available, device=self.device)
        historical = [slot for slot in self.available if slot != self.latest_slot]
        self.historical_tensor = torch.tensor(historical, device=self.device)

    def _sample_slots(self, shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
        uniform = self.available_tensor[
            torch.randint(len(self.available), shape, device=self.device)
        ]
        if (
            self.snapshots == 0
            or self.latest_slot not in self.available
            or not len(self.historical_tensor)
        ):
            return uniform
        historical = self.historical_tensor[
            torch.randint(len(self.historical_tensor), shape, device=self.device)
        ]
        newest = torch.full(shape, self.latest_slot, device=self.device)
        choose_newest = torch.rand(shape, device=self.device) < self.latest_probability
        return torch.where(choose_newest, newest, historical)

    @torch.no_grad()
    def advance(
        self, policy: Policy, slots: torch.Tensor, request_snapshot: bool
    ) -> dict[str, int]:
        if self.retiring is not None and not bool((slots == self.retiring).any()):
            self.models[self.retiring].load_state_dict(self.pending)
            self.latest_slot = self.retiring
            self.available.append(self.retiring)
            self.available.sort()
            self.retiring = None
            self.pending = None
            self.snapshots += 1
            self._refresh_available()
        if request_snapshot and self.retiring is None:
            self.retiring = self.next_slot
            self.next_slot = (self.next_slot + 1) % len(self.models)
            self.pending = {
                name: value.detach().clone() for name, value in policy.state_dict().items()
            }
            self.available.remove(self.retiring)
            self._refresh_available()
        return {
            "opponent_snapshots": self.snapshots,
            "opponent_pool_active": len(self.available),
            "opponent_pool_draining": int(self.retiring is not None),
        }

    def checkpoint(self) -> dict[str, object]:
        states = [model.state_dict() for model in self.models]
        latest_slot = self.latest_slot
        snapshots = self.snapshots
        if self.retiring is not None:
            states[self.retiring] = self.pending
            latest_slot = self.retiring
            snapshots += 1
        return {
            "models": states,
            "next_slot": self.next_slot,
            "latest_slot": latest_slot,
            "snapshots": snapshots,
        }

    @property
    def latest(self) -> Policy:
        return self.models[self.latest_slot]


class RolloutReplay:
    """Asynchronously capture one already-simulated training environment."""

    def __init__(
        self,
        env: RLEnv,
        length: int,
        action_repeat: int = 1,
        env_index: int = 0,
        opponent_mode: str = "self",
        learner_team: int = 0,
    ) -> None:
        if learner_team not in (0, 1):
            raise ValueError("learner team must be zero or one")
        self.env = env
        self.env_index = env_index
        self.action_repeat = action_repeat
        self.opponent_mode = opponent_mode
        self.learner_team = learner_team
        self.frames = length * action_repeat + 1
        soldiers = env.sim.num_soldiers
        device = env.device
        self.position = torch.empty((self.frames, soldiers, 2), device=device)
        self.angle = torch.empty((self.frames, soldiers), device=device)
        self.health = torch.empty((self.frames, soldiers), device=device)
        self.owners = torch.empty(
            (self.frames, TERRITORY_CELLS), dtype=torch.int32, device=device
        )
        self.control = torch.empty(
            (self.frames, TERRITORY_CELLS, 2), dtype=torch.float32, device=device
        )
        self.next_frame = 0

    def retarget(self, env_index: int, opponent_mode: str, learner_team: int) -> None:
        """Point the next capture at a different environment and matchup label."""
        if learner_team not in (0, 1):
            raise ValueError("learner team must be zero or one")
        self.env_index = env_index
        self.opponent_mode = opponent_mode
        self.learner_team = learner_team

    def start(self) -> None:
        self.next_frame = 0
        self.learner_mode = int(self.env.mode[self.env_index, self.learner_team])
        self.capture()

    def capture(self) -> None:
        sim = self.env.sim
        envs, soldiers = sim.num_envs, sim.num_soldiers
        frame = self.next_frame
        self.position[frame].copy_(
            wp.to_torch(sim.position).view(envs, soldiers, 2)[self.env_index]
        )
        self.angle[frame].copy_(
            wp.to_torch(sim.attack_angle).view(envs, soldiers)[self.env_index]
        )
        self.health[frame].copy_(
            wp.to_torch(sim.health).view(envs, soldiers)[self.env_index]
        )
        self.owners[frame].copy_(
            wp.to_torch(sim.territory_owner).view(envs, TERRITORY_CELLS)[self.env_index]
        )
        self.control[frame].copy_(
            wp.to_torch(sim.control_share).view(envs, TERRITORY_CELLS, 2)[
                self.env_index
            ]
        )
        self.next_frame += 1

    def replay(
        self, done: torch.Tensor, winner: torch.Tensor, update: int
    ) -> dict[str, object]:
        endings = torch.nonzero(done[:, self.env_index], as_tuple=False).flatten()
        terminal_step = int(endings[0]) if len(endings) else None
        frame_count = (
            (terminal_step + 1) * self.action_repeat + 1
            if terminal_step is not None
            else self.next_frame
        )
        position = self.position[:frame_count].cpu().numpy()
        angle = self.angle[:frame_count].cpu().numpy()
        health = self.health[:frame_count].cpu().numpy()
        owners = self.owners[:frame_count].cpu().numpy().astype(np.int8)
        control = self.control[:frame_count].cpu().numpy()
        sim = self.env.sim
        team = (
            wp.to_torch(sim.team)
            .view(sim.num_envs, sim.num_soldiers)[self.env_index]
            .cpu()
            .tolist()
        )
        config = self.env.config
        replay: dict[str, object] = {
            "config": {
                "world_width": config.world_width,
                "world_height": config.world_height,
                "soldier_radius": config.soldier_radius,
                "initial_health": config.initial_health,
                "decision_dt": config.decision_dt,
                "river_width": config.river_width,
                "bridge_width": config.bridge_width,
                "arena_shape": "hex",
                "territory_radius": TERRITORY_RADIUS,
                "territory_cells": TERRITORY_CELLS,
                "strongpoint_cells": STRONGPOINT_CELLS.tolist(),
                "strongpoint_weight": STRONGPOINT_WEIGHT,
            },
            "team": team,
            "frames": [
                {
                    "p": position[i],
                    "a": angle[i],
                    "h": health[i],
                    "o": owners[i],
                    "c": control[i],
                }
                for i in range(frame_count)
            ],
            "update": update,
            "opponent_mode": self.opponent_mode,
            "learner_team": self.learner_team,
            "learner_mode": getattr(self, "learner_mode", 0),
            "statistics": {
                "decision_steps": frame_count - 1,
                "simulated_seconds": (frame_count - 1) * config.decision_dt,
            },
        }
        if terminal_step is not None:
            replay["winner"] = int(winner[terminal_step, self.env_index])
        return replay


def write_rollout_replay(run_dir: Path, replay: dict[str, object]) -> None:
    """Write immutable history, then atomically point replay.html at it."""
    from viewer import write_replay

    update = int(replay["update"])
    history = run_dir / "replays" / f"update-{update}.html"
    history.parent.mkdir(parents=True, exist_ok=True)
    temporary = history.with_suffix(".html.tmp")
    write_replay(replay, temporary)
    os.replace(temporary, history)
    latest = run_dir / "replay.html"
    temporary_latest = run_dir / ".replay.html.tmp"
    temporary_latest.unlink(missing_ok=True)
    os.link(history, temporary_latest)
    os.replace(temporary_latest, latest)


def compile_policies(policy: Policy, pool: OpponentPool | None, mode: str) -> None:
    """Compile hot neural callables without wrapping checkpointed modules."""
    policy.act = torch.compile(policy.act, mode=mode)
    policy.value = torch.compile(policy.value, mode=mode)
    policy.evaluate_actions = torch.compile(policy.evaluate_actions, mode=mode)
    if pool is not None:
        for model in pool.models:
            model.actor_step = torch.compile(model.actor_step, mode=mode, dynamic=True)


def reward_from_facts(
    output: torch.Tensor,
    facts: dict[str, torch.Tensor],
    *,
    accumulate: bool = False,
    scale: float = 1.0,
) -> None:
    """Write the zero-sum weighted control advantage held this step.

    Under presence control the score is the time integral of held advantage,
    so the per-step reward is the current level, not the change. With
    ``scale = 1 / maximum_decision_steps`` the undiscounted episode return
    equals the episode's time-averaged control advantage in ``[-1, 1]``.
    """
    territory = facts["territory"]
    advantage = (territory[:, 0] - territory[:, 1]) * scale
    if accumulate:
        output[:, 0].add_(advantage)
    else:
        output[:, 0].copy_(advantage)
    torch.neg(output[:, 0], out=output[:, 1])


class ModeDiscriminator(nn.Module):
    """Predict a team's episode mode from its per-step behavior summary."""

    def __init__(self, input_size: int = SUMMARY_SIZE, mode_count: int = 16, hidden_size: int = 64):
        super().__init__()
        if mode_count < 2:
            raise ValueError("mode discrimination needs at least two modes")
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, mode_count),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.net(summary)


def mode_probe_update(
    discriminator: ModeDiscriminator,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    learner_teams: torch.Tensor,
    minibatch_size: int = 8192,
) -> dict[str, float]:
    """Fit the discriminator on learner-team summaries; report identifiability."""
    mask = learner_teams.unsqueeze(0).expand_as(rollout["mode"])
    inputs = rollout["summary"][mask]
    labels = rollout["mode"][mask]
    order = torch.randperm(len(labels), device=labels.device)
    totals = torch.zeros(2, device=labels.device)
    batches = 0
    for start in range(0, len(order), minibatch_size):
        batch = order[start : start + minibatch_size]
        logits = discriminator(inputs[batch])
        loss = F.cross_entropy(logits, labels[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = (logits.argmax(-1) == labels[batch]).float().mean()
            totals.add_(torch.stack((loss.detach(), accuracy)))
        batches += 1
    loss_mean, accuracy_mean = (totals / max(1, batches)).tolist()
    return {"mode_probe_loss": loss_mean, "mode_probe_accuracy": accuracy_mean}


@torch.no_grad()
def add_mode_intrinsic_reward(
    rollout: dict[str, torch.Tensor],
    discriminator: ModeDiscriminator,
    learner_teams: torch.Tensor,
    beta: float,
    mode_count: int,
) -> dict[str, float]:
    """Add the variational diversity bonus beta * (log q(z|summary) - log p(z))."""
    mask = learner_teams.unsqueeze(0).expand_as(rollout["mode"])
    log_prob = discriminator(rollout["summary"]).log_softmax(-1)
    gathered = log_prob.gather(-1, rollout["mode"].unsqueeze(-1)).squeeze(-1)
    bonus = beta * (gathered + math.log(mode_count))
    extrinsic = rollout["reward"][mask].abs().mean()
    rollout["reward"].add_(bonus * mask.to(bonus.dtype))
    intrinsic = bonus[mask]
    return {
        "mode_intrinsic_mean": float(intrinsic.mean()),
        "mode_intrinsic_fraction": float(
            intrinsic.abs().mean() / extrinsic.clamp_min(1e-8)
        ),
        "mode_intrinsic_beta": beta,
    }


@torch.no_grad()
def mode_outcome_metrics(
    rollout: dict[str, torch.Tensor], learner_teams: torch.Tensor, mode_count: int
) -> tuple[dict[str, float], torch.Tensor | None, torch.Tensor | None]:
    """Group finished episodes by the learner's mode; also feed the bandit."""
    terminal = rollout["done"]
    if not bool(terminal.any()):
        return {}, None, None
    length, envs = terminal.shape
    learner_index = learner_teams.long().argmax(-1)
    gather_index = learner_index.view(1, envs, 1).expand(length, -1, -1)
    modes = rollout["mode"].gather(-1, gather_index).squeeze(-1)[terminal]
    territory = rollout["territory"]
    learner_territory = territory.gather(-1, gather_index).squeeze(-1)
    advantage = (2 * learner_territory - territory.sum(-1))[terminal]
    win = (
        rollout["winner"] == learner_index.view(1, envs).to(rollout["winner"].dtype)
    ).float()[terminal]
    counts = torch.bincount(modes, minlength=mode_count).float()
    scale = counts.clamp_min(1)
    advantage_by_mode = (
        torch.zeros(mode_count, device=advantage.device).scatter_add_(0, modes, advantage)
        / scale
    )
    win_by_mode = (
        torch.zeros(mode_count, device=win.device).scatter_add_(0, modes, win) / scale
    )
    present = counts > 0
    metrics = {
        "mode_episodes_min": float(counts.min()),
        "mode_advantage_spread": float(
            advantage_by_mode[present].max() - advantage_by_mode[present].min()
        ),
        "mode_win_best": float(win_by_mode[present].max()),
        "mode_win_worst": float(win_by_mode[present].min()),
    }
    return metrics, advantage_by_mode, present


@torch.no_grad()
def mode_action_sensitivity(
    policy: Policy,
    rollout: dict[str, torch.Tensor],
    learner_teams: torch.Tensor,
    mode_count: int,
    envs: int = 8,
) -> dict[str, float]:
    """Mean action change when every team's mode is rerolled; zero means a dead latent."""
    state = rollout_state(rollout, 0)
    subset = LocalState(*(tensor[:envs] for tensor in state))
    rerolled = subset._replace(mode=(subset.mode + 1) % mode_count)
    baseline = policy.actor_actions(subset, deterministic=True)
    shifted = policy.actor_actions(rerolled, deterministic=True)
    agents = subset.alive.bool() & learner_teams[: subset.alive.shape[0]].unsqueeze(-1)
    if not bool(agents.any()):
        return {}
    delta = (baseline - shifted).abs().mean(-1)[agents].mean()
    return {"mode_action_delta": float(delta)}


def combat_metrics(
    rollout: dict[str, torch.Tensor], learner_teams: torch.Tensor
) -> dict[str, float]:
    """Damage and survival facts the territory reward never shows directly."""
    length, envs = rollout["done"].shape
    episode_damage = rollout["damage"].sum(0)
    learner_index = learner_teams.long().argmax(-1)
    gather_index = learner_index.view(1, envs, 1).expand(length, -1, -1)
    alive_fraction = rollout["alive"].float().mean(-1)
    learner_alive = alive_fraction.gather(-1, gather_index).squeeze(-1)
    opponent_alive = alive_fraction.gather(-1, 1 - gather_index).squeeze(-1)
    terminal = rollout["done"]
    if bool(terminal.any()):
        final_learner = learner_alive[terminal].mean()
        final_opponent = opponent_alive[terminal].mean()
    else:
        final_learner = learner_alive[-1].mean()
        final_opponent = opponent_alive[-1].mean()
    values = torch.stack(
        (
            episode_damage[~learner_teams].mean(),
            episode_damage[learner_teams].mean(),
            final_learner,
            final_opponent,
        )
    ).tolist()
    return dict(
        zip(
            (
                "episode_damage_dealt",
                "episode_damage_taken",
                "learner_final_alive_fraction",
                "opponent_final_alive_fraction",
            ),
            values,
        )
    )


class AnchorEvaluator:
    """Absolute-skill benchmark against the scripted nearest-enemy charger.

    Training metrics against a self-play pool sit near 50% by symmetry, so
    this periodically plays full deterministic episodes against the fixed
    charger, one batch entry per (episode mode, side) combination.
    """

    def __init__(
        self,
        config: Config,
        device: str,
        mode_count: int,
        episodes_per_mode: int = 4,
        action_repeat: int = 1,
    ) -> None:
        if episodes_per_mode < 2 or episodes_per_mode % 2:
            raise ValueError("--anchor-episodes must be a positive even number")
        self.mode_count = max(1, mode_count)
        self.action_repeat = action_repeat
        num_envs = self.mode_count * episodes_per_mode
        self.env = RLEnv(config, num_envs, device, mode_count=self.mode_count)
        index = torch.arange(num_envs, device=self.env.device)
        self.learner_index = index.remainder(2)
        learner_teams = torch.zeros((num_envs, 2), dtype=torch.bool, device=self.env.device)
        learner_teams[index, self.learner_index] = True
        self.learner_mask = learner_teams[:, :, None, None]
        self.opponent = NearestEnemyOpponent(self.env, learner_teams)
        self.learner_modes = (index // 2).remainder(self.mode_count)
        assigned = torch.zeros((num_envs, 2), dtype=torch.long, device=self.env.device)
        assigned[learner_teams] = self.learner_modes
        self.assigned_modes = assigned
        self.decisions = config.maximum_decision_steps // action_repeat

    @torch.no_grad()
    def run(self, policy: Policy) -> dict[str, float]:
        env = self.env
        env.reset()
        env.mode.copy_(self.assigned_modes)
        state = env.state()
        facts = env.facts
        memory = policy.initial_memory(state)
        for _ in range(self.decisions):
            learner, memory = policy.actor_step(state, memory, deterministic=True)
            actions = torch.where(self.learner_mask, learner, self.opponent.act())
            for repeat in range(self.action_repeat):
                state, facts = env.step(
                    actions, observe=repeat == self.action_repeat - 1
                )
        winner = facts["winner"]
        territory = facts["territory"]
        learner_territory = territory.gather(
            -1, self.learner_index.unsqueeze(-1)
        ).squeeze(-1)
        advantage = 2 * learner_territory - territory.sum(-1)
        win = (winner == self.learner_index.to(winner.dtype)).float()
        metrics = {
            "anchor_win_rate": float(win.mean()),
            "anchor_draw_rate": float((winner == -1).float().mean()),
            "anchor_territory_advantage": float(advantage.mean()),
        }
        if self.mode_count > 1:
            counts = torch.bincount(
                self.learner_modes, minlength=self.mode_count
            ).float().clamp_min(1)
            win_by_mode = (
                torch.zeros(self.mode_count, device=win.device).scatter_add_(
                    0, self.learner_modes, win
                )
                / counts
            )
            metrics["anchor_mode_win_best"] = float(win_by_mode.max())
            metrics["anchor_mode_win_worst"] = float(win_by_mode.min())
        return metrics


def make_rollout(
    length: int,
    state: LocalState,
    memory_size: int = 0,
    bptt_window: int = 1,
) -> dict[str, torch.Tensor]:
    envs, teams, soldiers, features = state.features.shape
    device = state.features.device
    memory = (
        {
            "memory": torch.zeros(
                (length // bptt_window, envs, teams, soldiers, memory_size),
                dtype=torch.float32,
                device=device,
            )
        }
        if memory_size
        else {}
    )
    return {
        **memory,
        "features": torch.empty(
            (length, envs, teams, soldiers, features),
            dtype=state.features.dtype,
            device=device,
        ),
        "cells": torch.empty(
            (length, envs, teams, soldiers), dtype=torch.int16, device=device
        ),
        "alive": torch.empty(
            (length, envs, teams, soldiers), dtype=torch.bool, device=device
        ),
        "owners": torch.empty((length, envs, state.owners.shape[-1]), dtype=torch.int8, device=device),
        "mirror_y": torch.empty((length, envs), dtype=torch.int8, device=device),
        "mode": torch.empty((length, envs, teams), dtype=torch.long, device=device),
        "neighbors": torch.empty(
            (length, envs, teams, soldiers, state.neighbors.shape[-1]),
            dtype=torch.int16,
            device=device,
        ),
        "summary": torch.empty(
            (length, envs, teams, SUMMARY_SIZE), dtype=torch.float32, device=device
        ),
        "damage": torch.empty((length, envs, teams), dtype=torch.float32, device=device),
        "actions": torch.empty(
            (length, envs, teams, soldiers, 4), dtype=torch.float32, device=device
        ),
        "log_prob": torch.empty(
            (length, envs, teams, soldiers), dtype=torch.float32, device=device
        ),
        "value": torch.empty(
            (length, envs, teams, soldiers), dtype=torch.float32, device=device
        ),
        "reward": torch.empty((length, envs, teams), dtype=torch.float32, device=device),
        "done": torch.empty((length, envs), dtype=torch.bool, device=device),
        "winner": torch.empty((length, envs), dtype=torch.int32, device=device),
        "territory": torch.empty((length, envs, teams), dtype=torch.float32, device=device),
        "advantage": torch.empty(
            (length, envs, teams, soldiers), dtype=torch.float32, device=device
        ),
    }


def rollout_state(rollout: dict[str, torch.Tensor], index) -> LocalState:
    return LocalState(*(rollout[name][index] for name in STATE_KEYS))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def scripted_environment_mask(
    num_envs: int, fraction: float, device: torch.device | str
) -> torch.Tensor:
    """Deterministically mark a scripted-opponent slice, balanced across sides.

    Environments alternate learner side by index, so the pattern picks one
    even and one odd index per period to keep the scripted slice side-fair.
    """
    if not 0 < fraction < 1:
        raise ValueError("scripted fraction must be strictly between zero and one")
    k = max(2, round(1 / fraction))
    phase = torch.arange(num_envs, device=device) % (2 * k)
    return (phase == 0) | (phase == k + 1)


@torch.no_grad()
def collect_rollout(
    env: RLEnv,
    policy: Policy,
    rollout: dict[str, torch.Tensor],
    state: LocalState,
    learner_teams: torch.Tensor,
    pool: OpponentPool | None,
    opponent_slots: torch.Tensor | None,
    action_repeat: int = 1,
    replay: RolloutReplay | None = None,
    nearest_opponent: NearestEnemyOpponent | None = None,
    scripted_envs: torch.Tensor | None = None,
    learner_memory: torch.Tensor | None = None,
    opponent_memory: torch.Tensor | None = None,
    bptt_window: int = 1,
) -> tuple[LocalState, torch.Tensor, float]:
    synchronize(state.features.device)
    started = time.perf_counter()
    learner_mask = learner_teams[:, :, None, None]
    reward_scale = 1.0 / env.config.maximum_decision_steps
    step_damage = torch.zeros_like(rollout["damage"][0])
    step_territory_delta = torch.zeros_like(step_damage)
    if replay is not None:
        replay.start()
    for step in range(rollout["features"].shape[0]):
        for name, tensor in zip(STATE_KEYS, state, strict=True):
            rollout[name][step].copy_(tensor)
        if learner_memory is not None and step % bptt_window == 0:
            rollout["memory"][step // bptt_window].copy_(learner_memory)
        learner_actions, log_prob, value, new_memory = policy.act(state, learner_memory)
        if learner_memory is not None:
            learner_memory.copy_(new_memory)
        if pool is not None and nearest_opponent is not None:
            opponent_actions = torch.where(
                scripted_envs[:, None, None, None],
                nearest_opponent.act(),
                pool.actions(state, opponent_slots, opponent_memory),
            )
            actions = torch.where(learner_mask, learner_actions, opponent_actions)
        elif pool is not None:
            actions = torch.where(
                learner_mask,
                learner_actions,
                pool.actions(state, opponent_slots, opponent_memory),
            )
        elif nearest_opponent is not None:
            actions = torch.where(
                learner_mask, learner_actions, nearest_opponent.act()
            )
        else:
            actions = learner_actions
        rollout["actions"][step].copy_(actions)
        rollout["log_prob"][step].copy_(log_prob)
        rollout["value"][step].copy_(value)
        for repeat in range(action_repeat):
            state, facts = env.step(
                actions,
                observe=repeat == action_repeat - 1,
                before_reset=replay.capture if replay is not None else None,
            )
            reward_from_facts(
                rollout["reward"][step],
                facts,
                accumulate=repeat > 0,
                scale=reward_scale,
            )
            if repeat == 0:
                rollout["done"][step].copy_(facts["done"])
                rollout["winner"][step].copy_(facts["winner"])
                step_damage.copy_(facts["damage_taken"])
                step_territory_delta.copy_(facts["territory_delta"])
            else:
                finished = facts["done"].bool()
                rollout["done"][step].logical_or_(finished)
                rollout["winner"][step].copy_(
                    torch.where(finished, facts["winner"], rollout["winner"][step])
                )
                step_damage.add_(facts["damage_taken"])
                step_territory_delta.add_(facts["territory_delta"])
            rollout["territory"][step].copy_(facts["territory"])
            if pool is not None:
                pool.resample_finished(opponent_slots, facts["done"].bool())
        rollout["damage"][step].copy_(step_damage)
        rollout["summary"][step].copy_(
            env.team_summary(step_damage, step_territory_delta)
        )
        finished = rollout["done"][step]
        if bool(finished.any()):
            if learner_memory is not None:
                learner_memory[finished] = 0.0
            if opponent_memory is not None:
                opponent_memory[finished] = 0.0
    bootstrap = policy.value(state, learner_memory)
    synchronize(state.features.device)
    return state, bootstrap, time.perf_counter() - started


@torch.no_grad()
def compute_gae(
    rollout: dict[str, torch.Tensor], bootstrap: torch.Tensor, gamma: float, gae_lambda: float
) -> None:
    advantage = torch.zeros_like(bootstrap)
    for step in range(rollout["reward"].shape[0] - 1, -1, -1):
        next_value = bootstrap if step == rollout["reward"].shape[0] - 1 else rollout["value"][step + 1]
        continuing = (~rollout["done"][step]).to(bootstrap.dtype).view(-1, 1, 1)
        reward = rollout["reward"][step].unsqueeze(-1)
        delta = reward + gamma * next_value * continuing - rollout["value"][step]
        advantage = delta + gamma * gae_lambda * continuing * advantage
        rollout["advantage"][step].copy_(advantage)


def _ppo_losses(
    ratio: torch.Tensor,
    log_ratio: torch.Tensor,
    agent_advantage: torch.Tensor,
    value: torch.Tensor,
    value_target: torch.Tensor,
    entropy: torch.Tensor,
    clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_loss = -torch.minimum(
        ratio * agent_advantage,
        ratio.clamp(1 - clip, 1 + clip) * agent_advantage,
    ).mean()
    value_loss = 0.5 * (value - value_target).square().mean()
    entropy_mean = entropy.mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
    with torch.no_grad():
        approximate_kl = (ratio - 1 - log_ratio).mean()
    return loss, policy_loss, value_loss, entropy_mean, approximate_kl


def ppo_update_sequences(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    learner_teams: torch.Tensor,
    epochs: int,
    minibatches: int,
    clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    maximum_gradient_norm: float,
    target_kl: float,
    bptt_window: int,
) -> dict[str, float]:
    """Recurrent PPO: minibatches of (window, env) chunks, backprop through time."""
    length, envs = rollout["done"].shape
    windows = length // bptt_window
    chunks = windows * envs
    minibatch_size = max(1, (chunks + minibatches - 1) // minibatches)
    device = rollout["features"].device
    alive_mask = rollout["alive"].bool() & learner_teams[None, :, :, None]
    selected_advantage = rollout["advantage"][alive_mask]
    advantage_mean = selected_advantage.mean()
    advantage_scale = selected_advantage.std(unbiased=False).clamp_min(1e-8)

    totals = torch.zeros(6, device=device)
    updates = 0
    stopped_early = False
    for _ in range(epochs):
        epoch_kl = torch.zeros((), device=device)
        epoch_updates = 0
        order = torch.randperm(chunks, device=device)
        for start in range(0, chunks, minibatch_size):
            batch = order[start : start + minibatch_size]
            window_index = batch // envs
            env_index = batch % envs
            memory = rollout["memory"][window_index, env_index].clone()
            new_parts, entropy_parts, value_parts = [], [], []
            old_log_parts, old_advantage_parts, old_value_parts = [], [], []
            for offset in range(bptt_window):
                step = window_index * bptt_window + offset
                state = LocalState(
                    *(rollout[name][step, env_index] for name in STATE_KEYS)
                )
                new_log_prob, entropy, value, memory = policy.evaluate_actions(
                    state, rollout["actions"][step, env_index], memory
                )
                agents = state.alive.bool() & learner_teams[env_index].unsqueeze(-1)
                new_parts.append(new_log_prob[agents])
                entropy_parts.append(entropy[agents])
                value_parts.append(value[agents])
                old_log_parts.append(rollout["log_prob"][step, env_index][agents])
                old_advantage_parts.append(rollout["advantage"][step, env_index][agents])
                old_value_parts.append(rollout["value"][step, env_index][agents])
                finished = rollout["done"][step, env_index]
                if bool(finished.any()):
                    memory = memory * (~finished).view(-1, 1, 1, 1).to(memory.dtype)
            log_ratio = torch.cat(new_parts) - torch.cat(old_log_parts)
            ratio = log_ratio.exp()
            old_advantage = torch.cat(old_advantage_parts)
            agent_advantage = (old_advantage - advantage_mean) / advantage_scale
            value_target = old_advantage + torch.cat(old_value_parts)
            loss, policy_loss, value_loss, entropy_mean, approximate_kl = _ppo_losses(
                ratio,
                log_ratio,
                agent_advantage,
                torch.cat(value_parts),
                value_target,
                torch.cat(entropy_parts),
                clip,
                value_coefficient,
                entropy_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), maximum_gradient_norm)
            optimizer.step()
            with torch.no_grad():
                epoch_kl.add_(approximate_kl)
                totals.add_(
                    torch.stack(
                        (
                            policy_loss,
                            value_loss,
                            entropy_mean,
                            approximate_kl,
                            ((ratio - 1).abs() > clip).float().mean(),
                            gradient_norm,
                        )
                    )
                )
            updates += 1
            epoch_updates += 1
        if target_kl > 0 and float(epoch_kl / epoch_updates) > target_kl:
            stopped_early = True
            break
    names = ("policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction", "gradient_norm")
    metrics = dict(zip(names, (totals / updates).tolist(), strict=True))
    metrics.update(optimizer_steps=updates, early_stopped=float(stopped_early))
    return metrics


def ppo_update(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    learner_teams: torch.Tensor,
    epochs: int,
    minibatches: int,
    clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    maximum_gradient_norm: float,
    target_kl: float,
    bptt_window: int = 1,
) -> dict[str, float]:
    if "memory" in rollout:
        return ppo_update_sequences(
            policy,
            optimizer,
            rollout,
            learner_teams,
            epochs,
            minibatches,
            clip,
            value_coefficient,
            entropy_coefficient,
            maximum_gradient_norm,
            target_kl,
            bptt_window,
        )
    length, envs = rollout["done"].shape
    batch_size = length * envs
    minibatch_size = max(1, (batch_size + minibatches - 1) // minibatches)
    flat = {name: tensor.flatten(0, 1) for name, tensor in rollout.items()}
    team_mask = learner_teams.unsqueeze(0).expand(length, -1, -1).flatten(0, 1)
    learner_agents = flat["alive"].bool() & team_mask.unsqueeze(-1)
    selected_advantage = flat["advantage"][learner_agents]
    advantage_mean = selected_advantage.mean()
    advantage_scale = selected_advantage.std(unbiased=False).clamp_min(1e-8)

    totals = torch.zeros(6, device=flat["features"].device)
    updates = 0
    stopped_early = False
    for _ in range(epochs):
        epoch_kl = torch.zeros((), device=totals.device)
        epoch_updates = 0
        order = torch.randperm(batch_size, device=totals.device)
        for start in range(0, batch_size, minibatch_size):
            batch = order[start : start + minibatch_size]
            state = LocalState(*(flat[name][batch] for name in STATE_KEYS))
            new_log_prob, entropy, value, _ = policy.evaluate_actions(state, flat["actions"][batch])
            agents = state.alive.bool() & team_mask[batch].unsqueeze(-1)
            log_ratio = new_log_prob[agents] - flat["log_prob"][batch][agents]
            ratio = log_ratio.exp()
            with torch.no_grad():
                approximate_kl = (ratio - 1 - log_ratio).mean()
            old_advantage = flat["advantage"][batch][agents]
            agent_advantage = (old_advantage - advantage_mean) / advantage_scale
            policy_loss = -torch.minimum(
                ratio * agent_advantage,
                ratio.clamp(1 - clip, 1 + clip) * agent_advantage,
            ).mean()
            value_target = old_advantage + flat["value"][batch][agents]
            value_loss = 0.5 * (value[agents] - value_target).square().mean()
            entropy_mean = entropy[agents].mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), maximum_gradient_norm)
            optimizer.step()

            with torch.no_grad():
                epoch_kl.add_(approximate_kl)
                totals.add_(
                    torch.stack(
                        (
                            policy_loss,
                            value_loss,
                            entropy_mean,
                            approximate_kl,
                            ((ratio - 1).abs() > clip).float().mean(),
                            gradient_norm,
                        )
                    )
                )
            updates += 1
            epoch_updates += 1
        if target_kl > 0 and float(epoch_kl / epoch_updates) > target_kl:
            stopped_early = True
            break
    names = ("policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction", "gradient_norm")
    metrics = dict(zip(names, (totals / updates).tolist(), strict=True))
    metrics.update(optimizer_steps=updates, early_stopped=float(stopped_early))
    return metrics


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


def write_evaluation_replay(
    path: Path,
    policy: Policy,
    config: Config,
    device: str,
    opponent: Policy | str | None = None,
    update: int | None = None,
    random_facing_seed: int | None = None,
    action_repeat: int = 1,
    learner_mode: int | None = None,
) -> None:
    from simulator_gpu import GpuSimulator
    from viewer import local_march_policy, nearest_enemy_policy, record_episode, write_replay

    scripted_policies = {"nearest": nearest_enemy_policy, "local_march": local_march_policy}
    if isinstance(opponent, str) and opponent not in scripted_policies:
        raise ValueError(f"unknown scripted opponent: {opponent}")
    if opponent is not None and not isinstance(opponent, (Policy, str)):
        raise ValueError("opponent must be a Policy, scripted policy name, or None")
    if action_repeat < 1:
        raise ValueError("action_repeat must be positive")

    simulator = GpuSimulator(config, device=device)
    env = RLEnv(simulator=simulator, mode_count=max(1, getattr(policy, "mode_count", 0)))
    env.mode.zero_()
    if learner_mode is not None:
        if not 0 <= learner_mode < env.mode_count:
            raise ValueError("learner_mode is outside this policy's mode range")
        env.mode[0, 0] = learner_mode
    if random_facing_seed is not None:
        generator = np.random.default_rng(random_facing_seed)
        simulator.reset(
            {
                "attack_angle": generator.uniform(
                    -np.pi, np.pi, config.soldier_count
                ).astype(np.float32)
            }
        )
    evaluation = Policy(**policy.model_kwargs).to(next(policy.parameters()).device)
    evaluation.load_state_dict(policy.state_dict())
    evaluation.use_bf16 = policy.use_bf16
    evaluation.eval()
    evaluation_opponent = None
    if isinstance(opponent, Policy):
        evaluation_opponent = Policy(**opponent.model_kwargs).to(
            next(opponent.parameters()).device
        )
        evaluation_opponent.load_state_dict(opponent.state_dict())
        evaluation_opponent.use_bf16 = opponent.use_bf16
        evaluation_opponent.eval()

    held_actions = None
    held_steps = 0
    held_memory = evaluation.initial_memory(env.state())
    held_opponent_memory = (
        evaluation_opponent.initial_memory(env.state())
        if evaluation_opponent is not None
        else None
    )

    def act(_snapshot, _config):
        nonlocal held_actions, held_steps, held_memory, held_opponent_memory
        if held_steps == 0:
            state = env.observe()
            with torch.no_grad():
                actions, _, _, held_memory = evaluation.act(
                    state, held_memory, deterministic=True
                )
                if evaluation_opponent is not None:
                    opponent_actions, held_opponent_memory = (
                        evaluation_opponent.actor_step(
                            state, held_opponent_memory, deterministic=True
                        )
                    )
                    actions[:, 1] = opponent_actions[:, 1]
                elif isinstance(opponent, str):
                    scripted = torch.as_tensor(
                        scripted_policies[opponent](_snapshot, _config), device=env.device
                    ).view(1, 2, config.soldiers_per_team, 4)
                    scripted[:, 1, :, 0].neg_()
                    scripted[:, 1, :, 2].neg_()
                    actions[:, 1].copy_(scripted[:, 1])
            held_actions = env.actions_to_sim(actions)
            if env.device.type == "cuda":
                torch.cuda.synchronize(env.device)
            held_steps = action_repeat
        held_steps -= 1
        return held_actions

    temporary = path.with_suffix(path.suffix + ".tmp")
    replay = record_episode(simulator, act)
    if update is not None:
        replay["update"] = update
    if isinstance(opponent, str):
        replay["opponent_mode"] = opponent
        replay["learner_team"] = 0
    if learner_mode is not None:
        replay["learner_mode"] = int(learner_mode)
    write_replay(replay, temporary)
    os.replace(temporary, path)


def episode_metrics(
    rollout: dict[str, torch.Tensor],
    learner_teams: torch.Tensor,
    env_mask: torch.Tensor | None = None,
    prefix: str = "",
) -> dict[str, int]:
    terminal = rollout["done"]
    if env_mask is not None:
        terminal = terminal & env_mask.unsqueeze(0)
    learner = learner_teams.long().argmax(-1).unsqueeze(0).expand_as(terminal)
    winner = rollout["winner"]
    return {
        f"{prefix}episodes": int(terminal.sum()),
        f"{prefix}wins": int((terminal & (winner == learner)).sum()),
        f"{prefix}draws": int((terminal & (winner == -1)).sum()),
        f"{prefix}losses": int((terminal & (winner >= 0) & (winner != learner)).sum()),
        f"{prefix}blue_wins": int((terminal & (winner == 0)).sum()),
        f"{prefix}red_wins": int((terminal & (winner == 1)).sum()),
    }


def territory_metrics(
    rollout: dict[str, torch.Tensor], learner_teams: torch.Tensor
) -> dict[str, float]:
    mask = learner_teams.unsqueeze(0).expand_as(rollout["territory"])
    learner = rollout["territory"][mask].float().mean()
    opponent = rollout["territory"][~mask].float().mean()
    blue = rollout["territory"][..., 0].float().mean()
    red = rollout["territory"][..., 1].float().mean()
    values = torch.stack((learner, opponent, learner - opponent, blue, red)).tolist()
    return dict(
        zip(
            (
                "learner_territory",
                "opponent_territory",
                "territory_advantage",
                "blue_territory",
                "red_territory",
            ),
            values,
        )
    )


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

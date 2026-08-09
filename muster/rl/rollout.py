"""Rollout collection, storage layout, and training replay capture."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from muster.rl.env import (
    SUMMARY_SIZE,
    LocalState,
    NearestEnemyOpponent,
    RLEnv,
)
from muster.rl.policy import Policy

if TYPE_CHECKING:
    from muster.rl.pool import OpponentPool
from muster.sim.constants import (
    ENTITY_RADIUS,
    STRONGPOINT_CELLS,
    STRONGPOINT_WEIGHT,
    TERRITORY_CELLS,
    TERRITORY_COLUMN_EXTENT,
    TERRITORY_COORDINATES,
)

STATE_KEYS = ("features", "cells", "alive", "owners", "mirror_y", "mode", "neighbors")

def select_state(state: LocalState, index) -> LocalState:
    return LocalState(*(tensor[index] for tensor in state))

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
                "territory_radius": TERRITORY_COLUMN_EXTENT,
                "territory_axial": TERRITORY_COORDINATES.tolist(),
                "hex_size": config.territory_tile_size,
                "territory_cells": TERRITORY_CELLS,
                "strongpoint_cells": STRONGPOINT_CELLS.tolist(),
                "strongpoint_weight": STRONGPOINT_WEIGHT,
                "entity_radius": ENTITY_RADIUS,
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
    from muster.viewer import write_replay

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

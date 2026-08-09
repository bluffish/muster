"""Absolute-skill benchmarks and evaluation replays."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from muster.rl.env import NearestEnemyOpponent, RLEnv
from muster.rl.policy import Policy
from muster.sim import Config

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
    from muster.sim.gpu import GpuSimulator
    from muster.viewer import local_march_policy, nearest_enemy_policy, record_episode, write_replay

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

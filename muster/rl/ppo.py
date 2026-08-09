"""GAE and the clipped-surrogate updates (flat and recurrent)."""

from __future__ import annotations

import torch
from torch import nn

from muster.rl.env import LocalState
from muster.rl.policy import Policy
from muster.rl.rollout import STATE_KEYS

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

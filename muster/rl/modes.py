"""Episode-mode diversity: discriminator, intrinsic bonus, diagnostics."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from muster.rl.env import SUMMARY_SIZE, LocalState
from muster.rl.policy import Policy
from muster.rl.rollout import STATE_KEYS, rollout_state

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

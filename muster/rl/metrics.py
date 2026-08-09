"""Episode, territory, and combat metrics."""

from __future__ import annotations

import torch

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

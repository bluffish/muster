"""Hand-coded benchmark policies."""

from __future__ import annotations

import numpy as np

from muster.sim import Config, strongpoint_world_centers, territory_indices
from muster.sim.constants import TERRITORY_COORDINATES
from muster.viewer.replay import Snapshot

def nearest_enemy_policy(state: Snapshot, config: Config) -> np.ndarray:
    """Move toward and aim at each living soldier's nearest living enemy.

    With no living enemy, occupy the nearest strongpoint instead of halting.
    """
    position = np.asarray(state["position"])
    team = np.asarray(state["team"])
    alive = np.asarray(state["alive"], dtype=bool)
    strongpoints = strongpoint_world_centers(config)
    actions = np.zeros((len(position), 4), np.float32)
    for soldier in np.flatnonzero(alive):
        enemies = np.flatnonzero(alive & (team != team[soldier]))
        if len(enemies) > 0:
            delta = position[enemies] - position[soldier]
            direction = delta[int(np.argmin(np.sum(delta * delta, axis=1)))]
            minimum_length = 1e-6
        else:
            delta = strongpoints - position[soldier]
            direction = delta[int(np.argmin(np.sum(delta * delta, axis=1)))]
            minimum_length = 1.0
        length = float(np.linalg.norm(direction))
        if length > minimum_length:
            direction = direction / length
            actions[soldier, :2] = direction
            actions[soldier, 2:] = direction
    return actions

def local_march_policy(
    state: Snapshot, config: Config, vision_radius: int = 2
) -> np.ndarray:
    """March forward, engaging only enemies visible within nearby hex tiles."""
    position = np.asarray(state["position"])
    team = np.asarray(state["team"])
    alive = np.asarray(state["alive"], dtype=bool)
    axial = TERRITORY_COORDINATES[territory_indices(position, config)]
    actions = np.zeros((len(position), 4), np.float32)
    actions[alive, 0] = np.where(team[alive] == 0, 1.0, -1.0)
    actions[alive, 2] = actions[alive, 0]
    for soldier in np.flatnonzero(alive):
        enemies = np.flatnonzero(alive & (team != team[soldier]))
        delta_hex = axial[enemies] - axial[soldier]
        distance = np.max(
            np.stack(
                (np.abs(delta_hex[:, 0]), np.abs(delta_hex[:, 1]), np.abs(delta_hex.sum(1)))
            ),
            axis=0,
        )
        visible = enemies[distance <= vision_radius]
        if len(visible) == 0:
            continue
        delta = position[visible] - position[soldier]
        direction = delta[np.argmin(np.sum(delta * delta, axis=1))]
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            direction = direction / length
            actions[soldier, :2] = direction
            actions[soldier, 2:] = direction
    return actions

"""Compact GPU-resident state and episode bookkeeping for reinforcement learning."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import NamedTuple

import torch
import warp as wp

from muster.sim import (
    TERRITORY_CELLS,
    TERRITORY_TOTAL_WEIGHT,
    Config,
    strongpoint_world_centers,
    territory_centers,
)
from muster.sim.gpu import ACTION_DTYPE, GpuSimulator

LOCAL_FEATURE_NAMES = (
    "tile_offset_x",
    "tile_offset_y",
    "velocity_x",
    "velocity_y",
    "facing_x",
    "facing_y",
    "health",
    "attack_recovery",
    "global_x",
    "global_y",
    "time_remaining",
    "score_advantage",
    "score_integral",
)
LOCAL_FEATURE_SIZE = len(LOCAL_FEATURE_NAMES)

SUMMARY_FEATURE_NAMES = (
    "centroid_x",
    "centroid_y",
    "dispersion",
    "velocity_x",
    "velocity_y",
    "speed",
    "alive_fraction",
    "health_fraction",
    "damage_dealt",
    "damage_taken",
    "territory",
    "territory_delta",
)
SUMMARY_SIZE = len(SUMMARY_FEATURE_NAMES)
SUMMARY_DAMAGE_SCALE = 50.0
SUMMARY_DELTA_SCALE = 50.0

from muster.sim.constants import ENTITY_NEIGHBORS, ENTITY_RADIUS  # noqa: E402


class LocalState(NamedTuple):
    """The losslessly stored input from which local actor views are built."""

    features: torch.Tensor
    cells: torch.Tensor
    alive: torch.Tensor
    owners: torch.Tensor
    mirror_y: torch.Tensor
    mode: torch.Tensor | None = None
    neighbors: torch.Tensor | None = None


def entity_neighbors(
    position: torch.Tensor,
    alive: torch.Tensor,
    soldiers_per_team: int,
    count: int = ENTITY_NEIGHBORS,
    radius: float = ENTITY_RADIUS,
) -> torch.Tensor:
    """Egocentric k-nearest living entities, indexed in each team's related space.

    ``position`` is ``(envs, 2 * soldiers_per_team, 2)`` world coordinates with
    team 0 first; ``alive`` matches its leading dimensions. Returns int16
    indices shaped ``(envs, 2, soldiers_per_team, count)`` where values in
    ``[0, soldiers_per_team)`` are allies, values in
    ``[soldiers_per_team, 2 * soldiers_per_team)`` are enemies, and ``-1``
    marks empty slots (dead, out of radius, or fewer than ``count`` in range).
    """
    total = 2 * soldiers_per_team
    if position.shape[-2] != total:
        raise ValueError("position must cover both teams' soldiers")
    unreachable = 4.0 * radius + 1.0
    distance = torch.cdist(position.float(), position.float())
    distance = distance.masked_fill(~alive.bool().unsqueeze(-2), unreachable)
    distance.diagonal(dim1=-2, dim2=-1).fill_(unreachable)
    count = min(count, total - 1)
    values, indices = distance.topk(count, dim=-1, largest=False)
    valid = values <= radius
    own = torch.where(valid, indices, indices.new_full((), -1))
    flipped = torch.where(
        valid, (indices + soldiers_per_team) % total, indices.new_full((), -1)
    )
    return torch.stack(
        (own[..., :soldiers_per_team, :], flipped[..., soldiers_per_team:, :]), dim=-3
    ).to(torch.int16)


@wp.struct
class _ObservationParams:
    num_soldiers: int
    speed_scale: float
    health_scale: float
    attack_recovery_ticks: int
    tile_size: float
    world_width: float
    world_height: float
    territory_cells: int
    territory_total_weight: float
    maximum_decision_steps: int


_NEIGHBOR_DISTANCES = wp.types.vector(length=ENTITY_NEIGHBORS, dtype=wp.float32)
_NEIGHBOR_ORDER = wp.types.vector(length=ENTITY_NEIGHBORS, dtype=wp.int32)


@wp.kernel
def _entity_neighbor_search(
    position: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    radius: float,
    count: int,
    neighbors: wp.array2d(dtype=wp.int32),
    p: _ObservationParams,
):
    """Per-soldier k-nearest living entities via in-register insertion sort.

    Matches ``entity_neighbors``: ascending distance, radius-capped, self
    excluded, dead excluded, related-space remap for team-one observers.
    """
    index = wp.tid()
    env = index // p.num_soldiers
    base = env * p.num_soldiers
    local = index - base
    soldiers_per_team = p.num_soldiers // 2
    me = position[index]
    distances = _NEIGHBOR_DISTANCES()
    order = _NEIGHBOR_ORDER()
    for slot in range(count):
        distances[slot] = 3.0e38
        order[slot] = -1
    limit = radius * radius
    for other_local in range(p.num_soldiers):
        if other_local == local:
            continue
        other = base + other_local
        if alive[other] == 0:
            continue
        delta = position[other] - me
        squared = wp.dot(delta, delta)
        if squared > limit or squared >= distances[count - 1]:
            continue
        slot = count - 1
        while slot > 0 and distances[slot - 1] > squared:
            distances[slot] = distances[slot - 1]
            order[slot] = order[slot - 1]
            slot -= 1
        distances[slot] = squared
        order[slot] = other_local
    observer_team = local // soldiers_per_team
    for slot in range(count):
        value = order[slot]
        if value >= 0 and observer_team == 1:
            value = (value + soldiers_per_team) % p.num_soldiers
        neighbors[index, slot] = value


@wp.kernel
def _clear_territory(territory: wp.array(dtype=float)):
    territory[wp.tid()] = 0.0


@wp.kernel
def _count_territory(
    share: wp.array(dtype=float),
    weights: wp.array(dtype=wp.int32),
    territory: wp.array(dtype=float),
    p: _ObservationParams,
):
    index = wp.tid()
    env = index // p.territory_cells
    cell = index - env * p.territory_cells
    weight = float(weights[cell]) / p.territory_total_weight
    wp.atomic_add(territory, env * 2, weight * share[index * 2])
    wp.atomic_add(territory, env * 2 + 1, weight * share[index * 2 + 1])


@wp.kernel
def _team_stats(
    team: wp.array(dtype=wp.int32),
    health: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    stats: wp.array2d(dtype=float),
    p: _ObservationParams,
):
    env = wp.tid()
    health_0 = float(0.0)
    health_1 = float(0.0)
    alive_0 = int(0)
    alive_1 = int(0)
    base = env * p.num_soldiers
    for local in range(p.num_soldiers):
        index = base + local
        if team[index] == 0:
            health_0 += health[index]
            alive_0 += alive[index]
        else:
            health_1 += health[index]
            alive_1 += alive[index]
    stats[env, 0] = health_0
    stats[env, 1] = health_1
    stats[env, 2] = float(alive_0)
    stats[env, 3] = float(alive_1)


@wp.func
def _store_feature(
    features: wp.array2d(dtype=wp.bfloat16), index: int, feature: int, value: float
):
    features[index, feature] = wp.bfloat16(value)


@wp.kernel
def _observe_local(
    team: wp.array(dtype=wp.int32),
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    angle: wp.array(dtype=float),
    health: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    attack_cooldown: wp.array(dtype=wp.int32),
    step_count: wp.array(dtype=wp.int32),
    territory: wp.array(dtype=float),
    advantage_integral: wp.array(dtype=float),
    vertical_sign: wp.array(dtype=float),
    territory_cell: wp.array(dtype=wp.int32),
    centers: wp.array(dtype=wp.vec2),
    features: wp.array2d(dtype=wp.bfloat16),
    p: _ObservationParams,
):
    index = wp.tid()
    for feature in range(LOCAL_FEATURE_SIZE):
        _store_feature(features, index, feature, 0.0)
    if alive[index] == 0:
        return

    env = index // p.num_soldiers
    x_sign = float(1.0) if team[index] == 0 else float(-1.0)
    y_sign = vertical_sign[env]
    center = centers[territory_cell[index]]
    offset = position[index] - center
    speed = wp.max(p.speed_scale, 1.0e-6)
    tile_size = wp.max(p.tile_size, 1.0e-6)
    _store_feature(features, index, 0, x_sign * offset[0] / tile_size)
    _store_feature(features, index, 1, y_sign * offset[1] / tile_size)
    _store_feature(features, index, 2, x_sign * velocity[index][0] / speed)
    _store_feature(features, index, 3, y_sign * velocity[index][1] / speed)
    _store_feature(features, index, 4, x_sign * wp.cos(angle[index]))
    _store_feature(features, index, 5, y_sign * wp.sin(angle[index]))
    _store_feature(features, index, 6, health[index] / wp.max(p.health_scale, 1.0e-6))
    _store_feature(
        features,
        index,
        7,
        float(attack_cooldown[index]) / float(wp.max(p.attack_recovery_ticks, 1)),
    )
    _store_feature(
        features,
        index,
        8,
        x_sign * (2.0 * position[index][0] / p.world_width - 1.0),
    )
    _store_feature(
        features,
        index,
        9,
        y_sign * (2.0 * position[index][1] / p.world_height - 1.0),
    )
    _store_feature(
        features,
        index,
        10,
        wp.max(
            0.0,
            1.0
            - float(step_count[env]) / float(wp.max(p.maximum_decision_steps, 1)),
        ),
    )
    team_sign = float(1.0)
    if team[index] == 1:
        team_sign = -1.0
    _store_feature(
        features,
        index,
        11,
        team_sign * (territory[env * 2] - territory[env * 2 + 1]),
    )
    _store_feature(
        features,
        index,
        12,
        team_sign
        * advantage_integral[env]
        / float(wp.max(p.maximum_decision_steps, 1)),
    )


@wp.kernel
def _nearest_enemy_actions(
    team: wp.array(dtype=wp.int32),
    position: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    learner_team: wp.array(dtype=wp.int32),
    vertical_sign: wp.array(dtype=float),
    strongpoints: wp.array(dtype=wp.vec2),
    actions: wp.array(dtype=ACTION_DTYPE),
    p: _ObservationParams,
):
    index = wp.tid()
    env = index // p.num_soldiers
    own_team = team[index]
    actions[index] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if alive[index] == 0 or own_team == learner_team[env]:
        return

    soldiers_per_team = p.num_soldiers // 2
    enemy_start = 0
    if own_team == 0:
        enemy_start = soldiers_per_team
    base = env * p.num_soldiers
    closest = int(-1)
    closest_distance = float(1.0e30)
    for enemy in range(soldiers_per_team):
        other = base + enemy_start + enemy
        if alive[other] != 0 and team[other] != own_team:
            delta = position[other] - position[index]
            distance = wp.dot(delta, delta)
            if distance < closest_distance:
                closest = other
                closest_distance = distance

    # With no living enemy, occupy the nearest strongpoint instead of halting;
    # a halted opponent lets territory painted before annihilation stand.
    target = position[index]
    if closest >= 0:
        target = position[closest]
    else:
        best_distance = float(1.0e30)
        for point in range(strongpoints.shape[0]):
            delta = strongpoints[point] - position[index]
            distance = wp.dot(delta, delta)
            if distance < best_distance:
                best_distance = distance
                target = strongpoints[point]

    offset = target - position[index]
    distance = wp.dot(offset, offset)
    if distance > 1.0e-12 and (closest >= 0 or distance > 1.0):
        direction = offset / wp.sqrt(distance)
        x_sign = float(1.0)
        if own_team == 1:
            x_sign = -1.0
        x = x_sign * direction[0]
        y = vertical_sign[env] * direction[1]
        actions[index] = wp.vec4(x, y, x, y)


@wp.kernel
def _collect_facts(
    previous: wp.array2d(dtype=float),
    current: wp.array2d(dtype=float),
    previous_territory: wp.array(dtype=float),
    territory: wp.array(dtype=float),
    done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    facts: wp.array2d(dtype=float),
    fact_done: wp.array(dtype=wp.int32),
    fact_winner: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    facts[env, 0] = wp.max(0.0, previous[env, 0] - current[env, 0])
    facts[env, 1] = wp.max(0.0, previous[env, 1] - current[env, 1])
    facts[env, 2] = current[env, 0]
    facts[env, 3] = current[env, 1]
    facts[env, 4] = current[env, 2]
    facts[env, 5] = current[env, 3]
    facts[env, 6] = territory[env * 2]
    facts[env, 7] = territory[env * 2 + 1]
    facts[env, 8] = territory[env * 2] - previous_territory[env * 2]
    facts[env, 9] = territory[env * 2 + 1] - previous_territory[env * 2 + 1]
    fact_done[env] = done[env]
    fact_winner[env] = winner[env]


class RLEnv:
    """Batched simulator exposing only local, team-canonical actor state."""

    def __init__(
        self,
        config: Config | None = None,
        num_envs: int = 1,
        device: str = "cuda",
        simulator: GpuSimulator | None = None,
        mode_count: int = 1,
    ):
        if mode_count < 1:
            raise ValueError("mode_count must be positive")
        self.sim = simulator or GpuSimulator(config, num_envs, device)
        self.config = self.sim.config
        self.num_envs = self.sim.num_envs
        self.soldiers_per_team = self.config.soldiers_per_team
        self.device = torch.device(str(self.sim.device))
        self.mode_count = int(mode_count)
        self.mode = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.mode_probabilities: torch.Tensor | None = None
        env_index = torch.arange(self.num_envs, device=self.device)
        self.mirror_y = (1 - 2 * ((env_index // 2) % 2)).to(torch.float32)
        self._vertical_sign = wp.from_torch(self.mirror_y)
        wp.synchronize_device(self.sim.device)
        self._stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
            if self.device.type == "cuda"
            else None
        )
        self.params = self._make_params()
        self._stats = wp.zeros((self.num_envs, 4), dtype=float, device=self.sim.device)
        self._previous_stats = wp.zeros((self.num_envs, 4), dtype=float, device=self.sim.device)
        self._territory = wp.zeros(self.num_envs * 2, dtype=float, device=self.sim.device)
        self._previous_territory = wp.zeros_like(self._territory)
        self._facts = wp.zeros((self.num_envs, 10), dtype=float, device=self.sim.device)
        self._fact_done = wp.zeros(self.num_envs, dtype=wp.int32, device=self.sim.device)
        self._fact_winner = wp.full(self.num_envs, -2, dtype=wp.int32, device=self.sim.device)
        self._centers = wp.array(
            territory_centers(self.config), dtype=wp.vec2, device=self.sim.device
        )
        self._features = wp.zeros(
            (self.sim.total_soldiers, LOCAL_FEATURE_SIZE),
            dtype=wp.bfloat16,
            device=self.sim.device,
        )

        shape = (self.num_envs, 2, self.soldiers_per_team)
        self.features = wp.to_torch(self._features).view(*shape, LOCAL_FEATURE_SIZE)
        self.cells = wp.to_torch(self.sim.territory_cell).view(shape)
        self.alive = wp.to_torch(self.sim.alive).view(shape)
        self._team_positions = wp.to_torch(self.sim.position).view(*shape, 2)
        self._team_velocities = wp.to_torch(self.sim.velocity).view(*shape, 2)
        self._team_healths = wp.to_torch(self.sim.health).view(shape)
        self._team_x_sign = torch.tensor((1.0, -1.0), device=self.device).view(1, 2)
        self._flat_positions = wp.to_torch(self.sim.position).view(self.num_envs, -1, 2)
        self._flat_alive = wp.to_torch(self.sim.alive).view(self.num_envs, -1)
        self.neighbor_count = min(ENTITY_NEIGHBORS, self.config.soldier_count - 1)
        self._neighbor_buffer = wp.zeros(
            (self.sim.total_soldiers, self.neighbor_count),
            dtype=wp.int32,
            device=self.sim.device,
        )
        self._neighbor_view = wp.to_torch(self._neighbor_buffer).view(
            self.num_envs, 2, self.soldiers_per_team, self.neighbor_count
        )
        self.neighbors = torch.full(
            (self.num_envs, 2, self.soldiers_per_team, self.neighbor_count),
            -1,
            dtype=torch.int16,
            device=self.device,
        )
        self.owners = wp.to_torch(self.sim.territory_owner).view(
            self.num_envs, TERRITORY_CELLS
        )
        self.territory = wp.to_torch(self._territory).view(self.num_envs, 2)
        fact_tensor = wp.to_torch(self._facts)
        self.facts = {
            "damage_taken": fact_tensor[:, 0:2],
            "health": fact_tensor[:, 2:4],
            "living": fact_tensor[:, 4:6],
            "territory": fact_tensor[:, 6:8],
            "territory_delta": fact_tensor[:, 8:10],
            "done": wp.to_torch(self._fact_done),
            "winner": wp.to_torch(self._fact_winner),
        }
        self._actions = torch.empty((*shape, 4), dtype=torch.float32, device=self.device)
        self._action_sign = torch.ones_like(self._actions)
        self._action_sign[:, 1, :, 0] = -1.0
        self._action_sign[:, 1, :, 2] = -1.0
        self._action_sign[..., 1] = self.mirror_y[:, None, None]
        self._action_sign[..., 3] = self.mirror_y[:, None, None]
        self._world_actions = wp.from_torch(self._actions.view(-1, 4), dtype=ACTION_DTYPE)
        self._resample_modes()
        self.observe()

    def _resample_modes(self, finished: torch.Tensor | None = None) -> None:
        """Draw fresh per-team episode modes, everywhere or where episodes ended."""
        shape = (self.num_envs, 2)
        if self.mode_probabilities is None:
            fresh = torch.randint(
                self.mode_count, shape, dtype=torch.long, device=self.device
            )
        else:
            fresh = torch.multinomial(
                self.mode_probabilities.expand(self.num_envs * 2, -1), 1
            ).view(shape)
        if finished is None:
            self.mode.copy_(fresh)
        else:
            self.mode.copy_(torch.where(finished.view(-1, 1).bool(), fresh, self.mode))

    def set_mode_distribution(self, probabilities: torch.Tensor | None) -> None:
        """Override uniform episode-mode sampling, or reset it with ``None``."""
        if probabilities is None:
            self.mode_probabilities = None
            return
        probabilities = probabilities.to(device=self.device, dtype=torch.float32).flatten()
        if probabilities.shape != (self.mode_count,):
            raise ValueError(f"expected {self.mode_count} mode probabilities")
        if not bool((probabilities >= 0).all()) or float(probabilities.sum()) <= 0:
            raise ValueError("mode probabilities must be non-negative and sum above zero")
        self.mode_probabilities = probabilities / probabilities.sum()

    def _make_params(self) -> _ObservationParams:
        c = self.config
        p = _ObservationParams()
        p.num_soldiers = c.soldier_count
        p.speed_scale = c.physics_speed_limit
        p.health_scale = c.initial_health
        p.attack_recovery_ticks = c.attack_recovery_ticks
        p.tile_size = c.territory_tile_size
        p.world_width = c.world_width
        p.world_height = c.world_height
        p.territory_cells = TERRITORY_CELLS
        p.territory_total_weight = float(TERRITORY_TOTAL_WEIGHT)
        p.maximum_decision_steps = c.maximum_decision_steps
        return p

    def _scope(self):
        if self._stream is None:
            return nullcontext()
        return wp.ScopedStream(self._stream, sync_enter=False)

    def _refresh_stats(self) -> None:
        wp.launch(
            _team_stats,
            self.num_envs,
            inputs=[self.sim.team, self.sim.health, self.sim.alive, self._stats, self.params],
            device=self.sim.device,
        )

    def _refresh_territory(self) -> None:
        wp.launch(
            _clear_territory,
            self._territory.size,
            inputs=[self._territory],
            device=self.sim.device,
        )
        wp.launch(
            _count_territory,
            self.num_envs * TERRITORY_CELLS,
            inputs=[
                self.sim.control_share,
                self.sim.territory_weights,
                self._territory,
                self.params,
            ],
            device=self.sim.device,
        )

    def _refresh(self) -> None:
        self._refresh_stats()
        self._refresh_territory()
        wp.launch(
            _observe_local,
            self.sim.total_soldiers,
            inputs=[
                self.sim.team,
                self.sim.position,
                self.sim.velocity,
                self.sim.attack_angle,
                self.sim.health,
                self.sim.alive,
                self.sim.attack_cooldown_ticks,
                self.sim.step_count,
                self._territory,
                self.sim.advantage_integral,
                self._vertical_sign,
                self.sim.territory_cell,
                self._centers,
                self._features,
                self.params,
            ],
            device=self.sim.device,
        )

    def _refresh_neighbors(self) -> None:
        with self._scope():
            wp.launch(
                _entity_neighbor_search,
                self.sim.total_soldiers,
                inputs=[
                    self.sim.position,
                    self.sim.alive,
                    ENTITY_RADIUS,
                    self.neighbor_count,
                    self._neighbor_buffer,
                    self.params,
                ],
                device=self.sim.device,
            )
        self.neighbors.copy_(self._neighbor_view)

    def state(self) -> LocalState:
        return LocalState(
            self.features,
            self.cells,
            self.alive,
            self.owners,
            self.mirror_y,
            self.mode,
            self.neighbors,
        )

    def observe(self) -> LocalState:
        with self._scope():
            self._refresh()
        self._refresh_neighbors()
        return self.state()

    def reset(self) -> LocalState:
        with self._scope():
            self.sim.reset()
            self._resample_modes()
            self._refresh()
        self._refresh_neighbors()
        return self.state()

    def actions_to_sim(self, actions: torch.Tensor) -> wp.array:
        expected = self._actions.shape
        if actions.shape != expected or actions.device != self.device:
            raise ValueError(f"expected actions with shape {tuple(expected)} on {self.device}")
        self._actions.copy_(actions)
        self._actions.mul_(self._action_sign)
        return self._world_actions

    def step(
        self,
        actions: torch.Tensor,
        *,
        reset_done: bool = True,
        observe: bool = True,
        before_reset: Callable[[], None] | None = None,
    ) -> tuple[LocalState, dict[str, torch.Tensor]]:
        world_actions = self.actions_to_sim(actions)
        with self._scope():
            wp.copy(self._previous_stats, self._stats)
            wp.copy(self._previous_territory, self._territory)
            self.sim.step(world_actions)
            self._refresh_stats()
            self._refresh_territory()
            wp.launch(
                _collect_facts,
                self.num_envs,
                inputs=[
                    self._previous_stats,
                    self._stats,
                    self._previous_territory,
                    self._territory,
                    self.sim.done,
                    self.sim.winner,
                    self._facts,
                    self._fact_done,
                    self._fact_winner,
                ],
                device=self.sim.device,
            )
            if before_reset is not None:
                before_reset()
            if reset_done:
                self.sim.reset(mask=self.sim.done)
                self._resample_modes(self.facts["done"])
            if observe:
                self._refresh()
        if observe:
            self._refresh_neighbors()
        return self.state(), self.facts

    def team_summary(
        self,
        damage_taken: torch.Tensor | None = None,
        territory_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-team canonical behavior statistics for mode discrimination.

        Callers holding one action across simulator steps pass damage and
        territory deltas accumulated over the held decision.
        """
        if damage_taken is None:
            damage_taken = self.facts["damage_taken"]
        if territory_delta is None:
            territory_delta = self.facts["territory_delta"]
        c = self.config
        alive = self.alive.to(torch.float32)
        count = alive.sum(-1).clamp_min(1.0)
        weight = alive.unsqueeze(-1)
        centroid = (self._team_positions * weight).sum(-2) / count.unsqueeze(-1)
        offsets = self._team_positions - centroid.unsqueeze(-2)
        spread = (offsets.square().sum(-1) * alive).sum(-1) / count
        mean_velocity = (self._team_velocities * weight).sum(-2) / count.unsqueeze(-1)
        speed = (
            self._team_velocities.square().sum(-1).clamp_min(1e-12).sqrt() * alive
        ).sum(-1) / count
        y_sign = self.mirror_y.view(-1, 1)
        health_pool = self.soldiers_per_team * c.initial_health
        damage_scale = SUMMARY_DAMAGE_SCALE / health_pool
        return torch.stack(
            (
                self._team_x_sign * (2.0 * centroid[..., 0] / c.world_width - 1.0),
                y_sign * (2.0 * centroid[..., 1] / c.world_height - 1.0),
                spread.sqrt() * (4.0 / c.world_width),
                self._team_x_sign * mean_velocity[..., 0] / c.maximum_running_speed,
                y_sign * mean_velocity[..., 1] / c.maximum_running_speed,
                speed / c.maximum_running_speed,
                alive.mean(-1),
                self._team_healths.sum(-1) / health_pool,
                damage_taken.flip(-1) * damage_scale,
                damage_taken * damage_scale,
                self.territory,
                territory_delta * SUMMARY_DELTA_SCALE,
            ),
            dim=-1,
        )


class NearestEnemyOpponent:
    """GPU policy that sends the fixed team directly at its nearest living enemy."""

    def __init__(self, env: RLEnv, learner_teams: torch.Tensor) -> None:
        expected = (env.num_envs, 2)
        if learner_teams.shape != expected or learner_teams.device != env.device:
            raise ValueError(
                f"expected learner teams with shape {expected} on {env.device}"
            )
        if not bool((learner_teams.sum(dim=1) == 1).all()):
            raise ValueError("each environment must have exactly one learner team")
        self.env = env
        self._learner_tensor = (
            learner_teams.to(torch.int32).argmax(dim=1).to(torch.int32)
        )
        self._learner_team = wp.from_torch(self._learner_tensor)
        self._strongpoints = wp.array(
            strongpoint_world_centers(env.config), dtype=wp.vec2, device=env.sim.device
        )
        self._output = wp.empty(
            env.sim.total_soldiers, dtype=ACTION_DTYPE, device=env.sim.device
        )
        self.actions = wp.to_torch(self._output).view(
            env.num_envs, 2, env.soldiers_per_team, 4
        )

    def act(self) -> torch.Tensor:
        with self.env._scope():
            wp.launch(
                _nearest_enemy_actions,
                self.env.sim.total_soldiers,
                inputs=[
                    self.env.sim.team,
                    self.env.sim.position,
                    self.env.sim.alive,
                    self._learner_team,
                    self.env._vertical_sign,
                    self._strongpoints,
                    self._output,
                    self.env.params,
                ],
                device=self.env.sim.device,
            )
        return self.actions


__all__ = [
    "ENTITY_NEIGHBORS",
    "ENTITY_RADIUS",
    "LOCAL_FEATURE_NAMES",
    "LOCAL_FEATURE_SIZE",
    "SUMMARY_FEATURE_NAMES",
    "SUMMARY_SIZE",
    "LocalState",
    "NearestEnemyOpponent",
    "RLEnv",
    "entity_neighbors",
]

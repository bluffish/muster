"""Compact NumPy reference implementation of the battle simulator rules."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from muster.sim.config import Config
from muster.sim.constants import (
    ARENA_NORMALS,
    INFLUENCE_FIXED_SCALE,
    INFLUENCE_NEUTRAL_FIXED,
    TERRITORY_CELLS,
    TERRITORY_INITIAL_OWNER,
    TERRITORY_TOTAL_WEIGHT,
    TERRITORY_WEIGHTS,
)
from muster.sim.geometry import arena_contains, territory_centers, territory_indices

def _limit_vectors(vectors: np.ndarray, maximum: float) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1)
    scale = np.ones_like(lengths)
    mask = lengths > maximum
    scale[mask] = maximum / lengths[mask]
    return vectors * scale[..., None]

def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi

class CpuSimulator:
    """Small correctness-first simulator. State arrays are public NumPy arrays."""

    def __init__(self, config: Config | None = None, num_envs: int = 1):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.config = config or Config()
        self.num_envs = num_envs
        self.num_soldiers = self.config.soldier_count
        self._pair_i, self._pair_j = np.triu_indices(self.num_soldiers, 1)
        self._pair_i = self._pair_i.astype(np.int32)
        self._pair_j = self._pair_j.astype(np.int32)
        shape = (num_envs, self.num_soldiers)
        self.team = np.empty(shape, np.int32)
        self.position = np.empty((*shape, 2), np.float32)
        self.velocity = np.empty((*shape, 2), np.float32)
        self.attack_angle = np.empty(shape, np.float32)
        self.health = np.empty(shape, np.float32)
        self.alive = np.empty(shape, bool)
        self.attack_cooldown_ticks = np.zeros(shape, np.int16)
        self.step_count = np.zeros(num_envs, np.int32)
        self.substep_count = np.zeros(num_envs, np.int32)
        self.done = np.zeros(num_envs, bool)
        self.winner = np.full(num_envs, -2, np.int8)  # -2 ongoing, -1 draw
        self.territory_owner = np.empty((num_envs, TERRITORY_CELLS), np.int8)
        self.advantage_integral = np.zeros(num_envs, np.float64)
        self.control_share = np.zeros((num_envs, TERRITORY_CELLS, 2), np.float32)
        self.invalid_action = np.zeros(num_envs, bool)
        self._territory_centers = territory_centers(self.config)
        self.reset()

    @property
    def state(self) -> dict[str, np.ndarray]:
        return {
            "team": self.team,
            "position": self.position,
            "velocity": self.velocity,
            "attack_angle": self.attack_angle,
            "health": self.health,
            "alive": self.alive,
            "attack_cooldown_ticks": self.attack_cooldown_ticks,
            "step_count": self.step_count,
            "done": self.done,
            "winner": self.winner,
            "territory_owner": self.territory_owner,
            "advantage_integral": self.advantage_integral,
            "control_share": self.control_share,
        }

    def snapshot(self, env: int = 0) -> dict[str, np.ndarray | int | bool]:
        """Copy one environment for rendering or inspection."""
        if not 0 <= env < self.num_envs:
            raise IndexError("environment index out of range")
        return {
            "team": self.team[env].copy(),
            "position": self.position[env].copy(),
            "velocity": self.velocity[env].copy(),
            "attack_angle": self.attack_angle[env].copy(),
            "health": self.health[env].copy(),
            "alive": self.alive[env].copy(),
            "step_count": int(self.step_count[env]),
            "done": bool(self.done[env]),
            "winner": int(self.winner[env]),
            "territory_owner": self.territory_owner[env].copy(),
            "control_share": self.control_share[env].copy(),
        }

    def _default_state(self) -> dict[str, np.ndarray]:
        c = self.config
        n = c.soldiers_per_team
        cols = math.ceil(math.sqrt(2 * n))
        rows = math.ceil(n / cols)
        spacing = 1.1 * c.diameter
        local = np.arange(n)
        col, row = local // rows, local % rows
        column_size = np.minimum(rows, n - col * rows)
        formation = np.stack(
            ((col - np.mean(col)) * spacing, (row - (column_size - 1) / 2) * spacing),
            axis=-1,
        )
        position = np.empty((self.num_soldiers, 2), np.float32)
        position[:n] = formation + (c.world_width * 0.25, c.world_height * 0.5)
        # Mirror the left formation exactly so the sides are float-identical.
        position[n:, 0] = np.float32(c.world_width) - position[:n, 0]
        position[n:, 1] = position[:n, 1]
        if not np.all(arena_contains(position, c, c.soldier_radius)):
            raise ValueError("default formations do not fit in the configured world")
        if np.max(position[:n, 0]) + c.diameter > np.min(position[n:, 0]):
            raise ValueError("default formations overlap")
        return {
            "team": np.concatenate((np.zeros(n, np.int32), np.ones(n, np.int32))),
            "position": position,
            "velocity": np.zeros((self.num_soldiers, 2), np.float32),
            "attack_angle": np.concatenate((np.zeros(n), np.full(n, np.pi))).astype(np.float32),
            "health": np.full(self.num_soldiers, c.initial_health, np.float32),
        }

    def _copy_state_value(self, value: object, trailing_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        target = (self.num_envs, self.num_soldiers, *trailing_shape)
        array = np.asarray(value, dtype=dtype)
        if array.shape == target[1:]:
            array = np.broadcast_to(array, target)
        if array.shape != target:
            raise ValueError(f"expected state shape {target}, got {array.shape}")
        return np.array(array, copy=True)

    def reset(self, state: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
        values = self._default_state()
        if state:
            values.update(state)
        self.team[:] = self._copy_state_value(values["team"], (), np.dtype(np.int32))
        self.position[:] = self._copy_state_value(values["position"], (2,), np.dtype(np.float32))
        self.velocity[:] = self._copy_state_value(values["velocity"], (2,), np.dtype(np.float32))
        self.attack_angle[:] = self._copy_state_value(values["attack_angle"], (), np.dtype(np.float32))
        self.health[:] = self._copy_state_value(values["health"], (), np.dtype(np.float32))
        if not all(
            np.isfinite(array).all()
            for array in (self.position, self.velocity, self.attack_angle, self.health)
        ):
            raise ValueError("initial state must be finite")
        if np.any((self.team != 0) & (self.team != 1)) or np.any(self.health < 0):
            raise ValueError("teams must be 0 or 1 and health cannot be negative")

        self.attack_angle[:] = _wrap_angle(self.attack_angle)
        self.alive[:] = self.health > 0
        self.velocity[~self.alive] = 0
        self.attack_cooldown_ticks.fill(0)
        self.step_count.fill(0)
        self.substep_count.fill(0)
        self.done.fill(False)
        self.winner.fill(-2)
        self.advantage_integral.fill(0.0)
        self.control_share.fill(0.0)
        self.territory_owner[:] = TERRITORY_INITIAL_OWNER
        if state and "territory_owner" in state:
            owner = np.asarray(state["territory_owner"], dtype=np.int8)
            expected = self.territory_owner.shape
            if owner.shape == expected[1:]:
                owner = np.broadcast_to(owner, expected)
            if owner.shape != expected or np.any((owner < -1) | (owner > 1)):
                raise ValueError(
                    f"expected territory_owner values -1, 0, or 1 with shape {expected}"
                )
            self.territory_owner[:] = owner
        self.invalid_action.fill(False)
        for env in range(self.num_envs):
            self._resolve_static(env, self.alive[env])
        return self.state

    def _resolve_static(self, env: int, mask: np.ndarray) -> None:
        c = self.config
        p, v = self.position[env], self.velocity[env]
        r = c.soldier_radius
        center = np.array((c.world_width / 2, c.world_height / 2), np.float32)
        apothems = c.arena_apothems
        for _ in range(2):
            for axis, apothem in zip(ARENA_NORMALS[:3], apothems, strict=True):
                limit = apothem - r
                signed_distance = (p - center) @ axis
                normal = np.where(signed_distance[:, None] < 0, -axis, axis)
                distance = np.abs(signed_distance) - limit
                crossed = mask & (distance > 0)
                p[crossed] -= distance[crossed, None] * normal[crossed]
                outward_speed = np.sum(v * normal, axis=1)
                bounce = crossed & (outward_speed > 0)
                v[bounce] -= (
                    (1 + c.wall_restitution) * outward_speed[bounce, None] * normal[bounce]
                )

        if c.river_width <= 0:
            return
        x_lo = c.world_width / 2 - c.river_width / 2 - r
        x_hi = c.world_width / 2 + c.river_width / 2 + r
        bridge_lo = c.world_height / 2 - c.bridge_width / 2 + r
        bridge_hi = c.world_height / 2 + c.bridge_width / 2 - r
        in_x = mask & (p[:, 0] > x_lo) & (p[:, 0] < x_hi)
        for index in np.flatnonzero(in_x & (p[:, 1] < bridge_lo)):
            choices = [p[index, 0] - x_lo, x_hi - p[index, 0], bridge_lo - p[index, 1]]
            if x_lo < r:
                choices[0] = math.inf
            if x_hi > c.world_width - r:
                choices[1] = math.inf
            edge = int(np.argmin(choices))
            if edge == 0:
                p[index, 0] = x_lo
                if v[index, 0] > 0:
                    v[index, 0] = 0
            elif edge == 1:
                p[index, 0] = x_hi
                if v[index, 0] < 0:
                    v[index, 0] = 0
            else:
                p[index, 1] = bridge_lo
                if v[index, 1] < 0:
                    v[index, 1] = 0
        for index in np.flatnonzero(in_x & (p[:, 1] > bridge_hi)):
            choices = [p[index, 0] - x_lo, x_hi - p[index, 0], p[index, 1] - bridge_hi]
            if x_lo < r:
                choices[0] = math.inf
            if x_hi > c.world_width - r:
                choices[1] = math.inf
            edge = int(np.argmin(choices))
            if edge == 0:
                p[index, 0] = x_lo
                if v[index, 0] > 0:
                    v[index, 0] = 0
            elif edge == 1:
                p[index, 0] = x_hi
                if v[index, 0] < 0:
                    v[index, 0] = 0
            else:
                p[index, 1] = bridge_hi
                if v[index, 1] > 0:
                    v[index, 1] = 0

    def _geometry(self, env: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.position[env]
        delta = p[self._pair_j] - p[self._pair_i]
        distance = np.linalg.norm(delta, axis=1)
        normal = np.empty_like(delta)
        separate = distance > 1e-12
        normal[separate] = delta[separate] / distance[separate, None]
        normal[~separate] = (1.0, 0.0)
        alive_pair = self.alive[env, self._pair_i] & self.alive[env, self._pair_j]
        touching = alive_pair & (distance < self.config.diameter)
        relative = self.velocity[env, self._pair_i] - self.velocity[env, self._pair_j]
        closing = np.sum(relative * normal, axis=1)
        allied = self.team[env, self._pair_i] == self.team[env, self._pair_j]
        return normal, distance, touching, closing, allied

    def _strikes(self, env: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        arc_cos = math.cos(math.radians(c.damaging_arc_degrees / 2))
        received = np.zeros(self.num_soldiers, np.float32)
        struck = np.zeros(self.num_soldiers, bool)
        position = self.position[env]
        velocity = self.velocity[env]
        facing = np.stack(
            (np.cos(self.attack_angle[env]), np.sin(self.attack_angle[env])), axis=-1
        )
        ready = self.alive[env] & (self.attack_cooldown_ticks[env] == 0)
        for attacker in np.flatnonzero(ready):
            delta = position - position[attacker]
            distance = np.linalg.norm(delta, axis=1)
            candidate = (
                self.alive[env]
                & (self.team[env] != self.team[env, attacker])
                & (distance < c.diameter)
            )
            candidate[attacker] = False
            targets = np.flatnonzero(candidate)
            if not len(targets):
                continue
            direction = np.empty((len(targets), 2), np.float32)
            separate = distance[targets] > 1e-12
            direction[separate] = (
                delta[targets[separate]] / distance[targets[separate], None]
            )
            coincident = ~separate
            direction[coincident, 0] = np.where(
                attacker < targets[coincident], 1.0, -1.0
            )
            direction[coincident, 1] = 0.0
            alignment = direction @ facing[attacker]
            eligible = alignment >= arc_cos
            targets = targets[eligible]
            if not len(targets):
                continue
            direction = direction[eligible]
            alignment = alignment[eligible]
            selected = int(np.argmax(alignment))
            target = int(targets[selected])
            attack_direction = direction[selected]
            closing = np.dot(velocity[attacker] - velocity[target], attack_direction)
            inward = max(
                0.0,
                float(np.dot(velocity[attacker], attack_direction))
                - c.minimum_damage_speed,
            )
            charge = c.damage_scale * inward**2 if closing > c.minimum_closing_speed else 0
            direction_to_attacker = -attack_direction
            protected = np.dot(facing[target], direction_to_attacker) >= arc_cos
            vulnerability = 1.0 if protected else c.flank_damage_multiplier
            received[target] += max(c.base_strike_damage, charge) * vulnerability
            struck[attacker] = True
        return received, struck

    def _collision_changes(
        self,
        env: int,
        normal: np.ndarray,
        distance: np.ndarray,
        touching: np.ndarray,
        closing: np.ndarray,
        allied: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        i, j = self._pair_i, self._pair_j
        approaching = touching & (closing > 0)
        restitution = np.where(allied, c.ally_restitution, c.enemy_restitution)
        impulse = np.where(approaching, 0.5 * (1 + restitution) * closing, 0).astype(np.float32)
        change_i = -impulse[:, None] * normal
        change_j = impulse[:, None] * normal

        ally_sum = np.zeros((self.num_soldiers, 2), np.float32)
        enemy_sum = np.zeros_like(ally_sum)
        ally_count = np.zeros(self.num_soldiers, np.int32)
        enemy_max = np.zeros(self.num_soldiers, np.float32)
        ally_approach = approaching & allied
        enemy_approach = approaching & ~allied
        np.add.at(ally_sum, i[ally_approach], change_i[ally_approach])
        np.add.at(ally_sum, j[ally_approach], change_j[ally_approach])
        np.add.at(ally_count, i[ally_approach], 1)
        np.add.at(ally_count, j[ally_approach], 1)
        np.add.at(enemy_sum, i[enemy_approach], change_i[enemy_approach])
        np.add.at(enemy_sum, j[enemy_approach], change_j[enemy_approach])
        np.maximum.at(enemy_max, i[enemy_approach], impulse[enemy_approach])
        np.maximum.at(enemy_max, j[enemy_approach], impulse[enemy_approach])
        ally_sum /= np.sqrt(np.maximum(1, ally_count))[:, None]
        enemy_length = np.linalg.norm(enemy_sum, axis=1)
        cap = (enemy_length > enemy_max) & (enemy_length > 0)
        enemy_sum[cap] *= (enemy_max[cap] / enemy_length[cap])[:, None]

        overlap = np.maximum(0, c.diameter - distance).astype(np.float32)
        raw_i = -0.5 * overlap[:, None] * normal
        raw_j = 0.5 * overlap[:, None] * normal
        position_sum = np.zeros((self.num_soldiers, 2), np.float32)
        contact_count = np.zeros(self.num_soldiers, np.int32)
        np.add.at(position_sum, i[touching], raw_i[touching])
        np.add.at(position_sum, j[touching], raw_j[touching])
        np.add.at(contact_count, i[touching], 1)
        np.add.at(contact_count, j[touching], 1)
        position_sum /= np.sqrt(np.maximum(1, contact_count))[:, None]
        position_sum = _limit_vectors(position_sum, 0.5 * c.soldier_radius)
        return ally_sum + enemy_sum, position_sum * c.collision_relaxation

    def _collision_iteration(self, env: int) -> None:
        normal, distance, touching, closing, allied = self._geometry(env)
        velocity_change, position_change = self._collision_changes(
            env, normal, distance, touching, closing, allied
        )
        living = self.alive[env]
        self.velocity[env, living] += velocity_change[living]
        self.velocity[env, living] = _limit_vectors(
            self.velocity[env, living], self.config.physics_speed_limit
        )
        self.position[env, living] += position_change[living]
        self._resolve_static(env, living)

    def _substep(self, move: np.ndarray, desired_angle: np.ndarray) -> None:
        c = self.config
        active = self.alive & ~self.done[:, None]
        self.attack_cooldown_ticks[active] = np.maximum(
            0, self.attack_cooldown_ticks[active] - 1
        )

        error = np.arctan2(np.sin(desired_angle - self.attack_angle), np.cos(desired_angle - self.attack_angle))
        error = np.where(np.isclose(error, -np.pi), np.pi, error)
        turn = np.clip(error, -c.maximum_turn_rate * c.physics_dt, c.maximum_turn_rate * c.physics_dt)
        self.attack_angle[active] = _wrap_angle(self.attack_angle[active] + turn[active])
        self.velocity[active] *= math.exp(-c.linear_drag * c.physics_dt)
        desired_velocity = c.maximum_running_speed * move
        change = _limit_vectors(desired_velocity - self.velocity, c.acceleration * c.physics_dt)
        self.velocity[active] += change[active]
        self.velocity[..., 0][active] -= c.slope_gravity * c.slope_height / c.world_width * c.physics_dt
        self.velocity[active] = _limit_vectors(self.velocity[active], c.physics_speed_limit)
        self.position[active] += self.velocity[active] * c.physics_dt

        for env in np.flatnonzero(~self.done):
            self._resolve_static(int(env), self.alive[env])
            normal, distance, touching, closing, allied = self._geometry(int(env))
            damage, struck = self._strikes(int(env))
            velocity_change, position_change = self._collision_changes(
                int(env), normal, distance, touching, closing, allied
            )
            previously_alive = self.alive[env].copy()
            self.health[env, previously_alive] = np.maximum(
                0, self.health[env, previously_alive] - damage[previously_alive]
            )
            self.alive[env] = self.health[env] > 0
            self.velocity[env, previously_alive] += velocity_change[previously_alive]
            self.velocity[env, previously_alive] = _limit_vectors(
                self.velocity[env, previously_alive], c.physics_speed_limit
            )
            self.position[env, previously_alive] += position_change[previously_alive]
            self.velocity[env, ~self.alive[env]] = 0
            self._resolve_static(int(env), self.alive[env])
            for _ in range(c.collision_iterations - 1):
                self._collision_iteration(int(env))
            self.attack_cooldown_ticks[env, struck & self.alive[env]] = (
                c.attack_recovery_ticks
            )
            self.attack_cooldown_ticks[env, ~self.alive[env]] = 0
            self.substep_count[env] += 1

    def _update_territory(self) -> None:
        """Health-weighted soft influence with order-invariant summation.

        Each living soldier contributes ``(h / H)^2 exp(-0.5 (d / sigma)^2)``
        influence to every cell, with ``h`` its current health, ``H`` the
        initial health, and ``sigma = control_radius / 2``, quantized to
        ``INFLUENCE_FIXED_SCALE``. Wounded soldiers project quadratically
        less control. A team's control share is its influence over the total
        plus the neutral mass ``kappa``: scoring requires presence, and
        unreached ground stays neutral. A cell displays as owned when a
        share exceeds one half.
        """
        sigma = np.float32(self.config.control_radius * 0.5)
        initial_health = np.float32(self.config.initial_health)
        for env in np.flatnonzero(~self.done):
            fixed = np.zeros((2, TERRITORY_CELLS), np.int64)
            living = np.flatnonzero(self.alive[env])
            for team in (0, 1):
                members = living[self.team[env, living] == team]
                if len(members):
                    deltas = (
                        self._territory_centers[:, None, :]
                        - self.position[env, members][None, :, :]
                    ).astype(np.float32)
                    squared = (deltas**2).sum(-1)
                    weight = np.minimum(
                        self.health[env, members].astype(np.float32) / initial_health,
                        np.float32(1.0),
                    )
                    influence = (
                        np.exp(np.float32(-0.5) * squared / (sigma * sigma)).astype(
                            np.float32
                        )
                        * weight[None, :]
                        * weight[None, :]
                    )
                    fixed[team] = (
                        np.floor(influence * INFLUENCE_FIXED_SCALE + 0.5)
                        .astype(np.int64)
                        .sum(axis=1)
                    )
            total = (fixed[0] + fixed[1] + INFLUENCE_NEUTRAL_FIXED).astype(np.float32)
            self.control_share[env, :, 0] = fixed[0].astype(np.float32) / total
            self.control_share[env, :, 1] = fixed[1].astype(np.float32) / total
            owner = np.full(TERRITORY_CELLS, -1, np.int8)
            owner[fixed[0] > fixed[1] + INFLUENCE_NEUTRAL_FIXED] = 0
            owner[fixed[1] > fixed[0] + INFLUENCE_NEUTRAL_FIXED] = 1
            self.territory_owner[env] = owner

    def _accumulate_control(self) -> None:
        for env in np.flatnonzero(~self.done):
            advantage = float(
                (
                    TERRITORY_WEIGHTS
                    * (self.control_share[env, :, 0] - self.control_share[env, :, 1])
                ).sum()
            )
            self.advantage_integral[env] += advantage / TERRITORY_TOTAL_WEIGHT

    def _finish_episodes(self) -> None:
        for env in range(self.num_envs):
            if self.done[env] or self.step_count[env] < self.config.maximum_decision_steps:
                continue
            integral = self.advantage_integral[env]
            self.done[env] = True
            self.winner[env] = 0 if integral > 1e-9 else 1 if integral < -1e-9 else -1

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.num_soldiers, 4)
        if actions.shape == expected[1:] and self.num_envs == 1:
            actions = actions[None]
        if actions.shape != expected:
            raise ValueError(f"expected actions with shape {expected}, got {actions.shape}")

        move, aim = actions[..., :2].copy(), actions[..., 2:].copy()
        valid_move = np.isfinite(move).all(axis=-1)
        valid_aim = np.isfinite(aim).all(axis=-1)
        self.invalid_action[:] = np.any(~valid_move | ~valid_aim, axis=1)
        move[~valid_move] = 0
        aim[~valid_aim] = 0
        np.clip(move, -1, 1, out=move)
        np.clip(aim, -1, 1, out=aim)
        move = _limit_vectors(move, 1.0)
        aim_length = np.linalg.norm(aim, axis=-1)
        desired_angle = self.attack_angle.copy()
        aimed = aim_length > 1e-6
        desired_angle[aimed] = np.arctan2(aim[..., 1][aimed], aim[..., 0][aimed])

        for _ in range(self.config.physics_substeps):
            self._substep(move, desired_angle)
        self._update_territory()
        self._accumulate_control()
        running = ~self.done
        self.step_count[running] += 1
        self._finish_episodes()
        return self.state

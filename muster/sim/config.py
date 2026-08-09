"""Simulation configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from muster.sim.constants import (
    ARENA_WALL_APOTHEMS_TILES,
    SQRT_3,
    WORLD_HEIGHT_TILES,
    WORLD_WIDTH_TILES,
)

@dataclass(frozen=True, slots=True)
class Config:
    soldiers_per_team: int = 256
    world_width: float = 88.5
    world_height: float = 88.5 * WORLD_HEIGHT_TILES / WORLD_WIDTH_TILES
    soldier_radius: float = 0.42
    initial_health: float = 100.0
    decision_dt: float = 0.10
    physics_substeps: int = 4
    collision_iterations: int = 10
    maximum_running_speed: float = 6.0
    physics_speed_limit: float = 9.0
    acceleration: float = 11.0
    linear_drag: float = 1.8
    maximum_turn_rate: float = 3.4
    ally_restitution: float = 0.12
    enemy_restitution: float = 1.25
    wall_restitution: float = 0.45
    collision_relaxation: float = 0.95
    attack_recovery_seconds: float = 0.40
    damaging_arc_degrees: float = 120.0
    base_strike_damage: float = 5.0
    minimum_damage_speed: float = 1.6
    minimum_closing_speed: float = 0.20
    damage_scale: float = 5.0
    flank_damage_multiplier: float = 2.0
    maximum_episode_seconds: float = 45.0
    control_radius: float = 8.0
    slope_height: float = 0.0
    slope_gravity: float = 24.0
    river_width: float = 0.0
    bridge_width: float = 10.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{field.name} must be finite")
        if self.soldiers_per_team < 1:
            raise ValueError("soldiers_per_team must be positive")
        if self.world_width <= 2 * self.soldier_radius or self.world_height <= 2 * self.soldier_radius:
            raise ValueError("world must be larger than one soldier")
        if not math.isclose(
            self.world_height,
            self.world_width * WORLD_HEIGHT_TILES / WORLD_WIDTH_TILES,
            rel_tol=1e-6,
        ):
            raise ValueError("world dimensions must match the elongated hex arena")
        if self.soldier_radius <= 0 or self.initial_health <= 0:
            raise ValueError("radius and initial health must be positive")
        if self.decision_dt <= 0 or self.physics_substeps < 1 or self.collision_iterations < 1:
            raise ValueError("timesteps and collision iterations must be positive")
        if not 0 < self.damaging_arc_degrees < 360:
            raise ValueError("damaging arc must be between 0 and 360 degrees")
        if self.physics_speed_limit < self.maximum_running_speed or self.maximum_running_speed < 0:
            raise ValueError("physics speed limit must cover maximum running speed")
        nonnegative = (
            self.acceleration,
            self.linear_drag,
            self.maximum_turn_rate,
            self.ally_restitution,
            self.enemy_restitution,
            self.wall_restitution,
            self.attack_recovery_seconds,
            self.base_strike_damage,
            self.minimum_damage_speed,
            self.minimum_closing_speed,
            self.damage_scale,
            self.flank_damage_multiplier,
            self.maximum_episode_seconds,
            self.slope_gravity,
            self.river_width,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("rates, speeds, damage values, and durations cannot be negative")
        if not 0 < self.collision_relaxation <= 1:
            raise ValueError("collision relaxation must be in (0, 1]")
        if self.control_radius <= 0:
            raise ValueError("control radius must be positive")
        if self.river_width >= self.world_width:
            raise ValueError("river must be narrower than the world")
        if self.river_width > 0 and not 2 * self.soldier_radius < self.bridge_width <= self.world_height:
            raise ValueError("an enabled river needs a bridge wider than one soldier")

    @property
    def soldier_count(self) -> int:
        return 2 * self.soldiers_per_team

    @property
    def diameter(self) -> float:
        return 2 * self.soldier_radius

    @property
    def physics_dt(self) -> float:
        return self.decision_dt / self.physics_substeps

    @property
    def attack_recovery_ticks(self) -> int:
        return math.ceil(self.attack_recovery_seconds / self.physics_dt - 1e-12)

    @property
    def maximum_decision_steps(self) -> int:
        return math.ceil(self.maximum_episode_seconds / self.decision_dt - 1e-12)

    @property
    def arena_apothems(self) -> tuple[float, float, float]:
        """Wall distances for the three point-symmetric wall pairs."""
        size = self.territory_tile_size
        return (
            size * float(ARENA_WALL_APOTHEMS_TILES[0]),
            size * float(ARENA_WALL_APOTHEMS_TILES[1]),
            size * float(ARENA_WALL_APOTHEMS_TILES[2]),
        )

    @property
    def territory_tile_size(self) -> float:
        return self.world_width / WORLD_WIDTH_TILES

"""World-space helpers over the territory grid."""

from __future__ import annotations

import numpy as np

from muster.sim.config import Config
from muster.sim.constants import (
    ARENA_NORMALS,
    ARENA_VERTICES_TILES,
    SQRT_3,
    STRONGPOINT_CENTERS,
    TERRITORY_COORDINATES,
    TERRITORY_LOOKUP,
    TERRITORY_LOOKUP_Q_DIAMETER,
    TERRITORY_LOOKUP_Q_RADIUS,
    TERRITORY_LOOKUP_R_RADIUS,
)

def arena_vertices(config: Config) -> np.ndarray:
    """Return the six world-space vertices, counter-clockwise from the east."""
    center = np.array((config.world_width / 2, config.world_height / 2), np.float32)
    return center + config.territory_tile_size * ARENA_VERTICES_TILES

def arena_contains(position: np.ndarray, config: Config, margin: float = 0.0) -> np.ndarray:
    """Whether points are within every wall, optionally inset by ``margin``."""
    relative = np.asarray(position) - (config.world_width / 2, config.world_height / 2)
    apothems = np.tile(np.asarray(config.arena_apothems, np.float32), 2)
    return np.max(relative @ ARENA_NORMALS.T - apothems, axis=-1) <= -margin + 1e-5

def territory_centers(config: Config) -> np.ndarray:
    """World-space centers in the same compact order as territory ownership."""
    q, r = TERRITORY_COORDINATES.T
    size = config.territory_tile_size
    return np.stack(
        (
            config.world_width / 2 + 1.5 * size * q,
            config.world_height / 2 + SQRT_3 * size * (r + 0.5 * q),
        ),
        axis=-1,
    ).astype(np.float32)

def strongpoint_world_centers(config: Config) -> np.ndarray:
    """World-space centers of the three strongpoint hexes."""
    matches = (TERRITORY_COORDINATES[:, None] == STRONGPOINT_CENTERS[None]).all(-1)
    return territory_centers(config)[matches.any(-1)]

def territory_indices(position: np.ndarray, config: Config) -> np.ndarray:
    """Map world positions to their closest valid axial territory tile."""
    local = (np.asarray(position) - (config.world_width / 2, config.world_height / 2))
    local = local / config.territory_tile_size
    fractional = np.stack(
        (2.0 * local[..., 0] / 3.0, -local[..., 0] / 3.0 + local[..., 1] / SQRT_3),
        axis=-1,
    )
    cube = np.concatenate((fractional, -fractional.sum(axis=-1, keepdims=True)), axis=-1)
    rounded = np.floor(cube + 0.5).astype(np.int32)
    difference = np.abs(rounded - cube)
    largest = np.argmax(difference, axis=-1)
    for axis in range(3):
        selected = largest == axis
        other = [value for value in range(3) if value != axis]
        rounded[..., axis][selected] = -rounded[..., other[0]][selected] - rounded[..., other[1]][selected]
    q = np.clip(rounded[..., 0], -TERRITORY_LOOKUP_Q_RADIUS, TERRITORY_LOOKUP_Q_RADIUS)
    r = np.clip(rounded[..., 1], -TERRITORY_LOOKUP_R_RADIUS, TERRITORY_LOOKUP_R_RADIUS)
    lookup = (r + TERRITORY_LOOKUP_R_RADIUS) * TERRITORY_LOOKUP_Q_DIAMETER
    lookup += q + TERRITORY_LOOKUP_Q_RADIUS
    return TERRITORY_LOOKUP[lookup]

"""Territory geometry constants and the hex lookup tables."""

from __future__ import annotations

import math

import numpy as np

SQRT_3 = math.sqrt(3.0)

ARENA_NORMALS = np.array(
    [(math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)) for i in range(6)],
    np.float32,
)

TERRITORY_RADIUS = 13

TERRITORY_DIAMETER = 2 * TERRITORY_RADIUS + 1

TERRITORY_COORDINATES = np.array(
    [
        (q, r)
        for q in range(-TERRITORY_RADIUS, TERRITORY_RADIUS + 1)
        for r in range(
            max(-TERRITORY_RADIUS, -q - TERRITORY_RADIUS),
            min(TERRITORY_RADIUS, -q + TERRITORY_RADIUS) + 1,
        )
    ],
    np.int32,
)

TERRITORY_CELLS = len(TERRITORY_COORDINATES)

TERRITORY_INITIAL_OWNER = np.where(
    TERRITORY_COORDINATES[:, 0] < 0,
    0,
    np.where(TERRITORY_COORDINATES[:, 0] > 0, 1, -1),
).astype(np.int32)

STRONGPOINT_CENTERS = np.array(((0, -8), (0, 0), (0, 8)), np.int32)

STRONGPOINT_RADIUS = 1

STRONGPOINT_WEIGHT = 30

_strongpoint_delta = TERRITORY_COORDINATES[:, None] - STRONGPOINT_CENTERS[None]

_strongpoint_distance = np.maximum.reduce(
    (
        np.abs(_strongpoint_delta[..., 0]),
        np.abs(_strongpoint_delta[..., 1]),
        np.abs(_strongpoint_delta.sum(axis=-1)),
    )
)

STRONGPOINT_MASK = np.any(_strongpoint_distance <= STRONGPOINT_RADIUS, axis=1)

STRONGPOINT_CELLS = np.flatnonzero(STRONGPOINT_MASK).astype(np.int32)

TERRITORY_WEIGHTS = np.where(STRONGPOINT_MASK, STRONGPOINT_WEIGHT, 1).astype(np.int32)

TERRITORY_TOTAL_WEIGHT = int(TERRITORY_WEIGHTS.sum())

# Influence contributions are quantized to fixed point before summation so
# control is exactly independent of soldier iteration order.
INFLUENCE_FIXED_SCALE = 1 << 20

# Per-soldier entity perception (used by the RL observation builder).
ENTITY_NEIGHBORS = 16
ENTITY_RADIUS = 5.0

INFLUENCE_NEUTRAL_FIXED = INFLUENCE_FIXED_SCALE >> 3  # kappa = 0.125

# Rounded axial coordinates can be one cell beyond the board near the six walls.
# This tiny table maps those coordinates to the closest valid territory cell.
TERRITORY_LOOKUP_RADIUS = TERRITORY_RADIUS + 2

TERRITORY_LOOKUP_DIAMETER = 2 * TERRITORY_LOOKUP_RADIUS + 1

_lookup_coordinates = np.array(
    [
        (q, r)
        for r in range(-TERRITORY_LOOKUP_RADIUS, TERRITORY_LOOKUP_RADIUS + 1)
        for q in range(-TERRITORY_LOOKUP_RADIUS, TERRITORY_LOOKUP_RADIUS + 1)
    ],
    np.float32,
)

_lookup_delta = _lookup_coordinates[:, None] - TERRITORY_COORDINATES[None]

_lookup_distance = (1.5 * _lookup_delta[..., 0]) ** 2 + 3.0 * (
    _lookup_delta[..., 1] + 0.5 * _lookup_delta[..., 0]
) ** 2

TERRITORY_LOOKUP = np.argmin(_lookup_distance, axis=1).astype(np.int32)

_coordinate_to_cell = {tuple(value): index for index, value in enumerate(TERRITORY_COORDINATES)}

def territory_neighborhood(radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return nearby cell indices and center offsets for every hex tile.

    Missing neighbors use ``TERRITORY_CELLS`` as a sentinel. Offsets are in
    tile-size units and are ordered with the tile itself first.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    offsets = sorted(
        (
            (q, r)
            for q in range(-radius, radius + 1)
            for r in range(-radius, radius + 1)
            if max(abs(q), abs(r), abs(q + r)) <= radius
        ),
        key=lambda value: (max(abs(value[0]), abs(value[1]), abs(sum(value))), value),
    )
    neighbors = np.array(
        [
            [
                _coordinate_to_cell.get((q + dq, r + dr), TERRITORY_CELLS)
                for dq, dr in offsets
            ]
            for q, r in TERRITORY_COORDINATES
        ],
        np.int32,
    )
    offsets = np.asarray(offsets, np.float32)
    centers = np.stack(
        (1.5 * offsets[:, 0], SQRT_3 * (offsets[:, 1] + 0.5 * offsets[:, 0])),
        axis=-1,
    ).astype(np.float32)
    return neighbors, centers

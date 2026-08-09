"""Territory geometry constants and the hex lookup tables.

Since rules 0.10 the arena is an elongated hexagon: flat north/south walls,
tapering to points at the east and west ends. The long axis is the attack
axis. Cells live at axial coordinates ``(q, r)`` with ``|q| <= 19`` columns
and vertical extent ``|r + q/2| <= min(9.5, 20 - |q|) - 0.5`` rows, giving
flat rows for ``|q| <= 10`` and one-row-per-column tapers to the points.
"""

from __future__ import annotations

import math

import numpy as np

SQRT_3 = math.sqrt(3.0)

# Lattice shape: columns span [-TERRITORY_COLUMN_EXTENT, +TERRITORY_COLUMN_EXTENT];
# the widest columns hold rows |r + q/2| <= TERRITORY_ROW_EXTENT.
TERRITORY_COLUMN_EXTENT = 19
TERRITORY_ROW_EXTENT = 9

# World size in tile units: x spans (3 * columns + 2) / 2 tile sizes per side,
# y spans sqrt(3) * (rows + 0.5) per side.
WORLD_WIDTH_TILES = 3.0 * TERRITORY_COLUMN_EXTENT + 2.0
WORLD_HEIGHT_TILES = SQRT_3 * (2.0 * TERRITORY_ROW_EXTENT + 1.0)


def _row_half_extent(q: int) -> float:
    return min(
        TERRITORY_ROW_EXTENT + 0.5,
        TERRITORY_COLUMN_EXTENT + 1 - abs(q),
    ) - 0.5


TERRITORY_COORDINATES = np.array(
    [
        (q, r)
        for q in range(-TERRITORY_COLUMN_EXTENT, TERRITORY_COLUMN_EXTENT + 1)
        for r in range(-TERRITORY_COLUMN_EXTENT * 2, TERRITORY_COLUMN_EXTENT * 2 + 1)
        if abs(r + q / 2.0) <= _row_half_extent(q) + 1e-9
    ],
    np.int32,
)

TERRITORY_CELLS = len(TERRITORY_COORDINATES)

TERRITORY_INITIAL_OWNER = np.where(
    TERRITORY_COORDINATES[:, 0] < 0,
    0,
    np.where(TERRITORY_COORDINATES[:, 0] > 0, 1, -1),
).astype(np.int32)

# Assault scoring (rules 0.11): each team owns one radius-1 base deep in its
# half and scores ONLY through influence on the enemy's base (plus a small
# symmetric value on plain ground). A team's own base is worth nothing to it
# except denial: defenders standing on it dilute the attacker's share.
# The pair is x-mirror symmetric (both centers sit at +0.5 rows), matching
# the actor's side canonicalization, which reflects across x.
BASE_CENTERS = np.array(((-13, 7), (13, -6)), np.int32)

BASE_RADIUS = 1

BASE_WEIGHT = 250

_base_delta = TERRITORY_COORDINATES[:, None] - BASE_CENTERS[None]

_base_distance = np.maximum.reduce(
    (
        np.abs(_base_delta[..., 0]),
        np.abs(_base_delta[..., 1]),
        np.abs(_base_delta.sum(axis=-1)),
    )
)

# BASE_MASKS[t] marks the tiles of team t's OWN base (west for team 0).
BASE_MASKS = (_base_distance <= BASE_RADIUS).T

BASE_CELLS_BY_TEAM = [np.flatnonzero(mask).astype(np.int32) for mask in BASE_MASKS]

# TEAM_TERRITORY_WEIGHTS[t, c] is what a unit of team t's control share on
# cell c is worth to team t: plain ground 1, the enemy base BASE_WEIGHT,
# its own base 0.
TEAM_TERRITORY_WEIGHTS = np.ones((2, len(TERRITORY_COORDINATES)), np.int32)
for _team in (0, 1):
    TEAM_TERRITORY_WEIGHTS[_team, BASE_MASKS[1 - _team]] = BASE_WEIGHT
    TEAM_TERRITORY_WEIGHTS[_team, BASE_MASKS[_team]] = 0

TEAM_TOTAL_WEIGHT = int(TEAM_TERRITORY_WEIGHTS[0].sum())
assert TEAM_TOTAL_WEIGHT == int(TEAM_TERRITORY_WEIGHTS[1].sum())

# Legacy aliases kept for display and scripted-opponent code: the union of
# both bases renders as highlighted ground, and the symmetric weight map is
# the per-cell maximum of the two team maps.
STRONGPOINT_CENTERS = BASE_CENTERS
STRONGPOINT_WEIGHT = BASE_WEIGHT
STRONGPOINT_MASK = BASE_MASKS.any(axis=0)
STRONGPOINT_CELLS = np.flatnonzero(STRONGPOINT_MASK).astype(np.int32)

# Arena walls: three point-symmetric pairs. Normals point outward for the
# positive member of each pair; the negative member mirrors through the
# center. Apothems are in tile-size units, computed as the support of the
# union of cell hexagons (cell center plus corner) in each normal direction.
_slant = np.array((SQRT_3, 1.5), np.float64)
_slant /= np.linalg.norm(_slant)
ARENA_NORMALS = np.array(
    [
        (0.0, 1.0),
        (_slant[0], _slant[1]),
        (-_slant[0], _slant[1]),
        (0.0, -1.0),
        (-_slant[0], -_slant[1]),
        (_slant[0], -_slant[1]),
    ],
    np.float32,
)

_cell_centers_tiles = np.stack(
    (
        1.5 * TERRITORY_COORDINATES[:, 0].astype(np.float64),
        SQRT_3
        * (
            TERRITORY_COORDINATES[:, 1].astype(np.float64)
            + 0.5 * TERRITORY_COORDINATES[:, 0].astype(np.float64)
        ),
    ),
    axis=-1,
)

_corner_angles = np.arange(6) * (math.pi / 3.0)
_cell_corners_tiles = np.stack(
    (np.cos(_corner_angles), np.sin(_corner_angles)), axis=-1
)

_support_points = (
    _cell_centers_tiles[:, None, :] + _cell_corners_tiles[None, :, :]
).reshape(-1, 2)

ARENA_WALL_APOTHEMS_TILES = np.max(
    _support_points @ ARENA_NORMALS[:3].T.astype(np.float64), axis=0
).astype(np.float32)

def _wall_intersection(first: int, second: int) -> tuple[float, float]:
    normals = np.array(
        [ARENA_NORMALS[first], ARENA_NORMALS[second]], np.float64
    )
    apothems = np.array(
        [
            ARENA_WALL_APOTHEMS_TILES[first % 3],
            ARENA_WALL_APOTHEMS_TILES[second % 3],
        ],
        np.float64,
    )
    return tuple(np.linalg.solve(normals, apothems))


# Counter-clockwise from the east point: the six wall-line intersections.
ARENA_VERTICES_TILES = np.array(
    [
        _wall_intersection(1, 5),  # east point (NE slant meets SE slant)
        _wall_intersection(0, 1),  # northeast (top meets NE slant)
        _wall_intersection(0, 2),  # northwest (top meets NW slant)
        _wall_intersection(2, 4),  # west point
        _wall_intersection(3, 4),  # southwest
        _wall_intersection(3, 5),  # southeast
    ],
    np.float32,
)

# Influence contributions are quantized to fixed point before summation so
# control is exactly independent of soldier iteration order.
INFLUENCE_FIXED_SCALE = 1 << 20

# Per-soldier entity perception (used by the RL observation builder).
ENTITY_NEIGHBORS = 16
ENTITY_RADIUS = 5.0

INFLUENCE_NEUTRAL_FIXED = INFLUENCE_FIXED_SCALE >> 3  # kappa = 0.125

# Rounded axial coordinates can be a cell or two beyond the board near the
# walls. This table maps any nearby coordinate to the closest valid cell.
TERRITORY_LOOKUP_Q_RADIUS = TERRITORY_COLUMN_EXTENT + 2

TERRITORY_LOOKUP_R_RADIUS = int(np.max(np.abs(TERRITORY_COORDINATES[:, 1]))) + 2

TERRITORY_LOOKUP_Q_DIAMETER = 2 * TERRITORY_LOOKUP_Q_RADIUS + 1

TERRITORY_LOOKUP_R_DIAMETER = 2 * TERRITORY_LOOKUP_R_RADIUS + 1

_lookup_coordinates = np.array(
    [
        (q, r)
        for r in range(-TERRITORY_LOOKUP_R_RADIUS, TERRITORY_LOOKUP_R_RADIUS + 1)
        for q in range(-TERRITORY_LOOKUP_Q_RADIUS, TERRITORY_LOOKUP_Q_RADIUS + 1)
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

"""Simulator package: constants, configuration, geometry, and the CPU reference."""

from muster.sim.constants import *  # noqa: F401,F403
from muster.sim.constants import territory_neighborhood  # noqa: F401
from muster.sim.config import Config  # noqa: F401
from muster.sim.geometry import (  # noqa: F401
    arena_contains,
    arena_vertices,
    strongpoint_world_centers,
    territory_centers,
    territory_indices,
)
from muster.sim.cpu import CpuSimulator  # noqa: F401

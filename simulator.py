"""Compatibility facade over :mod:`muster.sim`."""

from muster.sim import *  # noqa: F401,F403
from muster.sim.constants import (  # noqa: F401
    INFLUENCE_FIXED_SCALE,
    INFLUENCE_NEUTRAL_FIXED,
    TERRITORY_INITIAL_OWNER,
    TERRITORY_LOOKUP,
    TERRITORY_LOOKUP_DIAMETER,
    TERRITORY_LOOKUP_RADIUS,
)
from muster.sim import Config, CpuSimulator  # noqa: F401

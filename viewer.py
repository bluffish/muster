"""Compatibility facade over :mod:`muster.viewer`."""

from muster.viewer import *  # noqa: F401,F403
from muster.viewer.replay import record_episode, write_replay  # noqa: F401
from muster.viewer.scripted import local_march_policy, nearest_enemy_policy  # noqa: F401

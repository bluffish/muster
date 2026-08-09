"""Replay packing, browser template application, and episode recording."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from muster.sim import Config
from muster.sim.constants import (
    STRONGPOINT_CELLS,
    STRONGPOINT_WEIGHT,
    TERRITORY_CELLS,
    TERRITORY_RADIUS,
)

_REPLAY_HTML = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")

Snapshot = Mapping[str, object]

Policy = Callable[[Snapshot, Config], np.ndarray]

def _encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")

def _pack_replay(replay: Mapping[str, object]) -> dict[str, object]:
    """Quantize frame data into compact browser-native typed arrays."""
    frames = list(replay["frames"])
    if not frames:
        raise ValueError("replay needs at least one frame")
    config = replay["config"]
    positions = np.asarray([frame["p"] for frame in frames], dtype=np.float32)
    angles = np.asarray([frame["a"] for frame in frames], dtype=np.float32)
    health = np.asarray([frame["h"] for frame in frames], dtype=np.float32)
    territory_owner = (
        np.asarray([frame["o"] for frame in frames], dtype=np.int8)
        if all("o" in frame for frame in frames)
        else None
    )
    control_share = (
        np.asarray([frame["c"] for frame in frames], dtype=np.float32)
        if all("c" in frame for frame in frames)
        else None
    )
    expected = (len(frames), len(replay["team"]))
    if positions.shape != (*expected, 2) or angles.shape != expected or health.shape != expected:
        raise ValueError("replay frame shapes do not match its team array")
    if territory_owner is not None:
        territory_shape = (
            (len(frames), int(config["territory_cells"]))
            if config.get("territory_radius")
            else (
                len(frames),
                int(config["territory_height"]),
                int(config["territory_width"]),
            )
        )
        if territory_owner.shape != territory_shape:
            raise ValueError(f"expected territory frames with shape {territory_shape}")

    world_size = np.array(
        [float(config["world_width"]), float(config["world_height"])], dtype=np.float32
    )
    position_u16 = np.rint(np.clip(positions / world_size, 0, 1) * 65535).astype("<u2")
    wrapped_angles = (angles + math.pi) % (2 * math.pi) - math.pi
    angle_i16 = np.rint(wrapped_angles * (32767 / math.pi)).astype("<i2")
    initial_health = max(float(config["initial_health"]), 1e-6)
    health_u16 = np.rint(np.clip(health / initial_health, 0, 1) * 65535).astype("<u2")

    packed = {key: value for key, value in replay.items() if key != "frames"}
    packed.update(
        frame_count=len(frames),
        position_u16=_encode_array(position_u16),
        angle_i16=_encode_array(angle_i16),
        health_u16=_encode_array(health_u16),
    )
    if territory_owner is not None:
        packed["territory_i8"] = _encode_array(territory_owner)
    if control_share is not None:
        quantized = np.rint(np.clip(control_share, 0, 1) * 255).astype(np.uint8)
        packed["control_u8"] = _encode_array(quantized)
    return packed

def write_replay(replay: Mapping[str, object], path: str | Path) -> None:
    """Write a compact, self-contained browser replay."""
    payload = json.dumps(_pack_replay(replay), separators=(",", ":"))
    Path(path).write_text(_REPLAY_HTML.replace("__REPLAY__", payload), encoding="utf-8")

def record_episode(simulator: object, policy: Policy | None = None) -> dict[str, object]:
    """Run the current one-environment episode and return compact replay data."""
    if policy is None:
        from muster.viewer.scripted import nearest_enemy_policy as policy
    if simulator.num_envs != 1:
        raise ValueError("episode recording expects exactly one environment")
    config = simulator.config
    state = simulator.snapshot(0)
    started = time.perf_counter()
    steps = 0
    replay: dict[str, object] = {
        "config": {
            "world_width": config.world_width,
            "world_height": config.world_height,
            "soldier_radius": config.soldier_radius,
            "initial_health": config.initial_health,
            "decision_dt": config.decision_dt,
            "river_width": config.river_width,
            "bridge_width": config.bridge_width,
            "arena_shape": "hex",
            "territory_radius": TERRITORY_RADIUS,
            "territory_cells": TERRITORY_CELLS,
            "strongpoint_cells": STRONGPOINT_CELLS.tolist(),
            "strongpoint_weight": STRONGPOINT_WEIGHT,
        },
        "team": np.asarray(state["team"], dtype=int).tolist(),
        "frames": [],
    }
    frames = replay["frames"]
    for _ in range(config.maximum_decision_steps + 1):
        frames.append(
            {
                "p": np.asarray(state["position"]).round(4).tolist(),
                "a": np.asarray(state["attack_angle"]).round(4).tolist(),
                "h": np.asarray(state["health"]).round(3).tolist(),
                "o": np.asarray(state["territory_owner"], dtype=np.int8).tolist(),
                "c": np.asarray(state["control_share"], dtype=np.float32),
            }
        )
        if state["done"]:
            elapsed = time.perf_counter() - started
            replay["winner"] = int(state["winner"])
            replay["statistics"] = {
                "decision_steps": steps,
                "simulated_seconds": steps * config.decision_dt,
                "recording_wall_seconds": elapsed,
                "environment_decisions_per_second": steps / max(elapsed, 1e-9),
                "soldier_decisions_per_second": steps * simulator.num_soldiers / max(elapsed, 1e-9),
            }
            return replay
        simulator.step(policy(state, config))
        steps += 1
        state = simulator.snapshot(0)
    raise RuntimeError("episode did not finish within its configured limit")

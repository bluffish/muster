"""Optional interactive pygame viewer."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import math
import time

import numpy as np

from muster.sim import Config, CpuSimulator, territory_centers
from muster.sim.constants import (
    ARENA_NORMALS,
    SQRT_3,
    STRONGPOINT_CELLS,
    TERRITORY_CELLS,
)
from muster.viewer.replay import Policy, Snapshot, record_episode, write_replay
from muster.viewer.scripted import nearest_enemy_policy

class Viewer:
    """A read-only renderer. ``draw`` returns false after quit or Escape."""

    def __init__(self, config: Config, width: int = 1000, title: str = "Simulator"):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("install the viewer extra: pip install -e '.[viewer]'") from error
        self.pg = pygame
        self.config = config
        world_ratio = config.world_height / config.world_width
        self.size = width, max(300, int(width * world_ratio) + 50)
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        try:
            self.font = pygame.font.Font(None, 22)
        except (ImportError, NotImplementedError):
            self.font = None
        self.clock = pygame.time.Clock()
        self.closed = False

    def _point(self, position: np.ndarray, scale: float, left: float, bottom: float) -> tuple[int, int]:
        return round(left + position[0] * scale), round(bottom - position[1] * scale)

    def draw(self, state: Snapshot, caption: str = "") -> bool:
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.closed = True
        if self.closed:
            return False

        c = self.config
        margin, header = 16, 30
        scale = min((self.size[0] - 2 * margin) / c.world_width, (self.size[1] - header - margin) / c.world_height)
        left = (self.size[0] - c.world_width * scale) / 2
        top = header
        bottom = top + c.world_height * scale
        self.screen.fill((20, 20, 20))
        angles = np.arange(6) * math.pi / 3
        offsets = c.territory_tile_size * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        arena = [
            [self._point(point, scale, left, bottom) for point in center + offsets]
            for center in territory_centers(c)
        ]
        strongpoints = set(STRONGPOINT_CELLS.tolist())
        for cell, tile in enumerate(arena):
            color = (72, 59, 27) if cell in strongpoints else (43, 43, 43)
            pg.draw.polygon(self.screen, color, tile)

        if c.river_width > 0:
            river_x = left + (c.world_width - c.river_width) * 0.5 * scale
            river_w = c.river_width * scale
            bridge_lo = c.world_height * 0.5 - c.bridge_width * 0.5
            bridge_hi = c.world_height * 0.5 + c.bridge_width * 0.5
            pg.draw.rect(self.screen, (30, 70, 95), (river_x, bottom - bridge_lo * scale, river_w, bridge_lo * scale))
            pg.draw.rect(self.screen, (30, 70, 95), (river_x, top, river_w, (c.world_height - bridge_hi) * scale))

        position = np.asarray(state["position"])
        team = np.asarray(state["team"])
        angle = np.asarray(state["attack_angle"])
        health = np.asarray(state["health"])
        alive = np.asarray(state["alive"], dtype=bool)
        radius = max(2, round(c.soldier_radius * scale))
        colors = ((53, 167, 255), (255, 77, 95))
        for soldier in np.flatnonzero(alive):
            center = self._point(position[soldier], scale, left, bottom)
            color = colors[int(team[soldier])] if int(team[soldier]) in (0, 1) else (190, 190, 190)
            pg.draw.circle(self.screen, color, center, radius)
            pg.draw.circle(self.screen, (7, 17, 22), center, radius, max(1, round(radius * 0.14)))
            arc = float(angle[soldier]) + np.linspace(-math.pi / 3, math.pi / 3, 17)
            arc_position = position[soldier] + np.stack((np.cos(arc), np.sin(arc)), axis=-1) * c.soldier_radius * 1.17
            arc_points = [self._point(point, scale, left, bottom) for point in arc_position]
            pg.draw.lines(self.screen, (255, 224, 138), False, arc_points, max(1, round(radius * 0.25)))
            fraction = max(0.0, min(1.0, float(health[soldier]) / c.initial_health))
            if fraction < 1:
                health_arc = np.linspace(0, 2 * math.pi * fraction, max(2, math.ceil(25 * fraction)))
                health_position = position[soldier] + np.stack((np.cos(health_arc), np.sin(health_arc)), axis=-1) * c.soldier_radius * 0.72
                health_points = [self._point(point, scale, left, bottom) for point in health_position]
                pg.draw.lines(self.screen, (217, 242, 220), False, health_points, max(1, round(radius * 0.15)))

        status = f"step {state.get('step_count', '?')}"
        if state.get("done", False):
            winner = state.get("winner", -1)
            status += "  draw" if winner == -1 else f"  team {winner} wins"
        if caption:
            status += f"  {caption}"
        if self.font is not None:
            self.screen.blit(self.font.render(status, True, (225, 225, 225)), (margin, 7))
        for cell, tile in enumerate(arena):
            color = (255, 196, 64) if cell in strongpoints else (60, 60, 60)
            pg.draw.polygon(self.screen, color, tile, 2 if cell in strongpoints else 1)
        pg.display.flip()
        return True

    def close(self) -> None:
        if not self.closed:
            self.pg.quit()
            self.closed = True

def run(simulator: object, policy: Policy = nearest_enemy_policy, fps: int = 20) -> None:
    """Run a one-environment simulator with any ``policy(snapshot, config)`` callback."""
    if simulator.num_envs != 1:
        raise ValueError("standalone viewer run expects exactly one environment")
    viewer = Viewer(simulator.config)
    state = simulator.snapshot(0)
    try:
        while viewer.draw(state):
            if state["done"]:
                simulator.reset()
            else:
                simulator.step(policy(state, simulator.config))
            state = simulator.snapshot(0)
            viewer.clock.tick(fps)
    finally:
        viewer.close()


def load_checkpoint_policy(
    simulator: object, checkpoint: Mapping[str, object]
) -> Policy:
    """Build a deterministic snapshot callback from a training checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install the RL extra: pip install -e '.[rl]'") from error
    from policy import CHECKPOINT_VERSION, Policy as NeuralPolicy
    from rl_env import RLEnv

    env = RLEnv(simulator=simulator)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint predates the local hex policy and cannot be viewed")
    model = NeuralPolicy(**checkpoint.get("model_kwargs", {})).to(env.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    opponent = None
    pool = checkpoint.get("opponent_pool")
    if pool is not None:
        opponent = NeuralPolicy(**checkpoint.get("model_kwargs", {})).to(env.device)
        opponent.load_state_dict(pool["models"][int(pool["latest_slot"])])
        opponent.eval()

    def act(_state: Snapshot, _config: Config):
        state = env.observe()
        with torch.no_grad():
            actions, _, _ = model.act(state, deterministic=True)
            if opponent is not None:
                actions[:, 1] = opponent.actor_actions(state, deterministic=True)[:, 1]
        world_actions = env.actions_to_sim(actions)
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        return world_actions

    return act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--soldiers", type=int, default=256, help="soldiers per team")
    parser.add_argument("--record", metavar="HTML", help="record one episode as a browser replay")
    parser.add_argument("--checkpoint", metavar="PT", help="run a deterministic neural policy")
    args = parser.parse_args()
    checkpoint = None
    if args.checkpoint:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install the RL extra: pip install -e '.[rl]'") from error
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = Config(**checkpoint["sim_config"])
    else:
        config = Config(soldiers_per_team=args.soldiers)
    if args.device == "cpu" and checkpoint is None:
        simulator = CpuSimulator(config)
    else:
        from simulator_gpu import GpuSimulator

        simulator = GpuSimulator(config, device=args.device)
    selected_policy = (
        load_checkpoint_policy(simulator, checkpoint) if checkpoint is not None else nearest_enemy_policy
    )
    if args.record:
        replay = record_episode(simulator, selected_policy)
        write_replay(replay, args.record)
        print(args.record)
        statistics = replay["statistics"]
        print(
            f"{statistics['environment_decisions_per_second']:,.1f} recorded environment decisions/s; "
            f"{statistics['soldier_decisions_per_second']:,.0f} soldier decisions/s"
        )
    else:
        run(simulator, selected_policy, fps=args.fps)

if __name__ == "__main__":
    main()

"""Run a simple end-to-end simulator step benchmark."""

import argparse
import time

import numpy as np
import warp as wp

from simulator import Config
from simulator_gpu import ACTION_DTYPE, GpuSimulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--charge", action="store_true")
    args = parser.parse_args()

    sim = GpuSimulator(Config(), args.envs, args.device)
    if args.charge:
        raw = np.zeros((args.envs, sim.num_soldiers, 4), np.float32)
        split = sim.config.soldiers_per_team
        raw[:, :split, (0, 2)] = 1
        raw[:, split:, (0, 2)] = -1
        actions = wp.array(raw.reshape(-1, 4), dtype=ACTION_DTYPE, device=sim.device)
    else:
        actions = wp.zeros(sim.total_soldiers, dtype=ACTION_DTYPE, device=sim.device)
    for _ in range(2):
        sim.step(actions)
    wp.synchronize_device(sim.device)
    start = time.perf_counter()
    for _ in range(args.steps):
        sim.step(actions)
    wp.synchronize_device(sim.device)
    elapsed = time.perf_counter() - start
    decisions = args.envs * args.steps
    print(f"{decisions / elapsed:,.0f} environment decisions/s")
    print(f"{decisions * sim.num_soldiers / elapsed:,.0f} soldier decisions/s")
    print(f"{elapsed / args.steps * 1e3:.3f} ms/batched step")


if __name__ == "__main__":
    main()

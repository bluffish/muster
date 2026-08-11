"""Distill the perception-limited assault charger into the Policy network.

Teacher (per soldier): if the nearest living enemy is within visual range
(ENTITY_RADIUS), move toward it and face it; otherwise march on the ENEMY
base (stop within one unit) — under assault scoring that is the only
ground worth taking. Modes are sampled uniformly during cloning so the
student's behavior is mode-invariant.

Saves a warm-start-complete checkpoint (model, model_kwargs, fresh
optimizer state, sim_config, version) at runs/distilled_charger.pt,
relative to the repository root this script lives in.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from muster.rl.env import RLEnv
from muster.rl.policy import CHECKPOINT_VERSION, Policy
from muster.rl.rollout import collect_rollout, make_rollout
from muster.sim import Config, strongpoint_world_centers
from muster.sim.constants import ENTITY_RADIUS

DEVICE = "cuda"
NUM_ENVS = 64
STEPS = 3000
LOG_EVERY = 250
OUTPUT = REPO / "runs" / "distilled_charger.pt"
MODEL_KWARGS = {
    "hidden_size": 256,
    "entity_size": 16,
    "tile_size": 32,
    "local_radius": 2,
    "mode_count": 16,
    "mode_size": 16,
    "log_std_floor": -0.8,
    "memory_size": 64,
    "message_size": 8,
    "role_count": 8,
    "role_size": 8,
}


def teacher_actions(env, bases):
    """Canonical-frame assault-charger targets for every soldier."""
    positions = env._team_positions  # (envs, 2, S, 2) world frame
    alive = env.alive.bool()
    envs, teams, S, _ = positions.shape
    flat = positions.reshape(envs, teams * S, 2)
    flat_alive = alive.reshape(envs, teams * S)
    distance = torch.cdist(flat, flat)
    team_of = torch.arange(teams * S, device=positions.device) // S
    enemy_pair = team_of.view(-1, 1) != team_of.view(1, -1)
    mask = enemy_pair.unsqueeze(0) & flat_alive.unsqueeze(1)
    distance = distance.masked_fill(~mask, 1e9)
    nearest_dist, nearest = distance.min(-1)
    target = torch.take_along_dim(flat, nearest.unsqueeze(-1).expand(-1, -1, 2), dim=1)
    visible = nearest_dist <= ENTITY_RADIUS

    # March on the enemy base whenever no enemy is in sight.
    enemy_base = bases[1 - team_of]  # (2S, 2)
    base_delta = enemy_base.unsqueeze(0) - flat
    base_far = base_delta.norm(dim=-1) > 1.0

    delta = torch.where(visible.unsqueeze(-1), target - flat, base_delta)
    go = visible | base_far
    direction = torch.nn.functional.normalize(delta, dim=-1)
    direction = direction * (go & flat_alive).unsqueeze(-1)

    world = direction.view(envs, teams, S, 2)
    x_sign = torch.tensor([1.0, -1.0], device=positions.device).view(1, 2, 1)
    mirror = env.mirror_y.view(-1, 1, 1)
    cx = world[..., 0] * x_sign
    cy = world[..., 1] * mirror
    return torch.stack((cx, cy, cx, cy), dim=-1)


def main():
    student = Policy(**MODEL_KWARGS).to(DEVICE)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, 1e-5)

    env = RLEnv(
        Config(),
        NUM_ENVS,
        DEVICE,
        mode_count=MODEL_KWARGS["mode_count"],
        role_count=MODEL_KWARGS["role_count"],
    )
    bases = torch.as_tensor(
        strongpoint_world_centers(env.config), dtype=torch.float32, device=DEVICE
    )
    state = env.reset()
    memory = student.initial_memory(state)

    losses, agreements = [], []
    for step in range(STEPS):
        target = teacher_actions(env, bases)
        predicted, new_memory = student.actor_step(
            state, memory.detach(), deterministic=True
        )
        alive = state.alive.bool()
        weight = alive.unsqueeze(-1).float()
        loss = ((predicted - target * 0.95).square() * weight).sum() / weight.sum().clamp_min(1) / 4
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        memory = new_memory.detach()

        with torch.no_grad():
            move_p = torch.nn.functional.normalize(predicted[..., :2].float(), dim=-1)
            move_t = torch.nn.functional.normalize(target[..., :2].float() + 1e-9, dim=-1)
            moving = alive & (target[..., :2].norm(dim=-1) > 0.1)
            cosine = ((move_p * move_t).sum(-1) * moving.float()).sum() / moving.float().sum().clamp_min(1)
            losses.append(float(loss))
            agreements.append(float(cosine))
            state, facts = env.step(target)
            done = facts["done"].bool()
            if bool(done.any()):
                memory[done] = 0.0
        if (step + 1) % LOG_EVERY == 0:
            n = LOG_EVERY
            print(
                f"step {step + 1}: loss {sum(losses[-n:]) / n:.5f}  move-cosine {sum(agreements[-n:]) / n:.4f}",
                flush=True,
            )

    # Behavior cloning teaches the student to IGNORE roles (the teacher is
    # role-blind), which would zero out the exploration bias roles exist to
    # provide. Reinitialize the role pathway so the warm start carries the
    # charger behavior plus fresh, visible per-soldier role biases.
    if MODEL_KWARGS["role_count"]:
        torch.nn.init.orthogonal_(student.role_embedding.weight)
        torch.nn.init.orthogonal_(student.role_policy_bias.weight, 0.3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fresh = torch.optim.Adam(student.parameters(), lr=1e-3, eps=1e-5)
    torch.save(
        {
            "model": student.state_dict(),
            "model_kwargs": MODEL_KWARGS,
            "optimizer": fresh.state_dict(),
            "sim_config": asdict(Config()),
            "version": CHECKPOINT_VERSION,
            "update": 0,
            "distilled": "assault charger: attack within r=5, else enemy base",
        },
        OUTPUT,
    )
    print("saved", OUTPUT)

    # Validation: one mirror rollout through the exact training path.
    student.eval()
    with torch.no_grad():
        state = env.reset()
        rollout = make_rollout(150, state, student.memory_size, 15)
        learner_teams = torch.zeros((NUM_ENVS, 2), dtype=torch.bool, device=DEVICE)
        indices = torch.arange(NUM_ENVS, device=DEVICE)
        learner_teams[indices, indices.remainder(2)] = True
        collect_rollout(
            env,
            student,
            rollout,
            state,
            learner_teams,
            None,
            None,
            action_repeat=3,
            learner_memory=student.initial_memory(state),
            opponent_memory=None,
            bptt_window=15,
        )
        damage = rollout["damage"].sum(0)
        sides = indices.remainder(2)
        print(
            "validation mirror rollout: learner_taken=%.0f opponent_taken=%.0f"
            % (
                damage[indices, sides].mean().item(),
                damage[indices, 1 - sides].mean().item(),
            )
        )


if __name__ == "__main__":
    main()

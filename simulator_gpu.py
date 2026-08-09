"""GPU-resident Warp backend with spatially local collision queries."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import warp as wp

from simulator import (
    INFLUENCE_FIXED_SCALE,
    INFLUENCE_NEUTRAL_FIXED,
    TERRITORY_WEIGHTS,
    TERRITORY_CELLS,
    TERRITORY_INITIAL_OWNER,
    TERRITORY_LOOKUP,
    TERRITORY_LOOKUP_DIAMETER,
    TERRITORY_LOOKUP_RADIUS,
    TERRITORY_TOTAL_WEIGHT,
    Config,
    CpuSimulator,
    territory_centers,
)


ACTION_DTYPE = wp.vec4
COLLISION_FIXED_SCALE = 65536.0
POSITION_FIXED_SCALE = 16384.0


@wp.struct
class _Params:
    num_soldiers: int
    world_width: float
    world_height: float
    radius: float
    diameter: float
    physics_dt: float
    maximum_running_speed: float
    physics_speed_limit: float
    acceleration: float
    drag_factor: float
    maximum_turn_delta: float
    terrain_acceleration_x: float
    ally_restitution: float
    enemy_restitution: float
    wall_restitution: float
    collision_relaxation: float
    collision_fixed_scale: float
    collision_fixed_inverse: float
    attack_recovery_ticks: int
    arc_cos: float
    base_strike_damage: float
    minimum_damage_speed: float
    minimum_closing_speed: float
    damage_scale: float
    flank_multiplier: float
    river_width: float
    bridge_width: float
    maximum_decision_steps: int
    territory_tile_size: float
    territory_lookup_radius: int
    territory_lookup_diameter: int
    territory_cells: int
    territory_control_sigma: float
    territory_total_weight: float


@wp.func
def _limit(vector: wp.vec2, maximum: float) -> wp.vec2:
    length = wp.length(vector)
    if length > maximum and length > 0.0:
        return vector * (maximum / length)
    return vector


@wp.func
def _wrap(angle: float) -> float:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def _normal(delta: wp.vec2, local: int, other: int) -> wp.vec2:
    distance_sq = wp.dot(delta, delta)
    if distance_sq > 1.0e-20:
        return delta / wp.sqrt(distance_sq)
    if local < other:
        return wp.vec2(1.0, 0.0)
    return wp.vec2(-1.0, 0.0)


@wp.func
def _collision_fixed(value: float, scale: float) -> int:
    """Quantize a contact contribution with symmetric rounding."""
    scaled = value * scale
    if scaled >= 0.0:
        return int(wp.floor(scaled + 0.5))
    return -int(wp.floor(-scaled + 0.5))


@wp.func
def _facing(angle: float) -> wp.vec2:
    return wp.vec2(wp.cos(angle), wp.sin(angle))


@wp.func
def _resolve_static(position: wp.vec2, velocity: wp.vec2, p: _Params) -> wp.vec4:
    x = position[0]
    y = position[1]
    vx = velocity[0]
    vy = velocity[1]
    r = p.radius
    limit = p.world_width * 0.5 - r
    for iteration in range(2):
        for axis in range(3):
            nx = float(1.0)
            ny = float(0.0)
            if axis == 1:
                nx = 0.5
                ny = 0.8660254037844386
            elif axis == 2:
                nx = -0.5
                ny = 0.8660254037844386
            point = wp.vec2(x, y)
            signed_distance = x * nx + y * ny
            side = float(1.0)
            if signed_distance < 0.0:
                side = -1.0
            normal = wp.vec2(side * nx, side * ny)
            distance = wp.abs(signed_distance) - limit
            if distance > 0.0:
                point = point - distance * normal
                x = point[0]
                y = point[1]
                speed = vx * normal[0] + vy * normal[1]
                if speed > 0.0:
                    vx = vx - (1.0 + p.wall_restitution) * speed * normal[0]
                    vy = vy - (1.0 + p.wall_restitution) * speed * normal[1]

    if p.river_width > 0.0:
        x_lo = -p.river_width * 0.5 - r
        x_hi = p.river_width * 0.5 + r
        bridge_lo = -p.bridge_width * 0.5 + r
        bridge_hi = p.bridge_width * 0.5 - r
        if x > x_lo and x < x_hi and (y < bridge_lo or y > bridge_hi):
            left = x - x_lo
            right = x_hi - x
            bridge = bridge_lo - y
            upper = 0
            if y > bridge_hi:
                bridge = y - bridge_hi
                upper = 1
            if x_lo < -p.world_width * 0.5 + r:
                left = 1.0e30
            if x_hi > p.world_width * 0.5 - r:
                right = 1.0e30
            edge = 0
            best = left
            if right < best:
                edge = 1
                best = right
            if bridge < best:
                edge = 2
            if edge == 0:
                x = x_lo
                if vx > 0.0:
                    vx = 0.0
            elif edge == 1:
                x = x_hi
                if vx < 0.0:
                    vx = 0.0
            elif upper == 0:
                y = bridge_lo
                if vy < 0.0:
                    vy = 0.0
            else:
                y = bridge_hi
                if vy > 0.0:
                    vy = 0.0
    return wp.vec4(x, y, vx, vy)


@wp.func
def _territory_cell(
    position: wp.vec2,
    lookup: wp.array(dtype=wp.int32),
    world_width: float,
    world_height: float,
    tile_size: float,
    lookup_radius: int,
    lookup_diameter: int,
) -> int:
    x = (position[0] - world_width * 0.5) / tile_size
    y = (position[1] - world_height * 0.5) / tile_size
    q = 2.0 * x / 3.0
    r = -x / 3.0 + y * 0.5773502691896258
    s = -q - r
    rounded_q = int(wp.floor(q + 0.5))
    rounded_r = int(wp.floor(r + 0.5))
    rounded_s = int(wp.floor(s + 0.5))
    difference_q = wp.abs(float(rounded_q) - q)
    difference_r = wp.abs(float(rounded_r) - r)
    difference_s = wp.abs(float(rounded_s) - s)
    if difference_q > difference_r and difference_q > difference_s:
        rounded_q = -rounded_r - rounded_s
    elif difference_r > difference_s:
        rounded_r = -rounded_q - rounded_s
    rounded_q = wp.clamp(rounded_q, -lookup_radius, lookup_radius)
    rounded_r = wp.clamp(rounded_r, -lookup_radius, lookup_radius)
    index = (rounded_r + lookup_radius) * lookup_diameter + rounded_q + lookup_radius
    return lookup[index]


@wp.func
def _collision_delta(
    grid: wp.uint64,
    grid_position: wp.array(dtype=wp.vec3),
    env: int,
    local: int,
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    team: wp.array(dtype=wp.int32),
    p: _Params,
) -> wp.vec4:
    base = env * p.num_soldiers
    index = base + local
    own_position = position[index]
    own_velocity = velocity[index]
    # Hash-grid traversal order varies under reflection; integer addition does not.
    ally_x = int(0)
    ally_y = int(0)
    enemy_x = int(0)
    enemy_y = int(0)
    position_x = int(0)
    position_y = int(0)
    ally_count = int(0)
    touching_count = int(0)
    largest_enemy_impulse = int(0)

    for other_index in wp.hash_grid_query(grid, grid_position[index], p.diameter, env):
        other = other_index - base
        if other_index != index and alive[other_index] != 0:
            delta = position[other_index] - own_position
            distance = wp.length(delta)
            if distance < p.diameter:
                normal = _normal(delta, local, other)
                closing = wp.dot(own_velocity - velocity[other_index], normal)
                if closing > 0.0:
                    restitution = p.enemy_restitution
                    if team[index] == team[other_index]:
                        restitution = p.ally_restitution
                    impulse = 0.5 * (1.0 + restitution) * closing
                    change = -impulse * normal
                    fixed_x = _collision_fixed(change[0], p.collision_fixed_scale)
                    fixed_y = _collision_fixed(change[1], p.collision_fixed_scale)
                    if team[index] == team[other_index]:
                        ally_x += fixed_x
                        ally_y += fixed_y
                        ally_count += 1
                    else:
                        enemy_x += fixed_x
                        enemy_y += fixed_y
                        fixed_impulse = _collision_fixed(impulse, p.collision_fixed_scale)
                        largest_enemy_impulse = wp.max(
                            largest_enemy_impulse, fixed_impulse
                        )
                separation = -0.5 * (p.diameter - distance) * normal
                position_x += _collision_fixed(
                    separation[0], p.collision_fixed_scale
                )
                position_y += _collision_fixed(
                    separation[1], p.collision_fixed_scale
                )
                touching_count += 1

    inverse_scale = p.collision_fixed_inverse
    ally_sum = wp.vec2(
        float(ally_x) * inverse_scale,
        float(ally_y) * inverse_scale,
    )
    enemy_sum = wp.vec2(
        float(enemy_x) * inverse_scale,
        float(enemy_y) * inverse_scale,
    )
    position_sum = wp.vec2(
        float(position_x) * inverse_scale,
        float(position_y) * inverse_scale,
    )
    if ally_count > 0:
        ally_sum = ally_sum / wp.sqrt(float(ally_count))
    enemy_length = wp.length(enemy_sum)
    enemy_limit = float(largest_enemy_impulse) * inverse_scale
    if enemy_length > enemy_limit and enemy_length > 0.0:
        enemy_sum = enemy_sum * (enemy_limit / enemy_length)
    if touching_count > 0:
        position_sum = position_sum / wp.sqrt(float(touching_count))
    position_sum = _limit(position_sum, 0.5 * p.radius) * p.collision_relaxation
    velocity_change = ally_sum + enemy_sum
    return wp.vec4(velocity_change[0], velocity_change[1], position_sum[0], position_sum[1])


@wp.kernel
def _write_grid_positions(
    position: wp.array(dtype=wp.vec2),
    grid_position: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    point = position[index]
    grid_position[index] = wp.vec3(point[0], point[1], 0.0)


@wp.kernel
def _sync_world_positions(
    centered_position: wp.array(dtype=wp.vec2),
    world_position: wp.array(dtype=wp.vec2),
    p: _Params,
):
    index = wp.tid()
    point = centered_position[index]
    world_position[index] = point + wp.vec2(p.world_width * 0.5, p.world_height * 0.5)


@wp.kernel
def _clear_step_flags(invalid_action: wp.array(dtype=wp.int32)):
    invalid_action[wp.tid()] = 0


@wp.kernel
def _fill_int(values: wp.array(dtype=wp.int32), value: int):
    values[wp.tid()] = value


@wp.kernel
def _reset_soldiers(
    reset_mask: wp.array(dtype=wp.int32),
    initial_team: wp.array(dtype=wp.int32),
    initial_position: wp.array(dtype=wp.vec2),
    initial_velocity: wp.array(dtype=wp.vec2),
    initial_angle: wp.array(dtype=float),
    initial_health: wp.array(dtype=float),
    team: wp.array(dtype=wp.int32),
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    angle: wp.array(dtype=float),
    health: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    attack_cooldown: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.tid()
    env = index // p.num_soldiers
    if reset_mask[env] == 0:
        return
    local = index - env * p.num_soldiers
    team[index] = initial_team[local]
    position[index] = initial_position[local]
    velocity[index] = initial_velocity[local]
    angle[index] = initial_angle[local]
    next_health = initial_health[local]
    health[index] = next_health
    alive[index] = int(next_health > 0.0)
    attack_cooldown[index] = 0


@wp.kernel
def _reset_environments(
    reset_mask: wp.array(dtype=wp.int32),
    step_count: wp.array(dtype=wp.int32),
    substep_count: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    invalid_action: wp.array(dtype=wp.int32),
    advantage_integral: wp.array(dtype=float),
):
    env = wp.tid()
    if reset_mask[env] != 0:
        step_count[env] = 0
        substep_count[env] = 0
        done[env] = 0
        winner[env] = -2
        invalid_action[env] = 0
        advantage_integral[env] = 0.0


@wp.kernel
def _reset_territory(
    reset_mask: wp.array(dtype=wp.int32),
    initial_owner: wp.array(dtype=wp.int32),
    owner: wp.array(dtype=wp.int32),
    share: wp.array(dtype=float),
    p: _Params,
):
    index = wp.tid()
    env = index // p.territory_cells
    if reset_mask[env] == 0:
        return
    cell = index - env * p.territory_cells
    owner[index] = initial_owner[cell]
    share[index * 2] = 0.0
    share[index * 2 + 1] = 0.0


@wp.kernel
def _clear_territory_presence(presence: wp.array(dtype=wp.int32)):
    presence[wp.tid()] = 0


@wp.kernel
def _mark_territory_presence(
    team: wp.array(dtype=wp.int32),
    position: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    presence: wp.array(dtype=wp.int32),
    territory_cell: wp.array(dtype=wp.int32),
    lookup: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.tid()
    env = index // p.num_soldiers
    if alive[index] == 0:
        territory_cell[index] = -1
        return
    if done[env] != 0:
        return
    cell = _territory_cell(
        position[index],
        lookup,
        p.world_width,
        p.world_height,
        p.territory_tile_size,
        p.territory_lookup_radius,
        p.territory_lookup_diameter,
    )
    territory_cell[index] = cell
    wp.atomic_add(presence, (env * 2 + team[index]) * p.territory_cells + cell, 1)


@wp.kernel
def _locate_territory(
    position: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    territory_cell: wp.array(dtype=wp.int32),
    lookup: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.tid()
    territory_cell[index] = -1
    if alive[index] != 0:
        territory_cell[index] = _territory_cell(
            position[index],
            lookup,
            p.world_width,
            p.world_height,
            p.territory_tile_size,
            p.territory_lookup_radius,
            p.territory_lookup_diameter,
        )


@wp.kernel
def _control_territory_owner(
    position: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    team: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    centers: wp.array(dtype=wp.vec2),
    owner: wp.array(dtype=wp.int32),
    share: wp.array(dtype=float),
    p: _Params,
):
    """Soft influence control with order-invariant fixed-point summation."""
    index = wp.tid()
    env = index // p.territory_cells
    if done[env] != 0:
        return
    cell = index - env * p.territory_cells
    center = centers[cell]
    fixed_0 = int(0)
    fixed_1 = int(0)
    scale = -0.5 / (p.territory_control_sigma * p.territory_control_sigma)
    base = env * p.num_soldiers
    for local in range(p.num_soldiers):
        soldier = base + local
        if alive[soldier] == 0:
            continue
        delta = position[soldier] - center
        influence = wp.exp(wp.dot(delta, delta) * scale)
        quantized = int(wp.floor(influence * float(INFLUENCE_FIXED_SCALE) + 0.5))
        if team[soldier] == 0:
            fixed_0 += quantized
        else:
            fixed_1 += quantized
    total = float(fixed_0 + fixed_1 + INFLUENCE_NEUTRAL_FIXED)
    share[index * 2] = float(fixed_0) / total
    share[index * 2 + 1] = float(fixed_1) / total
    result = int(-1)
    if fixed_0 > fixed_1 + INFLUENCE_NEUTRAL_FIXED:
        result = 0
    elif fixed_1 > fixed_0 + INFLUENCE_NEUTRAL_FIXED:
        result = 1
    owner[index] = result


@wp.kernel
def _sanitize_actions(
    actions: wp.array(dtype=ACTION_DTYPE),
    angle: wp.array(dtype=float),
    done: wp.array(dtype=wp.int32),
    move: wp.array(dtype=wp.vec2),
    desired_angle: wp.array(dtype=float),
    invalid_action: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.tid()
    env = index // p.num_soldiers
    if done[env] != 0:
        move[index] = wp.vec2(0.0, 0.0)
        desired_angle[index] = angle[index]
        return
    action = actions[index]
    valid_move = wp.isfinite(action[0]) and wp.isfinite(action[1])
    valid_aim = wp.isfinite(action[2]) and wp.isfinite(action[3])
    if not valid_move or not valid_aim:
        wp.atomic_max(invalid_action, env, 1)
    movement = wp.vec2(0.0, 0.0)
    if valid_move:
        movement = wp.vec2(wp.clamp(action[0], -1.0, 1.0), wp.clamp(action[1], -1.0, 1.0))
        movement = _limit(movement, 1.0)
    move[index] = movement
    desired = angle[index]
    if valid_aim:
        aim = wp.vec2(wp.clamp(action[2], -1.0, 1.0), wp.clamp(action[3], -1.0, 1.0))
        if wp.length(aim) > 1.0e-6:
            desired = wp.atan2(aim[1], aim[0])
    desired_angle[index] = desired


@wp.kernel
def _integrate(
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    angle: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    attack_cooldown: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    move: wp.array(dtype=wp.vec2),
    desired_angle: wp.array(dtype=float),
    p: _Params,
):
    index = wp.tid()
    env = index // p.num_soldiers
    if done[env] != 0 or alive[index] == 0:
        if alive[index] == 0:
            velocity[index] = wp.vec2(0.0, 0.0)
        return

    attack_cooldown[index] = wp.max(0, attack_cooldown[index] - 1)
    error = wp.atan2(wp.sin(desired_angle[index] - angle[index]), wp.cos(desired_angle[index] - angle[index]))
    if error < -3.1415925:
        error = 3.1415927
    delta = wp.clamp(error, -p.maximum_turn_delta, p.maximum_turn_delta)
    angle[index] = _wrap(angle[index] + delta)

    current_velocity = velocity[index] * p.drag_factor
    error_velocity = p.maximum_running_speed * move[index] - current_velocity
    current_velocity = current_velocity + _limit(error_velocity, p.acceleration * p.physics_dt)
    current_velocity = current_velocity + wp.vec2(p.terrain_acceleration_x * p.physics_dt, 0.0)
    current_velocity = _limit(current_velocity, p.physics_speed_limit)
    current_position = position[index] + current_velocity * p.physics_dt
    resolved = _resolve_static(current_position, current_velocity, p)
    position[index] = wp.vec2(resolved[0], resolved[1])
    velocity[index] = wp.vec2(resolved[2], resolved[3])


@wp.kernel
def _choose_strike_targets(
    grid: wp.uint64,
    grid_position: wp.array(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec2),
    angle: wp.array(dtype=float),
    attack_cooldown: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    team: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.hash_grid_point_id(grid, wp.tid())
    env = index // p.num_soldiers
    local = index - env * p.num_soldiers
    base = env * p.num_soldiers
    selected = int(-1)
    best_alignment = float(-2.0)
    if done[env] == 0 and alive[index] != 0 and attack_cooldown[index] == 0:
        facing = _facing(angle[index])
        for other_index in wp.hash_grid_query(
            grid, grid_position[index], p.diameter, env
        ):
            other = other_index - base
            if other != local and alive[other_index] != 0 and team[other_index] != team[index]:
                delta = position[other_index] - position[index]
                if wp.length(delta) < p.diameter:
                    direction = _normal(delta, local, other)
                    alignment = wp.dot(facing, direction)
                    if alignment >= p.arc_cos and (
                        alignment > best_alignment
                        or (alignment == best_alignment and (selected < 0 or other < selected))
                    ):
                        best_alignment = alignment
                        selected = other
    target[index] = -1
    if selected >= 0:
        target[index] = base + selected


@wp.kernel
def _first_effects(
    grid: wp.uint64,
    grid_position: wp.array(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    angle: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    team: wp.array(dtype=wp.int32),
    strike_target: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    velocity_delta: wp.array(dtype=wp.vec2),
    position_delta: wp.array(dtype=wp.vec2),
    p: _Params,
):
    index = wp.hash_grid_point_id(grid, wp.tid())
    env = index // p.num_soldiers
    local = index - env * p.num_soldiers
    damage[index] = 0.0
    velocity_delta[index] = wp.vec2(0.0, 0.0)
    position_delta[index] = wp.vec2(0.0, 0.0)
    if done[env] != 0 or alive[index] == 0:
        return

    collision = _collision_delta(
        grid, grid_position, env, local, position, velocity, alive, team, p
    )
    velocity_delta[index] = wp.vec2(collision[0], collision[1])
    position_delta[index] = wp.vec2(collision[2], collision[3])
    own_facing = _facing(angle[index])
    base = env * p.num_soldiers
    total_damage = float(0.0)
    for other_index in wp.hash_grid_query(
        grid, grid_position[index], p.diameter, env
    ):
        other = other_index - base
        if (
            other != local
            and alive[other_index] != 0
            and team[other_index] != team[index]
            and strike_target[other_index] == index
        ):
            direction_to_attacker = _normal(
                position[other_index] - position[index], local, other
            )
            attacker_direction = -direction_to_attacker
            closing = wp.dot(
                velocity[other_index] - velocity[index], attacker_direction
            )
            inward = wp.max(
                0.0,
                wp.dot(velocity[other_index], attacker_direction)
                - p.minimum_damage_speed,
            )
            charge = float(0.0)
            if closing > p.minimum_closing_speed:
                charge = p.damage_scale * inward * inward
            vulnerability = p.flank_multiplier
            if wp.dot(own_facing, direction_to_attacker) >= p.arc_cos:
                vulnerability = 1.0
            total_damage += wp.max(p.base_strike_damage, charge) * vulnerability
    damage[index] = total_damage


@wp.kernel
def _commit_first(
    position: wp.array(dtype=wp.vec2),
    velocity: wp.array(dtype=wp.vec2),
    health: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    attack_cooldown: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    velocity_delta: wp.array(dtype=wp.vec2),
    position_delta: wp.array(dtype=wp.vec2),
    strike_target: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.tid()
    env = index // p.num_soldiers
    if done[env] != 0:
        return
    if alive[index] == 0:
        velocity[index] = wp.vec2(0.0, 0.0)
        attack_cooldown[index] = 0
        return
    next_health = wp.max(0.0, health[index] - damage[index])
    health[index] = next_health
    next_position = position[index] + position_delta[index]
    next_velocity = _limit(velocity[index] + velocity_delta[index], p.physics_speed_limit)
    if next_health > 0.0:
        alive[index] = 1
        resolved = _resolve_static(next_position, next_velocity, p)
        position[index] = wp.vec2(resolved[0], resolved[1])
        velocity[index] = wp.vec2(resolved[2], resolved[3])
        if strike_target[index] >= 0:
            attack_cooldown[index] = p.attack_recovery_ticks
    else:
        alive[index] = 0
        position[index] = next_position
        velocity[index] = wp.vec2(0.0, 0.0)
        attack_cooldown[index] = 0


@wp.kernel
def _collision_iteration(
    grid: wp.uint64,
    grid_position: wp.array(dtype=wp.vec3),
    position_in: wp.array(dtype=wp.vec2),
    velocity_in: wp.array(dtype=wp.vec2),
    position_out: wp.array(dtype=wp.vec2),
    velocity_out: wp.array(dtype=wp.vec2),
    alive: wp.array(dtype=wp.int32),
    team: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    p: _Params,
):
    index = wp.hash_grid_point_id(grid, wp.tid())
    env = index // p.num_soldiers
    local = index - env * p.num_soldiers
    if done[env] != 0 or alive[index] == 0:
        position_out[index] = position_in[index]
        velocity_out[index] = wp.vec2(0.0, 0.0) if alive[index] == 0 else velocity_in[index]
        return
    collision = _collision_delta(
        grid, grid_position, env, local, position_in, velocity_in, alive, team, p
    )
    next_position = position_in[index] + wp.vec2(collision[2], collision[3])
    next_velocity = _limit(
        velocity_in[index] + wp.vec2(collision[0], collision[1]), p.physics_speed_limit
    )
    resolved = _resolve_static(next_position, next_velocity, p)
    position_out[index] = wp.vec2(resolved[0], resolved[1])
    velocity_out[index] = wp.vec2(resolved[2], resolved[3])


@wp.kernel
def _advance_substeps(substep: wp.array(dtype=wp.int32), done: wp.array(dtype=wp.int32)):
    env = wp.tid()
    if done[env] == 0:
        substep[env] += 1


@wp.kernel
def _finish_step(
    share: wp.array(dtype=float),
    territory_weights: wp.array(dtype=wp.int32),
    step_count: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    advantage_integral: wp.array(dtype=float),
    p: _Params,
):
    env = wp.tid()
    if done[env] != 0:
        return
    advantage = float(0.0)
    base = env * p.territory_cells
    for cell in range(p.territory_cells):
        weight = float(territory_weights[cell])
        advantage += weight * (
            share[(base + cell) * 2] - share[(base + cell) * 2 + 1]
        )
    integral = advantage_integral[env] + advantage / p.territory_total_weight
    advantage_integral[env] = integral
    next_step = step_count[env] + 1
    step_count[env] = next_step
    if next_step >= p.maximum_decision_steps:
        done[env] = 1
        result = -1
        if integral > 1.0e-9:
            result = 0
        elif integral < -1.0e-9:
            result = 1
        winner[env] = result


class GpuSimulator:
    """Batched Warp simulator. ``device='cpu'`` is useful for kernel tests."""

    def __init__(self, config: Config | None = None, num_envs: int = 1, device: str = "cuda"):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        wp.init()
        self.device = wp.get_device(device)
        self.config = config or Config()
        self.num_envs = num_envs
        self.num_soldiers = self.config.soldier_count
        self.total_soldiers = num_envs * self.num_soldiers
        self.params = self._make_params()

        cell_size = self.config.diameter
        self.spatial_grid = wp.HashGrid(
            math.ceil(self.config.world_width / cell_size),
            math.ceil(self.config.world_height / cell_size),
            1,
            device=self.device,
        )
        self.spatial_position = wp.empty(
            self.total_soldiers, dtype=wp.vec3, device=self.device
        )
        groups = np.repeat(np.arange(self.num_envs, dtype=np.int32), self.num_soldiers)
        self.spatial_group = wp.array(groups, dtype=wp.int32, device=self.device)

        self.move = wp.zeros(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self.desired_angle = wp.zeros(self.total_soldiers, dtype=float, device=self.device)
        self.damage = wp.zeros(self.total_soldiers, dtype=float, device=self.device)
        self.velocity_delta = wp.zeros(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self.position_delta = wp.zeros(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self.strike_target = wp.full(
            self.total_soldiers, -1, dtype=wp.int32, device=self.device
        )
        self._position_tmp = wp.zeros(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self._velocity_tmp = wp.zeros(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self.invalid_action = wp.zeros(num_envs, dtype=wp.int32, device=self.device)
        self._reset_mask = wp.zeros(num_envs, dtype=wp.int32, device=self.device)
        self._action_ref = None

        initial = CpuSimulator(self.config)

        def flatten(array: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(array.reshape(-1))

        def vectors(array: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(array.reshape(-1, 2))

        self._initial_team = wp.array(flatten(initial.team), dtype=wp.int32, device=self.device)
        initial_position = vectors(initial.position).copy()
        initial_position -= (self.config.world_width * 0.5, self.config.world_height * 0.5)
        initial_position = (
            np.rint(initial_position * POSITION_FIXED_SCALE) / POSITION_FIXED_SCALE
        )
        self._initial_position = wp.array(initial_position, dtype=wp.vec2, device=self.device)
        self._initial_velocity = wp.array(vectors(initial.velocity), dtype=wp.vec2, device=self.device)
        self._initial_angle = wp.array(flatten(initial.attack_angle), dtype=float, device=self.device)
        self._initial_health = wp.array(flatten(initial.health), dtype=float, device=self.device)
        self._initial_territory = wp.array(
            TERRITORY_INITIAL_OWNER, dtype=wp.int32, device=self.device
        )
        self.territory_lookup = wp.array(TERRITORY_LOOKUP, dtype=wp.int32, device=self.device)
        self._territory_centers = wp.array(
            territory_centers(self.config), dtype=wp.vec2, device=self.device
        )
        self.advantage_integral = wp.zeros(self.num_envs, dtype=float, device=self.device)
        self.control_share = wp.zeros(
            self.num_envs * TERRITORY_CELLS * 2, dtype=float, device=self.device
        )
        self.territory_weights = wp.array(
            TERRITORY_WEIGHTS, dtype=wp.int32, device=self.device
        )

        self.team = wp.empty(self.total_soldiers, dtype=wp.int32, device=self.device)
        # Physics uses centered coordinates; the public array remains in world coordinates.
        self.position = wp.empty(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self._physics_position = wp.empty(
            self.total_soldiers, dtype=wp.vec2, device=self.device
        )
        self.velocity = wp.empty(self.total_soldiers, dtype=wp.vec2, device=self.device)
        self.attack_angle = wp.empty(self.total_soldiers, dtype=float, device=self.device)
        self.health = wp.empty(self.total_soldiers, dtype=float, device=self.device)
        self.alive = wp.empty(self.total_soldiers, dtype=wp.int32, device=self.device)
        self.attack_cooldown_ticks = wp.empty(
            self.total_soldiers, dtype=wp.int32, device=self.device
        )
        self.step_count = wp.empty(self.num_envs, dtype=wp.int32, device=self.device)
        self.substep_count = wp.empty(self.num_envs, dtype=wp.int32, device=self.device)
        self.done = wp.empty(self.num_envs, dtype=wp.int32, device=self.device)
        self.winner = wp.empty(self.num_envs, dtype=wp.int32, device=self.device)
        self.territory_owner = wp.empty(
            self.num_envs * TERRITORY_CELLS, dtype=wp.int32, device=self.device
        )
        self.territory_cell = wp.empty(
            self.total_soldiers, dtype=wp.int32, device=self.device
        )
        self._territory_presence = wp.zeros(
            self.num_envs * 2 * TERRITORY_CELLS, dtype=wp.int32, device=self.device
        )
        self.reset()

    def _make_params(self) -> _Params:
        c = self.config
        p = _Params()
        p.num_soldiers = c.soldier_count
        p.world_width = c.world_width
        p.world_height = c.world_height
        p.radius = c.soldier_radius
        p.diameter = c.diameter
        p.physics_dt = c.physics_dt
        p.maximum_running_speed = c.maximum_running_speed
        p.physics_speed_limit = c.physics_speed_limit
        p.acceleration = c.acceleration
        p.drag_factor = math.exp(-c.linear_drag * c.physics_dt)
        p.maximum_turn_delta = c.maximum_turn_rate * c.physics_dt
        p.terrain_acceleration_x = -c.slope_gravity * c.slope_height / c.world_width
        p.ally_restitution = c.ally_restitution
        p.enemy_restitution = c.enemy_restitution
        p.wall_restitution = c.wall_restitution
        p.collision_relaxation = c.collision_relaxation
        maximum_contacts = max(1, c.soldier_count - 1)
        maximum_impulse_sum = (
            maximum_contacts
            * (1.0 + max(c.ally_restitution, c.enemy_restitution))
            * c.physics_speed_limit
        )
        maximum_position_sum = maximum_contacts * c.soldier_radius
        fixed_bound = max(1.0, maximum_impulse_sum, maximum_position_sum)
        p.collision_fixed_scale = min(COLLISION_FIXED_SCALE, 1.0e9 / fixed_bound)
        p.collision_fixed_inverse = 1.0 / p.collision_fixed_scale
        p.attack_recovery_ticks = c.attack_recovery_ticks
        p.arc_cos = math.cos(math.radians(c.damaging_arc_degrees / 2))
        p.base_strike_damage = c.base_strike_damage
        p.minimum_damage_speed = c.minimum_damage_speed
        p.minimum_closing_speed = c.minimum_closing_speed
        p.damage_scale = c.damage_scale
        p.flank_multiplier = c.flank_damage_multiplier
        p.river_width = c.river_width
        p.bridge_width = c.bridge_width
        p.maximum_decision_steps = c.maximum_decision_steps
        p.territory_tile_size = c.territory_tile_size
        p.territory_lookup_radius = TERRITORY_LOOKUP_RADIUS
        p.territory_lookup_diameter = TERRITORY_LOOKUP_DIAMETER
        p.territory_cells = TERRITORY_CELLS
        p.territory_control_sigma = c.control_radius * 0.5
        p.territory_total_weight = float(TERRITORY_TOTAL_WEIGHT)
        return p

    @property
    def state(self) -> dict[str, wp.array]:
        return {
            "team": self.team,
            "position": self.position,
            "velocity": self.velocity,
            "attack_angle": self.attack_angle,
            "health": self.health,
            "alive": self.alive,
            "attack_cooldown_ticks": self.attack_cooldown_ticks,
            "step_count": self.step_count,
            "done": self.done,
            "winner": self.winner,
            "territory_owner": self.territory_owner,
            "advantage_integral": self.advantage_integral,
        }

    def reset(
        self,
        state: Mapping[str, object] | None = None,
        *,
        mask: object | None = None,
    ) -> dict[str, wp.array]:
        """Reset all or selected environments without replacing device arrays."""
        if state is not None and mask is not None:
            raise ValueError("custom state and reset mask cannot be combined")
        self._set_reset_mask(mask)
        wp.launch(
            _reset_soldiers,
            self.total_soldiers,
            inputs=[
                self._reset_mask,
                self._initial_team,
                self._initial_position,
                self._initial_velocity,
                self._initial_angle,
                self._initial_health,
                self.team,
                self._physics_position,
                self.velocity,
                self.attack_angle,
                self.health,
                self.alive,
                self.attack_cooldown_ticks,
                self.params,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_environments,
            self.num_envs,
            inputs=[
                self._reset_mask,
                self.step_count,
                self.substep_count,
                self.done,
                self.winner,
                self.invalid_action,
                self.advantage_integral,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_territory,
            self.num_envs * TERRITORY_CELLS,
            inputs=[
                self._reset_mask,
                self._initial_territory,
                self.territory_owner,
                self.control_share,
                self.params,
            ],
            device=self.device,
        )
        if state is not None:
            self._copy_custom_state(state)
        self.sync_world_positions()
        wp.launch(
            _locate_territory,
            self.total_soldiers,
            inputs=[
                self.position,
                self.alive,
                self.territory_cell,
                self.territory_lookup,
                self.params,
            ],
            device=self.device,
        )
        return self.state

    def _set_reset_mask(self, mask: object | None) -> None:
        if mask is None:
            wp.launch(
                _fill_int,
                self.num_envs,
                inputs=[self._reset_mask, 1],
                device=self.device,
            )
            return
        if isinstance(mask, wp.array):
            if mask.size != self.num_envs or mask.dtype != wp.int32:
                raise ValueError(f"expected {self.num_envs} int32 reset flags")
            wp.copy(self._reset_mask, mask)
            return
        array = np.asarray(mask, dtype=np.int32)
        if array.shape != (self.num_envs,):
            raise ValueError(f"expected reset mask with shape {(self.num_envs,)}")
        source = wp.array(array, dtype=wp.int32, device=self.device)
        wp.copy(self._reset_mask, source)

    def _copy_custom_state(self, state: Mapping[str, object]) -> None:
        cpu = CpuSimulator(self.config, self.num_envs)
        cpu.reset(state)
        physics_position = cpu.position.reshape(-1, 2).copy()
        physics_position -= (
            self.config.world_width * 0.5,
            self.config.world_height * 0.5,
        )
        physics_position = (
            np.rint(physics_position * POSITION_FIXED_SCALE) / POSITION_FIXED_SCALE
        ).astype(np.float32)
        values = {
            "team": (cpu.team.reshape(-1), self.team, wp.int32),
            "position": (physics_position, self._physics_position, wp.vec2),
            "velocity": (cpu.velocity.reshape(-1, 2), self.velocity, wp.vec2),
            "attack_angle": (cpu.attack_angle.reshape(-1), self.attack_angle, float),
            "health": (cpu.health.reshape(-1), self.health, float),
            "alive": (cpu.alive.astype(np.int32).reshape(-1), self.alive, wp.int32),
            "done": (cpu.done.astype(np.int32), self.done, wp.int32),
            "winner": (cpu.winner.astype(np.int32), self.winner, wp.int32),
            "territory_owner": (
                cpu.territory_owner.reshape(-1).astype(np.int32),
                self.territory_owner,
                wp.int32,
            ),
            "advantage_integral": (
                cpu.advantage_integral.astype(np.float32),
                self.advantage_integral,
                float,
            ),
            "control_share": (
                cpu.control_share.reshape(-1).astype(np.float32),
                self.control_share,
                float,
            ),
        }
        self._reset_sources = []
        for array, target, dtype in values.values():
            source = wp.array(np.ascontiguousarray(array), dtype=dtype, device=self.device)
            wp.copy(target, source)
            self._reset_sources.append(source)

    def _actions(self, actions: object) -> wp.array:
        if isinstance(actions, wp.array):
            if actions.size != self.total_soldiers or actions.dtype != ACTION_DTYPE:
                raise ValueError(f"expected {self.total_soldiers} vec4 actions")
            return actions
        if isinstance(actions, np.ndarray):
            array = np.asarray(actions, np.float32)
            expected = (self.num_envs, self.num_soldiers, 4)
            if array.shape == expected[1:] and self.num_envs == 1:
                array = array[None]
            if array.shape != expected:
                raise ValueError(f"expected actions with shape {expected}, got {array.shape}")
            result = wp.array(
                np.ascontiguousarray(array.reshape(-1, 4)),
                dtype=ACTION_DTYPE,
                device=self.device,
            )
            self._action_ref = result
            return result
        try:
            import torch

            if isinstance(actions, torch.Tensor):
                expected = (self.num_envs, self.num_soldiers, 4)
                if tuple(actions.shape) == expected[1:] and self.num_envs == 1:
                    actions = actions.unsqueeze(0)
                if tuple(actions.shape) != expected or not actions.is_cuda:
                    raise ValueError(f"expected CUDA actions with shape {expected}")
                tensor = actions.to(torch.float32).contiguous().view(-1, 4)
                result = wp.from_torch(tensor, dtype=ACTION_DTYPE)
                self._action_ref = (tensor, result)
                return result
        except ImportError:
            pass
        raise TypeError("actions must be a NumPy, Torch CUDA, or Warp vec4 array")

    def sync_world_positions(self) -> None:
        wp.launch(
            _sync_world_positions,
            self.total_soldiers,
            inputs=[self._physics_position, self.position, self.params],
            device=self.device,
        )

    def refresh_spatial_grid(self) -> None:
        wp.launch(
            _write_grid_positions,
            self.total_soldiers,
            inputs=[self._physics_position, self.spatial_position],
            device=self.device,
        )
        self.spatial_grid.build(
            self.spatial_position,
            self.config.diameter,
            groups=self.spatial_group,
        )

    def step(self, actions: object) -> dict[str, wp.array]:
        action_array = self._actions(actions)
        wp.launch(_clear_step_flags, self.num_envs, inputs=[self.invalid_action], device=self.device)
        wp.launch(
            _sanitize_actions,
            self.total_soldiers,
            inputs=[
                action_array,
                self.attack_angle,
                self.done,
                self.move,
                self.desired_angle,
                self.invalid_action,
                self.params,
            ],
            device=self.device,
        )
        for _ in range(self.config.physics_substeps):
            wp.launch(
                _integrate,
                self.total_soldiers,
                inputs=[
                    self._physics_position,
                    self.velocity,
                    self.attack_angle,
                    self.alive,
                    self.attack_cooldown_ticks,
                    self.done,
                    self.move,
                    self.desired_angle,
                    self.params,
                ],
                device=self.device,
            )
            self.refresh_spatial_grid()
            wp.launch(
                _choose_strike_targets,
                self.total_soldiers,
                inputs=[
                    self.spatial_grid.id,
                    self.spatial_position,
                    self._physics_position,
                    self.attack_angle,
                    self.attack_cooldown_ticks,
                    self.alive,
                    self.team,
                    self.done,
                    self.strike_target,
                    self.params,
                ],
                device=self.device,
            )
            wp.launch(
                _first_effects,
                self.total_soldiers,
                inputs=[
                    self.spatial_grid.id,
                    self.spatial_position,
                    self._physics_position,
                    self.velocity,
                    self.attack_angle,
                    self.alive,
                    self.team,
                    self.strike_target,
                    self.done,
                    self.damage,
                    self.velocity_delta,
                    self.position_delta,
                    self.params,
                ],
                device=self.device,
            )
            wp.launch(
                _commit_first,
                self.total_soldiers,
                inputs=[
                    self._physics_position,
                    self.velocity,
                    self.health,
                    self.alive,
                    self.attack_cooldown_ticks,
                    self.done,
                    self.damage,
                    self.velocity_delta,
                    self.position_delta,
                    self.strike_target,
                    self.params,
                ],
                device=self.device,
            )
            for _ in range(self.config.collision_iterations - 1):
                self.refresh_spatial_grid()
                wp.launch(
                    _collision_iteration,
                    self.total_soldiers,
                    inputs=[
                        self.spatial_grid.id,
                        self.spatial_position,
                        self._physics_position,
                        self.velocity,
                        self._position_tmp,
                        self._velocity_tmp,
                        self.alive,
                        self.team,
                        self.done,
                        self.params,
                    ],
                    device=self.device,
                )
                self._physics_position, self._position_tmp = (
                    self._position_tmp,
                    self._physics_position,
                )
                self.velocity, self._velocity_tmp = self._velocity_tmp, self.velocity
            wp.launch(
                _advance_substeps,
                self.num_envs,
                inputs=[self.substep_count, self.done],
                device=self.device,
            )
        self.sync_world_positions()
        wp.launch(
            _clear_territory_presence,
            self._territory_presence.size,
            inputs=[self._territory_presence],
            device=self.device,
        )
        wp.launch(
            _mark_territory_presence,
            self.total_soldiers,
            inputs=[
                self.team,
                self.position,
                self.alive,
                self.done,
                self._territory_presence,
                self.territory_cell,
                self.territory_lookup,
                self.params,
            ],
            device=self.device,
        )
        wp.launch(
            _control_territory_owner,
            self.num_envs * TERRITORY_CELLS,
            inputs=[
                self.position,
                self.alive,
                self.team,
                self.done,
                self._territory_centers,
                self.territory_owner,
                self.control_share,
                self.params,
            ],
            device=self.device,
        )
        wp.launch(
            _finish_step,
            self.num_envs,
            inputs=[
                self.control_share,
                self.territory_weights,
                self.step_count,
                self.done,
                self.winner,
                self.advantage_integral,
                self.params,
            ],
            device=self.device,
        )
        return self.state

    def numpy_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        shape = (self.num_envs, self.num_soldiers)
        return {
            "team": self.team.numpy().reshape(shape),
            "position": self.position.numpy().reshape(*shape, 2),
            "velocity": self.velocity.numpy().reshape(*shape, 2),
            "attack_angle": self.attack_angle.numpy().reshape(shape),
            "health": self.health.numpy().reshape(shape),
            "alive": self.alive.numpy().reshape(shape).astype(bool),
            "attack_cooldown_ticks": self.attack_cooldown_ticks.numpy().reshape(shape),
            "step_count": self.step_count.numpy(),
            "done": self.done.numpy().astype(bool),
            "winner": self.winner.numpy(),
            "territory_owner": self.territory_owner.numpy().reshape(
                self.num_envs, TERRITORY_CELLS
            ),
            "advantage_integral": self.advantage_integral.numpy(),
            "control_share": self.control_share.numpy().reshape(
                self.num_envs, TERRITORY_CELLS, 2
            ),
        }

    def snapshot(self, env: int = 0) -> dict[str, np.ndarray | int | bool]:
        """Copy only one environment to the host for rendering or inspection."""
        if not 0 <= env < self.num_envs:
            raise IndexError("environment index out of range")
        start = env * self.num_soldiers
        arrays = {
            "team": (self.team, wp.int32),
            "position": (self.position, wp.vec2),
            "velocity": (self.velocity, wp.vec2),
            "attack_angle": (self.attack_angle, float),
            "health": (self.health, float),
            "alive": (self.alive, wp.int32),
        }
        host = {
            name: wp.empty(self.num_soldiers, dtype=dtype, device="cpu")
            for name, (_, dtype) in arrays.items()
        }
        for name, (source, _) in arrays.items():
            wp.copy(host[name], source, src_offset=start, count=self.num_soldiers)
        owner = wp.empty(TERRITORY_CELLS, dtype=wp.int32, device="cpu")
        wp.copy(
            owner,
            self.territory_owner,
            src_offset=env * TERRITORY_CELLS,
            count=TERRITORY_CELLS,
        )
        share = wp.empty(TERRITORY_CELLS * 2, dtype=float, device="cpu")
        wp.copy(
            share,
            self.control_share,
            src_offset=env * TERRITORY_CELLS * 2,
            count=TERRITORY_CELLS * 2,
        )
        scalars = {
            "step_count": wp.empty(1, dtype=wp.int32, device="cpu"),
            "done": wp.empty(1, dtype=wp.int32, device="cpu"),
            "winner": wp.empty(1, dtype=wp.int32, device="cpu"),
        }
        wp.copy(scalars["step_count"], self.step_count, src_offset=env, count=1)
        wp.copy(scalars["done"], self.done, src_offset=env, count=1)
        wp.copy(scalars["winner"], self.winner, src_offset=env, count=1)
        wp.synchronize_device(self.device)
        return {
            "team": host["team"].numpy(),
            "position": host["position"].numpy(),
            "velocity": host["velocity"].numpy(),
            "attack_angle": host["attack_angle"].numpy(),
            "health": host["health"].numpy(),
            "alive": host["alive"].numpy().astype(bool),
            "step_count": int(scalars["step_count"].numpy()[0]),
            "done": bool(scalars["done"].numpy()[0]),
            "winner": int(scalars["winner"].numpy()[0]),
            "territory_owner": owner.numpy(),
            "control_share": share.numpy().reshape(TERRITORY_CELLS, 2),
        }


__all__ = ["ACTION_DTYPE", "GpuSimulator"]

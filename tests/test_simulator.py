import numpy as np
import pytest

from simulator import (
    ARENA_NORMALS,
    STRONGPOINT_CELLS,
    STRONGPOINT_CENTERS,
    STRONGPOINT_WEIGHT,
    TERRITORY_CELLS,
    TERRITORY_COORDINATES,
    TERRITORY_INITIAL_OWNER,
    TERRITORY_TOTAL_WEIGHT,
    TERRITORY_WEIGHTS,
    Config,
    CpuSimulator,
    arena_contains,
    territory_centers,
    territory_indices,
    territory_neighborhood,
)


def territory_cell(q: int, r: int) -> int:
    return int(np.flatnonzero(np.all(TERRITORY_COORDINATES == (q, r), axis=1))[0])


def two_soldier_strike(health: float = 100.0):
    config = Config(
        soldiers_per_team=1,
        decision_dt=0.025,
        physics_substeps=1,
        collision_iterations=3,
        maximum_running_speed=0,
        physics_speed_limit=9,
        acceleration=0,
        linear_drag=0,
        enemy_restitution=0,
        attack_recovery_seconds=0.05,
        minimum_damage_speed=0,
        minimum_closing_speed=0,
        damage_scale=1,
    )
    center = np.array([config.world_width / 2, config.world_height / 2], np.float32)
    state = {
        "team": np.array([0, 1]),
        "position": np.array(
            [center, center + (config.diameter + 0.1, 0)], np.float32
        ),
        "velocity": np.array([[4, 0], [-4, 0]], np.float32),
        "attack_angle": np.array([0, np.pi], np.float32),
        "health": np.full(2, health, np.float32),
    }
    return config, state, np.zeros((1, 2, 4), np.float32)


def test_movement_and_facing_are_independent():
    sim = CpuSimulator(Config(soldiers_per_team=1, linear_drag=0))
    actions = np.zeros((1, 2, 4), np.float32)
    actions[0, 0] = (0, 1, 1, 0)
    sim.step(actions)
    assert sim.velocity[0, 0, 1] > 0
    assert sim.velocity[0, 0, 0] == pytest.approx(0)
    assert sim.attack_angle[0, 0] == pytest.approx(0)


def test_default_formations_are_mirrored():
    config = Config()
    sim = CpuSimulator(config)
    left, right = np.split(sim.position[0], 2)
    np.testing.assert_allclose(left[:, 0], config.world_width - right[:, 0], atol=2e-6)
    np.testing.assert_allclose(left[:, 1], right[:, 1])
    assert arena_contains(sim.position[0], config, config.soldier_radius).all()


def test_hex_territory_starts_balanced_with_a_neutral_center_line():
    assert TERRITORY_CELLS == 547
    assert np.count_nonzero(TERRITORY_INITIAL_OWNER == 0) == 260
    assert np.count_nonzero(TERRITORY_INITIAL_OWNER == 1) == 260
    assert np.count_nonzero(TERRITORY_INITIAL_OWNER == -1) == 27
    assert STRONGPOINT_CENTERS.tolist() == [[0, -8], [0, 0], [0, 8]]
    assert len(STRONGPOINT_CELLS) == 21
    assert np.count_nonzero(TERRITORY_WEIGHTS == STRONGPOINT_WEIGHT) == 21
    assert TERRITORY_TOTAL_WEIGHT == 1156
    weighted = [
        TERRITORY_WEIGHTS[TERRITORY_INITIAL_OWNER == team].sum() for team in (0, 1)
    ]
    assert weighted == [434, 434]
    config = Config()
    np.testing.assert_array_equal(
        territory_indices(territory_centers(config), config),
        np.arange(TERRITORY_CELLS),
    )


def test_radius_two_hex_neighborhood_has_19_tiles_and_off_map_sentinel():
    neighbors, offsets = territory_neighborhood(2)
    center = territory_cell(0, 0)
    assert neighbors.shape == (TERRITORY_CELLS, 19)
    assert offsets.shape == (19, 2)
    np.testing.assert_array_equal(neighbors[:, 0], np.arange(TERRITORY_CELLS))
    np.testing.assert_array_equal(offsets[0], 0)
    assert (neighbors[center] < TERRITORY_CELLS).all()
    assert (neighbors[territory_cell(13, 0)] == TERRITORY_CELLS).any()


def test_contact_strikes_repeat_after_recovery_and_death_is_simultaneous():
    config, state, actions = two_soldier_strike()
    sim = CpuSimulator(config)
    sim.reset(state)
    sim.step(actions)
    np.testing.assert_allclose(sim.health, [[84, 84]])

    # Continuous contact cannot strike again until the two-tick recovery ends.
    touching = state["position"].copy()
    touching[1, 0] = touching[0, 0] + config.diameter - 0.01
    sim.position[:] = touching
    sim.velocity.fill(0)
    sim.step(actions)
    np.testing.assert_allclose(sim.health, [[84, 84]])
    sim.position[:] = touching
    sim.step(actions)
    np.testing.assert_allclose(sim.health, [[79, 79]])

    state["health"][:] = 15
    sim.reset(state)
    sim.step(actions)
    assert not sim.alive.any()
    assert not sim.done[0] and sim.winner[0] == -2


def test_attack_arc_is_binary_and_flank_doubles_the_whole_strike():
    config, state, actions = two_soldier_strike()
    state["velocity"].fill(0)
    state["position"][1, 0] = state["position"][0, 0] + config.diameter - 0.01
    state["attack_angle"] = np.deg2rad([59, 90]).astype(np.float32)
    sim = CpuSimulator(config)
    sim.reset(state)
    sim.step(actions)
    np.testing.assert_allclose(sim.health, [[100, 90]])

    state["attack_angle"][0] = np.deg2rad(61)
    sim.reset(state)
    sim.step(actions)
    np.testing.assert_allclose(sim.health, [[100, 100]])


def test_each_ready_soldier_strikes_only_the_most_centered_target():
    config = Config(
        soldiers_per_team=2,
        decision_dt=0.025,
        physics_substeps=1,
        collision_iterations=1,
        maximum_running_speed=0,
        acceleration=0,
        linear_drag=0,
    )
    center = np.array([config.world_width / 2, config.world_height / 2], np.float32)
    distance = config.diameter - 0.01
    state = {
        "position": np.array(
            [
                center,
                center + (-10, 0),
                center + (distance, 0),
                center + distance * np.array((np.cos(0.3), np.sin(0.3))),
            ],
            np.float32,
        ),
        "attack_angle": np.array([0, np.pi, 0, 0], np.float32),
    }
    sim = CpuSimulator(config)
    sim.reset(state)
    sim.step(np.zeros((1, 4, 4), np.float32))
    np.testing.assert_allclose(sim.health[0], [100, 100, 90, 100])


def test_timeout_winner_integrates_weighted_control():
    config = Config(soldiers_per_team=1, maximum_episode_seconds=0.1)
    centers = territory_centers(config)
    state = {
        "position": np.array(
            [centers[territory_cell(1, 0)], centers[territory_cell(10, 0)]], np.float32
        ),
        "health": np.array([1, 100], np.float32),
    }
    actions = np.zeros((1, 2, 4), np.float32)
    cpu = CpuSimulator(config)
    cpu.reset(state)
    cpu.step(actions)
    assert cpu.done[0] and cpu.winner[0] == 0
    assert cpu.advantage_integral[0] > 0
    owned_0 = int(TERRITORY_WEIGHTS[cpu.territory_owner[0] == 0].sum())
    owned_1 = int(TERRITORY_WEIGHTS[cpu.territory_owner[0] == 1].sum())
    assert owned_0 > 200 and 0 < owned_1 < 100

    wp = pytest.importorskip("warp")
    from simulator_gpu import GpuSimulator

    device = "cuda" if wp.is_cuda_available() else "cpu"
    warp_sim = GpuSimulator(config, device=device)
    warp_sim.reset(state)
    warp_sim.step(actions)
    gpu = warp_sim.numpy_state()
    assert gpu["done"][0] and gpu["winner"][0] == 0
    np.testing.assert_array_equal(gpu["territory_owner"], cpu.territory_owner)
    np.testing.assert_allclose(
        gpu["advantage_integral"], cpu.advantage_integral, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        gpu["control_share"], cpu.control_share, rtol=1e-4, atol=1e-5
    )


def test_strongpoint_control_outweighs_wider_plain_control():
    config = Config(soldiers_per_team=1, maximum_episode_seconds=0.1)
    centers = territory_centers(config)
    state = {
        "position": np.array(
            [centers[territory_cell(0, 0)], centers[territory_cell(10, 0)]], np.float32
        ),
    }
    actions = np.zeros((1, 2, 4), np.float32)
    cpu = CpuSimulator(config)
    cpu.reset(state)
    cpu.step(actions)
    counts = [np.count_nonzero(cpu.territory_owner[0] == team) for team in (0, 1)]
    assert abs(counts[0] - counts[1]) <= 3
    assert cpu.done[0] and cpu.winner[0] == 0
    assert cpu.advantage_integral[0] > 0

    wp = pytest.importorskip("warp")
    from simulator_gpu import GpuSimulator

    device = "cuda" if wp.is_cuda_available() else "cpu"
    warp_sim = GpuSimulator(config, device=device)
    warp_sim.reset(state)
    warp_sim.step(actions)
    assert warp_sim.numpy_state()["winner"][0] == 0


def test_control_requires_presence_and_equidistance_contests():
    config = Config(soldiers_per_team=1, maximum_episode_seconds=1)
    sim = CpuSimulator(config)
    actions = np.zeros((1, 2, 4), np.float32)
    centers = territory_centers(config)
    cell = territory_cell(0, 0)
    center = centers[cell]

    # Exactly equidistant opposing soldiers contest the cell.
    flanking = np.array(
        [center - (4.0, 0.0), center + (4.0, 0.0)], np.float32
    )
    sim.reset({"position": flanking})
    sim.step(actions)
    assert sim.territory_owner[0, cell] == -1

    # A sole nearby soldier controls it.
    sole = np.array([center - (4.0, 0.0), centers[territory_cell(10, 0)]], np.float32)
    sim.reset({"position": sole})
    sim.step(actions)
    assert sim.territory_owner[0, cell] == 0

    # Control fades the moment presence leaves: no ghost ownership.
    sim.position[0, 0] = centers[territory_cell(-10, 0)]
    sim.step(actions)
    assert sim.territory_owner[0, cell] == -1

    # Dead soldiers project no control.
    sim.reset(
        {
            "position": np.array(
                [center, centers[territory_cell(10, 0)]], np.float32
            ),
            "health": np.array([0.0, 100.0], np.float32),
        }
    )
    sim.step(actions)
    assert sim.territory_owner[0, cell] == -1

    # Mutual extinction leaves the whole field neutral.
    sim.reset(
        {
            "position": np.array(
                [center, centers[territory_cell(10, 0)]], np.float32
            ),
            "health": np.array([0.0, 0.0], np.float32),
        }
    )
    sim.step(actions)
    assert (sim.territory_owner[0] == -1).all()
    np.testing.assert_allclose(sim.control_share[0], 0.0, rtol=0, atol=1e-6)


def test_warp_matches_reference():
    wp = pytest.importorskip("warp")
    from simulator_gpu import GpuSimulator

    config, state, actions = two_soldier_strike()
    cpu = CpuSimulator(config)
    cpu.reset(state)
    cpu.step(actions)
    device = "cuda" if wp.is_cuda_available() else "cpu"
    warp_sim = GpuSimulator(config, device=device)
    warp_sim.reset(state)
    warp_sim.step(actions)
    gpu = warp_sim.numpy_state()
    for name in ("position", "velocity", "health"):
        np.testing.assert_allclose(gpu[name], getattr(cpu, name), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(gpu["alive"], cpu.alive)
    np.testing.assert_array_equal(gpu["done"], cpu.done)
    np.testing.assert_array_equal(gpu["territory_owner"], cpu.territory_owner)


def test_dense_warp_contacts_are_side_and_index_order_invariant():
    wp = pytest.importorskip("warp")
    from simulator_gpu import ACTION_DTYPE, GpuSimulator

    config = Config(base_strike_damage=0, damage_scale=0)
    reference = CpuSimulator(config, num_envs=2)
    n = config.soldiers_per_team
    permutation = np.r_[
        np.arange(n - 1, -1, -1), np.arange(2 * n - 1, n - 1, -1)
    ]
    state = {
        name: value.copy()
        for name, value in reference.state.items()
        if name in {"team", "position", "velocity", "attack_angle", "health"}
    }
    for value in state.values():
        value[1] = value[1, permutation]

    device = "cuda" if wp.is_cuda_available() else "cpu"
    sim = GpuSimulator(config, num_envs=2, device=device)
    sim.reset(state)
    actions = np.zeros((2, config.soldier_count, 4), np.float32)
    direction = np.where(state["team"] == 0, 1.0, -1.0)
    actions[..., 0] = direction
    actions[..., 2] = direction
    action_array = wp.array(
        np.ascontiguousarray(actions.reshape(-1, 4)), dtype=ACTION_DTYPE, device=device
    )
    for _ in range(config.maximum_decision_steps):
        sim.step(action_array)

    result = sim.numpy_state()
    scores = np.array(
        [
            [int(TERRITORY_WEIGHTS[owner == team].sum()) for team in (0, 1)]
            for owner in result["territory_owner"]
        ]
    )
    expected = [
        int(TERRITORY_WEIGHTS[TERRITORY_INITIAL_OWNER == team].sum())
        for team in (0, 1)
    ]
    del expected
    np.testing.assert_array_equal(
        result["territory_owner"][0], result["territory_owner"][1]
    )
    np.testing.assert_array_equal(result["winner"][0], result["winner"][1])
    np.testing.assert_array_equal(scores[0], scores[1])


def test_hex_wall_projection_matches_warp():
    wp = pytest.importorskip("warp")
    from simulator_gpu import GpuSimulator

    config = Config(soldiers_per_team=1, physics_substeps=1, collision_iterations=1)
    normal = ARENA_NORMALS[1]
    center = np.array((config.world_width / 2, config.world_height / 2), np.float32)
    distance = config.arena_apothem - config.soldier_radius - 0.02
    state = {"position": np.array([center + distance * normal, center - distance * normal])}
    actions = np.zeros((1, 2, 4), np.float32)
    actions[0, 0, :2] = normal
    actions[0, 1, :2] = -normal
    cpu = CpuSimulator(config)
    cpu.reset(state)
    cpu.step(actions)
    device = "cuda" if wp.is_cuda_available() else "cpu"
    warp_sim = GpuSimulator(config, device=device)
    warp_sim.reset(state)
    warp_sim.step(actions)
    gpu = warp_sim.numpy_state()
    assert arena_contains(cpu.position, config, config.soldier_radius).all()
    np.testing.assert_allclose(gpu["position"], cpu.position, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gpu["velocity"], cpu.velocity, rtol=1e-5, atol=1e-6)


def test_warp_masked_reset_keeps_arrays_and_other_environments():
    wp = pytest.importorskip("warp")
    from simulator_gpu import GpuSimulator

    device = "cuda" if wp.is_cuda_available() else "cpu"
    sim = GpuSimulator(Config(soldiers_per_team=1), num_envs=2, device=device)
    position = sim.position
    sim.step(np.zeros((2, 2, 4), np.float32))
    sim.reset(mask=np.array([1, 0], np.int32))
    state = sim.numpy_state()
    assert sim.position is position
    np.testing.assert_array_equal(state["step_count"], [0, 1])

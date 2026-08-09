import numpy as np
import pytest


def test_rl_step_and_policy_are_finite():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from rl_env import LOCAL_FEATURE_SIZE, RLEnv
    from simulator import TERRITORY_CELLS, Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.1), num_envs=2, device=device
    )
    state = env.reset()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).to(state.features.device)
    actions, log_prob, value, _ = policy.act(state)
    assert state.features.shape == (2, 2, 2, LOCAL_FEATURE_SIZE)
    assert state.cells.shape == (2, 2, 2)
    assert state.owners.shape == (2, TERRITORY_CELLS)
    assert actions.shape == (2, 2, 2, 4)
    assert value.shape == (2, 2, 2)
    assert log_prob.isfinite().all() and value.isfinite().all()
    assert policy.actor_actions(state).shape == (2, 2, 2, 4)
    _, facts = env.step(actions)
    assert facts["damage_taken"].shape == (2, 2)
    assert facts["territory"].shape == (2, 2) and facts["territory"].isfinite().all()
    assert facts["territory_delta"].shape == (2, 2)
    assert facts["done"].bool().all() and (facts["winner"] != -2).all()
    final = env.sim.numpy_state()
    assert not torch.from_numpy(final["done"]).any()


def test_presence_control_is_symmetric_and_in_local_state():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import (
        TERRITORY_CELLS,
        TERRITORY_COORDINATES,
        TERRITORY_INITIAL_OWNER,
        TERRITORY_TOTAL_WEIGHT,
        TERRITORY_WEIGHTS,
        Config,
        territory_centers,
    )

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(Config(soldiers_per_team=2, maximum_episode_seconds=0.1), device=device)
    state = env.reset()
    initial_territory = env.territory.detach().cpu().clone()
    # Control shares are zero until the first step; frame zero shows only the
    # initial display constant.
    torch.testing.assert_close(
        initial_territory, torch.zeros_like(initial_territory)
    )
    expected_fraction = (
        TERRITORY_WEIGHTS[TERRITORY_INITIAL_OWNER == 0].sum() / TERRITORY_TOTAL_WEIGHT
    )
    cell_owner = state.owners[:, None].expand(-1, 2, -1).gather(2, state.cells.long())
    assert (cell_owner[:, 0] == 0).all() and (cell_owner[:, 1] == 1).all()

    centers = territory_centers(env.config)
    cell = {tuple(value): index for index, value in enumerate(TERRITORY_COORDINATES)}
    spread = centers[[cell[(5, 0)], cell[(5, -1)], cell[(10, 0)], cell[(10, -1)]]]
    env.sim.reset({"position": spread})
    wp.synchronize_device(env.sim.device)
    actions = torch.zeros((1, 2, 2, 4), device=env.device)
    state, facts = env.step(actions, reset_done=False)
    assert facts["done"].bool().all()
    occupied_owner = state.owners[0].gather(0, state.cells[0, 0].long())
    assert (occupied_owner == 0).all()
    enemy_owner = state.owners[0].gather(0, state.cells[0, 1].long())
    assert (enemy_owner == 1).all()
    # Presence scoring: each team holds only its projection bubbles.
    assert 0 < float(facts["territory"][0, 0]) < expected_fraction
    assert 0 < float(facts["territory"][0, 1]) < expected_fraction
    expected_delta = facts["territory"].detach().cpu() - initial_territory
    torch.testing.assert_close(facts["territory_delta"].detach().cpu(), expected_delta)


def test_reward_is_scaled_control_advantage_level():
    torch = pytest.importorskip("torch")
    from train import reward_from_facts

    facts = {
        "done": torch.tensor([0], dtype=torch.int32),
        "territory": torch.tensor([[0.30, 0.20]]),
    }
    reward = torch.zeros((1, 2))
    reward_from_facts(reward, facts, scale=0.5)
    torch.testing.assert_close(reward, torch.tensor([[0.05, -0.05]]))
    facts["done"].fill_(1)
    reward_from_facts(reward, facts, scale=0.5)
    torch.testing.assert_close(reward, torch.tensor([[0.05, -0.05]]))

    reward_from_facts(reward, facts, accumulate=True, scale=0.5)
    torch.testing.assert_close(reward, torch.tensor([[0.10, -0.10]]))


def test_strongpoint_control_dominates_equal_plain_control():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import TERRITORY_COORDINATES, Config, territory_centers

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    config = Config(soldiers_per_team=1, maximum_episode_seconds=0.1)
    cell = {tuple(value): index for index, value in enumerate(TERRITORY_COORDINATES)}
    centers = territory_centers(config)
    env = RLEnv(config, device=device)
    env.sim.reset({"position": centers[[cell[(1, 0)], cell[(10, 0)]]]})
    env.observe()
    actions = torch.zeros((1, 2, 1, 4), device=env.device)
    _, facts = env.step(actions, reset_done=False)

    # Both soldiers project similar-sized discs, but only the first covers
    # the 7-tile center strongpoint cluster (weight 30 each).
    assert float(facts["territory"][0, 0] - facts["territory"][0, 1]) > 120 / 1156
    assert facts["done"].bool().all() and (facts["winner"] == 0).all()


def test_terminal_team_reward_produces_per_soldier_advantages():
    torch = pytest.importorskip("torch")
    from train import compute_gae

    rollout = {
        "reward": torch.tensor([[[0.0, 0.0]], [[0.4, -0.4]]]),
        "done": torch.tensor([[False], [True]]),
        "value": torch.zeros((2, 1, 2, 3)),
        "advantage": torch.empty((2, 1, 2, 3)),
    }
    compute_gae(rollout, torch.zeros((1, 2, 3)), gamma=1.0, gae_lambda=1.0)
    expected = torch.tensor([0.4, -0.4]).view(1, 1, 2, 1).expand(2, 1, 2, 3)
    torch.testing.assert_close(rollout["advantage"], expected)


def test_mappo_policy_and_value_heads_are_per_soldier():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import CHECKPOINT_VERSION, Policy

    state, _ = _synthetic_state(torch)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16)
    actions, _, values, _ = policy.act(state)
    assert policy.policy_head.in_features == policy.value_head.in_features == 32
    assert policy.global_value_encoder[-1].out_features == 32
    assert CHECKPOINT_VERSION == 13
    assert policy.tile_encoder[0].in_features == 9
    assert policy.backbone[0].in_features == 3 * 8 + 16
    assert policy.entity_query.out_features == 16
    assert policy.entity_output.out_features == 2 * 8
    assert policy.attention_heads == 4
    assert policy.global_tile_encoder[0].in_features == 2 * 8 + 8
    assert policy.cell_value.count_nonzero() == 21
    assert actions.shape == (1, 2, 12, 4)
    assert values.shape == (1, 2, 12)


def test_vertical_mirror_augmentation_is_paired():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(Config(soldiers_per_team=4), num_envs=4, device=device)
    state = env.reset()
    torch.testing.assert_close(
        state.features[2, :, :, [1, 3, 5, 9]],
        -state.features[0, :, :, [1, 3, 5, 9]],
    )
    torch.testing.assert_close(
        state.features[2, :, :, [0, 2, 4, 6, 7, 8, 10, 11, 12]],
        state.features[0, :, :, [0, 2, 4, 6, 7, 8, 10, 11, 12]],
    )
    assert (state.features[..., 8:10].abs() <= 1).all()
    positions = wp.to_torch(env.sim.position).view(4, 2, 4, 2)
    expected_global = torch.stack(
        (
            2 * positions[..., 0] / env.config.world_width - 1,
            2 * positions[..., 1] / env.config.world_height - 1,
        ),
        dim=-1,
    )
    expected_global[..., 0] *= torch.tensor(
        [1, -1], device=positions.device
    ).view(1, 2, 1)
    expected_global[..., 1] *= env.mirror_y.view(4, 1, 1)
    torch.testing.assert_close(
        state.features[..., 8:10].float(), expected_global, rtol=0, atol=4e-3
    )

    actions = torch.zeros((4, 2, 4, 4), device=state.features.device)
    actions[..., 1] = 1
    actions[..., 3] = 1
    world = env.actions_to_sim(actions).numpy().reshape(4, 8, 4)
    np.testing.assert_array_equal(world[:, 0, 1], [1, 1, -1, -1])
    np.testing.assert_array_equal(world[:, 0, 3], [1, 1, -1, -1])


def _synthetic_state(torch, soldiers=12, enemy_cell=(1, 0)):
    from rl_env import LOCAL_FEATURE_SIZE, LocalState, entity_neighbors
    from simulator import (
        TERRITORY_CELLS,
        TERRITORY_COORDINATES,
        TERRITORY_INITIAL_OWNER,
        Config,
        territory_centers,
    )

    config = Config()
    cell = {tuple(value): index for index, value in enumerate(TERRITORY_COORDINATES)}
    centers = territory_centers(config)
    features = torch.randn((1, 2, soldiers, LOCAL_FEATURE_SIZE))
    features[..., 6] = 1
    cells = torch.full((1, 2, soldiers), cell[(10, 0)], dtype=torch.int32)
    cells[0, 0, 0] = cell[(0, 0)]
    cells[0, 1] = cell[enemy_cell]
    positions = torch.empty((1, 2 * soldiers, 2))
    for team in range(2):
        for index in range(soldiers):
            center = centers[int(cells[0, team, index])]
            positions[0, team * soldiers + index, 0] = float(center[0]) + 0.1 * (index % 5) - 0.2
            positions[0, team * soldiers + index, 1] = float(center[1]) + 0.1 * (index // 5) - 0.2
    canonical_x = 2 * positions[0, :, 0] / config.world_width - 1
    canonical_y = 2 * positions[0, :, 1] / config.world_height - 1
    features[0, 0, :, 8] = canonical_x[:soldiers]
    features[0, 0, :, 9] = canonical_y[:soldiers]
    features[0, 1, :, 8] = -canonical_x[soldiers:]
    features[0, 1, :, 9] = canonical_y[soldiers:]
    alive = torch.ones((1, 2, soldiers), dtype=torch.bool)
    neighbors = entity_neighbors(positions, alive.view(1, -1), soldiers)
    owners = torch.from_numpy(TERRITORY_INITIAL_OWNER.copy()).view(1, TERRITORY_CELLS)
    state = LocalState(features, cells, alive, owners, torch.ones(1), None, neighbors)
    return state, cell


def test_actor_has_no_information_outside_its_two_hex_radius():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy
    from rl_env import LocalState

    torch.manual_seed(2)
    state, cell = _synthetic_state(torch, enemy_cell=(10, 0))
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).eval()
    assert not (state.neighbors[0, 0, 0] >= 0).any()
    before = policy.actor_actions(state, deterministic=True)[0, 0, 0]
    changed_features = state.features.clone()
    changed_features[0, 1, -1] *= 20
    changed_owners = state.owners.clone()
    changed_owners[0, cell[(10, 0)]] = 0
    changed = state._replace(features=changed_features, owners=changed_owners)
    after = policy.actor_actions(changed, deterministic=True)[0, 0, 0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_centralized_value_uses_information_outside_actor_radius():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy
    from rl_env import LocalState

    torch.manual_seed(4)
    state, cell = _synthetic_state(torch)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).eval()
    before_action = policy.actor_actions(state, deterministic=True)[0, 0, 0]
    before_value = policy.value(state)[0, 0, 0]

    owners = state.owners.clone()
    owners[0, cell[(-10, 0)]] = 1
    changed = state._replace(owners=owners)
    after_action = policy.actor_actions(changed, deterministic=True)[0, 0, 0]
    after_value = policy.value(changed)[0, 0, 0]

    torch.testing.assert_close(before_action, after_action, rtol=0, atol=0)
    assert not torch.equal(before_value, after_value)


def test_tile_entities_are_permutation_invariant_and_not_truncated():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy
    from rl_env import LocalState

    torch.manual_seed(3)
    state, _ = _synthetic_state(torch, soldiers=24)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).eval()
    baseline = policy.actor_actions(state, deterministic=True)[0, 0, 0]

    permutation = torch.randperm(24)
    inverse = torch.argsort(permutation)
    relabeled = state.neighbors.long().clone()
    team_zero = relabeled[:, 0]
    enemy_refs = team_zero >= 24
    team_zero[enemy_refs] = inverse[team_zero[enemy_refs] - 24] + 24
    team_one = relabeled[:, 1][:, permutation].clone()
    ally_refs = (team_one >= 0) & (team_one < 24)
    team_one[ally_refs] = inverse[team_one[ally_refs]]
    permuted = state._replace(
        features=torch.stack((state.features[:, 0], state.features[:, 1, permutation]), dim=1),
        cells=torch.stack((state.cells[:, 0], state.cells[:, 1, permutation]), dim=1),
        alive=torch.stack((state.alive[:, 0], state.alive[:, 1, permutation]), dim=1),
        neighbors=torch.stack((team_zero, team_one), dim=1).to(torch.int16),
    )
    shuffled = policy.actor_actions(permuted, deterministic=True)[0, 0, 0]
    torch.testing.assert_close(baseline, shuffled, rtol=1e-5, atol=1e-6)

    references = state.neighbors[0, 0, 0].long()
    visible_enemy = int(references[references >= 24][0]) - 24
    features = state.features.clone()
    features[0, 1, visible_enemy, 6] = 0.1
    changed = state._replace(features=features)
    changed_action = policy.actor_actions(changed, deterministic=True)[0, 0, 0]
    assert not torch.equal(baseline, changed_action)


def test_dead_soldiers_with_no_tile_are_safely_masked():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy

    state, _ = _synthetic_state(torch)
    state.cells[0, 0, -1] = -1
    state.alive[0, 0, -1] = False
    actions = Policy(hidden_size=32, entity_size=8, tile_size=16).actor_actions(state)
    assert actions.isfinite().all()
    assert not actions[0, 0, -1].any()


def test_local_feature_contract_includes_canonical_global_position():
    pytest.importorskip("torch")
    pytest.importorskip("warp")
    from rl_env import LOCAL_FEATURE_NAMES

    assert LOCAL_FEATURE_NAMES == (
        "tile_offset_x",
        "tile_offset_y",
        "velocity_x",
        "velocity_y",
        "facing_x",
        "facing_y",
        "health",
        "attack_recovery",
        "global_x",
        "global_y",
        "time_remaining",
        "score_advantage",
        "score_integral",
    )


def test_actor_is_side_equivariant_at_the_symmetric_start():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy
    from rl_env import RLEnv
    from simulator import Config

    torch.manual_seed(11)
    env = RLEnv(Config(soldiers_per_team=4), device="cpu")
    state = env.reset()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).eval()
    actions, _, values, _ = policy.act(state, deterministic=True)

    torch.testing.assert_close(state.features[:, 0], state.features[:, 1])
    torch.testing.assert_close(actions[:, 0], actions[:, 1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(values[:, 0], values[:, 1], rtol=1e-5, atol=1e-6)


def test_nearest_enemy_opponent_controls_only_the_fixed_team():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from rl_env import NearestEnemyOpponent, RLEnv
    from simulator import Config
    from train import collect_rollout, make_rollout

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.1),
        num_envs=2,
        device=device,
    )
    positions = np.array(
        [
            [[25, 30], [25, 40], [35, 30], [35, 39]],
            [[25, 30], [25, 40], [35, 31], [35, 40]],
        ],
        np.float32,
    )
    env.sim.reset({"position": positions})
    learner_teams = torch.tensor(
        [[True, False], [False, True]], device=env.device
    )
    opponent = NearestEnemyOpponent(env, learner_teams)
    actions = opponent.act()
    world = env.actions_to_sim(actions).numpy().reshape(2, 4, 4)

    diagonal = np.float32(1 / np.sqrt(101))
    expected = np.array(
        [
            [[0, 0], [0, 0], [-1, 0], [-10 * diagonal, diagonal]],
            [[10 * diagonal, diagonal], [1, 0], [0, 0], [0, 0]],
        ],
        np.float32,
    )
    np.testing.assert_allclose(world[..., :2], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(world[..., 2:], expected, rtol=1e-5, atol=1e-6)

    state = env.observe()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).to(env.device)
    rollout = make_rollout(1, state)
    collect_rollout(
        env,
        policy,
        rollout,
        state,
        learner_teams,
        None,
        None,
        nearest_opponent=opponent,
    )
    fixed_teams = ~learner_teams
    torch.testing.assert_close(rollout["actions"][0][fixed_teams], actions[fixed_teams])


def test_opponent_pool_replaces_snapshots_only_after_assigned_episodes_drain():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy
    from train import OpponentPool

    policy = Policy(hidden_size=32, entity_size=8, tile_size=16)
    pool = OpponentPool(policy, 4)
    slots = torch.tensor([0, 0, 1, 2])
    old = next(pool.models[0].parameters()).detach().clone()
    with torch.no_grad():
        next(policy.parameters()).add_(1)
    expected = next(policy.parameters()).detach().clone()

    pool.advance(policy, slots, request_snapshot=True)
    assert pool.retiring == 0 and pool.available == [1, 2, 3]
    assert torch.equal(next(pool.models[0].parameters()), old)
    pool.resample_finished(slots, torch.tensor([True, False, False, False]))
    pool.advance(policy, slots, request_snapshot=False)
    assert torch.equal(next(pool.models[0].parameters()), old)

    pool.resample_finished(slots, torch.ones(4, dtype=torch.bool))
    metrics = pool.advance(policy, slots, request_snapshot=False)
    assert torch.equal(next(pool.models[0].parameters()), expected)
    assert metrics["opponent_snapshots"] == 1


def test_opponent_pool_samples_newest_half_the_time_and_older_slots_uniformly():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy
    from train import OpponentPool

    torch.manual_seed(7)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16)
    pool = OpponentPool(policy, 8, latest_probability=0.5)
    slots = torch.arange(1, 8)
    pool.advance(policy, slots, request_snapshot=True)
    pool.advance(policy, slots, request_snapshot=False)

    counts = pool.initial_slots(40_000).bincount(minlength=8)
    assert 0.48 < counts[0] / counts.sum() < 0.52
    assert (counts[1:] > 2_500).all()


def test_rollout_replay_uses_the_existing_episode_and_keeps_history(tmp_path):
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import Config
    from train import RolloutReplay, write_rollout_replay

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.2),
        num_envs=1,
        device=device,
    )
    capture = RolloutReplay(env, length=2)
    actions = torch.zeros((1, 2, 2, 4), device=env.device)
    done, winner = [], []
    capture.start()
    for _ in range(2):
        _, facts = env.step(actions, before_reset=capture.capture)
        done.append(facts["done"].clone())
        winner.append(facts["winner"].clone())

    replay = capture.replay(torch.stack(done), torch.stack(winner), update=9)
    assert len(replay["frames"]) == 3
    assert replay["statistics"]["decision_steps"] == 2
    assert replay["update"] == 9 and replay["winner"] in (-1, 0, 1)
    assert replay["opponent_mode"] == "self" and replay["learner_team"] == 0

    write_rollout_replay(tmp_path, replay)
    history = tmp_path / "replays" / "update-9.html"
    latest = tmp_path / "replay.html"
    assert history.exists() and latest.exists()
    assert history.stat().st_ino == latest.stat().st_ino
    contents = history.read_text()
    assert '"update":9' in contents
    assert '"control_u8":' in contents


def test_action_repeat_holds_one_action_across_three_simulator_steps():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from rl_env import RLEnv
    from simulator import Config
    from train import RolloutReplay, collect_rollout, make_rollout

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.3),
        num_envs=2,
        device=device,
    )
    state = env.reset()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).to(env.device)
    rollout = make_rollout(1, state)
    learner_teams = torch.tensor([[True, False], [False, True]], device=env.device)
    capture = RolloutReplay(env, length=1, action_repeat=3)
    refresh = env._refresh
    refreshes = 0

    def count_refreshes():
        nonlocal refreshes
        refreshes += 1
        refresh()

    env._refresh = count_refreshes

    collect_rollout(
        env,
        policy,
        rollout,
        state,
        learner_teams,
        None,
        None,
        action_repeat=3,
        replay=capture,
    )
    replay = capture.replay(rollout["done"], rollout["winner"], update=1)

    assert rollout["done"].all()
    assert refreshes == 1
    assert len(replay["frames"]) == 4
    assert replay["statistics"]["decision_steps"] == 3
    assert replay["statistics"]["simulated_seconds"] == pytest.approx(0.3)


def test_mode_latent_is_in_state_and_holds_within_episodes():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    torch.manual_seed(5)
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.2),
        num_envs=32,
        device=device,
        mode_count=4,
    )
    state = env.reset()
    assert state.mode is env.mode and state.mode.shape == (32, 2)
    assert int(state.mode.min()) >= 0 and int(state.mode.max()) < 4
    assert len(torch.unique(state.mode)) > 1

    held = env.mode.detach().clone()
    actions = torch.zeros((32, 2, 2, 4), device=env.device)
    state, facts = env.step(actions)
    assert not facts["done"].bool().any()
    torch.testing.assert_close(env.mode, held)

    resampled = False
    for trial in range(5):
        env.reset()
        held = env.mode.detach().clone()
        env.step(actions)
        _, facts = env.step(actions)
        assert facts["done"].bool().all()
        resampled = resampled or not torch.equal(env.mode, held)
    assert resampled


def test_mode_distribution_override_controls_sampling():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.1),
        num_envs=16,
        device=device,
        mode_count=4,
    )
    env.set_mode_distribution(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    env.reset()
    assert (env.mode == 2).all()
    env.set_mode_distribution(None)
    with pytest.raises(ValueError):
        env.set_mode_distribution(torch.ones(3))


def test_mode_changes_action_means_and_values():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy

    torch.manual_seed(6)
    state, _ = _synthetic_state(torch)
    state = state._replace(mode=torch.zeros((1, 2), dtype=torch.long))
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, mode_count=4).eval()
    baseline = policy.actor_actions(state, deterministic=True)
    baseline_value = policy.value(state)
    shifted_state = state._replace(mode=torch.full((1, 2), 3, dtype=torch.long))
    shifted = policy.actor_actions(shifted_state, deterministic=True)
    shifted_value = policy.value(shifted_state)
    assert not torch.equal(baseline, shifted)
    assert (baseline - shifted).abs().max() > 1e-3
    assert not torch.equal(baseline_value, shifted_value)


def test_mode_free_policy_keeps_legacy_parameters_and_ignores_mode():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy

    torch.manual_seed(6)
    state, _ = _synthetic_state(torch)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).eval()
    assert policy.model_kwargs["mode_count"] == 0
    assert not any(name.startswith("mode_") for name in policy.state_dict())
    assert policy.backbone[0].in_features == 3 * 8 + 16
    assert policy.global_value_encoder[0].in_features == 2 * 16
    without_mode = policy.actor_actions(state, deterministic=True)
    with_mode = policy.actor_actions(
        state._replace(mode=torch.ones((1, 2), dtype=torch.long)), deterministic=True
    )
    torch.testing.assert_close(without_mode, with_mode, rtol=0, atol=0)


def test_team_summary_is_canonical_at_the_symmetric_start():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import SUMMARY_SIZE, RLEnv
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(Config(soldiers_per_team=4), num_envs=4, device=device)
    env.reset()
    summary = env.team_summary()
    assert summary.shape == (4, 2, SUMMARY_SIZE)
    assert summary.isfinite().all()
    torch.testing.assert_close(summary[:, 0], summary[:, 1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(summary[2], summary[0], rtol=1e-4, atol=1e-5)
    alive_fraction = summary[..., 6]
    torch.testing.assert_close(alive_fraction, torch.ones_like(alive_fraction))
    health_fraction = summary[..., 7]
    torch.testing.assert_close(health_fraction, torch.ones_like(health_fraction))
    assert (summary[..., 8:10] == 0).all()


def test_mode_intrinsic_reward_pays_only_learner_teams():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    import math

    from rl_env import SUMMARY_SIZE
    from train import ModeDiscriminator, add_mode_intrinsic_reward

    torch.manual_seed(8)
    length, envs, modes = 3, 2, 4
    rollout = {
        "mode": torch.randint(modes, (length, envs, 2)),
        "summary": torch.randn((length, envs, 2, SUMMARY_SIZE)),
        "reward": torch.full((length, envs, 2), 0.25),
    }
    learner_teams = torch.tensor([[True, False], [False, True]])
    discriminator = ModeDiscriminator(SUMMARY_SIZE, modes)
    metrics = add_mode_intrinsic_reward(rollout, discriminator, learner_teams, 0.5, modes)

    with torch.no_grad():
        log_prob = discriminator(rollout["summary"]).log_softmax(-1)
    expected_bonus = 0.5 * (
        log_prob.gather(-1, rollout["mode"].unsqueeze(-1)).squeeze(-1) + math.log(modes)
    )
    mask = learner_teams.unsqueeze(0).expand(length, -1, -1)
    torch.testing.assert_close(rollout["reward"][mask], 0.25 + expected_bonus[mask])
    torch.testing.assert_close(
        rollout["reward"][~mask], torch.full_like(rollout["reward"][~mask], 0.25)
    )
    assert metrics["mode_intrinsic_fraction"] >= 0


def test_anchor_evaluator_reports_per_mode_win_rates():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from train import AnchorEvaluator
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    torch.manual_seed(9)
    config = Config(soldiers_per_team=2, maximum_episode_seconds=0.2)
    with pytest.raises(ValueError):
        AnchorEvaluator(config, device, mode_count=2, episodes_per_mode=3)
    anchor = AnchorEvaluator(config, device, mode_count=2, episodes_per_mode=2)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, mode_count=2).to(
        anchor.env.device
    )
    metrics = anchor.run(policy)
    for name in ("anchor_win_rate", "anchor_draw_rate", "anchor_territory_advantage"):
        assert name in metrics
    assert 0.0 <= metrics["anchor_win_rate"] <= 1.0
    assert metrics["anchor_mode_win_best"] >= metrics["anchor_mode_win_worst"]


def test_collect_rollout_records_modes_and_summaries():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from rl_env import SUMMARY_SIZE, RLEnv
    from simulator import Config
    from train import collect_rollout, make_rollout

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    torch.manual_seed(10)
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.2),
        num_envs=2,
        device=device,
        mode_count=4,
    )
    state = env.reset()
    initial_modes = env.mode.detach().clone()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, mode_count=4).to(env.device)
    rollout = make_rollout(2, state)
    learner_teams = torch.tensor([[True, False], [False, True]], device=env.device)
    collect_rollout(env, policy, rollout, state, learner_teams, None, None)
    assert rollout["summary"].shape == (2, 2, 2, SUMMARY_SIZE)
    assert rollout["summary"].isfinite().all()
    assert (rollout["damage"] >= 0).all()
    torch.testing.assert_close(rollout["mode"][0], initial_modes)
    torch.testing.assert_close(rollout["mode"][1], initial_modes)


def test_time_remaining_feature_counts_down_and_resets():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import LOCAL_FEATURE_NAMES, RLEnv
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    assert LOCAL_FEATURE_NAMES[10] == "time_remaining"
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.3), num_envs=2, device=device
    )
    state = env.reset()
    clock = state.features[..., 10].float()
    torch.testing.assert_close(clock, torch.ones_like(clock))

    actions = torch.zeros((2, 2, 2, 4), device=env.device)
    state, facts = env.step(actions)
    clock = state.features[..., 10].float()
    torch.testing.assert_close(clock, torch.full_like(clock, 2 / 3), rtol=0, atol=4e-3)
    state, facts = env.step(actions)
    clock = state.features[..., 10].float()
    torch.testing.assert_close(clock, torch.full_like(clock, 1 / 3), rtol=0, atol=4e-3)

    state, facts = env.step(actions)
    assert facts["done"].bool().all()
    clock = state.features[..., 10].float()
    torch.testing.assert_close(clock, torch.ones_like(clock))


def test_nearest_charge_occupies_strongpoints_without_enemies():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import NearestEnemyOpponent, RLEnv
    from simulator import Config, strongpoint_world_centers

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    config = Config(soldiers_per_team=2, maximum_episode_seconds=0.1)
    env = RLEnv(config, num_envs=1, device=device)
    env.sim.reset({"health": np.array([0.0, 0.0, 100.0, 100.0], np.float32)})
    env.observe()
    learner_teams = torch.tensor([[True, False]], device=env.device)
    opponent = NearestEnemyOpponent(env, learner_teams)
    world = env.actions_to_sim(opponent.act()).numpy().reshape(1, 4, 4)

    strongpoints = strongpoint_world_centers(config)
    positions = wp.to_torch(env.sim.position).cpu().numpy().reshape(4, 2)
    for soldier in (2, 3):
        deltas = strongpoints - positions[soldier]
        nearest = deltas[int(np.argmin((deltas * deltas).sum(-1)))]
        expected = nearest / np.linalg.norm(nearest)
        np.testing.assert_allclose(world[0, soldier, :2], expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(world[0, soldier, 2:], expected, rtol=1e-4, atol=1e-5)
    assert not world[0, :2].any()


def test_nearest_charge_holds_inside_a_strongpoint():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import NearestEnemyOpponent, RLEnv
    from simulator import Config, strongpoint_world_centers

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    config = Config(soldiers_per_team=1, maximum_episode_seconds=0.1)
    center = strongpoint_world_centers(config)[1]
    env = RLEnv(config, num_envs=1, device=device)
    env.sim.reset(
        {
            "position": np.array([[5.0, center[1]], center], np.float32),
            "health": np.array([0.0, 100.0], np.float32),
        }
    )
    env.observe()
    opponent = NearestEnemyOpponent(
        env, torch.tensor([[True, False]], device=env.device)
    )
    world = env.actions_to_sim(opponent.act()).numpy().reshape(1, 2, 4)
    assert not world.any()


def test_scripted_environment_mask_is_balanced_across_sides():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from train import scripted_environment_mask

    mask = scripted_environment_mask(256, 0.25, "cpu")
    assert int(mask.sum()) == 64
    index = torch.arange(256)
    assert int(mask[index % 2 == 0].sum()) == int(mask[index % 2 == 1].sum())
    half = scripted_environment_mask(8, 0.5, "cpu")
    assert half.tolist() == [True, False, False, True] * 2
    with pytest.raises(ValueError):
        scripted_environment_mask(8, 0.0, "cpu")


def test_log_std_floor_bounds_exploration_noise():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    import math

    from policy import Policy

    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, log_std_floor=-0.8)
    assert policy.model_kwargs["log_std_floor"] == -0.8
    with torch.no_grad():
        policy.log_std.fill_(-5.0)
    distribution = policy._distribution(torch.zeros(2, 4))
    assert float(distribution.scale.min()) >= math.exp(-0.8) - 1e-6
    legacy = Policy(hidden_size=32, entity_size=8, tile_size=16)
    assert legacy.model_kwargs["log_std_floor"] == -5.0


def test_mixed_opponents_route_scripted_and_pool_environments():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from rl_env import NearestEnemyOpponent, RLEnv
    from simulator import Config
    from train import OpponentPool, collect_rollout, make_rollout, scripted_environment_mask

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    torch.manual_seed(12)
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.1),
        num_envs=4,
        device=device,
    )
    state = env.reset()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).to(env.device)
    pool = OpponentPool(policy, 2)
    slots = pool.initial_slots(4)
    learner_teams = torch.zeros((4, 2), dtype=torch.bool, device=env.device)
    learner_teams[torch.arange(4), torch.arange(4) % 2] = True
    opponent = NearestEnemyOpponent(env, learner_teams)
    scripted = scripted_environment_mask(4, 0.5, env.device)
    rollout = make_rollout(1, state)
    collect_rollout(
        env,
        policy,
        rollout,
        state,
        learner_teams,
        pool,
        slots,
        nearest_opponent=opponent,
        scripted_envs=scripted,
    )
    actions = rollout["actions"][0]
    assert actions.isfinite().all()
    opponent_mask = ~learner_teams
    for index in range(4):
        team = int(opponent_mask[index].long().argmax())
        move, aim = actions[index, team, :, :2], actions[index, team, :, 2:]
        if bool(scripted[index]):
            torch.testing.assert_close(move, aim)
        else:
            assert not torch.equal(move, aim)


def test_rollout_replay_retargets_environment_and_matchup():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import Config
    from train import RolloutReplay

    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.2),
        num_envs=3,
        device="cpu",
    )
    capture = RolloutReplay(env, length=2, opponent_mode="mixed")
    capture.retarget(2, "pool", 0)
    actions = torch.zeros((3, 2, 2, 4), device=env.device)
    done, winner = [], []
    capture.start()
    for _ in range(2):
        _, facts = env.step(actions, before_reset=capture.capture)
        done.append(facts["done"].clone())
        winner.append(facts["winner"].clone())
    replay = capture.replay(torch.stack(done), torch.stack(winner), update=3)
    assert replay["opponent_mode"] == "pool"
    assert replay["learner_team"] == 0
    with pytest.raises(ValueError):
        capture.retarget(0, "nearest", 2)


def test_warp_neighbor_search_matches_torch_reference():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv, entity_neighbors
    from simulator import Config, arena_contains

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    config = Config(soldiers_per_team=24, maximum_episode_seconds=0.1)
    env = RLEnv(config, num_envs=3, device=device)
    generator = np.random.default_rng(17)
    positions = np.empty((3 * 48, 2), np.float32)
    filled = 0
    while filled < 3 * 48:
        candidate = generator.uniform((5, 5), (55, 60), (2,)).astype(np.float32)
        if arena_contains(candidate, config, margin=config.soldier_radius):
            positions[filled] = candidate
            filled += 1
    positions = positions.reshape(3, 48, 2)
    health = generator.uniform(0, 100, (3, 48)).astype(np.float32)
    health[health < 30] = 0.0
    env.sim.reset({"position": positions, "health": health})
    state = env.observe()

    reference = entity_neighbors(
        torch.as_tensor(positions, device=env.device),
        torch.as_tensor(health > 0, device=env.device),
        config.soldiers_per_team,
        env.neighbor_count,
    )
    torch.testing.assert_close(state.neighbors.long(), reference.long())


def test_entity_neighbors_are_egocentric_living_and_radius_limited():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from rl_env import entity_neighbors

    positions = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [4.9, 0.0]]])
    alive = torch.ones((1, 4), dtype=torch.bool)
    neighbors = entity_neighbors(positions, alive, soldiers_per_team=2)
    assert neighbors.shape == (1, 2, 2, 3)
    assert set(neighbors[0, 0, 0].tolist()) == {2, 3, -1}
    assert neighbors[0, 0, 1].tolist() == [-1, -1, -1]
    assert set(neighbors[0, 1, 0].tolist()) == {2, 1, -1}

    alive[0, 0] = False
    neighbors = entity_neighbors(positions, alive, soldiers_per_team=2)
    assert set(neighbors[0, 1, 0].tolist()) == {1, -1}


def test_entity_attention_sees_individual_neighbor_geometry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy

    torch.manual_seed(13)
    state, _ = _synthetic_state(torch)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16).eval()
    baseline = policy.actor_actions(state, deterministic=True)[0, 0, 0]

    references = state.neighbors[0, 0, 0].long()
    visible_enemy = int(references[references >= 12][0]) - 12
    features = state.features.clone()
    features[0, 1, visible_enemy, 8] += 0.02
    moved = policy.actor_actions(state._replace(features=features), deterministic=True)[0, 0, 0]
    assert not torch.equal(baseline, moved)

    isolated = state._replace(neighbors=torch.full_like(state.neighbors, -1))
    lonely = policy.actor_actions(isolated, deterministic=True)
    assert lonely.isfinite().all()


def test_environment_neighbors_exclude_self_and_far_entities():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import ENTITY_RADIUS, RLEnv
    from simulator import Config

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    env = RLEnv(Config(soldiers_per_team=4), num_envs=2, device=device)
    state = env.reset()
    assert state.neighbors.shape == (2, 2, 4, 7)
    positions = wp.to_torch(env.sim.position).view(2, 8, 2)
    for team in range(2):
        for soldier in range(4):
            refs = state.neighbors[0, team, soldier].long()
            for reference in refs[refs >= 0].tolist():
                global_index = (
                    reference if team == 0 else (reference + 4) % 8
                )
                assert global_index != team * 4 + soldier
                distance = (
                    positions[0, global_index] - positions[0, team * 4 + soldier]
                ).norm()
                assert float(distance) <= ENTITY_RADIUS + 1e-4


def test_memory_policy_is_stateful_and_resets_dead_soldiers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy

    torch.manual_seed(21)
    state, _ = _synthetic_state(torch)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, memory_size=8).eval()
    memory = policy.initial_memory(state)
    assert memory.shape == (1, 2, 12, 8) and not memory.any()

    first, _, _, memory_one = policy.act(state, memory, deterministic=True)
    assert memory_one.abs().sum() > 0
    second, _, _, memory_two = policy.act(state, memory_one, deterministic=True)
    assert not torch.equal(first, second)
    assert not torch.equal(memory_one, memory_two)

    dead = state.alive.clone()
    dead[0, 0, -1] = False
    dead_state = state._replace(alive=dead)
    _, _, _, dead_memory = policy.act(dead_state, memory_one, deterministic=True)
    assert not dead_memory[0, 0, -1].any()
    assert dead_memory[0, 0, 0].any()

    legacy = Policy(hidden_size=32, entity_size=8, tile_size=16)
    assert legacy.initial_memory(state) is None
    actions, _, _, carried = legacy.act(state)
    assert carried is None and actions.shape == (1, 2, 12, 4)


def test_message_slot_is_reserved_but_dormant():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from policy import Policy

    torch.manual_seed(22)
    state, _ = _synthetic_state(torch)
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, message_size=8).eval()
    assert policy.entity_token[0].in_features == 8 + 3 + 8
    assert policy.model_kwargs["message_size"] == 8
    actions = policy.actor_actions(state, deterministic=True)
    assert actions.isfinite().all()


def test_sequence_evaluation_reproduces_collection_log_probs():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from policy import Policy
    from rl_env import RLEnv, LocalState
    from simulator import Config
    from train import STATE_KEYS, collect_rollout, make_rollout

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    torch.manual_seed(23)
    env = RLEnv(
        Config(soldiers_per_team=2, maximum_episode_seconds=0.4),
        num_envs=2,
        device=device,
    )
    state = env.reset()
    policy = Policy(hidden_size=32, entity_size=8, tile_size=16, memory_size=8).to(env.device)
    policy.use_bf16 = False
    window = 2
    rollout = make_rollout(4, state, memory_size=8, bptt_window=window)
    learner_teams = torch.tensor([[True, False], [False, True]], device=env.device)
    learner_memory = policy.initial_memory(state)
    collect_rollout(
        env,
        policy,
        rollout,
        state,
        learner_teams,
        None,
        None,
        learner_memory=learner_memory,
        bptt_window=window,
    )
    assert rollout["memory"].shape == (2, 2, 2, 2, 8)
    assert not rollout["memory"][0].any()

    for window_index in range(2):
        memory = rollout["memory"][window_index].clone()
        for offset in range(window):
            step = window_index * window + offset
            step_state = LocalState(*(rollout[name][step] for name in STATE_KEYS))
            log_prob, _, value, memory = policy.evaluate_actions(
                step_state, rollout["actions"][step], memory
            )
            alive = step_state.alive.bool()
            torch.testing.assert_close(
                log_prob[alive], rollout["log_prob"][step][alive], rtol=1e-3, atol=1e-4
            )
            torch.testing.assert_close(
                value[alive], rollout["value"][step][alive], rtol=1e-3, atol=1e-4
            )


def test_score_features_are_own_signed_and_accumulate():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rl_env import RLEnv
    from simulator import TERRITORY_COORDINATES, Config, territory_centers

    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
    config = Config(soldiers_per_team=1, maximum_episode_seconds=0.4)
    cell = {tuple(value): index for index, value in enumerate(TERRITORY_COORDINATES)}
    centers = territory_centers(config)
    env = RLEnv(config, device=device)
    env.sim.reset({"position": centers[[cell[(1, 0)], cell[(10, 0)]]]})
    env.observe()
    actions = torch.zeros((1, 2, 1, 4), device=env.device)

    state, facts = env.step(actions)
    advantage = state.features[..., 11].float()
    # Own-signed: the strongpoint holder sees a positive score, its opponent
    # the mirror-negative of the same magnitude.
    assert float(advantage[0, 0, 0]) > 0.1
    torch.testing.assert_close(
        advantage[0, 0, 0], -advantage[0, 1, 0], rtol=0, atol=5e-3
    )
    first_integral = float(state.features[0, 0, 0, 12])
    assert first_integral > 0

    state, facts = env.step(actions)
    second_integral = float(state.features[0, 0, 0, 12])
    assert second_integral > first_integral

import numpy as np

from viewer import local_march_policy, write_replay
from simulator import (
    STRONGPOINT_CELLS,
    STRONGPOINT_WEIGHT,
    TERRITORY_CELLS,
    TERRITORY_COORDINATES,
    TERRITORY_INITIAL_OWNER,
    TERRITORY_COLUMN_EXTENT,
    Config,
    territory_centers,
)


def test_local_march_moves_forward_until_an_enemy_is_within_two_hexes():
    config = Config(soldiers_per_team=1)
    centers = territory_centers(config)
    cell = {tuple(value): index for index, value in enumerate(TERRITORY_COORDINATES)}
    state = {
        "position": centers[[cell[(-5, 0)], cell[(5, 0)]]],
        "team": np.array([0, 1]),
        "alive": np.ones(2, bool),
    }
    np.testing.assert_array_equal(local_march_policy(state, config)[:, 0], [1, -1])

    state["position"] = centers[[cell[(0, 0)], cell[(0, 1)]]]
    actions = local_march_policy(state, config)
    assert actions[0, 1] > 0 and actions[1, 1] < 0


def test_browser_replay_contains_territory_overlay(tmp_path):
    replay = {
        "config": {
            "world_width": 72,
            "world_height": 42,
            "initial_health": 100,
            "territory_width": 2,
            "territory_height": 1,
        },
        "team": [0, 1],
        "frames": [
            {
                "p": [[18, 21], [54, 21]],
                "a": [0, 3.14],
                "h": [100, 100],
                "o": [[0, 1]],
            }
        ],
        "winner": -1,
        "update": 12,
        "opponent_mode": "nearest",
        "learner_team": 0,
    }
    path = tmp_path / "replay.html"
    write_replay(replay, path)
    html = path.read_text()
    assert "function updateTerritory" in html
    assert "load current" in html
    assert 'id="update"' in html
    assert 'id="updates"' in html
    assert 'id="matchup"' in html
    assert '"update":12' in html
    assert '"opponent_mode":"nearest"' in html
    assert '"learner_team":0' in html
    assert 'nearest:"nearest-charge"' in html
    assert "recorded steps/s" not in html
    assert 'location.replace("update-"+latestUpdate+".html")' in html
    assert 'fetch("manifest.json"' in html
    assert "setInterval(refreshUpdates,10000)" in html
    assert '"update-"+updates.value+".html"' in html
    assert "meanAdvantage" in html
    assert "territory<kbd>T</kbd>" in html
    assert "Uint16Array" in html
    assert 'maxDpr=mobile?1:2' in html
    assert 'targetFrameMs=1000/(mobile?20:30)' in html
    assert "function prepareTerritoryGeometry" in html
    assert "function scheduleDraw" in html
    assert "flex:1; min-height:0" in html
    assert '"frame_count":1' in html
    assert '"territory_i8":' in html
    assert '"frames":' not in html
    assert "__REPLAY__" not in html


def test_browser_replay_accepts_flat_hex_territory(tmp_path):
    config = Config(soldiers_per_team=1)
    replay = {
        "config": {
            "world_width": config.world_width,
            "world_height": config.world_height,
            "initial_health": config.initial_health,
            "decision_dt": config.decision_dt,
            "arena_shape": "hex",
            "territory_radius": TERRITORY_COLUMN_EXTENT,
            "territory_cells": TERRITORY_CELLS,
            "strongpoint_cells": STRONGPOINT_CELLS.tolist(),
            "strongpoint_weight": STRONGPOINT_WEIGHT,
        },
        "team": [0, 1],
        "frames": [
            {
                "p": [[15, config.world_height / 2], [45, config.world_height / 2]],
                "a": [0, 3.14],
                "h": [100, 100],
                "o": TERRITORY_INITIAL_OWNER.tolist(),
            }
        ],
        "winner": -1,
    }
    path = tmp_path / "hex.html"
    write_replay(replay, path)
    html = path.read_text()
    assert '"arena_shape":"hex"' in html
    assert f'"territory_cells":{TERRITORY_CELLS}' in html
    assert f'"strongpoint_weight":{STRONGPOINT_WEIGHT}' in html
    assert "strongpointCells" in html
    assert "territoryBoardPath" in html

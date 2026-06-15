import math
import importlib.util
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stateful_insertion_curriculum.py"
spec = importlib.util.spec_from_file_location("build_stateful_insertion_curriculum", SCRIPT)
stateful_curriculum = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stateful_curriculum)


def _episode(path):
    data = {
        "episode_id": "base",
        "scene": {
            "start_near_gate": {
                "target_gate_axis_world": [0.0, 0.0, -1.0],
                "target_gate_position": [0.9, 2.0, 3.0],
                "lateral_direction_world": [1.0, 0.0, 0.0],
                "body_start_position_world": [1.0, 2.0, 3.0],
                "tcp_start_position_world": [1.0, 2.0, 3.0],
                "body_start_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "reset_body_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "tcp_start_orientation_world": [1.0, 0.0, 0.0, 0.0],
                "reference_tip_center_position_world": [0.9, 2.0, 3.0],
                "reference_reward_body_start_position_world": [0.9, 2.0, 3.0],
                "reference_tcp_position": [0.9, 2.0, 3.0],
                "reference_body_position": [0.9, 2.0, 3.0],
                "reset_body_offset_from_reference_world": [0.1, 0.0, 0.0],
            }
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_stateful_curriculum_yaml_materializes_start_and_terminal_difficulty(tmp_path, monkeypatch):
    base_root = tmp_path / "base"
    episodes = base_root / "episodes"
    episodes.mkdir(parents=True)
    _episode(episodes / "episode_000001.yaml")
    out_root = tmp_path / "stateful"
    config = tmp_path / "stateful.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "base_config_dir": str(base_root),
                "output_root": str(out_root),
                "episodes": 5,
                "seed": 7,
                "start_near_gate": {
                    "axial_distance_m": {"initial": 0.003, "terminal": 0.040},
                    "lateral_distance_m": {"initial": 0.0, "terminal": 0.010},
                    "orientation_error_rad": {"initial": 0.0, "terminal": 0.060},
                },
                "progression": {"policy": "scheduled", "eval_every_episodes": 10},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["build_stateful_insertion_curriculum.py", "--config", str(config)])

    assert stateful_curriculum.main() == 0

    first = yaml.safe_load((out_root / "episodes" / "episode_000001.yaml").read_text(encoding="utf-8"))
    last = yaml.safe_load((out_root / "episodes" / "episode_000005.yaml").read_text(encoding="utf-8"))
    first_start = first["scene"]["start_near_gate"]
    last_start = last["scene"]["start_near_gate"]
    first_variant = first_start["stateful_curriculum_variant"]
    last_variant = last_start["stateful_curriculum_variant"]
    assert first_variant["requested_axial_distance_m"] == 0.003
    assert first_variant["requested_lateral_m"] == 0.0
    assert first_variant["requested_orientation_error_rad"] == 0.0
    assert last_variant["requested_axial_distance_m"] == 0.04
    assert last_variant["requested_lateral_m"] == 0.01
    assert last_variant["requested_orientation_error_rad"] == 0.06
    assert last_variant["progression_policy"] == "scheduled"
    assert math.isclose(last_start["axial_distance_m"], -0.04)
    assert math.isclose(last_start["achieved_axial_distance_m"], -0.04)
    assert math.isclose(last_start["lateral_distance_m"], 0.01)
    assert math.isclose(last_start["achieved_lateral_distance_m"], 0.01)


def test_stateful_curriculum_can_materialize_signed_inserted_start(tmp_path, monkeypatch):
    base_root = tmp_path / "base"
    episodes = base_root / "episodes"
    episodes.mkdir(parents=True)
    _episode(episodes / "episode_000001.yaml")
    out_root = tmp_path / "inserted"
    config = tmp_path / "inserted.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "base_config_dir": str(base_root),
                "output_root": str(out_root),
                "episodes": 1,
                "seed": 11,
                "start_near_gate": {
                    "signed_axial_distance": True,
                    "axial_distance_m": {"initial": 0.012, "terminal": 0.012},
                    "lateral_distance_m": {"initial": 0.0, "terminal": 0.0},
                    "orientation_error_rad": {"initial": 0.02, "terminal": 0.02},
                },
                "progression": {"policy": "scheduled", "eval_every_episodes": 10},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["build_stateful_insertion_curriculum.py", "--config", str(config)])

    assert stateful_curriculum.main() == 0

    data = yaml.safe_load((out_root / "episodes" / "episode_000001.yaml").read_text(encoding="utf-8"))
    start = data["scene"]["start_near_gate"]
    variant = start["stateful_curriculum_variant"]
    assert variant["signed_axial_distance"] is True
    assert variant["requested_axial_distance_m"] == 0.012
    assert math.isclose(start["axial_distance_m"], 0.012)
    assert math.isclose(start["achieved_axial_distance_m"], 0.012)
    assert math.isclose(variant["target_signed_depth_m"], 0.012)

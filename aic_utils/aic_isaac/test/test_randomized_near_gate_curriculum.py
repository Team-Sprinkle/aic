import math
from pathlib import Path

import yaml

from aic_utils.aic_isaac.scripts.build_randomized_near_gate_curriculum import Variant, _write_variant


def _episode(path: Path) -> None:
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


def test_randomized_variant_keeps_shifted_reference_and_reset_offset_consistent(tmp_path):
    base = tmp_path / "episode_000001.yaml"
    out = tmp_path / "episode_000002.yaml"
    _episode(base)

    _write_variant(
        base_episode=base,
        out_path=out,
        variant=Variant("near_gate", target_s_m=-0.004, lateral_m=0.001, rotvec_rad=(0.0, 0.0, 0.0)),
        base_settled_s_m=0.0,
        lateral_sign=1.0,
        rng=__import__("random").Random(1),
    )

    start = yaml.safe_load(out.read_text(encoding="utf-8"))["scene"]["start_near_gate"]
    reference = start["reference_reward_body_start_position_world"]
    body = start["body_start_position_world"]
    offset = start["reset_body_offset_from_reference_world"]

    assert offset == [0.1, 0.0, 0.0]
    assert all(math.isclose(body[i] - reference[i], offset[i], abs_tol=1e-9) for i in range(3))
    assert start["randomized_curriculum_variant"]["tip_preserving_rotation"] is False

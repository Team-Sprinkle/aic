from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.hybrid_schema import inspect_hybrid_schema  # noqa: E402


def write_fake_lerobot_dataset(root: Path, *, request_yaml: bool = True) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    info = {
        "fps": 20,
        "robot_type": "ur5e_aic",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "delta_position.x",
                    "delta_position.y",
                    "delta_position.z",
                    "delta_rotation.x",
                    "delta_rotation.y",
                    "delta_rotation.z",
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [31],
                "names": ["tcp_pose.position.x"],
            },
            "observation.images.center_rgb": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    if request_yaml:
        (root.parent / "request.yaml").write_text(
            "\n".join(
                [
                    "task_family: sfp_to_nic",
                    "generation:",
                    "  simulator_source: gazebo",
                ]
            ),
            encoding="utf-8",
        )


def test_hybrid_schema_infers_canonical_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "run" / "accepted_dataset"
    write_fake_lerobot_dataset(dataset_root)
    summary = inspect_hybrid_schema(dataset_root, action_horizon=8)
    data = summary.as_dict()

    assert data["task_family"] == "sfp_to_nic"
    assert data["simulator_source"] == "gazebo"
    assert data["action_mode"] == "cartesian"
    assert data["action_dim"] == 6
    assert data["action_horizon"] == 8
    assert data["obs_mode"] == "image_lowdim"
    assert data["obs_dim"] == 31
    assert data["camera_keys"] == ["observation.images.center_rgb"]
    assert data["lowdim_keys"] == ["observation.state"]
    assert data["validation"]["has_meta_info"] is True
    assert data["validation"]["has_data_dir"] is True
    assert data["validation"]["has_request_yaml"] is True


def test_inspect_hybrid_schema_cli_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "accepted_dataset"
    write_fake_lerobot_dataset(dataset_root, request_yaml=False)
    script = PACKAGE_DIR / "scripts" / "inspect_hybrid_schema.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--action-horizon",
            "2",
            "--simulator-source",
            "unknown",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["action_dim"] == 6
    assert data["action_horizon"] == 2
    assert data["simulator_source"] == "unknown"

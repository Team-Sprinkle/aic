from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pandas as pd
import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.offline_rl_dataset import (  # noqa: E402
    OfflineRLTransitionDataset,
    load_lerobot_transitions,
)
from lerobot_robot_aic.offline_serl import OfflineSERLConfig, OfflineSERLTrainer  # noqa: E402


def write_fake_lerobot_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 20,
        "robot_type": "ur5e_aic",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["delta_position.x", "delta_position.y"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [3],
                "names": ["x", "y", "z"],
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = []
    idx = 0
    for episode in range(2):
        for frame in range(3):
            rows.append(
                {
                    "action": [float(frame), float(episode)],
                    "observation.state": [float(frame), float(frame + 1), float(episode)],
                    "timestamp": frame / 20.0,
                    "frame_index": frame,
                    "episode_index": episode,
                    "index": idx,
                    "task_index": 0,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


def test_offline_dataset_constructs_transitions(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_fake_lerobot_dataset(root)
    arrays, schema = load_lerobot_transitions(root, reward_mode="final_success")
    assert schema.action_shape == [2]
    assert arrays.obs.shape == (6, 3)
    assert arrays.action.shape == (6, 2)
    assert arrays.done.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert arrays.reward.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert arrays.next_obs[0].tolist() == arrays.obs[1].tolist()


def test_offline_dataset_constructs_action_chunks(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_fake_lerobot_dataset(root)
    arrays, _ = load_lerobot_transitions(root, reward_mode="final_success", action_horizon=2)
    assert arrays.action.shape == (6, 4)
    assert arrays.action[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert arrays.action[2].tolist() == [2.0, 0.0, 2.0, 0.0]
    assert arrays.action[3].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_offline_serl_train_step_and_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_fake_lerobot_dataset(root)
    arrays, schema = load_lerobot_transitions(root, reward_mode="final_success")
    dataset = OfflineRLTransitionDataset(arrays)
    trainer = OfflineSERLTrainer(
        OfflineSERLConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim),
        device="cpu",
    )
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]], dim=0)
        for key in ("obs", "action", "reward", "next_obs", "done")
    }
    metrics = trainer.train_step(batch)
    assert "critic_loss" in metrics
    ckpt = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(
        ckpt,
        train_config={"steps": 1},
        schema_summary=schema.as_dict(),
        normalization_stats=dataset.stats.as_dict(),
        step=1,
    )
    assert ckpt.exists()


def test_train_offline_serl_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_fake_lerobot_dataset(root)
    script = PACKAGE_DIR / "scripts" / "train_offline_serl.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(root),
            "--output-dir",
            str(tmp_path / "out"),
            "--job-name",
            "dry",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--action-horizon",
            "2",
            "--hidden-dim",
            "64",
            "--num-layers",
            "3",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["training_config"]["obs_dim"] == 3
    assert summary["training_config"]["action_dim"] == 4
    assert summary["training_config"]["action_horizon"] == 2
    assert summary["training_config"]["hidden_dim"] == 64
    assert summary["training_config"]["num_layers"] == 3

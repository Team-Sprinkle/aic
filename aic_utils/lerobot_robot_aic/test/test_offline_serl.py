from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pandas as pd
import pytest
import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.offline_rl_dataset import (  # noqa: E402
    OfflineRLTransitionDataset,
    load_lerobot_transitions,
)
from lerobot_robot_aic.offline_serl import OfflineSERLConfig, OfflineSERLTrainer  # noqa: E402
from lerobot_robot_aic.task_encoding import TASK_VECTOR_DIM, encode_task_vector, task_encoding_schema  # noqa: E402


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


def write_task_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "accepted_episode_index": 0,
            "source_episode_index": 0,
            "task_family": "sfp_to_nic",
            "target_port_index": 1,
            "target_card_index": 3,
            "target_card_valid": 1,
            "task_vector": json.dumps(encode_task_vector(
                task_family="sfp_to_nic",
                target_port_index=1,
                target_card_index=3,
            ).astype(int).tolist()),
            "total_score": 1.0,
        },
        {
            "accepted_episode_index": 1,
            "source_episode_index": 1,
            "task_family": "sc_to_sc",
            "target_port_index": 0,
            "target_card_index": -1,
            "target_card_valid": 0,
            "task_vector": json.dumps(encode_task_vector(
                task_family="sc_to_sc",
                target_port_index=0,
                target_card_index=-1,
            ).astype(int).tolist()),
            "total_score": 1.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


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


def test_offline_dataset_appends_task_vector(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest = tmp_path / "manifests" / "accepted.csv"
    write_fake_lerobot_dataset(root)
    write_task_manifest(manifest)

    arrays, _ = load_lerobot_transitions(
        root,
        reward_mode="final_success",
        include_task_vector=True,
        task_metadata=manifest,
    )

    assert arrays.obs.shape == (6, 3 + TASK_VECTOR_DIM)
    assert arrays.next_obs.shape == (6, 3 + TASK_VECTOR_DIM)
    assert arrays.obs[0, :3].tolist() == [0.0, 1.0, 0.0]
    assert arrays.obs[0, 3:].tolist() == pytest.approx([1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    assert arrays.obs[3, 3:].tolist() == pytest.approx([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])


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


def test_offline_serl_act_bias_warmstart_repeats_horizon() -> None:
    trainer = OfflineSERLTrainer(
        OfflineSERLConfig(obs_dim=3, action_dim=4, action_horizon=2),
        device="cpu",
    )
    report = trainer.warm_start_actor_bias_from_action_head(torch.tensor([0.1, -0.2]))
    assert report["mode"] == "action_head_bias"
    assert trainer.actor.mean_head.bias.detach().tolist() == pytest.approx([0.1, -0.2, 0.1, -0.2])


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


def test_train_offline_serl_dry_run_with_task_vector(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest = tmp_path / "manifests" / "accepted.csv"
    write_fake_lerobot_dataset(root)
    write_task_manifest(manifest)
    script = PACKAGE_DIR / "scripts" / "train_offline_serl.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(root),
            "--task-metadata",
            str(manifest),
            "--include-task-vector",
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
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["training_config"]["original_obs_dim"] == 3
    assert summary["training_config"]["effective_obs_dim"] == 3 + TASK_VECTOR_DIM
    assert summary["task_conditioning"]["task_encoding_schema"] == task_encoding_schema()


def test_train_offline_serl_rejects_act_critic_init(tmp_path: Path) -> None:
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
            "--critic-init",
            "act",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ACT has no critic/value semantics" in result.stderr


def test_train_offline_serl_torchrun_cpu_smoke(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    out = tmp_path / "ddp_out"
    write_fake_lerobot_dataset(root)
    script = PACKAGE_DIR / "scripts" / "train_offline_serl.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(script),
            "--dataset-root",
            str(root),
            "--output-dir",
            str(out),
            "--job-name",
            "ddp_cpu",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--save-every",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "checkpoint_latest.pt").is_file()
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["distributed"]["world_size"] == 2


def test_materialize_task_conditioned_dataset_script(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest = tmp_path / "manifests" / "accepted.csv"
    out = tmp_path / "task_conditioned"
    write_fake_lerobot_dataset(root)
    write_task_manifest(manifest)
    script = PACKAGE_DIR / "scripts" / "materialize_task_conditioned_dataset.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(root),
            "--task-metadata",
            str(manifest),
            "--output-root",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["materialized_dataset_root"] == str(out.resolve())
    arrays, _ = load_lerobot_transitions(out, reward_mode="final_success")
    assert arrays.obs.shape == (6, 3 + TASK_VECTOR_DIM)


def test_materialized_task_conditioned_stats_use_safe_std_for_constant_task_bits(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest = tmp_path / "manifests" / "accepted.csv"
    out = tmp_path / "task_conditioned"
    write_fake_lerobot_dataset(root)
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [0.0, 1.0, 0.0],
                    "max": [2.0, 3.0, 1.0],
                    "mean": [1.0, 2.0, 0.5],
                    "std": [0.816, 0.816, 0.5],
                    "count": [6],
                    "q01": [0.0, 1.0, 0.0],
                    "q10": [0.0, 1.0, 0.0],
                    "q50": [1.0, 2.0, 0.5],
                    "q90": [2.0, 3.0, 1.0],
                    "q99": [2.0, 3.0, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    vector = encode_task_vector(
        task_family="sfp_to_nic",
        target_port_index=0,
        target_card_index=0,
    ).astype(int).tolist()
    pd.DataFrame(
        [
            {
                "accepted_episode_index": episode,
                "source_episode_index": episode,
                "task_family": "sfp_to_nic",
                "target_port_index": 0,
                "target_card_index": 0,
                "target_card_valid": 1,
                "task_vector": json.dumps(vector),
                "total_score": 1.0,
            }
            for episode in range(2)
        ]
    ).to_csv(manifest, index=False)

    from lerobot_robot_aic.task_metadata import append_task_vectors_to_observation_state_dataset

    append_task_vectors_to_observation_state_dataset(root, manifest, out)
    stats = json.loads((out / "meta" / "stats.json").read_text())
    state_mean = stats["observation.state"]["mean"]
    state_std = stats["observation.state"]["std"]

    assert state_mean[3:] == pytest.approx([0.0] * TASK_VECTOR_DIM)
    assert state_std[3:] == pytest.approx([1.0] * TASK_VECTOR_DIM)
    assert [
        (value - mean) / std
        for value, mean, std in zip(vector, state_mean[3:], state_std[3:], strict=True)
    ] == pytest.approx(vector)

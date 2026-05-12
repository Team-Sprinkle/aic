#!/usr/bin/env python3
"""Build the canonical 82D ACT dataset from synced clean AIC datasets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "aic_utils" / "lerobot_robot_aic"
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.task_encoding import encode_task_vector  # noqa: E402


FIELDNAMES = [
    "accepted_episode_index",
    "source_episode_index",
    "selected",
    "mode",
    "task_family",
    "target_port_index",
    "target_card_index",
    "target_card_valid",
    "task_family_sfp_to_nic",
    "task_family_sc_to_sc",
    "target_port_0",
    "target_port_1",
    "target_card_0",
    "target_card_1",
    "target_card_2",
    "target_card_3",
    "target_card_4",
    "task_vector",
    "dataset_root",
    "source_dataset_root",
    "trial_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-clean-root", type=Path, default=Path("outputs/s3_clean"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_indices(dataset_root: Path) -> list[int]:
    episodes_dir = dataset_root / "meta" / "episodes"
    vals: list[int] = []
    for path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(path, columns=["episode_index"])
        vals.extend(int(v) for v in df["episode_index"].tolist())
    return sorted(set(vals))


def _has_source_data(dataset_root: Path) -> bool:
    data_dir = dataset_root / "data"
    return any(path.suffix == ".parquet" and not path.name.endswith(".phased.parquet") for path in data_dir.rglob("*.parquet"))


def _fake_score_csv(dataset_root: Path, score_dir: Path) -> Path:
    episodes = _episode_indices(dataset_root)
    score_dir.mkdir(parents=True, exist_ok=True)
    out = score_dir / f"{dataset_root.parent.name}_{len(episodes)}_episodes_score_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "status", "total_score", "run_index"])
        writer.writeheader()
        for idx, _episode in enumerate(episodes, start=1):
            writer.writerow(
                {
                    "trial_id": f"accepted_{idx:06d}",
                    "status": "OK",
                    "total_score": 1.0,
                    "run_index": idx,
                }
            )
    return out


def _trial_yaml(run_root: Path, row: dict[str, str]) -> Path | None:
    trial_id = row.get("trial_id", "")
    candidates: list[Path] = []
    if trial_id.startswith("trial_"):
        candidates.append(run_root / "trials" / f"{trial_id}.yaml")
    run_index = row.get("run_index")
    if run_index:
        try:
            candidates.append(run_root / "trials" / f"trial_{int(run_index):06d}.yaml")
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _task_from_trial(path: Path) -> tuple[str, int, int, int]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    trial = next(iter((data.get("trials") or {}).values()))
    task = next(iter((trial.get("tasks") or {}).values()))
    plug_type = str(task.get("plug_type", ""))
    port_name = str(task.get("port_name", ""))
    target_module_name = str(task.get("target_module_name", ""))
    task_family = "sc_to_sc" if plug_type == "sc" and port_name.startswith("sc_") else "sfp_to_nic"
    if "port_1" in port_name:
        port_index = 1
    else:
        port_index = 0
    if task_family == "sfp_to_nic":
        digits = "".join(ch for ch in target_module_name if ch.isdigit())
        card_index = int(digits) if digits else 0
        card_valid = 1
    else:
        card_index = -1
        card_valid = 0
    return task_family, port_index, card_index, card_valid


def _fallback_task(run_root: Path) -> tuple[str, int, int, int]:
    parts = set(run_root.parts)
    if "sc_to_sc" in parts:
        return "sc_to_sc", 1 if "sc_ports_2" in parts else 0, -1, 0
    port_index = 1 if "port1" in run_root.name or "port_1" in run_root.name else 0
    card_index = 0
    for idx in range(5):
        if f"card{idx}" in run_root.name or f"card_{idx}" in run_root.name:
            card_index = idx
            break
    return "sfp_to_nic", port_index, card_index, 1


def _task_row(
    *,
    accepted_episode_index: int,
    source_episode_index: int,
    dataset_root: Path,
    run_root: Path,
    selection_row: dict[str, str],
) -> dict[str, Any]:
    trial_path = _trial_yaml(run_root, selection_row)
    if trial_path is not None:
        task_family, port_index, card_index, card_valid = _task_from_trial(trial_path)
    else:
        task_family, port_index, card_index, card_valid = _fallback_task(run_root)
    vector = encode_task_vector(
        task_family=task_family,
        target_port_index=port_index,
        target_card_index=card_index,
    ).astype(int).tolist()
    return {
        "accepted_episode_index": accepted_episode_index,
        "source_episode_index": source_episode_index,
        "selected": True,
        "mode": "clean",
        "task_family": task_family,
        "target_port_index": port_index,
        "target_card_index": card_index,
        "target_card_valid": card_valid,
        "task_family_sfp_to_nic": int(task_family == "sfp_to_nic"),
        "task_family_sc_to_sc": int(task_family == "sc_to_sc"),
        "target_port_0": int(port_index == 0),
        "target_port_1": int(port_index == 1),
        "target_card_0": int(card_valid == 1 and card_index == 0),
        "target_card_1": int(card_valid == 1 and card_index == 1),
        "target_card_2": int(card_valid == 1 and card_index == 2),
        "target_card_3": int(card_valid == 1 and card_index == 3),
        "target_card_4": int(card_valid == 1 and card_index == 4),
        "task_vector": json.dumps(vector),
        "dataset_root": str(dataset_root),
        "source_dataset_root": str(dataset_root),
        "trial_id": selection_row.get("trial_id", ""),
    }


def main() -> None:
    args = parse_args()
    clean_root = args.s3_clean_root.resolve()
    output_root = args.output_root.resolve()
    raw_root = output_root.parent / f"{output_root.name}_raw32"
    fake_score_dir = output_root.parent / f"{output_root.name}_fake_scores"

    if output_root.exists() or raw_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} or {raw_root} already exists")
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(raw_root, ignore_errors=True)
        shutil.rmtree(fake_score_dir, ignore_errors=True)

    roots = sorted(
        p.parent.parent
        for p in clean_root.glob("**/accepted_dataset/meta/info.json")
        if any(part in {"sfp_to_nic", "sc_to_sc"} for part in p.parts) and _has_source_data(p.parent.parent)
    )
    if not roots:
        raise RuntimeError(f"No accepted_dataset roots found under {clean_root}")

    score_csvs = [_fake_score_csv(root, fake_score_dir) for root in roots]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "scripts" / "filter_merge_lerobot_by_score.py"),
        "--datasets",
        *[str(root) for root in roots],
        "--score-csvs",
        *[str(path) for path in score_csvs],
        "--min-score",
        "0",
        "--status-allowlist",
        "OK",
        "--include-videos",
        "--output",
        str(raw_root),
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    manifest_path = raw_root / "manifests" / "accepted.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_index = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for root in roots:
            run_root = root.parent
            selected_rows: list[dict[str, str]] = []
            selection_path = root / "selection_report.csv"
            if selection_path.exists():
                with selection_path.open("r", encoding="utf-8", newline="") as sf:
                    selected_rows = [row for row in csv.DictReader(sf) if str(row.get("selected", "")).lower() == "true"]
            episodes = _episode_indices(root)
            if len(selected_rows) < len(episodes):
                selected_rows.extend({} for _ in range(len(episodes) - len(selected_rows)))
            for local_ep, selection_row in zip(episodes, selected_rows, strict=False):
                writer.writerow(
                    _task_row(
                        accepted_episode_index=accepted_index,
                        source_episode_index=local_ep,
                        dataset_root=root,
                        run_root=run_root,
                        selection_row=selection_row,
                    )
                )
                accepted_index += 1

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "scripts" / "materialize_aic_training_dataset.py"),
            "--dataset-root",
            str(raw_root),
            "--task-metadata",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    report = {
        "accepted_dataset_roots": len(roots),
        "output_root": str(output_root),
        "raw_root": str(raw_root),
        "manifest": str(manifest_path),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
    }
    (output_root / "meta" / "clean_dataset_prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

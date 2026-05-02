#!/usr/bin/env python3
"""Convert an external LeRobot dataset into expert-generator output layout.

The converter intentionally keeps the LeRobot dataset files unchanged.  It only
copies them into the same accepted_dataset_<mode>/ layout used by
scripts/generate_expert_trajectories.py and adds AIC expert metadata sidecars.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_URL = "https://huggingface.co/datasets/brucekimrok/sfp2nic_target_card0_port0_randomized"


@dataclass(frozen=True)
class ExpertEpisodeMetadata:
    episode_index: int
    mode: str
    scene_id: str
    candidate_index: int
    trajectory_path: str | None
    validation: dict[str, Any]
    vlm_strategy: dict[str, Any]
    moveit: dict[str, Any]
    phase_labels: list[dict[str, Any]]
    extra: dict[str, Any]


class DatasetMetadataWriter:
    """Small local writer for converted external LeRobot metadata sidecars."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.root / "episodes.jsonl"
        if not self.episodes_path.exists():
            self.episodes_path.write_text("", encoding="utf-8")

    def append_episode(self, metadata: ExpertEpisodeMetadata) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metadata), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap a LeRobot dataset in the AIC expert-generator accepted dataset layout.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Downloaded LeRobot dataset root, containing meta/info.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output run directory that will receive accepted_dataset_<mode>/.",
    )
    parser.add_argument(
        "--mode",
        choices=("nominal", "nominalrecovery", "recovery"),
        default="nominal",
        help="Expert dataset split/mode label to attach to the converted episodes.",
    )
    parser.add_argument("--source-repo-id", default="brucekimrok/sfp2nic_target_card0_port0_randomized")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--task-family", default=None)
    parser.add_argument("--target-card-index", type=int, default=None)
    parser.add_argument("--target-port-index", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing accepted_dataset_<mode> and accepted_metadata directory.",
    )
    return parser.parse_args()


def _require_lerobot_root(path: Path) -> dict[str, Any]:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing LeRobot metadata file: {info_path}")
    for child in ("data", "meta", "videos"):
        if not (path / child).exists():
            raise FileNotFoundError(f"missing required LeRobot directory: {path / child}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _infer_from_name(name: str) -> dict[str, Any]:
    normalized = name.lower()
    task_family = "sfp_to_nic" if "sfp2nic" in normalized or "sfp_to_nic" in normalized else None
    card_match = re.search(r"(?:card|target_card)(\d+)", normalized)
    port_match = re.search(r"(?:port|target_port)(\d+)", normalized)
    return {
        "task_family": task_family,
        "target_card_index": int(card_match.group(1)) if card_match else None,
        "target_port_index": int(port_match.group(1)) if port_match else None,
    }


def _copy_dataset(source: Path, dest: Path, *, overwrite: bool) -> None:
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for child in ("data", "meta", "videos"):
        shutil.copytree(source / child, dest / child)
    for optional in ("README.md", "selection_report.csv"):
        src = source / optional
        if src.exists():
            shutil.copy2(src, dest / optional)


def _episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"no LeRobot episode parquet files found under {dataset_root / 'meta' / 'episodes'}")
    frames = [pd.read_parquet(path) for path in episode_files]
    episodes = pd.concat(frames, ignore_index=True)
    rows = []
    for row in episodes.to_dict(orient="records"):
        rows.append({key: _jsonable(value) for key, value in row.items()})
    return rows


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _phase_label(row: dict[str, Any]) -> list[dict[str, Any]]:
    start_ts = float(row.get("timestamp", {}).get("min", 0.0)) if isinstance(row.get("timestamp"), dict) else 0.0
    end_ts = float(row.get("timestamp", {}).get("max", start_ts)) if isinstance(row.get("timestamp"), dict) else start_ts
    return [
        {
            "timestamp": start_ts,
            "phase": "nominal_external_episode",
            "source": "external_lerobot_dataset",
        },
        {
            "timestamp": end_ts,
            "phase": "episode_end",
            "source": "external_lerobot_dataset",
        },
    ]


def _write_sidecars(
    *,
    writer_root: Path,
    episode_rows: list[dict[str, Any]],
    mode: str,
    source_repo_id: str,
    source_url: str,
    task_family: str,
    target_card_index: int,
    target_port_index: int,
) -> None:
    writer = DatasetMetadataWriter(writer_root)
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        writer.append_episode(
            ExpertEpisodeMetadata(
                episode_index=episode_index,
                mode=mode,
                scene_id=f"external_hf_episode_{episode_index:06d}",
                candidate_index=episode_index,
                trajectory_path=None,
                validation={
                    "accepted": True,
                    "source_validation": "external_dataset_preserved",
                    "score": None,
                    "insertion_event": None,
                },
                vlm_strategy={
                    "source": "external_dataset_no_vlm_strategy",
                    "cable_risk": None,
                    "reason": "Converted from a previously recorded LeRobot dataset; no per-episode VLM strategy was provided.",
                    "preferred_approach_strategy": None,
                    "avoid_regions": [],
                },
                moveit={
                    "source": "external_dataset_no_moveit_plan",
                    "success": None,
                },
                phase_labels=_phase_label(row),
                extra={
                    "source_repo_id": source_repo_id,
                    "source_url": source_url,
                    "task_family": task_family,
                    "target_card_index": target_card_index,
                    "target_port_index": target_port_index,
                    "lerobot_episode_metadata": row,
                },
            )
        )


def _task_vector(task_family: str, target_card_index: int, target_port_index: int) -> list[int]:
    if task_family != "sfp_to_nic":
        raise ValueError(f"converter currently supports sfp_to_nic task vectors, got {task_family!r}")
    if target_port_index not in (0, 1):
        raise ValueError(f"target_port_index must be 0 or 1, got {target_port_index}")
    if target_card_index not in range(5):
        raise ValueError(f"target_card_index must be in 0..4, got {target_card_index}")
    vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    vector[2 + target_port_index] = 1
    vector[4 + target_card_index] = 1
    return vector


def _write_manifest(
    *,
    output_dir: Path,
    accepted_dataset: Path,
    episode_rows: list[dict[str, Any]],
    mode: str,
    task_family: str,
    target_card_index: int,
    target_port_index: int,
) -> None:
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    vector = _task_vector(task_family, target_card_index, target_port_index)
    fieldnames = [
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
    ]
    with (manifests / "accepted.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_rows:
            episode_index = int(row["episode_index"])
            writer.writerow(
                {
                    "accepted_episode_index": episode_index,
                    "source_episode_index": episode_index,
                    "selected": True,
                    "mode": mode,
                    "task_family": task_family,
                    "target_port_index": target_port_index,
                    "target_card_index": target_card_index,
                    "target_card_valid": 1,
                    "task_family_sfp_to_nic": vector[0],
                    "task_family_sc_to_sc": vector[1],
                    "target_port_0": vector[2],
                    "target_port_1": vector[3],
                    "target_card_0": vector[4],
                    "target_card_1": vector[5],
                    "target_card_2": vector[6],
                    "target_card_3": vector[7],
                    "target_card_4": vector[8],
                    "task_vector": json.dumps(vector),
                    "dataset_root": str(accepted_dataset),
                }
            )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    info = _require_lerobot_root(source)
    inferred = _infer_from_name(args.source_repo_id.rsplit("/", 1)[-1])

    task_family = args.task_family or inferred["task_family"]
    target_card_index = args.target_card_index if args.target_card_index is not None else inferred["target_card_index"]
    target_port_index = args.target_port_index if args.target_port_index is not None else inferred["target_port_index"]
    if task_family is None or target_card_index is None or target_port_index is None:
        raise ValueError(
            "could not infer task_family/target_card_index/target_port_index; pass them explicitly"
        )

    accepted_dataset = output_dir / f"accepted_dataset_{args.mode}"
    accepted_metadata = output_dir / "accepted_metadata"
    if accepted_metadata.exists() and args.overwrite:
        shutil.rmtree(accepted_metadata)
    elif accepted_metadata.exists() and not args.overwrite:
        raise FileExistsError(f"{accepted_metadata} already exists; pass --overwrite to replace it")

    _copy_dataset(source, accepted_dataset, overwrite=args.overwrite)
    rows = _episode_rows(accepted_dataset)
    sidecar_kwargs = {
        "episode_rows": rows,
        "mode": args.mode,
        "source_repo_id": args.source_repo_id,
        "source_url": args.source_url,
        "task_family": task_family,
        "target_card_index": target_card_index,
        "target_port_index": target_port_index,
    }
    _write_sidecars(writer_root=accepted_metadata, **sidecar_kwargs)
    _write_sidecars(writer_root=accepted_dataset, **sidecar_kwargs)
    _write_manifest(
        output_dir=output_dir,
        accepted_dataset=accepted_dataset,
        episode_rows=rows,
        mode=args.mode,
        task_family=task_family,
        target_card_index=target_card_index,
        target_port_index=target_port_index,
    )

    conversion = {
        "schema_version": "aic_external_lerobot_conversion/v1",
        "source": str(source),
        "source_repo_id": args.source_repo_id,
        "source_url": args.source_url,
        "accepted_dataset": str(accepted_dataset),
        "accepted_metadata": str(accepted_metadata),
        "accepted_manifest": str(output_dir / "manifests" / "accepted.csv"),
        "mode": args.mode,
        "task_family": task_family,
        "target_card_index": target_card_index,
        "target_port_index": target_port_index,
        "total_episodes": int(info.get("total_episodes", len(rows))),
        "total_frames": int(info.get("total_frames", 0)),
        "fps": int(info.get("fps", 0)),
        "robot_type": info.get("robot_type"),
        "action_representation": "relative_delta_gripper_tcp",
        "conversion_note": "LeRobot data/video/stat files were copied unchanged; only AIC sidecar metadata was added.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversion_report.json").write_text(
        json.dumps(conversion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (accepted_dataset / "meta" / "aic_dataset_conversion.json").write_text(
        json.dumps(conversion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(conversion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

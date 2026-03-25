#!/usr/bin/env python3
"""Split a phased LeRobot dataset into one dataset per phase label.

Input is a LeRobot dataset root containing phase labels in data parquet files
(for example created by `label_cheatcode_phases.py`).

The script creates one output LeRobot dataset per phase and rebuilds metadata
via LeRobot APIs (`create`, `add_frame`, `save_episode`, `finalize`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


AUTO_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a phased LeRobot dataset into separate per-phase datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to source LeRobot dataset root (contains meta/info.json and data/).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for new datasets. Defaults to the parent of --input. "
            "Each phase dataset is created as <dataset_name>_<phase> under this root."
        ),
    )
    parser.add_argument(
        "--phase-column",
        default="phase",
        help="Column containing phase labels (default: phase).",
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help="Optional explicit list of phases to export. Defaults to all discovered.",
    )
    parser.add_argument(
        "--suffix-template",
        default="{phase}",
        help="Template suffix appended to dataset name. Supports {phase}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output dataset directories if they already exist.",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Decode source videos and re-encode phase-split videos into outputs.",
    )
    return parser.parse_args()


def read_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def resolve_phase_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    phased = sorted(data_dir.rglob("*.phased.parquet"))
    if phased:
        return phased

    generic = sorted(data_dir.rglob("*.parquet"))
    if not generic:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    return generic


def resolve_source_data_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    files = sorted(
        p for p in data_dir.rglob("*.parquet") if not p.name.endswith(".phased.parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"No source data parquet files found under: {data_dir} (excluding *.phased.parquet)"
        )
    return files


def discover_phases(files: list[Path], phase_column: str) -> list[str]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires pandas. Run in the project environment."
        ) from exc

    phases: set[str] = set()
    for file in files:
        df = pd.read_parquet(file, columns=[phase_column])
        if phase_column in df.columns:
            phases.update(df[phase_column].dropna().astype(str).tolist())
    return sorted(phases)


def load_phase_map(files: list[Path], phase_column: str) -> dict[int, str]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires pandas. Run in the project environment."
        ) from exc

    phase_by_index: dict[int, str] = {}
    for file in files:
        df = pd.read_parquet(file)
        if phase_column not in df.columns or "index" not in df.columns:
            continue
        for _, row in df[["index", phase_column]].dropna().iterrows():
            phase_by_index[int(row["index"])] = str(row[phase_column])
    return phase_by_index


def load_task_lookup(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading tasks parquet requires pandas. Run in the project environment."
        ) from exc

    df = pd.read_parquet(tasks_path)
    lookup: dict[int, str] = {}

    if "task" in df.columns and "task_index" in df.columns:
        for _, row in df.iterrows():
            lookup[int(row["task_index"])] = str(row["task"])
    elif "task_index" in df.columns:
        for task_name, row in df.iterrows():
            lookup[int(row["task_index"])] = str(task_name)
    return lookup


def choose_feature_schema(
    info: dict[str, Any], include_videos: bool
) -> tuple[dict[str, Any], list[str], list[str]]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Invalid info.json: missing dict field 'features'")

    kept: dict[str, Any] = {}
    dropped_video: list[str] = []
    dropped_auto: list[str] = []

    for key, spec in features.items():
        if key in AUTO_FEATURE_KEYS:
            dropped_auto.append(key)
            continue
        if (
            not include_videos
            and isinstance(spec, dict)
            and spec.get("dtype") == "video"
        ):
            dropped_video.append(key)
            continue
        if isinstance(spec, dict):
            spec_copy = dict(spec)
            shape = spec_copy.get("shape")
            if isinstance(shape, list):
                spec_copy["shape"] = tuple(shape)
            kept[key] = spec_copy
        else:
            kept[key] = spec

    if not kept:
        raise ValueError("No exportable features remain.")
    return kept, dropped_video, dropped_auto


def normalize_feature_value(value: Any, spec: dict[str, Any]) -> Any:
    import numpy as np

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    expected_dtype = spec.get("dtype")
    if expected_dtype == "video":
        arr = np.asarray(value)
        shape = spec.get("shape")
        if isinstance(shape, tuple) and len(shape) == 3 and arr.ndim == 3:
            h, w, c = shape
            if arr.shape == (c, h, w):
                arr = np.transpose(arr, (1, 2, 0))
        return arr

    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    np_dtype = dtype_map.get(expected_dtype)
    if np_dtype is None:
        return value

    arr = np.asarray(value, dtype=np_dtype)
    shape = spec.get("shape")
    if isinstance(shape, tuple) and arr.shape != shape:
        arr = arr.reshape(shape)
    return arr


def build_clean_source_view(
    dataset_root: Path, source_files: list[Path]
) -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="lerobot_clean_view_")
    tmp_root = Path(tmp.name)

    for dirname in ("meta", "videos", "images"):
        src = dataset_root / dirname
        if src.exists():
            (tmp_root / dirname).symlink_to(src, target_is_directory=True)

    for src_file in source_files:
        rel = src_file.relative_to(dataset_root)
        dst = tmp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src_file)

    return tmp


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else input_root.parent.resolve()
    )

    info = read_info(input_root)
    phase_files = resolve_phase_files(input_root)
    source_data_files = resolve_source_data_files(input_root)
    phase_by_index = load_phase_map(phase_files, args.phase_column)
    discovered = discover_phases(phase_files, args.phase_column)
    phases = args.phases if args.phases else discovered
    phases = [p for p in phases if p in discovered]
    if not phases:
        raise ValueError(
            f"No phases selected. Discovered={discovered}, requested={args.phases}"
        )

    task_lookup = load_task_lookup(input_root)
    first_task_name = next(iter(task_lookup.values()), "task")

    feature_schema, dropped_video, dropped_auto = choose_feature_schema(
        info, include_videos=args.include_videos
    )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Could not import LeRobotDataset. Run in the project environment."
        ) from exc

    dataset_name = input_root.name
    fps = int(info.get("fps", 30))
    robot_type = str(info.get("robot_type", "unknown"))

    print(f"[info] input={input_root}")
    print(f"[info] phase files={len(phase_files)}")
    print(f"[info] source files={len(source_data_files)}")
    print(f"[info] discovered phases={discovered}")
    print(f"[info] include_videos={args.include_videos}")
    if dropped_video:
        print(f"[info] dropping video features={dropped_video}")
    print(f"[info] dropping auto features={dropped_auto}")
    print(f"[info] kept feature keys={list(feature_schema.keys())}")

    writers: dict[str, Any] = {}
    output_paths: dict[str, Path] = {}
    frame_counts: dict[str, int] = {p: 0 for p in phases}
    episode_counts: dict[str, int] = {p: 0 for p in phases}
    current_episode_by_phase: dict[str, int | None] = {p: None for p in phases}

    for phase in phases:
        suffix = args.suffix_template.format(phase=phase)
        repo_id = f"{dataset_name}_{suffix}"
        dataset_root = output_root / repo_id
        if dataset_root.exists():
            if args.overwrite:
                shutil.rmtree(dataset_root)
            else:
                raise FileExistsError(
                    f"Output dataset already exists: {dataset_root}. "
                    "Pass --overwrite to replace it."
                )
        writers[phase] = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=dataset_root,
            robot_type=robot_type,
            features=feature_schema,
            use_videos=args.include_videos,
        )
        output_paths[phase] = dataset_root
        print(f"[create] phase={phase} repo_id={repo_id} root={dataset_root}")

    with build_clean_source_view(input_root, source_data_files) as tmp_dir:
        source_root = Path(tmp_dir)
        source = LeRobotDataset(f"local/{dataset_name}", root=source_root)
        print(f"[info] source frames={len(source)} episodes={source.num_episodes}")

        for item in source:
            idx = item.get("index")
            ep = item.get("episode_index")
            if idx is None or ep is None:
                continue
            if hasattr(idx, "item"):
                idx = idx.item()
            if hasattr(ep, "item"):
                ep = ep.item()
            row_index = int(idx)
            source_episode = int(ep)

            phase = phase_by_index.get(row_index)
            if phase not in phases:
                continue

            current_ep = current_episode_by_phase[phase]
            if current_ep is None:
                current_episode_by_phase[phase] = source_episode
            elif current_ep != source_episode:
                writers[phase].save_episode()
                episode_counts[phase] += 1
                current_episode_by_phase[phase] = source_episode

            frame: dict[str, Any] = {}
            for key, spec in feature_schema.items():
                if key in item:
                    frame[key] = normalize_feature_value(item[key], spec)

            task_name = item.get("task", first_task_name)
            if not isinstance(task_name, str):
                task_name = first_task_name
            frame["task"] = task_name
            writers[phase].add_frame(frame)
            frame_counts[phase] += 1

    for phase in phases:
        if frame_counts[phase] > 0 and current_episode_by_phase[phase] is not None:
            writers[phase].save_episode()
            episode_counts[phase] += 1

    for phase, writer in writers.items():
        writer.finalize()
        print(
            f"[done] phase={phase} episodes={episode_counts[phase]} "
            f"frames={frame_counts[phase]} root={output_paths[phase]}"
        )

    print("[ok] phase split complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        import traceback

        traceback.print_exc()
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

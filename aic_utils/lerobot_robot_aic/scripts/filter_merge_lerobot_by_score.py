#!/usr/bin/env python3
"""Filter and merge LeRobot datasets by per-trial score.

This script takes one or more LeRobot dataset roots plus matching score
summaries (CSV), keeps episodes whose score is above a threshold, and writes a
single merged LeRobot dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUTO_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


@dataclass
class SelectionRow:
    dataset_root: Path
    score_csv: Path
    trial_id: str
    run_index: int
    status: str
    total_score: float
    mapped_episode_index: int | None
    selected: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter LeRobot episodes by score and merge into one dataset."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        type=Path,
        help="Input LeRobot dataset roots (for example outputs/sfp2nic_simtime_3_dataset).",
    )
    parser.add_argument(
        "--score-csvs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Optional score summary CSV paths, one per dataset. If omitted, each is "
            "inferred as sibling '<name_without__dataset>_scores/score_summary.csv'."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        required=True,
        help="Keep only episodes with total_score >= this threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--status-allowlist",
        nargs="+",
        default=["OK"],
        help=(
            "Allowed score status values for selection (case-insensitive). "
            "Default: OK"
        ),
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Include video features in output dataset (default: false).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    return parser.parse_args()


def infer_score_csv(dataset_root: Path) -> Path:
    name = dataset_root.name
    if not name.endswith("_dataset"):
        raise ValueError(
            f"Could not infer score CSV for dataset '{dataset_root}'. "
            "Pass --score-csvs explicitly."
        )
    score_dir = dataset_root.parent / f"{name[:-8]}_scores"
    return score_dir / "score_summary.csv"


def read_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


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


def load_episode_indices(dataset_root: Path) -> list[int]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires pandas. Run in the project environment."
        ) from exc

    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(
            f"No episode parquet files found under: {dataset_root / 'meta' / 'episodes'}"
        )

    vals: list[int] = []
    for file in episode_files:
        df = pd.read_parquet(file, columns=["episode_index"])
        vals.extend(int(v) for v in df["episode_index"].tolist())
    return sorted(set(vals))


def load_episode_meta(dataset_root: Path) -> dict[int, dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading episode parquet requires pandas. Run in the project environment."
        ) from exc

    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(
            f"No episode parquet files found under: {dataset_root / 'meta' / 'episodes'}"
        )

    meta: dict[int, dict[str, Any]] = {}
    for file in episode_files:
        df = pd.read_parquet(file)
        for row in df.to_dict(orient="records"):
            ep = int(row["episode_index"])
            meta[ep] = row
    return meta


def load_task_lookup(dataset_root: Path) -> dict[int, str]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading tasks parquet requires pandas. Run in the project environment."
        ) from exc

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}

    df = pd.read_parquet(tasks_path)
    lookup: dict[int, str] = {}
    if "task" in df.columns and "task_index" in df.columns:
        for _, row in df.iterrows():
            lookup[int(row["task_index"])] = str(row["task"])
        return lookup

    if "task_index" in df.columns:
        for idx, row in df.iterrows():
            lookup[int(row["task_index"])] = str(idx)
    return lookup


def video_path_for_episode(
    dataset_root: Path,
    episode_row: dict[str, Any],
    feature_key: str,
) -> Path:
    chunk_col = f"videos/{feature_key}/chunk_index"
    file_col = f"videos/{feature_key}/file_index"
    if chunk_col not in episode_row or file_col not in episode_row:
        raise ValueError(
            f"Missing video metadata columns for '{feature_key}' in episode row"
        )
    chunk_idx = int(episode_row[chunk_col])
    file_idx = int(episode_row[file_col])
    return (
        dataset_root
        / "videos"
        / feature_key
        / f"chunk-{chunk_idx:03d}"
        / f"file-{file_idx:03d}.mp4"
    )


def decode_video_frames(video_path: Path, expected_len: int) -> list[Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames: list[Any] = []
    decode_error: Exception | None = None

    try:
        import imageio.v3 as iio  # type: ignore

        for frame in iio.imiter(str(video_path)):
            frames.append(frame)
            if len(frames) >= expected_len:
                break
    except Exception as exc:
        decode_error = exc
        frames = []

    if not frames:
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(video_path))
            while len(frames) < expected_len:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()
        except Exception as exc:
            if decode_error is None:
                decode_error = exc

    if not frames:
        raise RuntimeError(
            f"Could not decode frames from video: {video_path}. "
            f"Decoder error: {decode_error}"
        )

    if len(frames) < expected_len:
        last = frames[-1]
        frames.extend([last] * (expected_len - len(frames)))
    elif len(frames) > expected_len:
        frames = frames[:expected_len]
    return frames


def load_selection_for_dataset(
    dataset_root: Path,
    score_csv: Path,
    min_score: float,
    status_allowlist: set[str],
) -> tuple[set[int], list[SelectionRow]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading score CSV requires pandas. Run in the project environment."
        ) from exc

    if not score_csv.exists():
        raise FileNotFoundError(f"Score CSV not found: {score_csv}")

    scores = pd.read_csv(score_csv)
    required = {"trial_id", "status", "total_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{score_csv} is missing required columns: {sorted(missing)}")

    if "run_index" not in scores.columns:
        scores["run_index"] = range(1, len(scores) + 1)
    scores = scores.sort_values("run_index").reset_index(drop=True)

    episode_indices = load_episode_indices(dataset_root)
    non_failed_mask = scores["status"].astype(str).str.upper() != "FAILED"
    non_failed_indices = list(scores.index[non_failed_mask])

    row_to_episode: dict[int, int] = {}
    if len(scores) == len(episode_indices):
        for i, episode in enumerate(episode_indices):
            row_to_episode[i] = episode
        mapping_mode = "all_rows"
    elif len(non_failed_indices) == len(episode_indices):
        for i, episode in enumerate(episode_indices):
            row_to_episode[int(non_failed_indices[i])] = episode
        mapping_mode = "non_failed_rows"
    else:
        raise RuntimeError(
            f"Could not align scores to episodes for dataset={dataset_root} "
            f"(score rows={len(scores)}, non_failed={len(non_failed_indices)}, "
            f"episodes={len(episode_indices)})."
        )

    selected_episodes: set[int] = set()
    report_rows: list[SelectionRow] = []
    for i, row in scores.iterrows():
        status = str(row["status"])
        status_ok = status.upper() in status_allowlist
        total_score = float(row["total_score"])
        trial_id = str(row["trial_id"])
        run_index = int(row["run_index"])
        mapped_episode = row_to_episode.get(i)

        if mapped_episode is None:
            selected = False
            reason = "no_mapped_episode"
        elif not status_ok:
            selected = False
            reason = "status_filtered"
        elif total_score < min_score:
            selected = False
            reason = "below_min_score"
        else:
            selected = True
            reason = "selected"
            selected_episodes.add(mapped_episode)

        report_rows.append(
            SelectionRow(
                dataset_root=dataset_root,
                score_csv=score_csv,
                trial_id=trial_id,
                run_index=run_index,
                status=status,
                total_score=total_score,
                mapped_episode_index=mapped_episode,
                selected=selected,
                reason=reason,
            )
        )

    print(
        f"[map] dataset={dataset_root.name} mode={mapping_mode} "
        f"score_rows={len(scores)} episodes={len(episode_indices)} selected={len(selected_episodes)}"
    )
    return selected_episodes, report_rows


def validate_compatibility(base_info: dict[str, Any], other_info: dict[str, Any]) -> None:
    base_fps = int(base_info.get("fps", 30))
    other_fps = int(other_info.get("fps", 30))
    if base_fps != other_fps:
        raise ValueError(f"Incompatible fps: {base_fps} vs {other_fps}")

    base_robot = str(base_info.get("robot_type", "unknown"))
    other_robot = str(other_info.get("robot_type", "unknown"))
    if base_robot != other_robot:
        raise ValueError(f"Incompatible robot_type: {base_robot} vs {other_robot}")

    base_features = base_info.get("features")
    other_features = other_info.get("features")
    if base_features != other_features:
        raise ValueError("Incompatible feature schema across datasets.")


def write_selection_report(rows: list[SelectionRow], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset_root",
                "score_csv",
                "trial_id",
                "run_index",
                "status",
                "total_score",
                "mapped_episode_index",
                "selected",
                "reason",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    str(row.dataset_root),
                    str(row.score_csv),
                    row.trial_id,
                    row.run_index,
                    row.status,
                    row.total_score,
                    row.mapped_episode_index,
                    row.selected,
                    row.reason,
                ]
            )


def main() -> int:
    args = parse_args()
    datasets = [p.resolve() for p in args.datasets]
    if args.score_csvs:
        score_csvs = [p.resolve() for p in args.score_csvs]
        if len(score_csvs) != len(datasets):
            raise ValueError("--score-csvs must have the same count as --datasets")
    else:
        score_csvs = [infer_score_csv(p).resolve() for p in datasets]

    output_root = args.output.resolve()
    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset exists: {output_root}. Pass --overwrite to replace it."
            )

    infos = [read_info(ds) for ds in datasets]
    for info in infos[1:]:
        validate_compatibility(infos[0], info)

    feature_schema, dropped_video, dropped_auto = choose_feature_schema(
        infos[0], include_videos=args.include_videos
    )
    fps = int(infos[0].get("fps", 30))
    robot_type = str(infos[0].get("robot_type", "unknown"))
    status_allowlist = {s.upper() for s in args.status_allowlist}

    try:
        import pandas as pd  # type: ignore
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Could not import required dependencies. Run in the project environment."
        ) from exc

    selected_by_dataset: dict[Path, set[int]] = {}
    report_rows: list[SelectionRow] = []
    for dataset_root, score_csv in zip(datasets, score_csvs, strict=True):
        selected_episodes, rows = load_selection_for_dataset(
            dataset_root=dataset_root,
            score_csv=score_csv,
            min_score=args.min_score,
            status_allowlist=status_allowlist,
        )
        selected_by_dataset[dataset_root] = selected_episodes
        report_rows.extend(rows)

    repo_id = output_root.name
    writer = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_root,
        robot_type=robot_type,
        features=feature_schema,
        use_videos=args.include_videos,
    )

    total_frames = 0
    total_episodes = 0

    print(f"[create] output={output_root}")
    print(f"[info] include_videos={args.include_videos}")
    if dropped_video:
        print(f"[info] dropped video features={dropped_video}")
    print(f"[info] dropped auto features={dropped_auto}")

    for dataset_root in datasets:
        selected_episodes = selected_by_dataset[dataset_root]
        source_files = resolve_source_data_files(dataset_root)
        task_lookup = load_task_lookup(dataset_root)
        episode_meta = load_episode_meta(dataset_root) if args.include_videos else {}
        video_feature_keys = [
            key
            for key, spec in feature_schema.items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"
        ]
        default_task = next(iter(task_lookup.values()), "task")
        print(
            f"[merge] dataset={dataset_root.name} selected_episodes={len(selected_episodes)}"
        )
        if not selected_episodes:
            continue

        current_source_episode: int | None = None
        current_selected = False
        frames_in_current = 0
        frame_idx_in_episode = 0
        current_video_frames: dict[str, list[Any]] = {}

        for source_file in source_files:
            df = pd.read_parquet(source_file)
            if "episode_index" not in df.columns:
                raise ValueError(f"Missing episode_index in {source_file}")
            if "task_index" not in df.columns:
                raise ValueError(f"Missing task_index in {source_file}")

            if "index" in df.columns:
                df = df.sort_values("index")

            for row in df.to_dict(orient="records"):
                source_episode = int(row["episode_index"])

                if current_source_episode is None:
                    current_source_episode = source_episode
                    current_selected = source_episode in selected_episodes
                    frame_idx_in_episode = 0
                    current_video_frames = {}
                    if current_selected and args.include_videos:
                        episode_row = episode_meta.get(source_episode)
                        if episode_row is None:
                            raise RuntimeError(
                                f"Missing meta row for episode {source_episode} in {dataset_root}"
                            )
                        expected_len = int(episode_row.get("length", 0))
                        for key in video_feature_keys:
                            vpath = video_path_for_episode(
                                dataset_root=dataset_root,
                                episode_row=episode_row,
                                feature_key=key,
                            )
                            current_video_frames[key] = decode_video_frames(
                                vpath, expected_len
                            )
                elif source_episode != current_source_episode:
                    if current_selected and frames_in_current > 0:
                        writer.save_episode()
                        total_episodes += 1
                    current_source_episode = source_episode
                    current_selected = source_episode in selected_episodes
                    frames_in_current = 0
                    frame_idx_in_episode = 0
                    current_video_frames = {}
                    if current_selected and args.include_videos:
                        episode_row = episode_meta.get(source_episode)
                        if episode_row is None:
                            raise RuntimeError(
                                f"Missing meta row for episode {source_episode} in {dataset_root}"
                            )
                        expected_len = int(episode_row.get("length", 0))
                        for key in video_feature_keys:
                            vpath = video_path_for_episode(
                                dataset_root=dataset_root,
                                episode_row=episode_row,
                                feature_key=key,
                            )
                            current_video_frames[key] = decode_video_frames(
                                vpath, expected_len
                            )

                if not current_selected:
                    continue

                frame: dict[str, Any] = {}
                for key, spec in feature_schema.items():
                    if (
                        isinstance(spec, dict)
                        and spec.get("dtype") == "video"
                        and args.include_videos
                    ):
                        if key not in current_video_frames:
                            raise RuntimeError(
                                f"No decoded frames for feature '{key}' in episode {source_episode}"
                            )
                        if frame_idx_in_episode >= len(current_video_frames[key]):
                            raise RuntimeError(
                                f"Video frame index out of range for '{key}' in episode {source_episode}"
                            )
                        frame[key] = normalize_feature_value(
                            current_video_frames[key][frame_idx_in_episode], spec
                        )
                    elif key in row:
                        frame[key] = normalize_feature_value(row[key], spec)

                task_idx = int(row["task_index"])
                frame["task"] = task_lookup.get(task_idx, default_task)

                writer.add_frame(frame)
                total_frames += 1
                frames_in_current += 1
                frame_idx_in_episode += 1

        if current_selected and frames_in_current > 0:
            writer.save_episode()
            total_episodes += 1

    if total_episodes == 0:
        raise RuntimeError(
            "No episodes selected. Check --min-score and --status-allowlist."
        )

    writer.finalize()
    report_path = output_root / "selection_report.csv"
    write_selection_report(report_rows, report_path)

    print(f"[done] episodes={total_episodes} frames={total_frames}")
    print(f"[done] output={output_root}")
    print(f"[done] report={report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        import traceback

        traceback.print_exc()
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Add CheatCode phase labels to recorded trajectory rows.

This script post-processes tabular rollout exports and annotates each row with a
phase label based on CheatCode timing:
- alignment: first 5.0 seconds (default)
- descent: everything after alignment

Supported formats:
- .jsonl (one JSON object per line)
- .csv
- .parquet (optional, requires pandas)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


EPISODE_CANDIDATES = ["episode_index", "episode_id", "episode", "episode_idx"]
TIMESTAMP_CANDIDATES = ["timestamp", "time", "ts", "t"]
STEP_CANDIDATES = ["frame_index", "step", "timestep", "action_index", "index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label rows as alignment/descent using CheatCode timing. "
            "If timestamps are available they are preferred; otherwise row index is used."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input file (.jsonl/.csv/.parquet) or LeRobot dataset directory "
            "(containing meta/info.json and data/)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to '<input_stem>.phased<suffix>'",
    )
    parser.add_argument(
        "--label-column",
        default="phase",
        help="Column name for phase labels (default: phase)",
    )
    parser.add_argument(
        "--alignment-duration-sec",
        type=float,
        default=5.5,
        help="CheatCode alignment duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Dataset frame rate used when timestamp is unavailable. "
            "If omitted, inferred from <dataset>/meta/info.json when possible, "
            "else defaults to 30.0."
        ),
    )
    parser.add_argument(
        "--sample-period-sec",
        type=float,
        default=None,
        help=(
            "Optional explicit sample period for step-based labeling. "
            "If omitted, uses 1.0 / --fps."
        ),
    )
    parser.add_argument(
        "--timestamp-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to timestamp values before differencing. "
            "Examples: 1.0 for seconds, 1e-3 for milliseconds, 1e-9 for nanoseconds."
        ),
    )
    parser.add_argument(
        "--episode-column",
        default=None,
        help="Override episode column name (default: auto-detect)",
    )
    parser.add_argument(
        "--timestamp-column",
        default=None,
        help="Override timestamp column name (default: auto-detect)",
    )
    parser.add_argument(
        "--step-column",
        default=None,
        help="Override step/index column name (default: auto-detect)",
    )
    return parser.parse_args()


def choose_column(candidates: list[str], keys: set[str]) -> str | None:
    lower_to_original = {k.lower(): k for k in keys}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Each JSONL line must be an object")
                rows.append(obj)
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet requires pandas in the current environment. "
                "Run this script from the project environment (for example: "
                "`pixi run python3 aic_utils/lerobot_robot_aic/scripts/label_cheatcode_phases.py ...`)."
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {suffix}")


def resolve_input_files(input_path: Path) -> tuple[list[Path], Path | None]:
    """Resolve one or more input files and optional LeRobot metadata path."""
    if input_path.is_file():
        return [input_path], input_path.parent / "meta" / "info.json"

    if not input_path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    info_path = input_path / "meta" / "info.json"
    if info_path.exists():
        data_dir = input_path / "data"
        if data_dir.exists():
            parquet_files = sorted(data_dir.rglob("*.parquet"))
            if parquet_files:
                return parquet_files, info_path

    generic_files = sorted(
        [
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jsonl", ".csv", ".parquet"}
        ]
    )
    if not generic_files:
        raise ValueError(f"No supported data files found under directory: {input_path}")
    return generic_files, info_path if info_path.exists() else None


def infer_fps(info_path: Path | None) -> float | None:
    if info_path is None or not info_path.exists():
        return None
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    fps = payload.get("fps")
    try:
        if fps is None:
            return None
        fps_f = float(fps)
        return fps_f if fps_f > 0.0 else None
    except (TypeError, ValueError):
        return None


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")))
                f.write("\n")
        return

    if suffix == ".csv":
        fieldnames: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Writing parquet requires pandas in the current environment. "
                "Run this script from the project environment (for example: "
                "`pixi run python3 aic_utils/lerobot_robot_aic/scripts/label_cheatcode_phases.py ...`)."
            ) from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {suffix}")


def annotate_rows(
    rows: list[dict[str, Any]],
    label_column: str,
    alignment_duration_sec: float,
    sample_period_sec: float,
    timestamp_scale: float,
    episode_column: str | None,
    timestamp_column: str | None,
    step_column: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {
            "episodes": 0,
            "rows": 0,
            "episode_column": episode_column,
            "timestamp_column": timestamp_column,
            "step_column": step_column,
        }

    keys = set()
    for row in rows:
        keys.update(row.keys())

    ep_col = episode_column or choose_column(EPISODE_CANDIDATES, keys)
    ts_col = timestamp_column or choose_column(TIMESTAMP_CANDIDATES, keys)
    st_col = step_column or choose_column(STEP_CANDIDATES, keys)

    grouped: "OrderedDict[Any, list[dict[str, Any]]]" = OrderedDict()
    if ep_col is None:
        grouped[0] = rows
    else:
        for row in rows:
            grouped.setdefault(row.get(ep_col), []).append(row)

    alignment_count = 0
    descent_count = 0

    for _, episode_rows in grouped.items():
        t0 = None
        if ts_col is not None and episode_rows:
            t0_raw = to_float(episode_rows[0].get(ts_col))
            if t0_raw is not None:
                t0 = t0_raw * timestamp_scale

        step0 = None
        if st_col is not None and episode_rows:
            step0 = to_float(episode_rows[0].get(st_col))

        for i, row in enumerate(episode_rows):
            elapsed_sec = None

            if ts_col is not None and t0 is not None:
                current_raw = to_float(row.get(ts_col))
                if current_raw is not None:
                    elapsed_sec = current_raw * timestamp_scale - t0

            if elapsed_sec is None:
                if st_col is not None and step0 is not None:
                    step_value = to_float(row.get(st_col))
                    if step_value is not None:
                        elapsed_sec = (step_value - step0) * sample_period_sec

            if elapsed_sec is None:
                elapsed_sec = i * sample_period_sec

            if elapsed_sec < alignment_duration_sec:
                row[label_column] = "alignment"
                alignment_count += 1
            else:
                row[label_column] = "descent"
                descent_count += 1

    summary = {
        "episodes": len(grouped),
        "rows": len(rows),
        "alignment_rows": alignment_count,
        "descent_rows": descent_count,
        "episode_column": ep_col,
        "timestamp_column": ts_col,
        "step_column": st_col,
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input path not found: {in_path}", file=sys.stderr)
        return 2

    try:
        input_files, info_path = resolve_input_files(in_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    fps = args.fps
    if fps is None:
        fps = infer_fps(info_path)
    if fps is None:
        fps = 30.0

    sample_period_sec = (
        args.sample_period_sec if args.sample_period_sec is not None else (1.0 / fps)
    )
    if sample_period_sec <= 0.0:
        print("sample period must be > 0", file=sys.stderr)
        return 2

    total_episodes = 0
    total_rows = 0
    total_alignment = 0
    total_descent = 0

    for i, input_file in enumerate(input_files):
        if args.output:
            if len(input_files) > 1:
                print(
                    "--output is only supported when --input resolves to a single file",
                    file=sys.stderr,
                )
                return 2
            out_path = Path(args.output)
        else:
            out_path = input_file.with_name(
                f"{input_file.stem}.phased{input_file.suffix}"
            )

        rows = read_rows(input_file)
        rows, summary = annotate_rows(
            rows=rows,
            label_column=args.label_column,
            alignment_duration_sec=args.alignment_duration_sec,
            sample_period_sec=sample_period_sec,
            timestamp_scale=args.timestamp_scale,
            episode_column=args.episode_column,
            timestamp_column=args.timestamp_column,
            step_column=args.step_column,
        )
        write_rows(out_path, rows)

        total_episodes += int(summary["episodes"])
        total_rows += int(summary["rows"])
        total_alignment += int(summary.get("alignment_rows", 0))
        total_descent += int(summary.get("descent_rows", 0))

        print(f"[{i + 1}/{len(input_files)}] Wrote labeled file: {out_path}")
        print(
            "Columns used: "
            f"episode={summary['episode_column']} "
            f"timestamp={summary['timestamp_column']} "
            f"step={summary['step_column']}"
        )

    print(
        "Summary: "
        f"files={len(input_files)} episodes={total_episodes} rows={total_rows} "
        f"alignment={total_alignment} descent={total_descent}"
    )
    print(
        "Timing used: "
        f"alignment_duration_sec={args.alignment_duration_sec} "
        f"sample_period_sec={sample_period_sec:.6f} "
        f"fps={fps:.3f}"
    )
    if info_path is not None and info_path.exists():
        print(f"Metadata source: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize_ns(name: str, values_ns: list[int]) -> None:
    values_ms = [float(value) / 1_000_000.0 for value in values_ns]
    if not values_ms:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: count={len(values_ms)} "
        f"mean={statistics.fmean(values_ms):.3f}ms "
        f"p50={percentile(values_ms, 50):.3f}ms "
        f"p95={percentile(values_ms, 95):.3f}ms "
        f"p99={percentile(values_ms, 99):.3f}ms "
        f"max={max(values_ms):.3f}ms"
    )


def summarize_float(name: str, values: list[float], unit: str = "") -> None:
    if not values:
        return
    suffix = unit if unit else ""
    print(
        f"{name}: count={len(values)} "
        f"mean={statistics.fmean(values):.6f}{suffix} "
        f"p50={percentile(values, 50):.6f}{suffix} "
        f"p95={percentile(values, 95):.6f}{suffix} "
        f"p99={percentile(values, 99):.6f}{suffix} "
        f"max={max(values):.6f}{suffix}"
    )


def summarize_axis(name: str, values: list[float], dt_sec: list[float]) -> None:
    if not values:
        return
    abs_values = [abs(value) for value in values]
    nonzero = sum(1 for value in abs_values if value > 1e-9)
    signed_integral = sum(value * dt for value, dt in zip(values, dt_sec))
    abs_integral = sum(abs(value) * dt for value, dt in zip(values, dt_sec))
    print(
        f"{name}: nonzero_ratio={nonzero / len(values):.3f} "
        f"mean={statistics.fmean(values):.6f} "
        f"abs_p95={percentile(abs_values, 95):.6f} "
        f"abs_max={max(abs_values):.6f} "
        f"signed_integral={signed_integral:.6f} "
        f"abs_integral={abs_integral:.6f}"
    )


def value(row: dict[str, str], *names: str) -> float:
    for name in names:
        raw = row.get(name)
        if raw not in ("", None):
            return float(raw)
    return 0.0


def interval_ns(rows: list[dict[str, str]]) -> list[int]:
    for name in (
        "wall_time_ns",
        "loop_start_ns",
        "read_start_ns",
        "send_action_start_ns",
    ):
        stamps = [int(row[name]) for row in rows if row.get(name) not in ("", None)]
        if len(stamps) > 1:
            return [b - a for a, b in zip(stamps, stamps[1:]) if b >= a]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TeleopOnly latency CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    columns: dict[str, list[int]] = {}
    nonzero_count = 0
    rows: list[dict[str, str]] = []
    linear_magnitudes: list[float] = []
    angular_magnitudes: list[float] = []
    all_magnitudes: list[float] = []
    axis_values: dict[str, list[float]] = {
        "linear_x": [],
        "linear_y": [],
        "linear_z": [],
        "angular_x": [],
        "angular_y": [],
        "angular_z": [],
    }

    with args.csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
            for name in (
                "get_action_duration_ns",
                "read_duration_ns",
                "action_duration_ns",
                "publish_duration_ns",
                "send_action_duration_ns",
                "loop_duration_ns",
            ):
                if name in row and row[name] not in ("", None):
                    columns.setdefault(name, []).append(int(row[name]))
            if "nonzero_action" in row:
                nonzero_count += int(row["nonzero_action"])

            linear_x = value(row, "linear_x", "linear.x")
            linear_y = value(row, "linear_y", "linear.y")
            linear_z = value(row, "linear_z", "linear.z")
            angular_x = value(row, "angular_x", "angular.x")
            angular_y = value(row, "angular_y", "angular.y")
            angular_z = value(row, "angular_z", "angular.z")
            axis_values["linear_x"].append(linear_x)
            axis_values["linear_y"].append(linear_y)
            axis_values["linear_z"].append(linear_z)
            axis_values["angular_x"].append(angular_x)
            axis_values["angular_y"].append(angular_y)
            axis_values["angular_z"].append(angular_z)
            linear_mag = math.sqrt(linear_x**2 + linear_y**2 + linear_z**2)
            angular_mag = math.sqrt(angular_x**2 + angular_y**2 + angular_z**2)
            all_mag = math.sqrt(linear_mag**2 + angular_mag**2)
            linear_magnitudes.append(linear_mag)
            angular_magnitudes.append(angular_mag)
            all_magnitudes.append(all_mag)

    total_count = len(rows)
    print(f"samples: total={total_count} nonzero_action={nonzero_count}")
    if total_count:
        print(f"nonzero_ratio: {nonzero_count / total_count:.3f}")

    intervals = interval_ns(rows)
    dt_sec = [0.0] * total_count
    if intervals:
        summarize_ns("sample_interval", intervals)
        duration_sec = sum(intervals) / 1_000_000_000.0
        if duration_sec > 0.0:
            print(f"estimated_rate_hz: {(len(intervals) / duration_sec):.3f}")
        interval_sec = [interval / 1_000_000_000.0 for interval in intervals]
        fill_dt = statistics.median(interval_sec)
        dt_sec = interval_sec + [fill_dt]

    if not columns:
        print("No known duration columns found.")
    else:
        for name, values in columns.items():
            summarize_ns(name.removesuffix("_duration_ns"), values)

    summarize_float("linear_command_magnitude", linear_magnitudes)
    summarize_float("angular_command_magnitude", angular_magnitudes)
    summarize_float("combined_command_magnitude", all_magnitudes)
    for name, values in axis_values.items():
        summarize_axis(name, values, dt_sec)


if __name__ == "__main__":
    main()

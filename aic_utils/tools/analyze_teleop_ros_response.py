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


def value(row: dict[str, str], name: str) -> float | None:
    raw = row.get(name)
    if raw in ("", None):
        return None
    return float(raw)


def summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: count={len(values)} "
        f"mean={statistics.fmean(values):.6f} "
        f"p50={percentile(values, 50):.6f} "
        f"p95={percentile(values, 95):.6f} "
        f"p99={percentile(values, 99):.6f} "
        f"max={max(values):.6f}"
    )


def summarize_signed_axis(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    abs_values = [abs(item) for item in values]
    print(
        f"{name}: mean={statistics.fmean(values):.6f} "
        f"abs_p95={percentile(abs_values, 95):.6f} "
        f"abs_max={max(abs_values):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze teleop ROS response CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    command_rows = 0
    state_rows = 0
    age_ms: list[float] = []
    linear_speed: list[float] = []
    angular_speed: list[float] = []
    elbow_abs: list[float] = []
    shoulder_elbow_wrist1_sum: list[float] = []
    response_ratios: dict[str, list[float]] = {
        "linear_x": [],
        "linear_y": [],
        "linear_z": [],
        "angular_x": [],
        "angular_y": [],
        "angular_z": [],
    }
    tcp_errors: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "z": [],
        "rx": [],
        "ry": [],
        "rz": [],
    }

    with args.csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("event") == "command":
                command_rows += 1
                continue
            if row.get("event") != "controller_state":
                continue
            state_rows += 1

            age = value(row, "age_since_last_command_ns")
            if age is not None and age >= 0:
                age_ms.append(age / 1_000_000.0)

            lin = [
                value(row, "state_tcp_linear_x") or 0.0,
                value(row, "state_tcp_linear_y") or 0.0,
                value(row, "state_tcp_linear_z") or 0.0,
            ]
            ang = [
                value(row, "state_tcp_angular_x") or 0.0,
                value(row, "state_tcp_angular_y") or 0.0,
                value(row, "state_tcp_angular_z") or 0.0,
            ]
            linear_speed.append(math.sqrt(sum(item * item for item in lin)))
            angular_speed.append(math.sqrt(sum(item * item for item in ang)))

            elbow = value(row, "elbow_abs_rad")
            if elbow is not None:
                elbow_abs.append(elbow)
            shoulder_sum = value(row, "shoulder_elbow_wrist1_sum")
            if shoulder_sum is not None:
                shoulder_elbow_wrist1_sum.append(shoulder_sum)

            for axis in response_ratios:
                ratio = value(row, f"cmd_to_state_{axis}_ratio")
                if ratio is not None:
                    response_ratios[axis].append(ratio)

            for axis in tcp_errors:
                error = value(row, f"tcp_error_{axis}")
                if error is not None:
                    tcp_errors[axis].append(error)

    print(f"rows: command={command_rows} controller_state={state_rows}")
    summarize("state_age_since_last_command_ms", age_ms)
    summarize("actual_tcp_linear_speed", linear_speed)
    summarize("actual_tcp_angular_speed", angular_speed)
    summarize("elbow_abs_rad", elbow_abs)
    summarize_signed_axis(
        "shoulder_lift_plus_elbow_plus_wrist1", shoulder_elbow_wrist1_sum
    )
    for axis, ratios in response_ratios.items():
        summarize_signed_axis(f"cmd_to_state_{axis}_ratio", ratios)
    for axis, errors in tcp_errors.items():
        summarize_signed_axis(f"tcp_error_{axis}", errors)


if __name__ == "__main__":
    main()

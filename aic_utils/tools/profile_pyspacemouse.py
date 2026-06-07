#!/usr/bin/env python3
"""Profile local SpaceMouse axis separation and read-loop latency.

Expected usage from the repository root:
    pixi run python aic_utils/tools/profile_pyspacemouse.py
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Iterable


AXES = ("x", "y", "z", "pitch", "roll", "yaw")
SEPARATION_HEADERS = (
    "Timestamp",
    "X",
    "Y",
    "Z",
    "Pitch",
    "Roll",
    "Yaw",
    "Buttons",
)
LATENCY_HEADERS = (
    "Loop_Start_Time",
    "Hardware_Timestamp",
    "Read_Latency",
    "Loop_Time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record SpaceMouse axis separation and latency profile CSVs."
    )
    parser.add_argument(
        "--output-dir",
        default="./profiles",
        help="Directory where generated CSV files are written.",
    )
    parser.add_argument(
        "--mode",
        choices=("separation", "latency", "both"),
        default="latency",
        help="Profiler mode to run.",
    )
    parser.add_argument(
        "--stage-duration",
        type=float,
        default=6.0,
        help="Seconds to record each axis after alignment.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.15,
        help="Minimum absolute SpaceMouse axis value used for dominant-axis detection.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=AXES,
        default=list(AXES),
        help="Axes to profile, in order.",
    )
    return parser.parse_args()


def get_csv_writer(output_dir: str, file_prefix: str, headers: Iterable[str]):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{file_prefix}.csv")
    file_handle = open(filename, mode="w", newline="")
    writer = csv.writer(file_handle)
    writer.writerow(headers)
    return file_handle, writer


def get_dominant_axis(state, signal_threshold: float) -> tuple[str | None, float]:
    axes = {
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "pitch": state.pitch,
        "roll": state.roll,
        "yaw": state.yaw,
    }

    dominant_axis = max(axes, key=lambda axis: abs(axes[axis]))
    max_value = axes[dominant_axis]

    if abs(max_value) < signal_threshold:
        return None, 0.0

    return dominant_axis, max_value


def wait_for_correct_axis(device, target_axis: str, signal_threshold: float) -> None:
    print(
        f"\n[WAITING] Push or twist the cap in the {target_axis.upper()} "
        "direction to confirm alignment..."
    )

    last_warning_time = 0.0

    while True:
        state = device.read()
        if state:
            dominant_axis, _ = get_dominant_axis(state, signal_threshold)

            if dominant_axis is not None:
                if dominant_axis == target_axis:
                    print("Correct axis detected. Hold on; recording starts in 1.5s.")
                    time.sleep(1.5)
                    return

                current_time = time.time()
                if current_time - last_warning_time > 1.5:
                    print(
                        f"WARNING: detected {dominant_axis.upper()}, but expected "
                        f"{target_axis.upper()}. Re-orient your hand."
                    )
                    last_warning_time = current_time

        time.sleep(0.01)


def run_separation_profile(
    device,
    output_dir: str,
    axes: Iterable[str],
    stage_duration: float,
    signal_threshold: float,
) -> None:
    for axis in axes:
        print("\n" + "=" * 60)
        print(f"STAGE: Separation Profiling [{axis.upper()}] Axis")
        print("=" * 60)

        wait_for_correct_axis(device, axis, signal_threshold)

        print(
            f"\nRECORDING SEPARATION [{axis.upper()}]. "
            "Move fully into both positive and negative directions..."
        )

        file_handle, writer = get_csv_writer(
            output_dir, f"profile_{axis}", SEPARATION_HEADERS
        )
        start_time = time.time()

        try:
            while time.time() - start_time < stage_duration:
                state = device.read()

                if state:
                    current_time = time.time()
                    dominant_axis, dominant_value = get_dominant_axis(
                        state, signal_threshold
                    )

                    row_data = dict.fromkeys(AXES, 0.0)
                    if dominant_axis:
                        row_data[dominant_axis] = dominant_value

                    writer.writerow(
                        [
                            current_time,
                            row_data["x"],
                            row_data["y"],
                            row_data["z"],
                            row_data["pitch"],
                            row_data["roll"],
                            row_data["yaw"],
                            state.buttons,
                        ]
                    )

                time.sleep(0.01)
        finally:
            file_handle.close()

        print(f"Finished recording profile_{axis}.csv")


def run_latency_profile(
    device,
    output_dir: str,
    axes: Iterable[str],
    stage_duration: float,
    signal_threshold: float,
) -> None:
    for axis in axes:
        print("\n" + "=" * 60)
        print(f"STAGE: Latency Profiling [{axis.upper()}] Axis")
        print("=" * 60)

        wait_for_correct_axis(device, axis, signal_threshold)

        print(f"\nRECORDING LATENCY [{axis.upper()}]. Keep moving it continuously...")

        file_handle, writer = get_csv_writer(
            output_dir, f"latency_profile_{axis}", LATENCY_HEADERS
        )
        start_time = time.time()
        last_loop_time = time.time()

        try:
            while time.time() - start_time < stage_duration:
                loop_start = time.time()
                loop_duration = loop_start - last_loop_time
                last_loop_time = loop_start

                state = device.read()

                if state:
                    hardware_time = getattr(state, "timestamp", None)

                    if hardware_time is None:
                        hardware_time = "N/A"
                        read_latency = "N/A"
                    elif hardware_time > loop_start:
                        read_latency = 0.0
                    else:
                        read_latency = loop_start - hardware_time

                    writer.writerow(
                        [loop_start, hardware_time, read_latency, loop_duration]
                    )

                time.sleep(0.001)
        finally:
            file_handle.close()

        print(f"Finished recording latency_profile_{axis}.csv")


def main() -> None:
    args = parse_args()

    try:
        import pyspacemouse
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyspacemouse. Run with "
            "`pixi run python aic_utils/tools/profile_pyspacemouse.py ...` "
            "from the repository root."
        ) from exc

    print("=== SpaceMouse Profiler Suite ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Axes: {', '.join(args.axes)}")

    with pyspacemouse.open() as device:
        if args.mode in ("separation", "both"):
            run_separation_profile(
                device,
                args.output_dir,
                args.axes,
                args.stage_duration,
                args.signal_threshold,
            )

        if args.mode in ("latency", "both"):
            run_latency_profile(
                device,
                args.output_dir,
                args.axes,
                args.stage_duration,
                args.signal_threshold,
            )

    print(f"\nProfiling session completed. Output directory: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess terminated by user.")

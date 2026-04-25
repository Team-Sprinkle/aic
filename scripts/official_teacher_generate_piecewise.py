#!/usr/bin/env python3
"""Generate a first official teacher PiecewiseTrajectory JSON artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))

from aic_teacher_official.generate_piecewise import (
    PiecewiseGeneratorConfig,
    generate_piecewise_file,
)
from aic_teacher_official.context import (
    OfficialTeacherContext,
    capture_context_from_ros,
)
from aic_teacher_official.vlm_planner import call_gpt5_mini_delta_planner


def _vector(text: str, *, length: int, name: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != length:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {length} comma-separated numbers"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Piecewise trajectory JSON.")
    parser.add_argument("--task-name", default="Insert cable into target port")
    parser.add_argument(
        "--start-position",
        default="-0.35,0.35,0.32",
        help="Initial TCP position x,y,z in base_link. TODO: auto-read from TF.",
    )
    parser.add_argument(
        "--port-position",
        default="-0.10,0.45,0.12",
        help="Target port position x,y,z in base_link. TODO: auto-read from task TF.",
    )
    parser.add_argument(
        "--orientation-xyzw",
        default="1,0,0,0",
        help="TCP orientation x,y,z,w in base_link.",
    )
    parser.add_argument(
        "--approach-offset",
        default="-0.08,-0.08,0.22",
        help="Placeholder VLM approach offset from port x,y,z.",
    )
    parser.add_argument("--alignment-height", type=float, default=0.16)
    parser.add_argument("--pre-insertion-height", type=float, default=0.03)
    parser.add_argument("--insertion-depth", type=float, default=-0.015)
    parser.add_argument("--approach-duration", type=float, default=2.0)
    parser.add_argument("--alignment-duration", type=float, default=2.0)
    parser.add_argument("--pre-insertion-duration", type=float, default=1.0)
    parser.add_argument(
        "--insertion-duration",
        type=float,
        default=12.0,
        help="Slow final insertion duration. 12s matched the high-scoring CheatCode reference.",
    )
    parser.add_argument("--context-json", help="Optional saved official context JSON.")
    parser.add_argument(
        "--auto-context",
        action="store_true",
        help="Capture current TCP and port TF from a running official ROS/Gazebo sim.",
    )
    parser.add_argument("--target-module-name", default="", help="Task target module for --auto-context.")
    parser.add_argument("--port-name", default="", help="Task target port for --auto-context.")
    parser.add_argument("--use-vlm", action="store_true", help="Call GPT-5 mini for approach/alignment delta waypoints.")
    parser.add_argument("--vlm-model", default="gpt-5-mini")
    parser.add_argument("--max-vlm-calls", type=int, default=20)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path for VLM planning. Repeatable.",
    )
    args = parser.parse_args()

    context = None
    if args.context_json:
        context = OfficialTeacherContext.load_json(args.context_json)
    elif args.auto_context:
        if not args.target_module_name or not args.port_name:
            raise SystemExit("--auto-context requires --target-module-name and --port-name")
        context = capture_context_from_ros(
            target_module_name=args.target_module_name,
            port_name=args.port_name,
        )

    vlm_delta_plan = None
    if args.use_vlm:
        vlm_context = context or OfficialTeacherContext(
            start_position=_vector(args.start_position, length=3, name="--start-position"),
            port_position=_vector(args.port_position, length=3, name="--port-position"),
            orientation_xyzw=_vector(args.orientation_xyzw, length=4, name="--orientation-xyzw"),
            diagnostics={"source": "explicit_cli"},
        )
        vlm_delta_plan = call_gpt5_mini_delta_planner(
            vlm_context,
            image_paths=[Path(path) for path in args.image],
            max_calls=args.max_vlm_calls,
            model=args.vlm_model,
        )

    config = PiecewiseGeneratorConfig(
        start_position=_vector(args.start_position, length=3, name="--start-position"),
        port_position=_vector(args.port_position, length=3, name="--port-position"),
        orientation_xyzw=_vector(args.orientation_xyzw, length=4, name="--orientation-xyzw"),
        approach_offset=_vector(args.approach_offset, length=3, name="--approach-offset"),
        alignment_height=args.alignment_height,
        pre_insertion_height=args.pre_insertion_height,
        insertion_depth=args.insertion_depth,
        approach_duration=args.approach_duration,
        alignment_duration=args.alignment_duration,
        pre_insertion_duration=args.pre_insertion_duration,
        insertion_duration=args.insertion_duration,
        task_name=args.task_name,
        context=context,
        vlm_delta_plan=vlm_delta_plan,
    )
    trajectory = generate_piecewise_file(config, args.output)
    print(f"Wrote {len(trajectory.waypoints)} piecewise waypoints to {args.output}")


if __name__ == "__main__":
    main()

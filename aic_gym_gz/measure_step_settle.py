"""Measure how well one env.step() reaches dense trajectory targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.teacher.types import TeacherPlan, TeacherWaypoint
from aic_gym_gz.trajectory.smoothing import MinimumJerkSmoother
from aic_gym_gz.utils import to_jsonable


def _make_plan(observation: dict[str, np.ndarray]) -> TeacherPlan:
    plug = np.asarray(observation["plug_pose"][:3], dtype=np.float64)
    entrance = np.asarray(observation["target_port_entrance_pose"][:3], dtype=np.float64)
    delta = entrance - plug
    distance = float(np.linalg.norm(delta))
    if distance > 1e-9:
        delta = delta / distance
    target = plug + delta * min(0.03, max(distance * 0.35, 0.01))
    return TeacherPlan(
        next_phase="pre_insert_align",
        waypoints=(TeacherWaypoint(position_xyz=tuple(float(v) for v in target.tolist()), yaw=0.0, speed_scale=1.0),),
        motion_mode="fine_cartesian",
        caution_flag=False,
        should_probe=False,
        segment_horizon_steps=1,
        segment_granularity="fine",
        rationale_summary="Single-step settle probe toward the entrance.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--attach-to-existing", action="store_true")
    parser.add_argument("--transport-backend", choices=("transport", "cli"), default="transport")
    parser.add_argument("--live-timeout", type=float, default=20.0)
    parser.add_argument("--attach-ready-timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ticks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--output", default="aic_gym_gz/artifacts/context_audit/step_settle_measurement.json")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    recommended_ticks: int | None = None
    for ticks_per_step in args.ticks:
        env = (
            make_live_env(
                include_images=False,
                enable_randomization=False,
                ticks_per_step=ticks_per_step,
                attach_to_existing=args.attach_to_existing,
                transport_backend=args.transport_backend,
                timeout=args.live_timeout,
                attach_ready_timeout=args.attach_ready_timeout,
            )
            if args.live
            else make_default_env(
                include_images=False,
                enable_randomization=False,
                ticks_per_step=ticks_per_step,
            )
        )
        try:
            observation, _ = env.reset(seed=args.seed)
            assert env._state is not None
            plan = _make_plan(observation)
            smoother = MinimumJerkSmoother(base_dt=float(ticks_per_step) * 0.002)
            segment = smoother.smooth(state=env._state, plan=plan)
            point_errors: list[float] = []
            for point in segment.points:
                observation, _, terminated, truncated, _ = env.step(np.asarray(point.action, dtype=np.float32))
                tcp = np.asarray(observation["tcp_pose"][:3], dtype=np.float64)
                target = np.asarray(point.target_tcp_pose[:3], dtype=np.float64)
                point_errors.append(float(np.linalg.norm(tcp - target)))
                if terminated or truncated:
                    break
            if not point_errors:
                continue
            row = {
                "ticks_per_step": int(ticks_per_step),
                "step_dt_s": float(ticks_per_step) * 0.002,
                "dense_point_count": len(segment.points),
                "mean_target_error_m": float(np.mean(point_errors)),
                "max_target_error_m": float(np.max(point_errors)),
                "final_target_error_m": float(point_errors[-1]),
            }
            rows.append(row)
            if recommended_ticks is None and row["final_target_error_m"] <= 0.01:
                recommended_ticks = int(ticks_per_step)
        finally:
            env.close()

    payload = {
        "artifact_type": "step_settle_measurement",
        "results": rows,
        "recommended_ticks_per_step": recommended_ticks,
        "selection_rule": "smallest ticks_per_step with final_target_error_m <= 0.01",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

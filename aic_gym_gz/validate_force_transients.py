"""Validate transient within-step force/contact observability under coarse stepping."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .env import AicInsertionEnv
from .io import MockGazeboIO
from .randomizer import AicEnvRandomizer
from .runtime import AicGazeboRuntime, MockStepperBackend, MockTransientContactConfig
from .task import AicInsertionTask


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _make_validation_env(
    *,
    ticks_per_step: int,
    contact_band_z: tuple[float, float] | None,
    peak_force_newtons: float = 30.0,
    peak_torque_newton_meters: float = 3.0,
) -> AicInsertionEnv:
    return AicInsertionEnv(
        runtime=AicGazeboRuntime(
            backend=MockStepperBackend(
                transient_contact_config=MockTransientContactConfig(
                    contact_band_z=contact_band_z,
                    peak_force_newtons=peak_force_newtons,
                    peak_torque_newton_meters=peak_torque_newton_meters,
                )
            ),
            ticks_per_step=ticks_per_step,
        ),
        task=AicInsertionTask(
            hold_action_ticks=ticks_per_step,
            include_images=False,
            max_episode_steps=64,
        ),
        io=MockGazeboIO(),
        randomizer=AicEnvRandomizer(enable_randomization=False),
    )


def _action(z_velocity: float = 0.0, x_velocity: float = 0.0) -> np.ndarray:
    action = np.zeros(6, dtype=np.float32)
    action[0] = x_velocity
    action[2] = z_velocity
    return action


def _run_sequence(
    *,
    name: str,
    ticks_per_step: int,
    contact_band_z: tuple[float, float] | None,
    actions: list[np.ndarray],
    seed: int,
    peak_force_newtons: float = 30.0,
) -> dict[str, Any]:
    env = _make_validation_env(
        ticks_per_step=ticks_per_step,
        contact_band_z=contact_band_z,
        peak_force_newtons=peak_force_newtons,
    )
    try:
        observation, reset_info = env.reset(seed=seed)
        del reset_info
        records: list[dict[str, Any]] = []
        for step_index, action in enumerate(actions):
            observation, reward, terminated, truncated, info = env.step(action)
            aux = info["auxiliary_force_contact_summary"]
            current_force = float(np.linalg.norm(observation["wrench"][:3]))
            max_force = float(aux["wrench_max_force_abs_recent"])
            aliasing_detected = bool(
                aux["had_contact_recent"] and current_force <= 1e-6 and max_force > current_force + 1e-6
            )
            records.append(
                {
                    "scenario": name,
                    "step_index": step_index,
                    "sim_tick": int(observation["sim_tick"]),
                    "sim_time": float(observation["sim_time"]),
                    "current_wrench": observation["wrench"].astype(np.float32).copy(),
                    "current_wrench_force_l2_norm": current_force,
                    "current_contact": bool(observation["off_limit_contact"][0] > 0.5),
                    "reward": float(reward),
                    "had_contact_recent": bool(aux["had_contact_recent"]),
                    "max_contact_indicator_recent": float(aux["max_contact_indicator_recent"]),
                    "wrench_max_abs_recent": aux["wrench_max_abs_recent"].astype(np.float32).copy(),
                    "wrench_mean_recent": aux["wrench_mean_recent"].astype(np.float32).copy(),
                    "wrench_max_force_abs_recent": max_force,
                    "wrench_max_torque_abs_recent": float(aux["wrench_max_torque_abs_recent"]),
                    "first_wrench_recent": aux["first_wrench_recent"].astype(np.float32).copy(),
                    "last_wrench_recent": aux["last_wrench_recent"].astype(np.float32).copy(),
                    "time_of_peak_within_step": aux["time_of_peak_within_step"],
                    "aux_source": str(aux["source"]),
                    "aux_sample_count": int(aux["sample_count"]),
                    "aliasing_detected": aliasing_detected,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                break
        return {
            "name": name,
            "ticks_per_step": ticks_per_step,
            "contact_band_z": None if contact_band_z is None else list(contact_band_z),
            "records": records,
            "summary": {
                "step_count": len(records),
                "aliasing_detected_step_count": sum(
                    1 for record in records if record["aliasing_detected"]
                ),
                "had_contact_recent_step_count": sum(
                    1 for record in records if record["had_contact_recent"]
                ),
                "current_contact_step_count": sum(
                    1 for record in records if record["current_contact"]
                ),
                "max_current_force_l2_norm": max(
                    (record["current_wrench_force_l2_norm"] for record in records),
                    default=0.0,
                ),
                "max_recent_force_l2_norm": max(
                    (record["wrench_max_force_abs_recent"] for record in records),
                    default=0.0,
                ),
            },
        }
    finally:
        env.close()


def _isaac_expectation_check(
    transient_report: dict[str, Any],
    no_contact_report: dict[str, Any],
    weak_report: dict[str, Any],
    strong_report: dict[str, Any],
) -> dict[str, Any]:
    transient_aliasing = transient_report["summary"]["aliasing_detected_step_count"] > 0
    no_contact_quiet = (
        no_contact_report["summary"]["had_contact_recent_step_count"] == 0
        and no_contact_report["summary"]["max_recent_force_l2_norm"] <= 1e-6
    )
    monotonic_contact_strength = (
        strong_report["summary"]["max_recent_force_l2_norm"]
        >= weak_report["summary"]["max_recent_force_l2_norm"]
    )
    return {
        "direct_isaac_lab_parity_tested": False,
        "conceptual_expectation_check_passed": bool(
            transient_aliasing and no_contact_quiet and monotonic_contact_strength
        ),
        "checks": {
            "hidden_transient_visible_in_auxiliary_summary": transient_aliasing,
            "no_contact_motion_stays_quiet": no_contact_quiet,
            "stronger_contact_produces_no_weaker_aggregate_signal": monotonic_contact_strength,
        },
        "notes": [
            "Direct Isaac Lab execution is not performed here.",
            "The conceptual check requires hidden transient contacts to appear in aggregated summaries even when the final policy sample is quiet.",
            "The no-contact control must remain quiet and stronger contact should not reduce the aggregate signal.",
        ],
    }


def _official_path_check() -> dict[str, Any]:
    rclpy_available = importlib.util.find_spec("rclpy") is not None
    official_trace_available = importlib.util.find_spec("aic_gym_gz.official_trace") is not None
    ros2_available = shutil.which("ros2") is not None
    gz_available = shutil.which("gz") is not None
    direct_possible = bool(
        rclpy_available and official_trace_available and ros2_available and gz_available
    )
    limitations: list[str] = []
    if not rclpy_available:
        limitations.append(
            "Direct official-path parity requires the ROS 2 runtime (`rclpy`) and the official controller/services."
        )
    if not official_trace_available:
        limitations.append(
            "The official trace capture module is not importable in the current environment."
        )
    if not ros2_available:
        limitations.append("`ros2` is not on PATH in the current environment.")
    if not gz_available:
        limitations.append("`gz` is not on PATH in the current environment.")
    return {
        "direct_official_parity_tested": False,
        "direct_official_parity_possible_in_current_environment": direct_possible,
        "what_was_compared": [
            "The public observation still exposes only the current `wrench` sample.",
            "Auxiliary within-step summaries live in `step_info` under `auxiliary_force_contact_summary` and are explicitly marked non-official.",
            "Aliasing is detected by comparing the current observation sample to within-step aggregated maxima.",
        ],
        "limitations": limitations
        or [
            "Direct official-path execution was not attempted by this validator; use `aic_gym_gz.official_trace` inside the official ROS/Gazebo environment for direct comparison."
        ],
    }


def _write_csv(path: Path, scenarios: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "step_index",
                "sim_tick",
                "sim_time",
                "current_wrench_force_l2_norm",
                "current_contact",
                "had_contact_recent",
                "max_contact_indicator_recent",
                "wrench_max_force_abs_recent",
                "wrench_max_torque_abs_recent",
                "time_of_peak_within_step",
                "aux_source",
                "aux_sample_count",
                "aliasing_detected",
                "terminated",
                "truncated",
            ]
        )
        for scenario in scenarios:
            for record in scenario["records"]:
                writer.writerow(
                    [
                        record["scenario"],
                        record["step_index"],
                        record["sim_tick"],
                        record["sim_time"],
                        record["current_wrench_force_l2_norm"],
                        int(record["current_contact"]),
                        int(record["had_contact_recent"]),
                        record["max_contact_indicator_recent"],
                        record["wrench_max_force_abs_recent"],
                        record["wrench_max_torque_abs_recent"],
                        record["time_of_peak_within_step"],
                        record["aux_source"],
                        record["aux_sample_count"],
                        int(record["aliasing_detected"]),
                        int(record["terminated"]),
                        int(record["truncated"]),
                    ]
                )


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Force Transient Validation",
        "",
        "## Observation Contract",
        "- Official-compatible observation remains current-sample-only: `observation['wrench']` is the final sample of the step.",
        "- Auxiliary within-step summaries are non-official and are emitted only in `step_info['auxiliary_force_contact_summary']`.",
        "",
        "## Scenario Results",
    ]
    for scenario in payload["scenarios"]:
        summary = scenario["summary"]
        lines.extend(
            [
                f"### {scenario['name']}",
                f"- ticks_per_step: {scenario['ticks_per_step']}",
                f"- aliasing_detected_step_count: {summary['aliasing_detected_step_count']}",
                f"- had_contact_recent_step_count: {summary['had_contact_recent_step_count']}",
                f"- current_contact_step_count: {summary['current_contact_step_count']}",
                f"- max_current_force_l2_norm: {summary['max_current_force_l2_norm']:.4f}",
                f"- max_recent_force_l2_norm: {summary['max_recent_force_l2_norm']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Isaac-Lab-Style Expectation Check",
            f"- direct_isaac_lab_parity_tested: {payload['isaac_lab_style_check']['direct_isaac_lab_parity_tested']}",
            f"- conceptual_expectation_check_passed: {payload['isaac_lab_style_check']['conceptual_expectation_check_passed']}",
            "",
            "## Official Path Check",
            f"- direct_official_parity_tested: {payload['official_path_check']['direct_official_parity_tested']}",
            f"- direct_official_parity_possible_in_current_environment: {payload['official_path_check']['direct_official_parity_possible_in_current_environment']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_force_transient_validation(
    *,
    ticks_per_step: int,
    seed: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    transient = _run_sequence(
        name="scenario_a_obstacle_contact_transient",
        ticks_per_step=ticks_per_step,
        contact_band_z=(1.292, 1.294),
        actions=[_action(z_velocity=-0.25)],
        seed=seed,
    )
    no_contact = _run_sequence(
        name="scenario_b_no_contact_control",
        ticks_per_step=ticks_per_step,
        contact_band_z=(1.292, 1.294),
        actions=[_action(x_velocity=0.02), _action(x_velocity=-0.02), _action()],
        seed=seed,
    )
    repeated = _run_sequence(
        name="scenario_c_repeated_coarse_boundary_crossing",
        ticks_per_step=ticks_per_step,
        contact_band_z=(1.297, 1.299),
        actions=[
            _action(z_velocity=-0.25),
            _action(z_velocity=0.25),
            _action(z_velocity=-0.25),
            _action(z_velocity=0.25),
        ],
        seed=seed,
    )
    weak_contact = _run_sequence(
        name="scenario_strength_weak_contact",
        ticks_per_step=ticks_per_step,
        contact_band_z=(1.297, 1.299),
        actions=[_action(z_velocity=-0.125)],
        seed=seed,
    )
    strong_contact = _run_sequence(
        name="scenario_strength_strong_contact",
        ticks_per_step=ticks_per_step,
        contact_band_z=(1.297, 1.299),
        actions=[_action(z_velocity=-0.25)],
        seed=seed,
    )
    payload = {
        "observation_contract": {
            "official_compatible_current_sample_only": True,
            "auxiliary_force_contact_summary_is_non_official": True,
        },
        "scenarios": [transient, no_contact, repeated, weak_contact, strong_contact],
        "isaac_lab_style_check": _isaac_expectation_check(
            transient_report=transient,
            no_contact_report=no_contact,
            weak_report=weak_contact,
            strong_report=strong_contact,
        ),
        "official_path_check": _official_path_check(),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "force_transient_validation_report.json").write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(output_dir / "force_transient_validation_records.csv", payload["scenarios"])
        (output_dir / "force_transient_validation_report.md").write_text(
            _markdown_summary(payload),
            encoding="utf-8",
        )
    return _json_safe(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks-per-step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    payload = run_force_transient_validation(
        ticks_per_step=args.ticks_per_step,
        seed=args.seed,
        output_dir=output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

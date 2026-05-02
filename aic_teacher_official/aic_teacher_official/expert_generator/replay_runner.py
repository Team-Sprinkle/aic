"""Gazebo replay runner and score parser for expert candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
import subprocess
import os
from pathlib import Path
from typing import Any

import yaml

from aic_teacher_official.trajectory import SmoothTrajectory


@dataclass(frozen=True)
class OfficialReplayConfig:
    repo_root: Path
    engine_config: Path
    output_dir: Path
    dataset_repo_id_prefix: str = "local/aic_expert"
    policy_class: str = "aic_teacher_official.OfficialTeacherReplay"
    action_mode: str = "relative_delta_gripper_tcp"
    gazebo_gui: bool = False
    launch_rviz: bool = False
    startup_delay_sec: int = 8
    recorder_drain_sec: int = 120
    per_trial_timeout_sec: int = 0
    sim_distrobox: str = ""
    require_recorder_save_log: bool = True
    remove_bag_data: bool = True
    expert_mode: str = "nominal"
    ft_threshold_n: float | None = None
    recovery_backoff_distance_m: float | None = None
    recovery_min_backoff_distance_m: float | None = None
    recovery_max_retries: int | None = None
    recovery_release_force_threshold_n: float | None = None
    force_confirm_sec: float | None = None


class OfficialRecordingReplayRunner:
    def __init__(self, config: OfficialReplayConfig):
        self.config = config

    def replay_and_score(
        self,
        trajectory: SmoothTrajectory | Any,
        *,
        attempt_index: int,
        candidate_index: int,
        variant_label: str | None = None,
    ) -> dict[str, Any]:
        suffix = f"_candidate_{candidate_index:02d}"
        if variant_label:
            suffix += f"_{variant_label}"
        attempt_dir = self.config.output_dir / f"attempt_{attempt_index:06d}{suffix}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = attempt_dir / "smooth_trajectory.json"
        if hasattr(trajectory, "save_json"):
            trajectory.save_json(trajectory_path)
        else:
            raise TypeError("OfficialRecordingReplayRunner requires a SmoothTrajectory-like object with save_json")
        cmd = self.build_command(
            trajectory_path=trajectory_path,
            attempt_dir=attempt_dir,
            attempt_index=attempt_index,
            candidate_index=candidate_index,
        )
        env = os.environ.copy()
        source_paths = [
            self.config.repo_root / "aic_teacher_official",
            self.config.repo_root / "aic_example_policies",
        ]
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(
            [str(path) for path in source_paths] + ([existing_pythonpath] if existing_pythonpath else [])
        )
        env["AIC_EXPERT_MODE"] = self.config.expert_mode
        env["AIC_OFFICIAL_TEACHER_RUNTIME_TRACE"] = str(attempt_dir / "runtime_trace.jsonl")
        if self.config.ft_threshold_n is not None:
            env["AIC_OFFICIAL_TEACHER_FT_THRESHOLD_N"] = str(self.config.ft_threshold_n)
        if self.config.recovery_backoff_distance_m is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M"] = str(
                self.config.recovery_backoff_distance_m
            )
        if self.config.recovery_min_backoff_distance_m is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_MIN_BACKOFF_DISTANCE_M"] = str(
                self.config.recovery_min_backoff_distance_m
            )
        if self.config.recovery_max_retries is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_MAX_RETRIES"] = str(self.config.recovery_max_retries)
        if self.config.recovery_release_force_threshold_n is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N"] = str(
                self.config.recovery_release_force_threshold_n
            )
        if self.config.force_confirm_sec is not None:
            env["AIC_OFFICIAL_TEACHER_FORCE_CONFIRM_SEC"] = str(self.config.force_confirm_sec)
        with (attempt_dir / "replay_stdout.txt").open("w", encoding="utf-8") as stdout, (
            attempt_dir / "replay_stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                cmd,
                cwd=self.config.repo_root,
                env=env,
                text=True,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        metrics = metrics_from_scoring_yaml(attempt_dir / "results" / "trial_1_trial_000001" / "scoring.yaml")
        metrics.update(
            {
                "replay_returncode": result.returncode,
                "replay_command": " ".join(shlex.quote(part) for part in cmd),
                "trajectory_path": str(trajectory_path),
                "runtime_trace_path": str(attempt_dir / "runtime_trace.jsonl"),
            }
        )
        return metrics

    def build_command(
        self,
        *,
        trajectory_path: Path,
        attempt_dir: Path,
        attempt_index: int,
        candidate_index: int,
    ) -> list[str]:
        cmd = [
            "bash",
            "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
            "--engine-config",
            str(self.config.engine_config),
            "--policy-class",
            self.config.policy_class,
            "--teacher-trajectory",
            str(trajectory_path),
            "--teacher-action-mode",
            self.config.action_mode,
            "--dataset-repo-id",
            f"{self.config.dataset_repo_id_prefix}_{attempt_index:06d}_{candidate_index:02d}",
            "--dataset-root",
            str(attempt_dir / "dataset"),
            "--results-root",
            str(attempt_dir / "results"),
            "--tmp-dir",
            str(attempt_dir / "tmp"),
            "--gazebo-gui",
            str(self.config.gazebo_gui).lower(),
            "--launch-rviz",
            str(self.config.launch_rviz).lower(),
            "--startup-delay-sec",
            str(self.config.startup_delay_sec),
            "--recorder-drain-sec",
            str(self.config.recorder_drain_sec),
            "--per-trial-timeout-sec",
            str(self.config.per_trial_timeout_sec),
            "--require-recorder-save-log",
            str(self.config.require_recorder_save_log).lower(),
            "--remove-bag-data",
            str(self.config.remove_bag_data).lower(),
        ]
        if self.config.sim_distrobox:
            cmd.extend(["--sim-distrobox", self.config.sim_distrobox])
        return cmd


def metrics_from_scoring_yaml(path: str | Path) -> dict[str, Any]:
    scoring_path = Path(path)
    if not scoring_path.exists():
        return {
            "score": None,
            "insertion_event_reached": False,
            "max_force_n": None,
            "offlimit_contact_count": None,
            "scoring_yaml": str(scoring_path),
            "scoring_missing": True,
        }
    data = yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
    text = scoring_path.read_text(encoding="utf-8")
    official_max_force = _extract_float(r"Max detected force:\s*([0-9.]+)N", text)
    force_penalty_applied = official_max_force is not None and "Penalty not applied" not in text
    max_force = official_max_force if force_penalty_applied else 0.0
    if official_max_force is None and "No excessive force detected" in text:
        official_max_force = 0.0
        max_force = 0.0
    contacts_ok = "No contact detected" in text
    insertion = "Cable insertion successful" in text
    duration = _extract_float(r"Task duration:\s*([0-9.]+)\s*seconds", text)
    official_total = _coerce_float(data.get("total"))
    task_score = _task_score_from_scoring(data, official_total=official_total)
    trial = _first_trial_block(data) or {}
    tier_1 = trial.get("tier_1") if isinstance(trial, dict) else {}
    tier_2 = trial.get("tier_2") if isinstance(trial, dict) else {}
    tier_3 = trial.get("tier_3") if isinstance(trial, dict) else {}
    tier_1_score = _coerce_float(tier_1.get("score")) if isinstance(tier_1, dict) else None
    tier_2_score = _coerce_float(tier_2.get("score")) if isinstance(tier_2, dict) else None
    tier_3_score = _coerce_float(tier_3.get("score")) if isinstance(tier_3, dict) else None
    tier_3_message = str(tier_3.get("message", "")) if isinstance(tier_3, dict) else ""
    return {
        # Keep `score` aligned with the official scorer. Partial insertion and
        # proximity points are represented in the top-level total, and expert
        # acceptance still relies on insertion/validation gates to reject
        # model-validity-only runs where total=1.
        "score": official_total,
        "official_total_score": official_total,
        "task_score_excluding_tier_1": task_score,
        "score_source": "official_total",
        "tier_1_score": tier_1_score,
        "tier_2_score": tier_2_score,
        "tier_3_score": tier_3_score,
        "tier_3_message": tier_3_message,
        "has_partial_or_full_task_progress": bool((tier_3_score or 0.0) > 0.0 or (tier_2_score or 0.0) > 0.0),
        "insertion_event_reached": insertion,
        "max_force_n": max_force,
        "official_max_force_n": official_max_force,
        "insertion_force_penalty_applied": force_penalty_applied,
        "ft_impulse_ns": None,
        "max_tracking_error_m": None,
        "offlimit_contact_count": 0 if contacts_ok else 1,
        "trajectory_duration_s": duration,
        "scoring_yaml": str(scoring_path),
        "scoring_missing": False,
    }


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_score_from_scoring(data: dict[str, Any], *, official_total: float | None) -> float | None:
    """Return task score excluding tier-1 model-validation credit.

    The official YAML top-level `total` includes non-task bookkeeping such as
    `tier_1.score: 1` for model validation. For expert trajectory acceptance and
    debugging, that makes failed rollouts look like they scored 1 instead of 0.
    Prefer explicit non-tier-1 score fields when present; otherwise subtract
    tier-1 credit from the official total. If no tier information exists, fall
    back to the raw total for older/minimal score fixtures.
    """

    trial = _first_trial_block(data)
    if not isinstance(trial, dict):
        return official_total

    non_tier_1_scores: list[float] = []
    tier_1_score = 0.0
    saw_tier_score = False
    for name, block in trial.items():
        if not str(name).startswith("tier_") or not isinstance(block, dict):
            continue
        score = _coerce_float(block.get("score"))
        if score is None:
            continue
        saw_tier_score = True
        if str(name) == "tier_1":
            tier_1_score += score
        else:
            non_tier_1_scores.append(score)

    if non_tier_1_scores:
        return float(sum(non_tier_1_scores))
    if saw_tier_score:
        return 0.0
    if official_total is not None and tier_1_score:
        return max(0.0, official_total - tier_1_score)
    return official_total


def _first_trial_block(data: dict[str, Any]) -> dict[str, Any] | None:
    for key, value in data.items():
        if str(key).startswith("trial_") and isinstance(value, dict):
            return value
    return None

"""Gazebo replay runner and score parser for expert candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
import subprocess
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


class OfficialRecordingReplayRunner:
    def __init__(self, config: OfficialReplayConfig):
        self.config = config

    def replay_and_score(
        self,
        trajectory: SmoothTrajectory | Any,
        *,
        attempt_index: int,
        candidate_index: int,
    ) -> dict[str, Any]:
        attempt_dir = self.config.output_dir / f"attempt_{attempt_index:06d}_candidate_{candidate_index:02d}"
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
        with (attempt_dir / "replay_stdout.txt").open("w", encoding="utf-8") as stdout, (
            attempt_dir / "replay_stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(cmd, cwd=self.config.repo_root, text=True, stdout=stdout, stderr=stderr, check=False)
        metrics = metrics_from_scoring_yaml(attempt_dir / "results" / "trial_1_trial_000001" / "scoring.yaml")
        metrics.update(
            {
                "replay_returncode": result.returncode,
                "replay_command": " ".join(shlex.quote(part) for part in cmd),
                "trajectory_path": str(trajectory_path),
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
    max_force = _extract_float(r"Max detected force:\s*([0-9.]+)N", text)
    contacts_ok = "No contact detected" in text
    insertion = "Cable insertion successful" in text
    duration = _extract_float(r"Task duration:\s*([0-9.]+)\s*seconds", text)
    return {
        "score": float(data["total"]) if data.get("total") is not None else None,
        "insertion_event_reached": insertion,
        "max_force_n": max_force,
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

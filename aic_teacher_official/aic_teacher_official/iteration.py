"""Iteration helpers for official teacher trajectory improvement loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any


def run_name_for_loop(base_run_name: str, loop_index: int) -> str:
    if loop_index < 1:
        raise ValueError("loop_index must be >= 1")
    return f"{base_run_name}_loop_{loop_index}"


@dataclass(frozen=True)
class LoopRoots:
    planner_root: Path
    postprocessed_root: Path

    @property
    def piecewise_path(self) -> Path:
        return self.planner_root / "piecewise_trajectory.json"

    @property
    def smooth_path(self) -> Path:
        return self.postprocessed_root / "smooth_trajectory.json"

    @property
    def dataset_root(self) -> Path:
        return self.postprocessed_root / "raw_dataset"

    @property
    def scores_root(self) -> Path:
        return self.postprocessed_root / "scores"

    @property
    def scoring_path(self) -> Path:
        return self.scores_root / "trial_1_trial_000001" / "scoring.yaml"


def loop_roots(
    *,
    root_dir: str | Path,
    task_family: str,
    scene_count_label: str,
    attempt_label: str,
    base_run_name: str,
    loop_index: int,
) -> LoopRoots:
    run_name = run_name_for_loop(base_run_name, loop_index)
    root = Path(root_dir)
    return LoopRoots(
        planner_root=root
        / task_family
        / "vlm_planner"
        / scene_count_label
        / attempt_label
        / run_name,
        postprocessed_root=root
        / task_family
        / "vlm_planner_postprocessed"
        / scene_count_label
        / attempt_label
        / run_name,
    )


def parse_total_score(scoring_path: str | Path) -> float | None:
    path = Path(scoring_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("total:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def build_recording_command(
    *,
    engine_config: str | Path,
    sim_distrobox: str,
    smooth_path: str | Path,
    dataset_repo_id: str,
    dataset_root: str | Path,
    scores_root: str | Path,
    tmp_dir: str | Path,
    action_mode: str = "relative_delta_gripper_tcp",
    gazebo_gui: bool = False,
    launch_rviz: bool = False,
    startup_delay_sec: int = 8,
    per_trial_timeout_sec: int = 0,
    recorder_drain_sec: int = 120,
) -> list[str]:
    return [
        "bash",
        "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
        "--engine-config",
        str(engine_config),
        "--sim-distrobox",
        sim_distrobox,
        "--policy-class",
        "aic_teacher_official.OfficialTeacherReplay",
        "--teacher-trajectory",
        str(smooth_path),
        "--teacher-action-mode",
        action_mode,
        "--dataset-repo-id",
        dataset_repo_id,
        "--dataset-root",
        str(dataset_root),
        "--results-root",
        str(scores_root),
        "--gazebo-gui",
        str(gazebo_gui).lower(),
        "--launch-rviz",
        str(launch_rviz).lower(),
        "--startup-delay-sec",
        str(startup_delay_sec),
        "--per-trial-timeout-sec",
        str(per_trial_timeout_sec),
        "--recorder-drain-sec",
        str(recorder_drain_sec),
        "--require-recorder-save-log",
        "false",
        "--remove-bag-data",
        "true",
        "--tmp-dir",
        str(tmp_dir),
    ]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_loop_manifest(path: str | Path, data: dict[str, Any]) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

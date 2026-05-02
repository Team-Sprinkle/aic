"""Live planner-recording runner for OfficialExpertGeneratorPlanner."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpertPlannerRunConfig:
    repo_root: Path
    engine_config: Path
    output_dir: Path
    expert_mode: str
    strategy_model: str = "gpt-5-mini"
    candidates_per_scene: int = 5
    image_sample_period_sec: float = 0.5
    image_capture_duration_sec: float = 2.0
    max_images: int = 8
    gazebo_gui: bool = False
    launch_rviz: bool = False
    startup_delay_sec: int = 8
    recorder_drain_sec: int = 45
    per_trial_timeout_sec: int = 0
    sim_distrobox: str = ""
    remove_bag_data: bool = True
    launch_moveit: bool = True
    moveit_launch_file: str = "aic_moveit_config moveit.launch.py"
    ft_threshold_n: float | None = None


class ExpertPlannerRecordingRunner:
    def __init__(self, config: ExpertPlannerRunConfig):
        self.config = config

    def run_planner(
        self,
        *,
        attempt_index: int,
        candidate_index: int,
        seed: int,
    ) -> dict[str, Any]:
        attempt_dir = self.config.output_dir / f"attempt_{attempt_index:06d}_candidate_{candidate_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        piecewise_path = attempt_dir / "piecewise_trajectory.json"
        debug_dir = attempt_dir / "planner_debug"
        cmd = self.build_command(attempt_dir=attempt_dir)
        env = {
            **os.environ,
            "AIC_EXPERT_MODE": self.config.expert_mode,
            "AIC_EXPERT_PIECEWISE_OUTPUT": str(piecewise_path),
            "AIC_EXPERT_DEBUG_OUTPUT_DIR": str(debug_dir),
            "AIC_EXPERT_ENGINE_CONFIG": str(self.config.engine_config),
            "AIC_EXPERT_RUN_ID": self.config.output_dir.name,
            "AIC_EXPERT_SEED": str(seed),
            "AIC_EXPERT_CANDIDATE_INDEX": str(candidate_index),
            "AIC_EXPERT_CANDIDATES_PER_SCENE": str(self.config.candidates_per_scene),
            "AIC_EXPERT_STRATEGY_MODEL": self.config.strategy_model,
            "AIC_EXPERT_IMAGE_SAMPLE_PERIOD_SEC": str(self.config.image_sample_period_sec),
            "AIC_EXPERT_IMAGE_CAPTURE_DURATION_SEC": str(self.config.image_capture_duration_sec),
            "AIC_EXPERT_MAX_IMAGES": str(self.config.max_images),
        }
        if self.config.ft_threshold_n is not None:
            env["AIC_EXPERT_FT_THRESHOLD_N"] = str(self.config.ft_threshold_n)
            env["AIC_EXPERT_FT_SOFT_THRESHOLD_N"] = str(self.config.ft_threshold_n)
            env["AIC_EXPERT_FT_HARD_THRESHOLD_N"] = str(self.config.ft_threshold_n * 3.0)
        with (attempt_dir / "planner_stdout.txt").open("w", encoding="utf-8") as stdout, (
            attempt_dir / "planner_stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                cmd,
                cwd=self.config.repo_root,
                text=True,
                stdout=stdout,
                stderr=stderr,
                env=env,
                check=False,
            )
        return {
            "attempt_dir": str(attempt_dir),
            "piecewise_path": str(piecewise_path),
            "piecewise_exists": piecewise_path.exists(),
            "debug_dir": str(debug_dir),
            "returncode": result.returncode,
            "command": " ".join(shlex.quote(part) for part in cmd),
        }

    def build_command(self, *, attempt_dir: Path) -> list[str]:
        cmd = [
            "bash",
            "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
            "--engine-config",
            str(self.config.engine_config),
            "--policy-class",
            "aic_teacher_official.OfficialExpertGeneratorPlanner",
            "--dataset-repo-id",
            f"local/aic_expert_planner_{attempt_dir.name}",
            "--dataset-root",
            str(attempt_dir / "planner_dataset"),
            "--results-root",
            str(attempt_dir / "planner_results"),
            "--tmp-dir",
            str(attempt_dir / "planner_tmp"),
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
            "false",
            "--remove-bag-data",
            str(self.config.remove_bag_data).lower(),
            "--launch-moveit",
            str(self.config.launch_moveit).lower(),
            "--moveit-launch-file",
            self.config.moveit_launch_file,
        ]
        if self.config.sim_distrobox:
            cmd.extend(["--sim-distrobox", self.config.sim_distrobox])
        return cmd

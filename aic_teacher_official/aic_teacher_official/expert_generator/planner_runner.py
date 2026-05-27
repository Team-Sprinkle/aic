"""Live planner-recording runner for OfficialExpertGeneratorPlanner."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


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
    record_dataset: bool = True


class ExpertPlannerRecordingRunner:
    def __init__(self, config: ExpertPlannerRunConfig):
        self.config = config

    def engine_config_for_attempt(self, attempt_index: int) -> Path:
        trial_config = self.config.engine_config.parent / "trials" / f"trial_{attempt_index:06d}.yaml"
        if trial_config.exists():
            return trial_config
        return self.config.engine_config

    def run_planner(
        self,
        *,
        attempt_index: int,
        candidate_index: int,
        seed: int,
    ) -> dict[str, Any]:
        attempt_dir = self.config.output_dir / f"attempt_{attempt_index:06d}_candidate_{candidate_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = self._runtime_attempt_dir(attempt_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        engine_config = self._write_attempt_engine_config(attempt_dir=runtime_dir, attempt_index=attempt_index)
        piecewise_path = attempt_dir / "piecewise_trajectory.json"
        debug_dir = attempt_dir / "planner_debug"
        runtime_results = runtime_dir / "planner_results"
        runtime_tmp = runtime_dir / "planner_tmp"
        cmd = self.build_command(
            attempt_dir=attempt_dir,
            engine_config=engine_config,
            results_root=runtime_results,
            tmp_dir=runtime_tmp,
        )
        env = {
            **os.environ,
            "AIC_EXPERT_MODE": self.config.expert_mode,
            "AIC_EXPERT_PIECEWISE_OUTPUT": str(piecewise_path),
            "AIC_EXPERT_DEBUG_OUTPUT_DIR": str(debug_dir),
            "AIC_EXPERT_ENGINE_CONFIG": str(engine_config),
            "AIC_EXPERT_RUN_ID": self.config.output_dir.name,
            "AIC_EXPERT_SEED": str(seed),
            "AIC_EXPERT_CANDIDATE_INDEX": str(candidate_index),
            "AIC_EXPERT_CANDIDATES_PER_SCENE": str(self.config.candidates_per_scene),
            "AIC_EXPERT_STRATEGY_MODEL": self.config.strategy_model,
            "AIC_EXPERT_IMAGE_SAMPLE_PERIOD_SEC": str(self.config.image_sample_period_sec),
            "AIC_EXPERT_IMAGE_CAPTURE_DURATION_SEC": str(self.config.image_capture_duration_sec),
            "AIC_EXPERT_MAX_IMAGES": str(self.config.max_images),
        }
        env.update(_registry_env_for_engine_config(engine_config))
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
        self._copy_runtime_artifacts(runtime_results, attempt_dir / "planner_results")
        self._copy_runtime_artifacts(runtime_tmp, attempt_dir / "planner_tmp")
        return {
            "attempt_dir": str(attempt_dir),
            "piecewise_path": str(piecewise_path),
            "piecewise_exists": piecewise_path.exists(),
            "debug_dir": str(debug_dir),
            "engine_config": str(engine_config),
            "returncode": result.returncode,
            "command": " ".join(shlex.quote(part) for part in cmd),
        }

    def _runtime_attempt_dir(self, attempt_dir: Path) -> Path:
        try:
            relative = attempt_dir.relative_to(self.config.repo_root)
        except ValueError:
            relative = Path(self.config.output_dir.name) / attempt_dir.name
        return self.config.repo_root / "outputs" / "trajectory_runtime" / relative

    @staticmethod
    def _copy_runtime_artifacts(src: Path, dst: Path) -> None:
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def _write_attempt_engine_config(self, *, attempt_dir: Path, attempt_index: int) -> Path:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_load(self.config.engine_config.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Engine config must be a map: {self.config.engine_config}")
        trials = data.get("trials")
        if not isinstance(trials, dict) or not trials:
            raise ValueError(f"Engine config must contain non-empty trials: {self.config.engine_config}")
        trial_ids = list(trials)
        trial_id = trial_ids[(attempt_index - 1) % len(trial_ids)]
        single = dict(data)
        single["trials"] = {trial_id: trials[trial_id]}
        single["number_of_trials"] = 1
        out_path = attempt_dir / f"engine_{attempt_index:06d}_{trial_id}.yaml"
        out_path.write_text(yaml.safe_dump(single, sort_keys=False), encoding="utf-8")
        return out_path

    def build_command(
        self,
        *,
        attempt_dir: Path,
        engine_config: Path | None = None,
        results_root: Path | None = None,
        tmp_dir: Path | None = None,
    ) -> list[str]:
        engine_config = engine_config or self.config.engine_config
        results_root = results_root or attempt_dir / "planner_results"
        tmp_dir = tmp_dir or attempt_dir / "planner_tmp"
        cmd = [
            "bash",
            "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
            "--engine-config",
            str(engine_config),
            "--policy-class",
            "aic_teacher_official.OfficialExpertGeneratorPlanner",
            "--dataset-repo-id",
            f"local/aic_expert_planner_{attempt_dir.name}",
            "--dataset-root",
            str(attempt_dir / "planner_dataset"),
            "--results-root",
            str(results_root),
            "--tmp-dir",
            str(tmp_dir),
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
            "--record-episode",
            str(self.config.record_dataset).lower(),
        ]
        if self.config.sim_distrobox:
            cmd.extend(["--sim-distrobox", self.config.sim_distrobox])
        return cmd


def _rail_indices(task_board: dict[str, Any], prefix: str) -> list[int]:
    indices: list[int] = []
    for key, value in task_board.items():
        if not key.startswith(prefix) or not isinstance(value, dict):
            continue
        if value.get("entity_present"):
            try:
                indices.append(int(key.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(indices)


def _parse_index(text: str, prefix: str) -> int | None:
    if not text.startswith(prefix):
        return None
    try:
        return int(text[len(prefix) :])
    except ValueError:
        return None


def _registry_suffix_from_trial(task_family: str, trial: dict[str, Any]) -> str | None:
    task = trial.get("tasks", {}).get("task_1", {})
    task_board = trial.get("scene", {}).get("task_board", {})
    if not isinstance(task, dict) or not isinstance(task_board, dict):
        return None
    if task_family == "sc_to_sc":
        present = _rail_indices(task_board, "sc_rail_")
        target = _parse_index(str(task.get("target_module_name", "")), "sc_port_")
        if not present or target is None:
            return None
        nic_count = len(_rail_indices(task_board, "nic_rail_"))
        present_label = "".join(str(idx) for idx in present)
        return f"matrix_sc2sc_sc{len(present)}_present{present_label}_target{target}_nic{nic_count}"
    if task_family == "sfp_to_nic":
        present = _rail_indices(task_board, "nic_rail_")
        target = _parse_index(str(task.get("target_module_name", "")), "nic_card_mount_")
        port = _parse_index(str(task.get("port_name", "")), "sfp_port_")
        if not present or target is None or port is None:
            return None
        present_label = "".join(str(idx) for idx in present)
        return f"matrix_sfp2nic_cards{len(present)}_present{present_label}_target{target}_port{port}"
    return None


def _single_trial_registry_suffix(engine_config: Path) -> str | None:
    task_family = os.environ.get("AIC_EXPERT_TASK_FAMILY", "")
    if not task_family:
        return None
    try:
        config = yaml.safe_load(engine_config.read_text(encoding="utf-8"))
    except Exception:
        return None
    trials = config.get("trials") if isinstance(config, dict) else None
    if not isinstance(trials, dict) or not trials:
        return None
    for trial in trials.values():
        if isinstance(trial, dict):
            return _registry_suffix_from_trial(task_family, trial)
    return None


def _registry_env_for_engine_config(engine_config: Path) -> dict[str, str]:
    raw = os.environ.get("AIC_EXPERT_REGISTRY_MODE_ENV_BY_SUFFIX", "")
    if not raw:
        return {}
    suffix = _single_trial_registry_suffix(engine_config)
    if not suffix:
        return {}
    try:
        env_by_suffix = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    mode_env = env_by_suffix.get(suffix) if isinstance(env_by_suffix, dict) else None
    if not isinstance(mode_env, dict):
        return {}
    return {str(key): str(value) for key, value in mode_env.items()}

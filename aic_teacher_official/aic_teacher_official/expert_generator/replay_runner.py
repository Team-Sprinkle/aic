"""Gazebo replay runner and score parser for expert candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import shutil
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
    # The VLM/MoveIt expert generator records free-space transport as MoveIt
    # joint states, then hands off to online CheatCode geometry. Keeping this
    # default avoids replaying base-link trajectory poses as TCP-frame deltas.
    action_mode: str = "joint_position_then_cheatcode"
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
    recovery_backoff_increment_m: float | None = None
    recovery_backoff_sec: float | None = None
    recovery_min_backoff_distance_m: float | None = None
    recovery_max_retries: int | None = None
    recovery_release_force_threshold_n: float | None = None
    force_confirm_sec: float | None = None
    cartesian_stiffness: str | None = None
    cartesian_damping: str | None = None
    recovery_cartesian_stiffness: str | None = None
    recovery_cartesian_damping: str | None = None
    joint_stiffness: str | None = None
    joint_damping: str | None = None


class OfficialRecordingReplayRunner:
    def __init__(self, config: OfficialReplayConfig):
        self.config = config

    def engine_config_for_attempt(self, attempt_index: int) -> Path:
        trial_config = self.config.engine_config.parent / "trials" / f"trial_{attempt_index:06d}.yaml"
        if trial_config.exists():
            return trial_config
        return self.config.engine_config

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
        runtime_dir = self._runtime_attempt_dir(attempt_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = attempt_dir / "smooth_trajectory.json"
        if hasattr(trajectory, "save_json"):
            trajectory.save_json(trajectory_path)
        else:
            raise TypeError("OfficialRecordingReplayRunner requires a SmoothTrajectory-like object with save_json")
        engine_config, trial_id = self._write_attempt_engine_config(
            attempt_dir=runtime_dir,
            attempt_index=attempt_index,
        )
        runtime_results = runtime_dir / "results"
        runtime_tmp = runtime_dir / "tmp"
        cmd = self.build_command(
            trajectory_path=trajectory_path,
            attempt_dir=attempt_dir,
            attempt_index=attempt_index,
            candidate_index=candidate_index,
            engine_config=engine_config,
            results_root=runtime_results,
            tmp_dir=runtime_tmp,
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
        env.update(_registry_env_for_engine_config(engine_config))
        env["AIC_EXPERT_MODE"] = self.config.expert_mode
        env["AIC_OFFICIAL_TEACHER_RUNTIME_TRACE"] = str(attempt_dir / "runtime_trace.jsonl")
        if self.config.ft_threshold_n is not None:
            env["AIC_OFFICIAL_TEACHER_FT_THRESHOLD_N"] = str(self.config.ft_threshold_n)
        if self.config.recovery_backoff_distance_m is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M"] = str(
                self.config.recovery_backoff_distance_m
            )
        if self.config.recovery_backoff_increment_m is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_DISTANCE_M"] = str(
                self.config.recovery_backoff_increment_m
            )
        if self.config.recovery_backoff_sec is not None:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_SEC"] = str(self.config.recovery_backoff_sec)
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
        if self.config.cartesian_stiffness:
            env["AIC_OFFICIAL_TEACHER_CARTESIAN_STIFFNESS"] = self.config.cartesian_stiffness
        if self.config.cartesian_damping:
            env["AIC_OFFICIAL_TEACHER_CARTESIAN_DAMPING"] = self.config.cartesian_damping
        if self.config.recovery_cartesian_stiffness:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_STIFFNESS"] = (
                self.config.recovery_cartesian_stiffness
            )
        if self.config.recovery_cartesian_damping:
            env["AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_DAMPING"] = (
                self.config.recovery_cartesian_damping
            )
        if self.config.joint_stiffness:
            env["AIC_OFFICIAL_TEACHER_JOINT_STIFFNESS"] = self.config.joint_stiffness
        if self.config.joint_damping:
            env["AIC_OFFICIAL_TEACHER_JOINT_DAMPING"] = self.config.joint_damping
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
        self._copy_runtime_artifacts(runtime_results, attempt_dir / "results")
        self._copy_runtime_artifacts(runtime_tmp, attempt_dir / "tmp")
        metrics = metrics_from_scoring_yaml(runtime_results / f"trial_1_{trial_id}" / "scoring.yaml")
        metrics.update(
            {
                "replay_returncode": result.returncode,
                "replay_command": " ".join(shlex.quote(part) for part in cmd),
                "engine_config": str(engine_config),
                "trajectory_path": str(trajectory_path),
                "runtime_trace_path": str(attempt_dir / "runtime_trace.jsonl"),
            }
        )
        return metrics

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

    def _write_attempt_engine_config(self, *, attempt_dir: Path, attempt_index: int) -> tuple[Path, str]:
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
        return out_path, trial_id

    def build_command(
        self,
        *,
        trajectory_path: Path,
        attempt_dir: Path,
        attempt_index: int,
        candidate_index: int,
        engine_config: Path | None = None,
        results_root: Path | None = None,
        tmp_dir: Path | None = None,
    ) -> list[str]:
        engine_config = engine_config or self.config.engine_config
        results_root = results_root or attempt_dir / "results"
        tmp_dir = tmp_dir or attempt_dir / "tmp"
        cmd = [
            "bash",
            "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
            "--engine-config",
            str(engine_config),
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
            str(self.config.require_recorder_save_log).lower(),
            "--remove-bag-data",
            str(self.config.remove_bag_data).lower(),
        ]
        if self.config.sim_distrobox:
            cmd.extend(["--sim-distrobox", self.config.sim_distrobox])
        return cmd


def _scoring_yaml_for_attempt(attempt_dir: Path) -> Path:
    scoring_paths = sorted((attempt_dir / "results").glob("trial_*_*/scoring.yaml"))
    if scoring_paths:
        return scoring_paths[0]
    return attempt_dir / "results" / "trial_1_trial_000001" / "scoring.yaml"


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

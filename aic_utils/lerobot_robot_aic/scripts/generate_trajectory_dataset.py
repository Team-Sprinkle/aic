#!/usr/bin/env python3
"""Generate user-facing trajectory datasets with the LeRobot recorder pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_SCRIPT_DIR = REPO_ROOT / "aic_engine" / "scripts"
sys.path.insert(0, str(ENGINE_SCRIPT_DIR))

from generate_random_trials_config import (  # noqa: E402
    PROFILE_QUALIFICATION_EVAL_LIKE,
    _build_trial,
    _profile_defaults,
)

TASK_FAMILIES = {"sfp_to_nic", "sc_to_sc"}
POLICY_CLASS = {"cheatcode": "aic_example_policies.ros.CheatCode"}
NIC_RAILS = [f"nic_rail_{i}" for i in range(5)]
SC_RAILS = [f"sc_rail_{i}" for i in range(2)]
MOUNT_RAILS = [
    "lc_mount_rail_0",
    "sfp_mount_rail_0",
    "sc_mount_rail_0",
    "lc_mount_rail_1",
    "sfp_mount_rail_1",
    "sc_mount_rail_1",
]
MOUNT_ENTITY_NAMES = {
    "lc_mount_rail_0": "lc_mount_0",
    "sfp_mount_rail_0": "sfp_mount_0",
    "sc_mount_rail_0": "sc_mount_0",
    "lc_mount_rail_1": "lc_mount_1",
    "sfp_mount_rail_1": "sfp_mount_1",
    "sc_mount_rail_1": "sc_mount_1",
}
LIMITS = {
    "nic_translation": (-0.0215, 0.0234),
    "nic_yaw_deg": (-10.0, 10.0),
    "sc_translation": (-0.06, 0.055),
    "fixture_translation": (-0.09425, 0.09425),
    "fixture_yaw_deg": (-60.0, 60.0),
}
DEFAULT_TEMPLATE = REPO_ROOT / "aic_engine" / "config" / "sample_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-yaml", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-trials-override", type=int, default=None)
    parser.add_argument("--target-accepted-override", type=int, default=None)
    parser.add_argument("--max-attempts-override", type=int, default=None)
    parser.add_argument("--skip-recording", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument(
        "--inspect-reference-dataset",
        default=None,
        help="Optional Hugging Face LeRobot reference dataset repo id.",
    )
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        request = yaml.safe_load(f)
    if not isinstance(request, dict):
        raise ValueError(f"Request YAML must be a map: {path}")
    return request


def require_path(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Missing required field: {dotted}")
        current = current[part]
    return current


def validate_request(request: dict[str, Any]) -> None:
    require_path(request, "root_dir")
    task_family = require_path(request, "task_family")
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"task_family must be one of {sorted(TASK_FAMILIES)}")
    require_path(request, "generation.target_accepted_trajectories")
    require_path(request, "generation.max_attempts")
    policy = require_path(request, "generation.policy")
    if policy not in POLICY_CLASS:
        raise ValueError(f"Unsupported generation.policy '{policy}'. Supported: {sorted(POLICY_CLASS)}")
    require_path(request, "acceptance.min_score")
    if task_family == "sfp_to_nic":
        require_path(request, "scene.nic_cards.count")
    else:
        require_path(request, "scene.sc_ports.count")


def sample_value(spec: Any, default: Any, rng: random.Random) -> Any:
    if spec is None:
        if isinstance(default, (tuple, list)) and len(default) == 2:
            return rng.uniform(float(default[0]), float(default[1]))
        return default
    if isinstance(spec, list):
        if not spec:
            raise ValueError("List override must not be empty")
        return rng.choice(spec)
    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        lo = float(spec["min"])
        hi = float(spec["max"])
        if lo > hi:
            raise ValueError(f"Invalid range override: min {lo} > max {hi}")
        return rng.uniform(lo, hi)
    return spec


def _as_count(value: Any, field: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field} must resolve to an integer count, got {value!r}")
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def _count_label(task_family: str, request: dict[str, Any]) -> str:
    if task_family == "sfp_to_nic":
        spec = request["scene"]["nic_cards"]["count"]
        label = spec if isinstance(spec, int) else "mixed"
        return f"nic_cards_{label}"
    spec = request["scene"]["sc_ports"]["count"]
    label = spec if isinstance(spec, int) else "mixed"
    return f"sc_ports_{label}"


def derive_output_dir(request: dict[str, Any]) -> Path:
    policy = request["generation"]["policy"]
    target = int(request["generation"]["target_accepted_trajectories"])
    suffix = str(request.get("suffix", "dataset"))
    return (
        Path(request["root_dir"])
        / request["task_family"]
        / policy
        / _count_label(request["task_family"], request)
        / f"n{target}__{suffix}"
    )


def derived_dataset_name(output_dir: Path) -> str:
    parts = output_dir.parts[-5:]
    return "__".join(p.replace("/", "_") for p in parts)


def _validate_spec_range(spec: Any, lo: float, hi: float, field: str) -> None:
    if spec is None:
        return
    vals: list[float]
    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        vals = [float(spec["min"]), float(spec["max"])]
    elif isinstance(spec, list):
        vals = [float(v) for v in spec]
    else:
        vals = [float(spec)]
    bad = [v for v in vals if v < lo or v > hi]
    if bad:
        raise ValueError(f"{field} override {bad[0]} is outside official limits [{lo}, {hi}]")


def validate_override_limits(request: dict[str, Any]) -> None:
    scene = request.get("scene", {})
    nic = scene.get("nic_cards", {}) if isinstance(scene.get("nic_cards", {}), dict) else {}
    sc = scene.get("sc_ports", {}) if isinstance(scene.get("sc_ports", {}), dict) else {}
    mounts = scene.get("fixture_mounts", {}) if isinstance(scene.get("fixture_mounts", {}), dict) else {}
    _validate_spec_range(nic.get("translation"), *LIMITS["nic_translation"], "scene.nic_cards.translation")
    _validate_spec_range(nic.get("yaw_deg"), *LIMITS["nic_yaw_deg"], "scene.nic_cards.yaw_deg")
    _validate_spec_range(sc.get("translation"), *LIMITS["sc_translation"], "scene.sc_ports.translation")
    _validate_spec_range(mounts.get("translation"), *LIMITS["fixture_translation"], "scene.fixture_mounts.translation")
    _validate_spec_range(mounts.get("yaw_deg"), *LIMITS["fixture_yaw_deg"], "scene.fixture_mounts.yaw_deg")


def _deg_field(section: dict[str, Any], name: str, default_rad: Any, rng: random.Random) -> float:
    spec = section.get(f"{name}_deg")
    if spec is None:
        return round(float(sample_value(None, default_rad, rng)), 5)
    return round(math.radians(float(sample_value(spec, None, rng))), 5)


def _range_or_fixed(value: Any) -> tuple[float, float] | float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if float(value[0]) == float(value[1]):
            return float(value[0])
        return (float(value[0]), float(value[1]))
    return float(value)


def _sample_pose_on_rail(
    section: dict[str, Any],
    defaults: dict[str, Any],
    translation_default: tuple[float, float],
    rng: random.Random,
) -> dict[str, float]:
    return {
        "translation": round(float(sample_value(section.get("translation"), translation_default, rng)), 5),
        "roll": _deg_field(section, "roll", defaults["roll"], rng),
        "pitch": _deg_field(section, "pitch", defaults["pitch"], rng),
        "yaw": _deg_field(section, "yaw", defaults["yaw"], rng),
    }


def _selected_rails(section: dict[str, Any], all_rails: list[str], count: int, rng: random.Random) -> list[str]:
    rails_spec = section.get("rails", all_rails)
    rails = list(rails_spec)
    unknown = sorted(set(rails) - set(all_rails))
    if unknown:
        raise ValueError(f"Unknown rails in request: {unknown}")
    if count > len(rails):
        raise ValueError(f"Requested count {count} exceeds available rails {rails}")
    return sorted(rng.sample(rails, count))


def _target_index(target_spec: Any, present_rails: list[str], prefix: str, rng: random.Random) -> int:
    if target_spec in (None, "auto"):
        rail = rng.choice(present_rails)
    elif isinstance(target_spec, int):
        rail = f"{prefix}_{target_spec}"
    else:
        text = str(target_spec)
        rail = text if text.startswith(prefix) else f"{prefix}_{text.split('_')[-1]}"
    if rail not in present_rails:
        raise ValueError(f"Target {target_spec!r} must be among present rails {present_rails}")
    return int(rail.rsplit("_", 1)[1])


def _apply_board_overrides(trial: dict[str, Any], section: dict[str, Any], rng: random.Random) -> None:
    if not section:
        return
    pose = trial["scene"]["task_board"]["pose"]
    for key in ("x", "y", "z"):
        if key in section:
            pose[key] = round(float(sample_value(section[key], pose[key], rng)), 5)
    for key in ("roll", "pitch", "yaw"):
        deg_key = f"{key}_deg"
        if deg_key in section:
            pose[key] = round(math.radians(float(sample_value(section[deg_key], None, rng))), 5)


def _apply_nic_overrides(
    trial: dict[str, Any],
    section: dict[str, Any],
    required_exact: bool,
    profile_cfg: dict[str, Any],
    rng: random.Random,
) -> int | None:
    if "count" not in section and not required_exact:
        return None
    count = _as_count(sample_value(section.get("count"), None, rng), "scene.nic_cards.count")
    present = _selected_rails(section, NIC_RAILS, count, rng)
    nic_defaults = profile_cfg["nic_pose"]
    for rail in NIC_RAILS:
        if rail not in present:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        idx = int(rail.rsplit("_", 1)[1])
        trial["scene"]["task_board"][rail] = {
            "entity_present": True,
            "entity_name": f"nic_card_{idx}",
            "entity_pose": _sample_pose_on_rail(section, nic_defaults, LIMITS["nic_translation"], rng),
        }
    if not present:
        return None
    return _target_index(section.get("target_card"), present, "nic_rail", rng)


def _apply_sc_overrides(
    trial: dict[str, Any],
    section: dict[str, Any],
    required_exact: bool,
    profile_cfg: dict[str, Any],
    rng: random.Random,
) -> int | None:
    if "count" not in section and not required_exact:
        return None
    count = _as_count(sample_value(section.get("count"), None, rng), "scene.sc_ports.count")
    present = _selected_rails(section, SC_RAILS, count, rng)
    sc_defaults = profile_cfg["sc_pose"]
    for rail in SC_RAILS:
        if rail not in present:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        idx = int(rail.rsplit("_", 1)[1])
        trial["scene"]["task_board"][rail] = {
            "entity_present": True,
            "entity_name": f"sc_mount_{idx}",
            "entity_pose": _sample_pose_on_rail(section, sc_defaults, LIMITS["sc_translation"], rng),
        }
    if not present:
        return None
    return _target_index(section.get("target_port"), present, "sc_rail", rng)


def _apply_fixture_mount_overrides(
    trial: dict[str, Any], section: dict[str, Any], profile_cfg: dict[str, Any], rng: random.Random
) -> None:
    if not section:
        return
    rails = list(section.get("rails", MOUNT_RAILS))
    unknown = sorted(set(rails) - set(MOUNT_RAILS))
    if unknown:
        raise ValueError(f"Unknown fixture_mounts rails in request: {unknown}")
    present_prob = float(sample_value(section.get("present_probability"), profile_cfg["mount_pose"]["present_prob"], rng))
    for rail in MOUNT_RAILS:
        if rail not in rails:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        if rng.random() > present_prob:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        trial["scene"]["task_board"][rail] = {
            "entity_present": True,
            "entity_name": MOUNT_ENTITY_NAMES[rail],
            "entity_pose": _sample_pose_on_rail(section, profile_cfg["mount_pose"], LIMITS["fixture_translation"], rng),
        }


def _apply_family_task_and_cable(
    trial: dict[str, Any],
    task_family: str,
    target_nic: int | None,
    target_sc: int | None,
    scene: dict[str, Any],
    rng: random.Random,
) -> None:
    cable_section = scene.get("cable", {}) if isinstance(scene.get("cable", {}), dict) else {}
    if task_family == "sfp_to_nic":
        if target_nic is None:
            raise ValueError("sfp_to_nic requires at least one present NIC card")
        cable_name = "cable_0"
        cable_type = str(sample_value(cable_section.get("cable_type"), "sfp_sc_cable", rng))
        port_name = str(sample_value(scene.get("nic_cards", {}).get("target_port"), "sfp_port_0", rng))
        if port_name == "auto":
            port_name = rng.choice(["sfp_port_0", "sfp_port_1"])
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": port_name,
            "target_module_name": f"nic_card_mount_{target_nic}",
            "time_limit": 180,
        }
        default_offset = {"x": 0.0, "y": 0.015385, "z": 0.04245}
    else:
        if target_sc is None:
            raise ValueError("sc_to_sc requires at least one present SC port")
        cable_name = "cable_1"
        cable_type = str(sample_value(cable_section.get("cable_type"), "sfp_sc_cable_reversed", rng))
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{target_sc}",
            "time_limit": 180,
        }
        default_offset = {"x": 0.0, "y": 0.015385, "z": 0.04045}

    existing_pose = next(iter(trial["scene"]["cables"].values()))["pose"]
    offset_section = cable_section.get("gripper_offset", {})
    pose = {
        "gripper_offset": {
            axis: round(float(sample_value(offset_section.get(axis), default_offset[axis], rng)), 5)
            for axis in ("x", "y", "z")
        },
        "roll": round(math.radians(float(sample_value(cable_section.get("roll_deg"), math.degrees(existing_pose["roll"]), rng))), 5),
        "pitch": round(math.radians(float(sample_value(cable_section.get("pitch_deg"), math.degrees(existing_pose["pitch"]), rng))), 5),
        "yaw": round(math.radians(float(sample_value(cable_section.get("yaw_deg"), math.degrees(existing_pose["yaw"]), rng))), 5),
    }
    trial["scene"]["cables"] = {
        cable_name: {
            "pose": pose,
            "attach_cable_to_gripper": True,
            "cable_type": cable_type,
        }
    }
    trial["tasks"] = {"task_1": task}


def generate_trials(request: dict[str, Any], num_trials: int) -> dict[str, Any]:
    base = yaml.safe_load(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    limits = copy.deepcopy(base.get("task_board_limits", {}))
    profile_cfg = _profile_defaults(PROFILE_QUALIFICATION_EVAL_LIKE)
    rng = random.Random(request.get("generation", {}).get("seed"))
    scene = request.get("scene", {})
    generated: dict[str, Any] = {}
    for idx in range(1, num_trials + 1):
        raw = _build_trial(
            rng,
            idx,
            limits,
            profile_cfg=profile_cfg,
            sfp_to_nic_weight=1.0 if request["task_family"] == "sfp_to_nic" else 0.0,
            sc_to_sc_weight=1.0 if request["task_family"] == "sc_to_sc" else 0.0,
        )
        _apply_board_overrides(raw, scene.get("board", {}), rng)
        target_nic = _apply_nic_overrides(
            raw,
            scene.get("nic_cards", {}),
            request["task_family"] == "sfp_to_nic",
            profile_cfg,
            rng,
        )
        target_sc = _apply_sc_overrides(
            raw,
            scene.get("sc_ports", {}),
            request["task_family"] == "sc_to_sc",
            profile_cfg,
            rng,
        )
        _apply_fixture_mount_overrides(raw, scene.get("fixture_mounts", {}), profile_cfg, rng)
        _apply_family_task_and_cable(raw, request["task_family"], target_nic, target_sc, scene, rng)
        generated[f"trial_{idx:06d}"] = raw
    return generated


def write_engine_configs(request: dict[str, Any], output_dir: Path, num_trials: int) -> Path:
    base = yaml.safe_load(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    trials = generate_trials(request, num_trials)
    engine_config = copy.deepcopy(base)
    engine_config["trials"] = trials
    engine_config["generated"] = {
        "script": "aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py",
        "seed": request.get("generation", {}).get("seed"),
        "task_family": request["task_family"],
        "num_trials": num_trials,
        "request_yaml_semantics": {
            "missing": "team default randomization from generate_random_trials_config.py",
            "list": "uniform categorical choice",
            "min_max": "continuous uniform range; min == max is fixed",
            "degrees": "request fields ending in _deg are converted to radians in engine_config.yaml",
        },
    }
    out_path = output_dir / "engine_config.yaml"
    out_path.write_text(yaml.safe_dump(engine_config, sort_keys=False), encoding="utf-8")
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    for trial_id, trial in trials.items():
        single = copy.deepcopy(engine_config)
        single["trials"] = {trial_id: trial}
        (trials_dir / f"{trial_id}.yaml").write_text(
            yaml.safe_dump(single, sort_keys=False), encoding="utf-8"
        )
    return out_path


def run_command(cmd: list[str], dry_run: bool) -> dict[str, Any]:
    rendered = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {rendered}")
        return {"cmd": cmd, "skipped": True, "returncode": None}
    print(f"[run] {rendered}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {rendered}")
    return {"cmd": cmd, "skipped": False, "returncode": result.returncode}


def count_selected(selection_report: Path) -> int | None:
    if not selection_report.exists():
        return None
    with selection_report.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for row in csv.DictReader(f) if str(row.get("selected", "")).lower() == "true")


def compare_reference(local_dataset: Path, reference_repo_id: str) -> dict[str, Any]:
    if not (local_dataset / "meta" / "info.json").exists():
        return {"status": "skipped", "reason": f"missing {local_dataset / 'meta' / 'info.json'}"}
    cmd = [
        "pixi",
        "run",
        "aic-validate-dataset-compat",
        f"--base.repo_id={reference_repo_id}",
        f"--candidate.repo_id={local_dataset.name}",
        f"--candidate.root={local_dataset}",
        "--json",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "status": "succeeded" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "cmd": cmd,
    }


def main() -> int:
    args = parse_args()
    request = load_request(args.request_yaml)
    if args.target_accepted_override is not None:
        request.setdefault("generation", {})["target_accepted_trajectories"] = args.target_accepted_override
    if args.max_attempts_override is not None:
        request.setdefault("generation", {})["max_attempts"] = args.max_attempts_override
    validate_request(request)
    validate_override_limits(request)

    target = int(request["generation"]["target_accepted_trajectories"])
    max_attempts = int(request["generation"]["max_attempts"])
    if target <= 0 or max_attempts <= 0:
        raise ValueError("target_accepted_trajectories and max_attempts must be > 0")
    num_trials = args.num_trials_override or max_attempts
    if num_trials <= 0:
        raise ValueError("--num-trials-override must be > 0")
    if num_trials > max_attempts:
        raise ValueError("--num-trials-override cannot exceed generation.max_attempts")

    output_dir = derive_output_dir(request)
    append_if_exists = bool(request.get("generation", {}).get("append_if_exists", False))
    if output_dir.exists() and not append_if_exists and not args.dry_run:
        raise FileExistsError(f"Output directory exists and append_if_exists is false: {output_dir}")
    for child in ("scores", "trials", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    if args.dry_run or args.skip_recording:
        for child in ("raw_dataset", "accepted_dataset"):
            (output_dir / child).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.request_yaml, output_dir / "request.yaml")
    engine_config_path = write_engine_configs(request, output_dir, num_trials)

    commands: list[dict[str, Any]] = []
    dataset_repo_id = f"local/{derived_dataset_name(output_dir)}"
    recording_cmd = [
        "bash",
        str(REPO_ROOT / "aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh"),
        "--engine-config",
        str(engine_config_path),
        "--dataset-repo-id",
        dataset_repo_id,
        "--dataset-root",
        str(output_dir / "raw_dataset"),
        "--results-root",
        str(output_dir / "scores"),
        "--policy-class",
        POLICY_CLASS[request["generation"]["policy"]],
        "--gazebo-gui",
        "false",
        "--launch-rviz",
        "false",
        "--require-recorder-save-log",
        "true",
        "--remove-bag-data",
        "true",
        "--tmp-dir",
        str(output_dir / "logs" / "per_trial_tmp"),
    ]
    if not args.skip_recording:
        commands.append(run_command(recording_cmd, args.dry_run))

    filter_cmd = [
        "pixi",
        "run",
        "python",
        str(REPO_ROOT / "aic_utils/lerobot_robot_aic/scripts/filter_merge_lerobot_by_score.py"),
        "--datasets",
        str(output_dir / "raw_dataset"),
        "--score-csvs",
        str(output_dir / "scores" / "score_summary.csv"),
        "--min-score",
        str(float(request["acceptance"]["min_score"])),
        "--output",
        str(output_dir / "accepted_dataset"),
        "--include-videos",
        "--overwrite",
    ]
    can_filter_existing = (
        (output_dir / "raw_dataset" / "meta" / "info.json").exists()
        and (output_dir / "scores" / "score_summary.csv").exists()
    )
    if not args.skip_filter and (not args.skip_recording or can_filter_existing):
        commands.append(run_command(filter_cmd, args.dry_run))

    report_src = output_dir / "accepted_dataset" / "selection_report.csv"
    if report_src.exists():
        shutil.copy2(report_src, output_dir / "selection_report.csv")
    accepted = count_selected(report_src)
    schema_comparison = None
    if args.inspect_reference_dataset:
        schema_comparison = compare_reference(output_dir / "accepted_dataset", args.inspect_reference_dataset)

    summary = {
        "request_yaml": str(args.request_yaml),
        "output_dir": str(output_dir),
        "task_family": request["task_family"],
        "policy": request["generation"]["policy"],
        "count_label": _count_label(request["task_family"], request),
        "target_accepted_trajectories": target,
        "max_attempts": max_attempts,
        "min_score": float(request["acceptance"]["min_score"]),
        "seed": request.get("generation", {}).get("seed"),
        "raw_dataset": str(output_dir / "raw_dataset"),
        "accepted_dataset": str(output_dir / "accepted_dataset"),
        "scores": str(output_dir / "scores"),
        "number_attempted": num_trials,
        "number_accepted": accepted,
        "generated_engine_config": str(engine_config_path),
        "command_lines_run": commands,
        "schema_comparison": schema_comparison,
        "notes": {
            "dataset_format": "raw_dataset and accepted_dataset are native LeRobot dataset roots.",
            "cable_jitter": (
                "Missing cable fields inherit the existing generate_random_trials_config.py "
                "internal cable jitter. Explicit cable fields in request YAML override it."
            ),
            "attempt_strategy": "This first implementation generates max_attempts upfront unless --num-trials-override is used.",
        },
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote trajectory dataset request artifacts under: {output_dir}")
    if accepted is not None and accepted < target:
        print(f"Accepted {accepted}/{target}; generate additional attempts or adjust acceptance criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

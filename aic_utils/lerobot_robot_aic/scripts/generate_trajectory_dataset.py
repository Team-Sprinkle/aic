#!/usr/bin/env python3
"""Generate user-facing trajectory datasets with the LeRobot recorder pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
PACKAGE_ROOT = REPO_ROOT / "aic_utils" / "lerobot_robot_aic"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(ENGINE_SCRIPT_DIR))

from generate_random_trials_config import (  # noqa: E402
    PROFILE_QUALIFICATION_EVAL_LIKE,
    _build_trial,
    _profile_defaults,
)
from lerobot_robot_aic.task_metadata import TASK_VECTOR_NAMES, task_vector_from_fields  # noqa: E402

TASK_FAMILIES = {"sfp_to_nic", "sc_to_sc"}
POLICY_CLASS = {"cheatcode": "aic_example_policies.ros.CheatCode"}
ACTION_MODES = {"cartesian", "joint"}
TASK_CABLE_TYPES = {
    "sfp_to_nic": "sfp_sc_cable",
    "sc_to_sc": "sfp_sc_cable_reversed",
}
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
    action_mode = request.get("generation", {}).get("action_mode", "cartesian")
    if action_mode not in ACTION_MODES:
        raise ValueError(f"Unsupported generation.action_mode '{action_mode}'. Supported: {sorted(ACTION_MODES)}")
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
    explicit_output_dir = request.get("output_dir")
    if explicit_output_dir:
        return Path(str(explicit_output_dir))
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


def derived_dataset_repo_id(output_dir: Path) -> str:
    name = derived_dataset_name(output_dir)
    max_name_len = 96 - len("local/")
    if len(name) > max_name_len:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: max_name_len - len(digest) - 1]}_{digest}"
    return f"local/{name}"


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


def _rail_from_target(target_spec: Any, prefix: str) -> str | None:
    if target_spec in (None, "auto"):
        return None
    if isinstance(target_spec, int):
        return f"{prefix}_{target_spec}"
    text = str(target_spec)
    return text if text.startswith(prefix) else f"{prefix}_{text.rsplit('_', 1)[-1]}"


def _selected_rails(
    section: dict[str, Any],
    all_rails: list[str],
    count: int,
    rng: random.Random,
    *,
    required_rail: str | None = None,
) -> list[str]:
    rails_spec = section.get("rails", all_rails)
    rails = list(rails_spec)
    unknown = sorted(set(rails) - set(all_rails))
    if unknown:
        raise ValueError(f"Unknown rails in request: {unknown}")
    if count > len(rails):
        raise ValueError(f"Requested count {count} exceeds available rails {rails}")
    if required_rail is None:
        return sorted(rng.sample(rails, count))
    if required_rail not in rails:
        raise ValueError(f"Target rail {required_rail!r} must be among candidate rails {rails}")
    if count < 1:
        raise ValueError(f"Requested count {count} cannot include target rail {required_rail!r}")
    remaining = [rail for rail in rails if rail != required_rail]
    return sorted([required_rail, *rng.sample(remaining, count - 1)])


def _target_index(target_spec: Any, present_rails: list[str], prefix: str, rng: random.Random) -> int:
    rail = _rail_from_target(target_spec, prefix)
    if rail is None:
        rail = rng.choice(present_rails)
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
    required_rail = _rail_from_target(section.get("target_card"), "nic_rail")
    present = _selected_rails(section, NIC_RAILS, count, rng, required_rail=required_rail)
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
    required_rail = _rail_from_target(section.get("target_port"), "sc_rail")
    present = _selected_rails(section, SC_RAILS, count, rng, required_rail=required_rail)
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
    expected_cable_type = TASK_CABLE_TYPES[task_family]
    if task_family == "sfp_to_nic":
        if target_nic is None:
            raise ValueError("sfp_to_nic requires at least one present NIC card")
        cable_name = "cable_0"
        cable_type = str(sample_value(cable_section.get("cable_type"), expected_cable_type, rng))
        target_port_spec = scene.get("nic_cards", {}).get("target_port")
        if target_port_spec in (None, "auto"):
            port_name = rng.choice(["sfp_port_0", "sfp_port_1"])
        else:
            port_name = str(sample_value(target_port_spec, None, rng))
        if port_name not in {"sfp_port_0", "sfp_port_1"}:
            raise ValueError(f"sfp_to_nic target_port must be sfp_port_0 or sfp_port_1, got {port_name!r}")
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
        cable_type = str(sample_value(cable_section.get("cable_type"), expected_cable_type, rng))
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

    if cable_type != expected_cable_type:
        raise ValueError(
            f"{task_family} requires scene.cable.cable_type={expected_cable_type!r}; "
            f"got {cable_type!r}"
        )

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


def write_engine_configs(request: dict[str, Any], output_dir: Path, num_trials: int) -> tuple[Path, dict[str, Any]]:
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
    return out_path, trials


def run_command(cmd: list[str], dry_run: bool, *, check: bool = True) -> dict[str, Any]:
    rendered = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {rendered}")
        return {"cmd": cmd, "skipped": True, "returncode": None}
    print(f"[run] {rendered}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {rendered}")
    if result.returncode != 0:
        print(f"[warn] Command failed with exit code {result.returncode}; continuing: {rendered}")
    return {"cmd": cmd, "skipped": False, "returncode": result.returncode}


def recording_outputs_complete(output_dir: Path, expected_trials: int) -> tuple[bool, str]:
    score_csv = output_dir / "scores" / "score_summary.csv"
    raw_info = output_dir / "raw_dataset" / "meta" / "info.json"
    if not score_csv.exists():
        return False, f"missing {score_csv}"
    if not raw_info.exists():
        return False, f"missing {raw_info}"
    with score_csv.open("r", encoding="utf-8", newline="") as f:
        attempted = sum(1 for _ in csv.DictReader(f))
    if attempted < expected_trials:
        return False, f"score_summary.csv has {attempted}/{expected_trials} trial rows"
    return True, f"score_summary.csv has {attempted}/{expected_trials} trial rows"


def count_selected(selection_report: Path) -> int | None:
    if not selection_report.exists():
        return None
    with selection_report.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for row in csv.DictReader(f) if str(row.get("selected", "")).lower() == "true")


def _parse_index_from_suffix(text: str, prefix: str) -> int:
    if not text.startswith(prefix):
        raise ValueError(f"Expected {text!r} to start with {prefix!r}")
    return int(text.rsplit("_", 1)[1])


def _present_rails(task_board: dict[str, Any], rails: list[str]) -> list[str]:
    return [rail for rail in rails if task_board.get(rail, {}).get("entity_present")]


def _task_metadata_from_trial(task_family: str, trial: dict[str, Any]) -> dict[str, Any]:
    task = trial["tasks"]["task_1"]
    scene = trial["scene"]
    task_board = scene["task_board"]
    target_module_name = str(task["target_module_name"])
    port_name = str(task["port_name"])
    if task_family == "sfp_to_nic":
        target_card_index = _parse_index_from_suffix(target_module_name, "nic_card_mount_")
        target_port_index = _parse_index_from_suffix(port_name, "sfp_port_")
        target_card_valid = 1
    else:
        target_card_index = -1
        target_port_index = _parse_index_from_suffix(target_module_name, "sc_port_")
        target_card_valid = 0
    task_vector = task_vector_from_fields(task_family, target_port_index, target_card_index)
    cable = next(iter(scene.get("cables", {}).values()), {})
    return {
        "task": {
            "task_family": task_family,
            "plug_type": task["plug_type"],
            "plug_name": task["plug_name"],
            "port_type": task["port_type"],
            "port_name": port_name,
            "target_module_name": target_module_name,
            "target_port_index": target_port_index,
            "target_card_index": target_card_index,
            "target_card_valid": target_card_valid,
            "task_vector": task_vector,
        },
        "scene_summary": {
            "present_nic_rails": _present_rails(task_board, NIC_RAILS),
            "present_sc_rails": _present_rails(task_board, SC_RAILS),
            "board_pose": task_board.get("pose", {}),
            "cable_type": cable.get("cable_type"),
        },
    }


def _read_csv_by_trial(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row.get("trial_id", "")): dict(row) for row in csv.DictReader(f)}


def _read_selection_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv_by_trial(path)
    accepted_idx = 0
    for row in rows.values():
        if str(row.get("selected", "")).lower() != "true":
            row["accepted_episode_index"] = ""
            continue
        row["accepted_episode_index"] = str(accepted_idx)
        accepted_idx += 1
    return rows


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def write_task_manifests(
    output_dir: Path,
    task_family: str,
    trials: dict[str, Any],
    *,
    score_csv: Path | None = None,
    selection_report: Path | None = None,
) -> dict[str, str]:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    score_rows = _read_csv_by_trial(score_csv or output_dir / "scores" / "score_summary.csv")
    selection_rows = _read_selection_rows(selection_report or output_dir / "accepted_dataset" / "selection_report.csv")

    attempts_path = manifests_dir / "attempts.csv"
    accepted_path = manifests_dir / "accepted.csv"
    jsonl_path = manifests_dir / "episode_task_metadata.jsonl"
    vector_fieldnames = [name for name in TASK_VECTOR_NAMES if name != "target_card_valid"]
    fieldnames = [
        "run_index",
        "trial_id",
        "status",
        "total_score",
        "selected",
        "rejection_reason",
        "source_episode_index",
        "accepted_episode_index",
        "trial_yaml_path",
        "scoring_yaml_path",
        "task_family",
        "plug_type",
        "plug_name",
        "port_type",
        "port_name",
        "target_module_name",
        "cable_type",
        "target_port_index",
        "target_card_index",
        "target_card_valid",
        *vector_fieldnames,
        "task_vector",
    ]
    rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    for fallback_index, (trial_id, trial) in enumerate(sorted(trials.items()), start=1):
        score = score_rows.get(trial_id, {})
        selection = selection_rows.get(trial_id, {})
        metadata = _task_metadata_from_trial(task_family, trial)
        task = metadata["task"]
        scene_summary = metadata["scene_summary"]
        run_index = int(score.get("run_index") or fallback_index)
        source_episode_index = _optional_int(selection.get("mapped_episode_index") or score.get("episode_index"))
        if source_episode_index is None and (score or selection):
            source_episode_index = run_index - 1
        selected = str(selection.get("selected", "")).lower() == "true"
        accepted_episode_index = _optional_int(selection.get("accepted_episode_index"))
        total_score = score.get("total_score") or selection.get("total_score") or ""
        row = {
            "run_index": run_index,
            "trial_id": trial_id,
            "status": score.get("status", "dry_run" if not score_rows else ""),
            "total_score": total_score,
            "selected": selected,
            "rejection_reason": selection.get("reason", ""),
            "source_episode_index": source_episode_index if source_episode_index is not None else "",
            "accepted_episode_index": accepted_episode_index if accepted_episode_index is not None else "",
            "trial_yaml_path": str(output_dir / "trials" / f"{trial_id}.yaml"),
            "scoring_yaml_path": score.get("scoring_yaml", ""),
            "task_family": task["task_family"],
            "plug_type": task["plug_type"],
            "plug_name": task["plug_name"],
            "port_type": task["port_type"],
            "port_name": task["port_name"],
            "target_module_name": task["target_module_name"],
            "cable_type": scene_summary["cable_type"],
            "target_port_index": task["target_port_index"],
            "target_card_index": task["target_card_index"],
            "target_card_valid": task["target_card_valid"],
            "task_vector": json.dumps(task["task_vector"]),
        }
        for name, value in zip(TASK_VECTOR_NAMES, task["task_vector"], strict=True):
            row[name] = value
        rows.append(row)
        json_rows.append(
            {
                "trial_id": trial_id,
                "source_episode_index": source_episode_index,
                "accepted_episode_index": accepted_episode_index,
                "selected": selected,
                "status": row["status"],
                "total_score": float(total_score) if total_score not in ("", None) else None,
                "trial_yaml_path": row["trial_yaml_path"],
                "task": task,
                "scene_summary": scene_summary,
            }
        )

    with attempts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with accepted_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row for row in rows if row["selected"]])
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in json_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "attempts_csv": str(attempts_path),
        "accepted_csv": str(accepted_path),
        "episode_task_metadata_jsonl": str(jsonl_path),
    }


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
    if args.skip_recording and not args.dry_run:
        for child in ("raw_dataset", "accepted_dataset"):
            (output_dir / child).mkdir(parents=True, exist_ok=True)
    request_copy_path = output_dir / "request.yaml"
    if args.request_yaml.resolve() != request_copy_path.resolve():
        shutil.copy2(args.request_yaml, request_copy_path)
    engine_config_path, trials = write_engine_configs(request, output_dir, num_trials)

    commands: list[dict[str, Any]] = []
    dataset_repo_id = derived_dataset_repo_id(output_dir)
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
        "--action-mode",
        request.get("generation", {}).get("action_mode", "cartesian"),
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
    if "startup_delay_sec" in request.get("generation", {}):
        recording_cmd.extend(
            ["--startup-delay-sec", str(int(request["generation"]["startup_delay_sec"]))]
        )
    if "per_trial_timeout_sec" in request.get("generation", {}):
        recording_cmd.extend(
            ["--per-trial-timeout-sec", str(int(request["generation"]["per_trial_timeout_sec"]))]
        )
    if "restart_sim_container" in request.get("generation", {}):
        restart_sim_container = str(bool(request["generation"]["restart_sim_container"])).lower()
        recording_cmd.extend(["--restart-sim-container", restart_sim_container])
    if not args.skip_recording:
        recording_result = run_command(recording_cmd, args.dry_run, check=False)
        commands.append(recording_result)
        if recording_result["returncode"] not in (0, None):
            complete, reason = recording_outputs_complete(output_dir, num_trials)
            if not complete:
                rendered = " ".join(str(c) for c in recording_cmd)
                raise RuntimeError(
                    f"Recording command failed with exit code {recording_result['returncode']} "
                    f"and recording outputs are incomplete ({reason}): {rendered}"
                )
            print(
                "[warn] Recording command reported failed trials but completed all requested attempts; "
                f"continuing to filtering ({reason})."
            )

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
        "--max-selected-episodes",
        str(target),
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
    manifest_paths = write_task_manifests(
        output_dir,
        request["task_family"],
        trials,
        score_csv=output_dir / "scores" / "score_summary.csv",
        selection_report=report_src,
    )
    schema_comparison = None
    if args.inspect_reference_dataset:
        schema_comparison = compare_reference(output_dir / "accepted_dataset", args.inspect_reference_dataset)

    summary = {
        "request_yaml": str(args.request_yaml),
        "output_dir": str(output_dir),
        "task_family": request["task_family"],
        "policy": request["generation"]["policy"],
        "action_mode": request.get("generation", {}).get("action_mode", "cartesian"),
        "count_label": _count_label(request["task_family"], request),
        "target_accepted_trajectories": target,
        "max_attempts": max_attempts,
        "min_score": float(request["acceptance"]["min_score"]),
        "seed": request.get("generation", {}).get("seed"),
        "raw_dataset": str(output_dir / "raw_dataset"),
        "accepted_dataset": str(output_dir / "accepted_dataset"),
        "scores": str(output_dir / "scores"),
        "manifests": manifest_paths,
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

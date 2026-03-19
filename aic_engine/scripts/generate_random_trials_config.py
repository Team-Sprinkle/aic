#!/usr/bin/env python3

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Generate randomized AIC engine trial configs for data collection."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import yaml


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _rail_pose(rng: random.Random, lo: float, hi: float) -> dict[str, float]:
    return {
        "translation": round(rng.uniform(lo, hi), 5),
        "roll": round(rng.uniform(-0.08, 0.08), 5),
        "pitch": round(rng.uniform(-0.08, 0.08), 5),
        "yaw": round(rng.uniform(-0.2, 0.2), 5),
    }


def _maybe_mount(
    rng: random.Random,
    name: str,
    lo: float,
    hi: float,
    present_prob: float,
) -> dict[str, Any]:
    if rng.random() > present_prob:
        return {"entity_present": False}
    return {
        "entity_present": True,
        "entity_name": name,
        "entity_pose": _rail_pose(rng, lo, hi),
    }


def _rand_board_pose(rng: random.Random) -> dict[str, float]:
    return {
        "x": round(rng.uniform(0.10, 0.20), 5),
        "y": round(rng.uniform(-0.25, 0.20), 5),
        "z": 1.14,
        "roll": round(rng.uniform(-0.04, 0.04), 5),
        "pitch": round(rng.uniform(-0.04, 0.04), 5),
        "yaw": round(rng.uniform(2.85, 3.25), 5),
    }


def _build_trial(
    rng: random.Random,
    trial_idx: int,
    limits: dict[str, dict[str, float]],
) -> dict[str, Any]:
    nic_lo = float(limits["nic_rail"]["min_translation"])
    nic_hi = float(limits["nic_rail"]["max_translation"])
    sc_lo = float(limits["sc_rail"]["min_translation"])
    sc_hi = float(limits["sc_rail"]["max_translation"])
    mount_lo = float(limits["mount_rail"]["min_translation"])
    mount_hi = float(limits["mount_rail"]["max_translation"])

    scenario = rng.choice(["sfp_to_nic", "sc_to_sc"])
    target_nic = rng.randint(0, 4)
    target_sc = rng.randint(0, 1)

    task_board: dict[str, Any] = {"pose": _rand_board_pose(rng)}

    for i in range(5):
        key = f"nic_rail_{i}"
        if i == target_nic or rng.random() < 0.25:
            task_board[key] = {
                "entity_present": True,
                "entity_name": f"nic_card_{i}",
                "entity_pose": _rail_pose(rng, nic_lo, nic_hi),
            }
        else:
            task_board[key] = {"entity_present": False}

    for i in range(2):
        key = f"sc_rail_{i}"
        if i == target_sc or rng.random() < 0.25:
            task_board[key] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{i}",
                "entity_pose": _rail_pose(rng, sc_lo, sc_hi),
            }
        else:
            task_board[key] = {"entity_present": False}

    task_board["lc_mount_rail_0"] = _maybe_mount(
        rng, "lc_mount_0", mount_lo, mount_hi, present_prob=0.75
    )
    task_board["sfp_mount_rail_0"] = _maybe_mount(
        rng, "sfp_mount_0", mount_lo, mount_hi, present_prob=0.75
    )
    task_board["sc_mount_rail_0"] = _maybe_mount(
        rng, "sc_mount_0", mount_lo, mount_hi, present_prob=0.75
    )
    task_board["lc_mount_rail_1"] = _maybe_mount(
        rng, "lc_mount_1", mount_lo, mount_hi, present_prob=0.75
    )
    task_board["sfp_mount_rail_1"] = _maybe_mount(
        rng, "sfp_mount_1", mount_lo, mount_hi, present_prob=0.75
    )
    task_board["sc_mount_rail_1"] = _maybe_mount(
        rng, "sc_mount_1", mount_lo, mount_hi, present_prob=0.75
    )

    if scenario == "sfp_to_nic":
        cable_name = "cable_0"
        cable_type = "sfp_sc_cable"
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": "sfp_port_0",
            "target_module_name": f"nic_card_mount_{target_nic}",
            "time_limit": 180,
        }
        base_z = 0.04245
    else:
        cable_name = "cable_1"
        cable_type = "sfp_sc_cable_reversed"
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
        base_z = 0.04045

    cable_roll = 0.4432 + rng.uniform(-0.12, 0.12)
    cable_pitch = -0.4838 + rng.uniform(-0.12, 0.12)
    cable_yaw = 1.3303 + rng.uniform(-0.15, 0.15)
    cable_offset_x = _clamp(rng.uniform(-0.003, 0.003), -0.01, 0.01)
    cable_offset_y = _clamp(0.015385 + rng.uniform(-0.002, 0.002), -0.03, 0.03)
    cable_offset_z = _clamp(base_z + rng.uniform(-0.004, 0.004), 0.03, 0.06)

    trial: dict[str, Any] = {
        "scene": {
            "task_board": task_board,
            "cables": {
                cable_name: {
                    "pose": {
                        "gripper_offset": {
                            "x": round(cable_offset_x, 5),
                            "y": round(cable_offset_y, 5),
                            "z": round(cable_offset_z, 5),
                        },
                        "roll": round(cable_roll, 5),
                        "pitch": round(cable_pitch, 5),
                        "yaw": round(cable_yaw, 5),
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": cable_type,
                }
            },
        },
        "tasks": {"task_1": task},
    }
    _ = trial_idx
    return trial


def parse_args() -> argparse.Namespace:
    default_template = (
        Path(__file__).resolve().parents[1] / "config" / "sample_config.yaml"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Generate randomized aic_engine config with N trials for policy "
            "data collection."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=default_template,
        help=f"Template config path (default: {default_template})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output YAML file path.",
    )
    parser.add_argument(
        "--num_trials",
        "-n",
        type=int,
        required=True,
        help="Number of randomized trials to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible trial generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be > 0")

    with args.template.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    if not isinstance(base, dict):
        raise ValueError(f"Template config must be a YAML map: {args.template}")
    if "trials" not in base:
        raise ValueError(f"Template missing required key 'trials': {args.template}")

    limits = copy.deepcopy(base.get("task_board_limits", {}))
    defaults = {
        "nic_rail": {"min_translation": -0.048, "max_translation": 0.036},
        "sc_rail": {"min_translation": -0.055, "max_translation": 0.055},
        "mount_rail": {"min_translation": -0.09625, "max_translation": 0.09625},
    }
    for k, v in defaults.items():
        if k not in limits:
            limits[k] = copy.deepcopy(v)
            continue
        limits[k].setdefault("min_translation", v["min_translation"])
        limits[k].setdefault("max_translation", v["max_translation"])

    rng = random.Random(args.seed)

    generated_trials: dict[str, Any] = {}
    for idx in range(1, args.num_trials + 1):
        generated_trials[f"trial_{idx}"] = _build_trial(rng, idx, limits)

    out_cfg = copy.deepcopy(base)
    out_cfg["trials"] = generated_trials
    out_cfg["generated"] = {
        "script": "aic_engine/scripts/generate_random_trials_config.py",
        "seed": args.seed,
        "num_trials": args.num_trials,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print(f"Wrote randomized config: {args.output}")
    print(f"Trials generated: {args.num_trials}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

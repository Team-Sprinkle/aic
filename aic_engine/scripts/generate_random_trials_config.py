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
import math
import random
from pathlib import Path
from typing import Any

import yaml

PROFILE_QUALIFICATION_EVAL_LIKE = "qualification_eval_like"
PROFILE_TRAINING_BROAD = "training_broad"


def _profile_defaults(profile: str) -> dict[str, Any]:
    if profile == PROFILE_QUALIFICATION_EVAL_LIKE:
        # Rationale for eval-like bounds:
        # - docs/qualification_phase.md:
        #   - randomized task board pose/orientation each trial
        #   - NIC cards randomize translation + yaw offset
        #   - SC ports randomize translation along rail
        #   - grasp deviations are small (~2 mm, ~0.04 rad)
        # - aic_bringup/README.md:
        #   - evaluation keeps task board/components roll & pitch fixed to 0.0
        #   - SC port yaw fixed to 0.0 during evaluation
        # - docs/scoring.md + docs/qualification_phase.md:
        #   - robot starts close to target and target remains visible;
        #     keep board XY sampling conservative around nominal pose.
        return {
            "board_pose": {
                # Conservative eval-like board window centered near sample_config
                # nominal (x=0.15, y=-0.20, z=1.14), with yaw around pi.
                "x": (0.12, 0.18),
                "y": (-0.24, -0.12),
                "z": 1.14,
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (2.95, 3.20),
            },
            "board_nominal_xy": (0.15, -0.20),
            # Reject extreme XY samples to keep "starts within a few cm" / in-view
            # behavior closer to evaluation descriptions.
            "max_board_offset_xy": 0.10,
            "nic_pose": {
                # NIC roll/pitch fixed for eval-like mode; only yaw offset randomized.
                # ±10 deg comes from task_board_description NIC orientation limits.
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.radians(10.0), math.radians(10.0)),
                "extra_present_prob": 0.25,
            },
            "sc_pose": {
                # SC ports: translation randomized, orientation fixed in eval-like mode.
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                "extra_present_prob": 0.25,
            },
            "mount_pose": {
                # Evaluation notes specify fixed roll/pitch (and SC port yaw fixed).
                # Keep mount orientation fixed for better scoring-environment match.
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                "present_prob": 0.75,
            },
            "cable_jitter": {
                # Qualification docs: small grasp perturbations (~2 mm, ~0.04 rad).
                "roll": 0.04,
                "pitch": 0.04,
                "yaw": 0.04,
                "offset_x": 0.002,
                "offset_y": 0.002,
                "offset_z": 0.002,
            },
        }

    if profile == PROFILE_TRAINING_BROAD:
        return {
            "board_pose": {
                "x": (0.10, 0.20),
                "y": (-0.25, 0.20),
                "z": 1.14,
                "roll": (-0.04, 0.04),
                "pitch": (-0.04, 0.04),
                "yaw": (2.85, 3.25),
            },
            "board_nominal_xy": (0.15, -0.20),
            "max_board_offset_xy": None,
            "nic_pose": {
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.2, 0.2),
                "extra_present_prob": 0.25,
            },
            "sc_pose": {
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.2, 0.2),
                "extra_present_prob": 0.25,
            },
            "mount_pose": {
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.2, 0.2),
                "present_prob": 0.75,
            },
            "cable_jitter": {
                "roll": 0.12,
                "pitch": 0.12,
                "yaw": 0.15,
                "offset_x": 0.003,
                "offset_y": 0.002,
                "offset_z": 0.004,
            },
        }

    raise ValueError(f"Unsupported profile: {profile}")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _rand_ranged(
    rng: random.Random, lo_hi: tuple[float, float] | list[float]
) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return rng.uniform(lo, hi)


def _rail_pose(
    rng: random.Random,
    lo: float,
    hi: float,
    roll_range: tuple[float, float],
    pitch_range: tuple[float, float],
    yaw_range: tuple[float, float],
) -> dict[str, float]:
    return {
        "translation": round(rng.uniform(lo, hi), 5),
        "roll": round(_rand_ranged(rng, roll_range), 5),
        "pitch": round(_rand_ranged(rng, pitch_range), 5),
        "yaw": round(_rand_ranged(rng, yaw_range), 5),
    }


def _maybe_mount(
    rng: random.Random,
    name: str,
    lo: float,
    hi: float,
    present_prob: float,
    roll_range: tuple[float, float],
    pitch_range: tuple[float, float],
    yaw_range: tuple[float, float],
) -> dict[str, Any]:
    if rng.random() > present_prob:
        return {"entity_present": False}
    return {
        "entity_present": True,
        "entity_name": name,
        "entity_pose": _rail_pose(
            rng, lo, hi, roll_range=roll_range, pitch_range=pitch_range, yaw_range=yaw_range
        ),
    }


def _rand_board_pose(rng: random.Random, board_cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round(_rand_ranged(rng, board_cfg["x"]), 5),
        "y": round(_rand_ranged(rng, board_cfg["y"]), 5),
        "z": float(board_cfg["z"]),
        "roll": round(_rand_ranged(rng, board_cfg["roll"]), 5),
        "pitch": round(_rand_ranged(rng, board_cfg["pitch"]), 5),
        "yaw": round(_rand_ranged(rng, board_cfg["yaw"]), 5),
    }


def _build_trial(
    rng: random.Random,
    trial_idx: int,
    limits: dict[str, dict[str, float]],
    profile_cfg: dict[str, Any],
    sfp_to_nic_weight: float,
    sc_to_sc_weight: float,
) -> dict[str, Any]:
    nic_lo = float(limits["nic_rail"]["min_translation"])
    nic_hi = float(limits["nic_rail"]["max_translation"])
    sc_lo = float(limits["sc_rail"]["min_translation"])
    sc_hi = float(limits["sc_rail"]["max_translation"])
    mount_lo = float(limits["mount_rail"]["min_translation"])
    mount_hi = float(limits["mount_rail"]["max_translation"])

    scenario = rng.choices(
        ["sfp_to_nic", "sc_to_sc"],
        weights=[sfp_to_nic_weight, sc_to_sc_weight],
        k=1,
    )[0]
    target_nic = rng.randint(0, 4)
    target_sc = rng.randint(0, 1)
    target_sfp_port_name = rng.choice(["sfp_port_0", "sfp_port_1"])

    board_cfg = profile_cfg["board_pose"]
    nominal_x, nominal_y = profile_cfg["board_nominal_xy"]
    max_board_offset_xy = profile_cfg["max_board_offset_xy"]

    # Keep board samples conservative around nominal pose for eval-like realism.
    board_pose: dict[str, float] | None = None
    for _ in range(128):
        candidate = _rand_board_pose(rng, board_cfg)
        if max_board_offset_xy is None:
            board_pose = candidate
            break
        dx = candidate["x"] - nominal_x
        dy = candidate["y"] - nominal_y
        if math.hypot(dx, dy) <= max_board_offset_xy:
            board_pose = candidate
            break
    if board_pose is None:
        raise RuntimeError(
            "Unable to sample a board pose satisfying offset constraints. "
            "Adjust board ranges or max offset."
        )

    task_board: dict[str, Any] = {"pose": board_pose}

    nic_pose_cfg = profile_cfg["nic_pose"]
    sc_pose_cfg = profile_cfg["sc_pose"]
    mount_pose_cfg = profile_cfg["mount_pose"]

    for i in range(5):
        key = f"nic_rail_{i}"
        if i == target_nic or rng.random() < nic_pose_cfg["extra_present_prob"]:
            task_board[key] = {
                "entity_present": True,
                "entity_name": f"nic_card_{i}",
                "entity_pose": _rail_pose(
                    rng,
                    nic_lo,
                    nic_hi,
                    roll_range=nic_pose_cfg["roll"],
                    pitch_range=nic_pose_cfg["pitch"],
                    yaw_range=nic_pose_cfg["yaw"],
                ),
            }
        else:
            task_board[key] = {"entity_present": False}

    for i in range(2):
        key = f"sc_rail_{i}"
        if i == target_sc or rng.random() < sc_pose_cfg["extra_present_prob"]:
            task_board[key] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{i}",
                "entity_pose": _rail_pose(
                    rng,
                    sc_lo,
                    sc_hi,
                    roll_range=sc_pose_cfg["roll"],
                    pitch_range=sc_pose_cfg["pitch"],
                    yaw_range=sc_pose_cfg["yaw"],
                ),
            }
        else:
            task_board[key] = {"entity_present": False}

    task_board["lc_mount_rail_0"] = _maybe_mount(
        rng,
        "lc_mount_0",
        mount_lo,
        mount_hi,
        present_prob=mount_pose_cfg["present_prob"],
        roll_range=mount_pose_cfg["roll"],
        pitch_range=mount_pose_cfg["pitch"],
        yaw_range=mount_pose_cfg["yaw"],
    )
    task_board["sfp_mount_rail_0"] = _maybe_mount(
        rng,
        "sfp_mount_0",
        mount_lo,
        mount_hi,
        present_prob=mount_pose_cfg["present_prob"],
        roll_range=mount_pose_cfg["roll"],
        pitch_range=mount_pose_cfg["pitch"],
        yaw_range=mount_pose_cfg["yaw"],
    )
    task_board["sc_mount_rail_0"] = _maybe_mount(
        rng,
        "sc_mount_0",
        mount_lo,
        mount_hi,
        present_prob=mount_pose_cfg["present_prob"],
        roll_range=mount_pose_cfg["roll"],
        pitch_range=mount_pose_cfg["pitch"],
        yaw_range=mount_pose_cfg["yaw"],
    )
    task_board["lc_mount_rail_1"] = _maybe_mount(
        rng,
        "lc_mount_1",
        mount_lo,
        mount_hi,
        present_prob=mount_pose_cfg["present_prob"],
        roll_range=mount_pose_cfg["roll"],
        pitch_range=mount_pose_cfg["pitch"],
        yaw_range=mount_pose_cfg["yaw"],
    )
    task_board["sfp_mount_rail_1"] = _maybe_mount(
        rng,
        "sfp_mount_1",
        mount_lo,
        mount_hi,
        present_prob=mount_pose_cfg["present_prob"],
        roll_range=mount_pose_cfg["roll"],
        pitch_range=mount_pose_cfg["pitch"],
        yaw_range=mount_pose_cfg["yaw"],
    )
    task_board["sc_mount_rail_1"] = _maybe_mount(
        rng,
        "sc_mount_1",
        mount_lo,
        mount_hi,
        present_prob=mount_pose_cfg["present_prob"],
        roll_range=mount_pose_cfg["roll"],
        pitch_range=mount_pose_cfg["pitch"],
        yaw_range=mount_pose_cfg["yaw"],
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
            "port_name": target_sfp_port_name,
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

    cable_jitter_cfg = profile_cfg["cable_jitter"]
    cable_roll = 0.4432 + rng.uniform(-cable_jitter_cfg["roll"], cable_jitter_cfg["roll"])
    cable_pitch = -0.4838 + rng.uniform(
        -cable_jitter_cfg["pitch"], cable_jitter_cfg["pitch"]
    )
    cable_yaw = 1.3303 + rng.uniform(-cable_jitter_cfg["yaw"], cable_jitter_cfg["yaw"])
    cable_offset_x = _clamp(
        rng.uniform(-cable_jitter_cfg["offset_x"], cable_jitter_cfg["offset_x"]),
        -0.01,
        0.01,
    )
    cable_offset_y = _clamp(
        0.015385 + rng.uniform(-cable_jitter_cfg["offset_y"], cable_jitter_cfg["offset_y"]),
        -0.03,
        0.03,
    )
    cable_offset_z = _clamp(
        base_z + rng.uniform(-cable_jitter_cfg["offset_z"], cable_jitter_cfg["offset_z"]),
        0.03,
        0.06,
    )

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
        help=(
            "Number of randomized board setups to generate. "
            "Total trial count is num_trials * episodes_per_setup."
        ),
    )
    parser.add_argument(
        "--episodes_per_setup",
        type=int,
        default=1,
        help=(
            "Number of repeated episodes to run for each randomized board setup "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible trial generation.",
    )
    parser.add_argument(
        "--profile",
        choices=[PROFILE_QUALIFICATION_EVAL_LIKE, PROFILE_TRAINING_BROAD],
        default=PROFILE_QUALIFICATION_EVAL_LIKE,
        help=(
            "Randomization profile. "
            f"'{PROFILE_QUALIFICATION_EVAL_LIKE}' matches qualification docs "
            "more closely; "
            f"'{PROFILE_TRAINING_BROAD}' uses wider domain randomization."
        ),
    )
    parser.add_argument(
        "--sfp_to_nic_weight",
        type=float,
        default=2.0,
        help="Relative sampling weight for SFP->NIC scenarios (default: 2.0).",
    )
    parser.add_argument(
        "--sc_to_sc_weight",
        type=float,
        default=1.0,
        help="Relative sampling weight for SC->SC scenarios (default: 1.0).",
    )
    parser.add_argument(
        "--board_x_min",
        type=float,
        default=None,
        help="Optional override for board x min bound.",
    )
    parser.add_argument(
        "--board_x_max",
        type=float,
        default=None,
        help="Optional override for board x max bound.",
    )
    parser.add_argument(
        "--board_y_min",
        type=float,
        default=None,
        help="Optional override for board y min bound.",
    )
    parser.add_argument(
        "--board_y_max",
        type=float,
        default=None,
        help="Optional override for board y max bound.",
    )
    parser.add_argument(
        "--board_yaw_min",
        type=float,
        default=None,
        help="Optional override for board yaw min bound.",
    )
    parser.add_argument(
        "--board_yaw_max",
        type=float,
        default=None,
        help="Optional override for board yaw max bound.",
    )
    parser.add_argument(
        "--max_board_offset_xy",
        type=float,
        default=None,
        help=(
            "Optional max XY distance from nominal board pose for acceptance. "
            "Overrides profile default when provided."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be > 0")
    if args.episodes_per_setup <= 0:
        raise ValueError("--episodes_per_setup must be > 0")
    if args.sfp_to_nic_weight <= 0:
        raise ValueError("--sfp_to_nic_weight must be > 0")
    if args.sc_to_sc_weight <= 0:
        raise ValueError("--sc_to_sc_weight must be > 0")

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

    profile_cfg = _profile_defaults(args.profile)
    board_cfg = profile_cfg["board_pose"]
    if args.board_x_min is not None:
        board_cfg["x"] = (args.board_x_min, board_cfg["x"][1])
    if args.board_x_max is not None:
        board_cfg["x"] = (board_cfg["x"][0], args.board_x_max)
    if args.board_y_min is not None:
        board_cfg["y"] = (args.board_y_min, board_cfg["y"][1])
    if args.board_y_max is not None:
        board_cfg["y"] = (board_cfg["y"][0], args.board_y_max)
    if args.board_yaw_min is not None:
        board_cfg["yaw"] = (args.board_yaw_min, board_cfg["yaw"][1])
    if args.board_yaw_max is not None:
        board_cfg["yaw"] = (board_cfg["yaw"][0], args.board_yaw_max)
    if (
        board_cfg["x"][0] > board_cfg["x"][1]
        or board_cfg["y"][0] > board_cfg["y"][1]
        or board_cfg["yaw"][0] > board_cfg["yaw"][1]
    ):
        raise ValueError("Invalid board bounds: min must be <= max")
    if args.max_board_offset_xy is not None:
        if args.max_board_offset_xy <= 0:
            raise ValueError("--max_board_offset_xy must be > 0 when provided")
        profile_cfg["max_board_offset_xy"] = args.max_board_offset_xy

    rng = random.Random(args.seed)

    generated_trials: dict[str, Any] = {}
    trial_idx = 1
    for setup_idx in range(1, args.num_trials + 1):
        setup_trial = _build_trial(
            rng,
            setup_idx,
            limits,
            profile_cfg=profile_cfg,
            sfp_to_nic_weight=args.sfp_to_nic_weight,
            sc_to_sc_weight=args.sc_to_sc_weight,
        )
        for _ in range(args.episodes_per_setup):
            generated_trials[f"trial_{trial_idx}"] = copy.deepcopy(setup_trial)
            trial_idx += 1

    out_cfg = copy.deepcopy(base)
    out_cfg["trials"] = generated_trials
    out_cfg["generated"] = {
        "script": "aic_engine/scripts/generate_random_trials_config.py",
        "seed": args.seed,
        "profile": args.profile,
        "num_board_setups": args.num_trials,
        "episodes_per_setup": args.episodes_per_setup,
        "num_trials": len(generated_trials),
        "scenario_weights": {
            "sfp_to_nic": args.sfp_to_nic_weight,
            "sc_to_sc": args.sc_to_sc_weight,
        },
        "board_pose_bounds": {
            "x": list(board_cfg["x"]),
            "y": list(board_cfg["y"]),
            "yaw": list(board_cfg["yaw"]),
            "roll": list(board_cfg["roll"]),
            "pitch": list(board_cfg["pitch"]),
        },
        "max_board_offset_xy": profile_cfg["max_board_offset_xy"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print(f"Wrote randomized config: {args.output}")
    print(f"Board setups generated: {args.num_trials}")
    print(f"Episodes per board setup: {args.episodes_per_setup}")
    print(f"Total trials generated: {len(generated_trials)}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

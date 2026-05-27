#!/usr/bin/env python3
"""Materialize fixed expert-generation configs for sfp2nic and sc2sc matrices."""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "scripts"))

from generate_trajectory_dataset import (  # noqa: E402
    derive_output_dir,
    validate_override_limits,
    validate_request,
    write_engine_configs,
)


def _range(min_value: float, max_value: float) -> dict[str, float]:
    return {"min": min_value, "max": max_value}


def _base_request(*, task_family: str, suffix: str, seed: int, score_threshold: float) -> dict[str, Any]:
    return {
        "root_dir": "outputs/trajectory_datasets",
        "task_family": task_family,
        "suffix": suffix,
        "generation": {
            "target_accepted_trajectories": 3,
            "max_attempts": 3,
            "policy": "cheatcode",
            "seed": seed,
            "append_if_exists": True,
        },
        "acceptance": {
            "success_only": True,
            "min_score": score_threshold,
        },
        "scene": {
            "board": {
                "x": _range(0.12, 0.18),
                "y": _range(-0.24, -0.12),
                "z": _range(1.14, 1.14),
                "roll_deg": _range(0.0, 0.0),
                "pitch_deg": _range(0.0, 0.0),
                "yaw_deg": _range(169.0, 183.0),
            },
            "fixture_mounts": {
                "present_probability": _range(0.75, 0.75),
                "rails": [
                    "lc_mount_rail_0",
                    "sfp_mount_rail_0",
                    "sc_mount_rail_0",
                    "lc_mount_rail_1",
                    "sfp_mount_rail_1",
                    "sc_mount_rail_1",
                ],
                "translation": _range(-0.09425, 0.09425),
                "yaw_deg": _range(-60.0, 60.0),
            },
        },
    }


def _sfp_request(
    *,
    present_cards: tuple[int, ...],
    target_card: int,
    target_port: int,
    seed: int,
    score_threshold: float,
) -> dict[str, Any]:
    suffix = "matrix_sfp2nic_cards{}_present{}_target{}_port{}".format(
        len(present_cards),
        "".join(str(card) for card in present_cards),
        target_card,
        target_port,
    )
    request = _base_request(
        task_family="sfp_to_nic",
        suffix=suffix,
        seed=seed,
        score_threshold=score_threshold,
    )
    request["scene"].update(
        {
            "nic_cards": {
                "count": len(present_cards),
                "rails": [f"nic_rail_{card}" for card in present_cards],
                "target_card": target_card,
                "target_port": f"sfp_port_{target_port}",
                "translation": _range(-0.0215, 0.0234),
                "roll_deg": _range(0.0, 0.0),
                "pitch_deg": _range(0.0, 0.0),
                "yaw_deg": _range(-10.0, 10.0),
            },
            "sc_ports": {
                "count": 0,
                "rails": ["sc_rail_0", "sc_rail_1"],
                "translation": _range(-0.06, 0.055),
                "yaw_deg": _range(0.0, 0.0),
            },
            "cable": {
                "cable_type": "sfp_sc_cable",
                "gripper_offset": {
                    "x": _range(-0.002, 0.002),
                    "y": _range(0.013385, 0.017385),
                    "z": _range(0.04045, 0.04445),
                },
                "roll_deg": _range(25.4, 25.4),
                "pitch_deg": _range(-27.7, -27.7),
                "yaw_deg": _range(76.2, 76.2),
            },
        }
    )
    return request


def _sc_request(
    *,
    present_sc_ports: tuple[int, ...],
    target_sc_port: int,
    nic_distractor_count: int,
    seed: int,
    score_threshold: float,
) -> dict[str, Any]:
    suffix = "matrix_sc2sc_sc{}_present{}_target{}_nic{}".format(
        len(present_sc_ports),
        "".join(str(port) for port in present_sc_ports),
        target_sc_port,
        nic_distractor_count,
    )
    request = _base_request(
        task_family="sc_to_sc",
        suffix=suffix,
        seed=seed,
        score_threshold=score_threshold,
    )
    request["scene"].update(
        {
            "sc_ports": {
                "count": len(present_sc_ports),
                "rails": [f"sc_rail_{port}" for port in present_sc_ports],
                "target_port": target_sc_port,
                "translation": _range(-0.06, 0.055),
                "roll_deg": _range(0.0, 0.0),
                "pitch_deg": _range(0.0, 0.0),
                "yaw_deg": _range(0.0, 0.0),
            },
            "nic_cards": {
                "count": nic_distractor_count,
                "rails": [f"nic_rail_{card}" for card in range(5)],
                "translation": _range(-0.0215, 0.0234),
                "yaw_deg": _range(-10.0, 10.0),
            },
            "cable": {
                "cable_type": "sfp_sc_cable_reversed",
                "gripper_offset": {
                    "x": _range(0.0, 0.0),
                    "y": _range(0.015385, 0.015385),
                    "z": _range(0.04045, 0.04045),
                },
                "roll_deg": _range(25.4, 25.4),
                "pitch_deg": _range(-27.7, -27.7),
                "yaw_deg": _range(76.2, 76.2),
            },
        }
    )
    return request


def _iter_sfp_requests(seed: int, score_threshold: float) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for count in (1, 3, 5):
        for present_cards in itertools.combinations(range(5), count):
            for target_card in present_cards:
                for target_port in (0, 1):
                    requests.append(
                        _sfp_request(
                            present_cards=present_cards,
                            target_card=target_card,
                            target_port=target_port,
                            seed=seed + len(requests),
                            score_threshold=score_threshold,
                        )
                    )
    return requests


def _iter_sc_requests(seed: int, score_threshold: float) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for count in (1, 2):
        for present_ports in itertools.combinations(range(2), count):
            for target_port in present_ports:
                for nic_distractor_count in (0, 1, 2):
                    requests.append(
                        _sc_request(
                            present_sc_ports=present_ports,
                            target_sc_port=target_port,
                            nic_distractor_count=nic_distractor_count,
                            seed=seed + 1000 + len(requests),
                            score_threshold=score_threshold,
                        )
                    )
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/expert_matrix_configs"))
    parser.add_argument("--families", nargs="+", choices=["sfp_to_nic", "sc_to_sc"], default=["sfp_to_nic", "sc_to_sc"])
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--score-threshold", type=float, default=92.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke generation.")
    parser.add_argument(
        "--trials-per-config",
        type=int,
        default=1,
        help="Number of engine trials per fixed setting config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials_per_config <= 0:
        raise ValueError("--trials-per-config must be > 0")
    requests: list[dict[str, Any]] = []
    if "sfp_to_nic" in args.families:
        requests.extend(_iter_sfp_requests(args.seed, args.score_threshold))
    if "sc_to_sc" in args.families:
        requests.extend(_iter_sc_requests(args.seed, args.score_threshold))
    if args.limit > 0:
        requests = requests[: args.limit]

    manifest: list[dict[str, Any]] = []
    for index, request in enumerate(requests, start=1):
        validate_request(request)
        validate_override_limits(request)
        output_dir = args.output_root / request["task_family"] / request["suffix"]
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / "request.yaml"
        request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
        engine_config = write_engine_configs(
            copy.deepcopy(request),
            output_dir,
            num_trials=args.trials_per_config,
        )
        manifest.append(
            {
                "index": index,
                "task_family": request["task_family"],
                "suffix": request["suffix"],
                "request_yaml": str(request_path),
                "engine_config": str(engine_config),
                "derived_dataset_dir": str(derive_output_dir(request)),
                "trials_per_config": args.trials_per_config,
            }
        )

    manifest_path = args.output_root / "matrix_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump({"settings": manifest}, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(manifest)} setting configs under {args.output_root}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

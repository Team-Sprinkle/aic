#!/usr/bin/env python3
"""Build repeated post-fix SC trials for targeted failure reproduction.

This is a production-family diagnostic suite.  It intentionally repeats the
released SC trial and a few documented variants so intermittent cable and
port-contact failures have more than one opportunity to appear.
"""

from copy import deepcopy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "official_qualification" / "eval_config.yaml"
OUT = ROOT / "targeted_failure_reproduction"


def set_cards(trial, rails):
    board = trial["scene"]["task_board"]
    for index in range(5):
        if index in rails:
            board[f"nic_rail_{index}"] = {
                "entity_present": True,
                "entity_name": f"nic_card_{index}",
                "entity_pose": {
                    "translation": 0.005,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
            }
        else:
            board[f"nic_rail_{index}"] = {"entity_present": False}


def main():
    config = yaml.safe_load(SOURCE.read_text())
    base = config["trials"]["trial_3"]
    cases = []

    for repeat in range(3):
        cases.append((f"exact_q3_repeat_{repeat + 1}", deepcopy(base), {
            "source": "exact_trial_3", "card_rails": [0, 1, 2],
            "target": "sc1", "reason": "intermittent released-scene failure",
        }))

    for repeat in range(2):
        trial = deepcopy(base)
        set_cards(trial, range(5))
        cases.append((f"five_cards_sc1_repeat_{repeat + 1}", trial, {
            "source": "card-occupancy variant", "card_rails": list(range(5)),
            "target": "sc1", "reason": "maximum intervening-card occupancy",
        }))

    for repeat in range(2):
        trial = deepcopy(base)
        set_cards(trial, range(5))
        trial["tasks"]["task_1"]["target_module_name"] = "sc_port_0"
        cases.append((f"five_cards_sc0_repeat_{repeat + 1}", trial, {
            "source": "target-rail variant", "card_rails": list(range(5)),
            "target": "sc0", "reason": "opposite cable route across card field",
        }))

    for repeat in range(3):
        trial = deepcopy(base)
        set_cards(trial, range(5))
        pose = trial["scene"]["cables"]["cable_1"]["pose"]
        pose["gripper_offset"]["y"] -= 0.002
        pose["pitch"] -= 0.04
        cases.append((f"five_cards_grasp_y_pitch_repeat_{repeat + 1}", trial, {
            "source": "documented grasp-norm variant",
            "card_rails": list(range(5)), "target": "sc1",
            "translation_jitter_m": [0.0, -0.002, 0.0],
            "rotation_jitter_rad": [0.0, -0.04, 0.0],
            "reason": "previously reproduced grasp-sensitive partial insertion",
        }))

    trials = {}
    manifest = []
    for index, (label, trial, details) in enumerate(cases, 1):
        key = f"trial_{index:02d}_{label}"
        trials[key] = trial
        manifest.append({"trial": key, "family": "sc_to_sc", **details})

    config["trials"] = trials
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT / "manifest.json").write_text(json.dumps({
        "classification": "targeted post-fix Gazebo failure reproduction; not exact leaderboard replay",
        "source_config": str(SOURCE),
        "trial_count": len(trials),
        "trials": manifest,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

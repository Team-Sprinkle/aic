#!/usr/bin/env python3
"""Build a stratified ordinary-development CheatCode audit configuration.

This deliberately samples task/card-count/target combinations, not the sealed
qualification trials. It includes two historical randomized SC scenes whose
videos show possible cable/card interaction.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "official_qualification/eval_config.yaml"
ARCHIVE = ROOT.parents[1] / "outputs/trajectory_datasets/clean_including_no_insert_trajs/sc_to_sc/agent"


def cards(trial, count):
    board = trial["scene"]["task_board"]
    for index in range(5):
        board[f"nic_rail_{index}"] = (
            {
                "entity_present": True,
                "entity_name": f"nic_card_{index}",
                "entity_pose": {
                    "translation": round(0.005 + 0.003 * ((index + count) % 3), 5),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": round((-0.08, 0.0, 0.08)[(index + count) % 3], 5),
                },
            }
            if index < count else {"entity_present": False}
        )


def add(config, manifest, name, trial, **details):
    key = f"trial_{len(manifest) + 1:02d}_{name}"
    config["trials"][key] = trial
    manifest.append({"trial": key, **details})


def main(out):
    config = yaml.safe_load(BASE.read_text())
    source = deepcopy(config["trials"])
    config["trials"] = {}
    manifest = []

    for count in range(1, 6):
        trial = deepcopy(source["trial_1"] if count % 2 else source["trial_2"])
        cards(trial, count)
        target = count - 1
        port = count % 2
        task = trial["tasks"]["task_1"]
        task["target_module_name"] = f"nic_card_mount_{target}"
        task["port_name"] = f"sfp_port_{port}"
        add(config, manifest, f"sfp_cards{count}_nic{target}_port{port}", trial,
            family="sfp_to_nic", card_count=count, target=f"nic{target}/port{port}",
            source="qualification scaffold with card count, target, and port changed")

    for count in range(6):
        for port in (0, 1):
            trial = deepcopy(source["trial_3"])
            cards(trial, count)
            trial["tasks"]["task_1"]["target_module_name"] = f"sc_port_{port}"
            add(config, manifest, f"sc_cards{count}_sc{port}", trial,
                family="sc_to_sc", card_count=count, target=f"sc{port}",
                source="qualification scaffold with card count and target changed")

    historical = [
        ("sc_ports_2/n20__sc_ports2_nic5_full_insert_n20/trials/trial_000012.yaml", "sc_historical_cards5_seed12"),
        ("sc_ports_1/n100__sc_ports1_nic3_stop_near_gate_05mm_n100/trials/trial_000004.yaml", "sc_historical_cards3_seed4"),
    ]
    for relative, label in historical:
        path = ARCHIVE / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        old = yaml.safe_load(path.read_text())
        trial = next(iter(old["trials"].values()))
        board = trial["scene"]["task_board"]
        count = sum(bool(board[f"nic_rail_{i}"].get("entity_present")) for i in range(5))
        target = trial["tasks"]["task_1"]["target_module_name"]
        add(config, manifest, label, trial, family="sc_to_sc", card_count=count,
            target=target, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            historical_policy="joint_position_then_cheatcode; current replay uses stock CheatCode only")

    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out / "manifest.json").write_text(json.dumps({
        "classification": "bounded ordinary-development replay; not qualification frequency estimate",
        "source_config": str(BASE),
        "source_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "policy": "installed aic_example_policies.ros.CheatCode",
        "trial_count": len(manifest),
        "trials": manifest,
    }, indent=2) + "\n")
    print(out / "eval_config.yaml", len(manifest), "trials")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

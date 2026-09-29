#!/usr/bin/env python3
"""Derive controlled SC card-count fidelity scenes from a five-card scene.

Use the Isaac Python environment, which includes PyYAML. All port, board,
start and cable settings remain identical; absent cards move to the task's
hidden-part position in both human-readable and reset specifications.
"""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card-count", type=int, choices=range(6), required=True)
    args = parser.parse_args()
    episode = yaml.safe_load(args.source.read_text())
    if episode["task_context"]["nic_card_count"] != 5:
        raise ValueError("Source must be the validated five-card SC scene")
    board = episode["scene"]["task_board"]["position_world"]
    cards = ["nic_card"] + [f"nic_card_{i}" for i in range(1, 5)]
    by_name = {part["scene_name"]: part for part in episode["isaac_randomization"]["parts"]}
    episode["task_context"]["nic_card_count"] = args.card_count
    episode["episode_id"] = f"sc_fidelity_cards_{args.card_count}"
    for index, name in enumerate(cards):
        entry = by_name[name]
        visible = index < args.card_count
        entry["present"] = visible
        if not visible:
            entry["offset"] = [5.0, 5.0, -2.0]
        episode["scene"]["parts"][name]["offset_from_board"] = list(entry["offset"])
        episode["scene"]["parts"][name]["position_world"] = [
            board[axis] + entry["offset"][axis] for axis in range(3)
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(episode, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()

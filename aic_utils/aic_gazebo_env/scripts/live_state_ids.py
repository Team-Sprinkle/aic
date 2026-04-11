#!/usr/bin/env python3
"""Dump selected entity components from Gazebo world state."""

from __future__ import annotations

import json
import re
import subprocess

IDS = {32, 50, 79}


def main() -> None:
    completed = subprocess.run(
        ["gz", "topic", "-e", "-n", "1", "-t", "/world/aic_world/state"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = completed.stdout
    blocks = re.findall(
        r"entities \{\n    key: (\d+)\n    value \{(.*?)\n    \}\n  \}",
        text,
        re.S,
    )
    result: dict[int, list[dict[str, str]]] = {}
    for key_text, body in blocks:
        key = int(key_text)
        if key not in IDS:
            continue
        result[key] = []
        for match in re.finditer(
            r'type: (\d+)\n\s+component: "((?:\\.|[^"])*)"',
            body,
            re.S,
        ):
            result[key].append(
                {
                    "type": match.group(1),
                    "component": match.group(2),
                }
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print selected entity ids from Gazebo pose/info."""

from __future__ import annotations

import json
import re
import subprocess

NAMES = ["ati/tool_link", "tabletop", "wrist_3_link"]


def main() -> None:
    completed = subprocess.run(
        ["gz", "topic", "-e", "-n", "1", "-t", "/world/aic_world/pose/info"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = completed.stdout
    result: dict[str, int | None] = {}
    for name in NAMES:
        match = re.search(
            rf'pose \{{\n  name: "{re.escape(name)}"\n  id: (\d+)\n',
            text,
            re.S,
        )
        result[name] = int(match.group(1)) if match else None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

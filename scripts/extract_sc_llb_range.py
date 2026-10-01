#!/usr/bin/env python3
"""Extract a disjoint range of verified SC LLB bags into the resumable cache.

Run only ranges well ahead of the main materialization worker. Each completed
case is renamed atomically; the main worker skips it when it reaches that case.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from run_sc_llb_cleaned_materialization import IMAGE, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--streams", type=Path, required=True)
    p.add_argument("--start", type=int, required=True, help="zero-based inclusive index")
    p.add_argument("--end", type=int, required=True, help="zero-based exclusive index")
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    if spec.get("source_episode_count") != 104 or not 0 <= a.start < a.end <= 104:
        p.error("Expected valid range of the audited 104 episodes")
    streams = a.streams.resolve()
    streams.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DOCKER_HOST=os.environ.get("DOCKER_HOST", f"unix:///run/user/{os.getuid()}/docker.sock"))
    for episode in spec["episodes"][a.start:a.end]:
        case = episode["case"]
        dest = streams / f"{case}.json"
        if dest.exists():
            saved = json.loads(dest.read_text())
            if saved.get("official_tier3") != 75 or Path(saved["audit"]).name != case:
                raise ValueError(f"Invalid existing stream: {dest}")
            continue
        partial = streams / f"{case}.partial.json"
        if partial.exists():
            raise ValueError(f"Another worker owns partial stream: {partial}")
        cmd = ["docker", "run", "--rm", "--net=none", "--entrypoint", "/bin/bash",
               "-v", f"{ROOT}:{ROOT}:ro", "-v", "/home/nas:/home/nas:ro",
               "-v", f"{streams}:{streams}", "-w", str(ROOT), IMAGE, "-lc",
               "source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; "
               "python3 scripts/extract_moveit_training_stream.py \"$1\" \"$2\"",
               "_", episode["source_episode"], str(partial)]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        saved = json.loads(partial.read_text())
        if saved.get("official_tier3") != 75 or Path(saved["audit"]).name != case:
            raise ValueError(f"Wrong provenance: {case}")
        partial.rename(dest)
        print(f"completed {case}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resume scored-bag extraction, build raw BC chunks, then filter the 104 SC cases.

Requires the project's rootless Docker socket for the official ROS bag reader.
It never modifies the score-verified source episodes or overwrites completed
streams. Run only in an environment with that socket and NAS mounted.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923"


def run(args, log):
    with log.open("a") as output:
        subprocess.run(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec.get("status") != "metadata_only_action_chunks_pending" or len(spec["episodes"]) != 104:
        parser.error("Expected the audited 104-case cleaning specification")
    # Fail before writing anything if the restricted session cannot reach Docker.
    env = dict(os.environ, DOCKER_HOST=os.environ.get("DOCKER_HOST", "unix:///run/user/1008/docker.sock"))
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], cwd=ROOT,
                   env=env, check=True, stdout=subprocess.DEVNULL)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    streams = output / "streams"
    streams.mkdir(exist_ok=True)
    log = output / "materialization.log"
    for episode in spec["episodes"]:
        case = episode["case"]
        stream = streams / (case + ".json")
        if stream.exists():
            saved = json.loads(stream.read_text())
            if saved.get("official_tier3") != 75 or Path(saved["audit"]).name != case:
                raise ValueError(f"Existing extracted stream has wrong provenance: {stream}")
            continue
        partial = streams / (case + ".partial.json")
        if partial.exists():
            partial.unlink()
        command = ["docker", "run", "--rm", "--net=none", "--entrypoint", "/bin/bash",
                   "-v", f"{ROOT}:{ROOT}:ro", "-v", "/home/nas:/home/nas:ro",
                   "-v", f"{streams}:{streams}", "-w", str(ROOT), IMAGE, "-lc",
                   "source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; "
                   "python3 scripts/extract_moveit_training_stream.py \"$1\" \"$2\"",
                   "_", episode["source_episode"], str(partial)]
        with log.open("a") as output_log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=output_log,
                           stderr=subprocess.STDOUT, check=True)
        saved = json.loads(partial.read_text())
        if saved.get("official_tier3") != 75 or Path(saved["audit"]).name != case:
            raise ValueError(f"Extracted stream has wrong provenance: {case}")
        partial.rename(stream)
    raw = output / "raw_chunks"
    if not (raw / "manifest.json").exists():
        if raw.exists():
            raise ValueError(f"Incomplete raw materialization, inspect before retry: {raw}")
        train = [str(streams / (row["case"] + ".json")) for row in spec["episodes"] if row["split"] == "train"]
        validation = [str(streams / (row["case"] + ".json")) for row in spec["episodes"] if row["split"] == "validation"]
        run(["bash", "scripts/run_materialize_transport_bc.sh", "--streams", *train,
             "--validation-full-success-streams", *validation, "--output", str(raw)], log)
    cleaned = output / "bc_chunks"
    if not (cleaned / "manifest.json").exists():
        if cleaned.exists():
            raise ValueError(f"Incomplete cleaned materialization, inspect before retry: {cleaned}")
        run([sys.executable, "scripts/filter_sc_llb_bc_chunks.py", "--spec", str(args.spec),
             "--raw-transport", str(raw / "transport_chunks.jsonl"),
             "--raw-local", str(raw / "local_chunks.jsonl"), "--output", str(cleaned)], log)
    print(json.dumps({"cleaned_manifest": str(cleaned / "manifest.json"),
                      "episodes": len(spec["episodes"])}))


if __name__ == "__main__":
    main()

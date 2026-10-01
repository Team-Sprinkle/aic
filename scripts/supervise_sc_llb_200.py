#!/usr/bin/env python3
"""Continue a passed 30-case LLB pilot to 200 attempted cases, resumably."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiments/2026-09-30_sc_llb_collection"
PILOT = BASE / "pilot30"
CONTINUATION = BASE / "continuation170"
ARCHIVE = Path("/home/nas/chmin/aic_llb_collection_20260930/continuation170")
STATUS = BASE / "supervisor_status.json"


def write(status, **extra):
    STATUS.write_text(json.dumps({"schema": "aic_sc_llb_200_supervisor/v1",
                                  "status": status, "time_unix": time.time(), **extra}, indent=2) + "\n")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    if not (PILOT / "manifest.json").exists() or not (CONTINUATION / "manifest.json").exists():
        raise FileNotFoundError("Both deterministic scene manifests are required")
    start_manifest_path = CONTINUATION / "start_variant_manifest.json"
    if not start_manifest_path.exists():
        raise FileNotFoundError("Actual Gazebo start-variant xacros are required for continuation")
    start_manifest = json.loads(start_manifest_path.read_text())
    if len(start_manifest["cases"]) != 170 or not all(
        Path(case["xacro"]).exists() for case in start_manifest["cases"]
    ):
        raise RuntimeError("Incomplete 170-case Gazebo start-variant preparation")
    while True:
        path = PILOT / "collection_progress.json"
        if path.exists():
            pilot = json.loads(path.read_text())
            count = len(pilot["attempts"])
            successes = pilot["full_successes"]
            write("waiting_for_pilot", pilot_attempts=count, pilot_full_successes=successes)
            if count >= 30:
                if count != 30 or pilot["pilot_gate"] != "pass" or successes < 10:
                    write("pilot_failed_stop", pilot_attempts=count, pilot_full_successes=successes)
                    return 0
                break
        else:
            count = 0
        pid_path = PILOT / "collector.pid"
        if pid_path.exists() and not pid_alive(int(pid_path.read_text())) and count < 30:
            write("pilot_runner_stopped_early", pilot_attempts=count)
            return 1
        time.sleep(30)
    # The runner writes its last progress entry after the rootless replay
    # launcher returns and cleans its own container. Avoid racing the final exit.
    pid = int((PILOT / "collector.pid").read_text())
    while pid_alive(pid):
        time.sleep(3)
    write("pilot_passed_starting_continuation", pilot_attempts=30,
          pilot_full_successes=successes, continuation_requested=170)
    command = [str(ROOT / ".pixi/envs/default/bin/python"),
               str(ROOT / "scripts/run_sc_llb_collection.py"), str(CONTINUATION),
               "--passed-pilot-progress", str(PILOT / "collection_progress.json"),
               "--archive-root", str(ARCHIVE)]
    with (CONTINUATION / "collection.log").open("a") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                check=False)
    progress = CONTINUATION / "collection_progress.json"
    details = json.loads(progress.read_text()) if progress.exists() else {}
    completed = len(details.get("attempts", {}))
    write("continuation_complete" if result.returncode == 0 and completed == 170 else
          "continuation_stopped_early", pilot_attempts=30,
          pilot_full_successes=successes, continuation_attempts=completed,
          continuation_full_successes=details.get("full_successes", 0),
          returncode=result.returncode, archive_root=str(ARCHIVE))
    return result.returncode if result.returncode else (0 if completed == 170 else 2)


if __name__ == "__main__":
    sys.exit(main())

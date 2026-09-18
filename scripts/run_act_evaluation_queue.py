#!/usr/bin/env python3
"""Run an explicit saved list of evaluation commands sequentially before a deadline."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    deadline = datetime.fromisoformat(manifest["deadline_utc"]).timestamp()
    parent = args.manifest.parent
    wait_pid = manifest.get("wait_for_pid")
    if wait_pid:
        try:
            process = psutil.Process(wait_pid)
            expected = manifest["wait_for_command_contains"]
            if not any(expected in arg for arg in process.cmdline()):
                raise ValueError("Wait PID does not match the declared evaluation")
            while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                if time.time() >= deadline:
                    return
                time.sleep(2)
        except psutil.NoSuchProcess:
            pass
    with args.manifest.with_suffix(".results.jsonl").open("a") as results:
        for job in manifest["jobs"]:
            remaining = deadline - time.time()
            if remaining < job["maximum_seconds"]:
                results.write(json.dumps({"job": job["name"], "skipped": "Insufficient time before queue deadline"}) + "\n")
                break
            start = datetime.now(timezone.utc).isoformat()
            with (parent / job["log"]).open("w") as log:
                completed = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                           timeout=job["maximum_seconds"])
            row = {"job": job["name"], "started_utc": start, "returncode": completed.returncode,
                   "finished_utc": datetime.now(timezone.utc).isoformat()}
            results.write(json.dumps(row) + "\n"); results.flush()
            print(json.dumps(row), flush=True)
            if completed.returncode:
                raise RuntimeError(f"Evaluation failed: {job['name']}; inspect {job['log']}")


if __name__ == "__main__":
    main()

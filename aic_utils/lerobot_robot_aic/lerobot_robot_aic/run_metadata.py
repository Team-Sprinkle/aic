"""Run directory and metadata helpers for hybrid training scripts."""

from __future__ import annotations

from datetime import datetime
import json
import subprocess
from pathlib import Path
from typing import Any


def make_run_dir(
    output_root: Path,
    *,
    task_family: str,
    dataset_tag: str,
    model_family: str,
    stage: str,
    run_name: str,
    now: datetime | None = None,
) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return output_root / task_family / dataset_tag / model_family / stage / f"{stamp}_{run_name}"


def git_info(repo_root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        result = subprocess.run(["git", *args], cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["branch", "--show-current"]),
        "status_short": run(["status", "--short"]),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

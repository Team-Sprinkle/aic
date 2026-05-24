from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SERVO_SCRIPT = REPO / "aic_utils" / "aic_isaac" / "scripts" / "privileged_insertion_servo_sweep.py"
AUDIT_SCRIPT = REPO / "aic_utils" / "aic_isaac" / "scripts" / "audit_phase_reward_funnel.py"


def test_privileged_servo_sweep_writes_reproducible_run_folder(tmp_path: Path) -> None:
    out = tmp_path / "servo"
    subprocess.run(
        [
            sys.executable,
            str(SERVO_SCRIPT),
            "--output-root",
            str(out),
            "--run-name",
            "smoke",
            "--lateral-starts-mm",
            "1",
            "--axial-starts-mm",
            "3",
            "--orientation-starts",
            "small",
            "--max-steps",
            "30",
        ],
        cwd=REPO,
        check=True,
    )
    run_dir = out / "smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "git_status.txt").is_file()
    assert (run_dir / "git_diff.patch").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "summary.md").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["summaries"]
    assert {"final_s_m", "final_r_m", "final_theta_rad", "failure_mode"} <= set(metrics["summaries"][0])


def test_reward_funnel_audit_flags_no_bad_forward_reward(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--output-root",
            str(out),
            "--run-name",
            "smoke",
        ],
        cwd=REPO,
        check=True,
    )
    summary = json.loads((out / "smoke" / "summary.json").read_text(encoding="utf-8"))
    assert summary["bad_forward_surface_count"] == 0

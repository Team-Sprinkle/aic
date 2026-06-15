#!/usr/bin/env python3
"""Run a bounded wrist-reset compensation sweep.

The sweep is intentionally conservative: it uses existing calibration and
reset-settle validation scripts, records every command, and promotes only
configs that produce accepted post-step reset states under strict geometry
thresholds.  It does not train a policy.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = REPO_ROOT / ".pixi/envs/default/bin/python"
CALIBRATE = REPO_ROOT / "aic_utils/aic_isaac/scripts/calibrate_randomized_curriculum_from_reset_metrics.py"
VALIDATE = REPO_ROOT / "aic_utils/aic_isaac/scripts/validate_serl_reset_settle.py"


def _run(cmd: list[str], *, cwd: Path, timeout_s: float | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": cmd,
        "command_text": " ".join(shlex.quote(str(x)) for x in cmd),
        "started_utc": started,
        "returncode": proc.returncode,
        "output": proc.stdout,
    }


def _latest_train_run(output: str) -> str | None:
    for line in output.splitlines():
        if '"train_run_dir"' in line:
            try:
                return str(json.loads("{" + line.strip().rstrip(",")).get("train_run_dir"))
            except Exception:
                pass
    try:
        parsed = json.loads(output[output.rfind("{") :])
    except Exception:
        return None
    value = parsed.get("train_run_dir")
    return str(value) if value else None


def _summarize_metrics(train_run: Path) -> dict[str, Any]:
    metrics = train_run / "metrics.jsonl"
    if not metrics.is_file():
        return {"error": f"missing {metrics}"}
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not rows:
        return {"error": "metrics has no rows"}
    out: dict[str, Any] = {"row_count": len(rows)}
    for label, row in (("step1", rows[0]), ("final", rows[-1])):
        geom = row.get("post_step_insertion_geometry") or {}
        packed: dict[str, Any] = {"step": row.get("step"), "reward_mean": row.get("reward_mean")}
        for key in (
            "signed_depth_m_by_env",
            "lateral_error_m_by_env",
            "orientation_error_rad_by_env",
            "consistency_final_axial_error_m_by_env",
            "consistency_gate_by_env",
        ):
            values = [float(v) for v in geom.get(key, []) if isinstance(v, int | float)]
            if values:
                packed[key.replace("_by_env", "")] = {
                    "min": min(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                }
        strict = geom.get("strict_success_by_env")
        if isinstance(strict, list):
            packed["strict_success_count"] = sum(1 for item in strict if bool(item))
            packed["env_count"] = len(strict)
        out[label] = packed
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-dir", required=True, type=Path)
    parser.add_argument("--validation-run", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/agentic_reward_curriculum_20260529")
    parser.add_argument("--run-prefix", default="wrist_comp")
    parser.add_argument("--source-step", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--position-gains", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    parser.add_argument("--orientation-gains", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    parser.add_argument("--seed-base", type=int, default=67330)
    parser.add_argument("--max-wall-time-minutes", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = (args.output_root / "reset_compensation_sweeps" / f"{timestamp}_{args.run_prefix}").resolve()
    generated_root = (args.output_root / "generated_episode_configs").resolve()
    reset_root = (args.output_root / "reset_validation").resolve()
    train_root = (args.output_root / "policy_train_runs").resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    decisions: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    idx = 0
    for pos_gain in args.position_gains:
        for ori_gain in args.orientation_gains:
            idx += 1
            tag = f"{args.run_prefix}_p{pos_gain:g}_o{ori_gain:g}".replace(".", "p")
            config_dir = generated_root / tag
            calibrate_cmd = [
                str(PYTHON),
                str(CALIBRATE),
                "--input-config-dir",
                str(args.base_config_dir),
                "--validation-run",
                str(args.validation_run),
                "--output-config-dir",
                str(config_dir),
                "--step",
                str(int(args.source_step)),
                "--max-envs",
                str(int(args.num_envs)),
                "--position-gain",
                str(float(pos_gain)),
                "--overwrite",
            ]
            if float(ori_gain) != 0.0:
                calibrate_cmd.extend(["--calibrate-orientation", "--orientation-gain", str(float(ori_gain))])
            else:
                calibrate_cmd.append("--no-calibrate-orientation")
            cal = _run(calibrate_cmd, cwd=REPO_ROOT)

            validate_cmd = [
                str(PYTHON),
                str(VALIDATE),
                "--episode-config-dir",
                str(config_dir),
                "--output-root",
                str(reset_root),
                "--train-output-root",
                str(train_root),
                "--run-name",
                f"validate_{tag}_zeroaction_reset{int(args.steps)}_env{int(args.num_envs)}",
                "--num-envs",
                str(int(args.num_envs)),
                "--steps",
                str(int(args.steps)),
                "--zero-action",
                "--max-lateral-m",
                "0.003",
                "--max-theta-rad",
                "0.030",
                "--min-s-m",
                "-0.004",
                "--max-s-m",
                "0.004",
                "--max-wall-time-minutes",
                str(float(args.max_wall_time_minutes)),
                "--seed",
                str(int(args.seed_base) + idx),
            ]
            val = _run(validate_cmd, cwd=REPO_ROOT)
            train_run = _latest_train_run(val["output"] or "")
            metrics = _summarize_metrics(Path(train_run)) if train_run else {"error": "train_run_dir not found"}
            accepted = 0
            try:
                parsed = json.loads((val["output"] or "")[(val["output"] or "").rfind("{") :])
                accepted = int(parsed.get("accepted_count", 0))
            except Exception:
                accepted = 0
            decision = {
                "tag": tag,
                "position_gain": float(pos_gain),
                "orientation_gain": float(ori_gain),
                "config_dir": str(config_dir),
                "calibrate": cal,
                "validate": val,
                "train_run_dir": train_run,
                "accepted_count": accepted,
                "metrics": metrics,
                "decision": "promote" if accepted > 0 else "reject",
            }
            decisions.append(decision)
            if best is None or accepted > int(best.get("accepted_count", 0)):
                best = decision
            (sweep_dir / f"{idx:03d}_{tag}.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
            if accepted > 0:
                break
        if decisions and int(decisions[-1].get("accepted_count", 0)) > 0:
            break

    summary = {
        "sweep_dir": str(sweep_dir),
        "base_config_dir": str(args.base_config_dir.resolve()),
        "validation_run": str(args.validation_run.resolve()),
        "candidate_count": len(decisions),
        "best": best,
        "decisions": [
            {
                "tag": item["tag"],
                "position_gain": item["position_gain"],
                "orientation_gain": item["orientation_gain"],
                "accepted_count": item["accepted_count"],
                "train_run_dir": item["train_run_dir"],
                "decision": item["decision"],
                "metrics": item["metrics"],
            }
            for item in decisions
        ],
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if decisions else 1


if __name__ == "__main__":
    raise SystemExit(main())

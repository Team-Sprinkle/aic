#!/usr/bin/env python3
"""Canonical live parity runner for the training-only Gazebo runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from aic_gazebo_env.live_runtime import (
    DEFAULT_WORLD_NAME,
    LiveRuntimeManager,
    perform_live_parity_sequence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--world-name", default=DEFAULT_WORLD_NAME)
    parser.add_argument("--world-path", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        payload = perform_live_parity_sequence(
            world_name=args.world_name,
            world_path=args.world_path,
        )
    else:
        manager = LiveRuntimeManager(
            world_name=args.world_name,
            world_path=args.world_path,
        )
        preflight = manager.preflight()
        context = manager.prepare(
            auto_build=args.auto_build,
            auto_launch=args.auto_launch,
        )
        health = manager.wait_for_health(context, timeout_s=120.0).to_dict()
        if health.get("no_op_step_ok") and health.get("action_step_ok"):
            script_path = Path(__file__).resolve()
            command = (
                f"PYTHONPATH={manager.repo_root / 'aic_utils' / 'aic_gazebo_env'} "
                f"{sys.executable} "
                f"{script_path} --worker --world-name {args.world_name} "
                f"--world-path {args.world_path or manager.world_path} --json-only"
            )
            result = manager.run_context_command(context, command, timeout_s=120.0)
            parity = json.loads(result.stdout) if result.returncode == 0 else {
                "ok": False,
                "error": result.stderr or result.stdout,
            }
        else:
            parity = {
                "ok": False,
                "error": health.get("diagnostics", {}).get(
                    "last_error",
                    "live health checks did not pass",
                ),
            }
        payload = {
            "preflight": preflight,
            "context": context.to_dict(),
            "health": health,
            "parity": parity,
        }
    rendered = json.dumps(payload, indent=None if args.json_only else 2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload.get("ok", payload.get("parity", {}).get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

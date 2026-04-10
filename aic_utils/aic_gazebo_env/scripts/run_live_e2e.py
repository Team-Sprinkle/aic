#!/usr/bin/env python3
"""Canonical live e2e runner for the training-only Gazebo runtime."""

from __future__ import annotations

import argparse
import json
import sys

from aic_gazebo_env.live_runtime import (
    DEFAULT_WORLD_NAME,
    LiveRuntimeManager,
    perform_live_health_check,
    perform_live_parity_sequence,
    perform_live_smoke_sequence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--world-name", default=DEFAULT_WORLD_NAME)
    parser.add_argument("--world-path", default=None)
    parser.add_argument("--worker", choices=("health", "smoke", "parity"), default=None)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker == "health":
        print(
            json.dumps(
                perform_live_health_check(
                    world_name=args.world_name,
                    world_path=args.world_path,
                ),
                indent=None if args.json_only else 2,
                sort_keys=True,
            )
        )
        return 0
    if args.worker == "smoke":
        print(
            json.dumps(
                perform_live_smoke_sequence(
                    world_name=args.world_name,
                    world_path=args.world_path,
                ),
                indent=None if args.json_only else 2,
                sort_keys=True,
            )
        )
        return 0
    if args.worker == "parity":
        print(
            json.dumps(
                perform_live_parity_sequence(
                    world_name=args.world_name,
                    world_path=args.world_path,
                ),
                indent=None if args.json_only else 2,
                sort_keys=True,
            )
        )
        return 0

    manager = LiveRuntimeManager(
        world_name=args.world_name,
        world_path=args.world_path,
    )
    preflight = manager.preflight()
    context = manager.prepare(
        auto_build=args.auto_build,
        auto_launch=args.auto_launch,
    )
    result = manager.run_e2e(context)
    payload = {
        "preflight": preflight,
        "context": context.to_dict(),
        "result": result,
    }
    if not args.json_only:
        if preflight.get("setup_script"):
            print(f"live_e2e_setup_script: {preflight['setup_script']}", file=sys.stderr)
        print(f"live_e2e_context_mode: {context.mode}", file=sys.stderr)
    print(json.dumps(payload, indent=None if args.json_only else 2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

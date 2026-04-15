#!/usr/bin/env python3
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run aic_model while forcing a source-tree policy package to the front of sys.path."
    )
    parser.add_argument(
        "--policy-source-root",
        required=True,
        help="Path to the source root that contains the aic_example_policies package",
    )
    parser.add_argument(
        "--policy-class",
        required=True,
        help="Fully-qualified policy module path",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.policy_source_root)

    from aic_model.aic_model import main as aic_model_main

    sys.argv = [
        "aic_model",
        "--ros-args",
        "-p",
        "use_sim_time:=true",
        "-p",
        f"policy:={args.policy_class}",
    ]
    return aic_model_main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a reproducible SERL command variant by replacing flags exactly.

This is a small guardrail for iterative Isaac experiments.  Appending override
flags works with argparse in many cases, but it leaves ambiguous command files
and makes it easy to train on an old episode config.  This helper removes the
old flag occurrence before inserting the requested value.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    negated = f"--no-{flag[2:]}" if flag.startswith("--") else None
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == flag or (negated is not None and token == negated):
            idx += 1
            if idx < len(argv) and not argv[idx].startswith("--"):
                idx += 1
            continue
        out.append(token)
        idx += 1
    return out


def _set_flag(argv: list[str], flag: str, values: list[str]) -> list[str]:
    argv = _strip_flag(argv, flag)
    return [*argv, flag, *values]


def _enable_flag(argv: list[str], flag: str) -> list[str]:
    argv = _strip_flag(argv, flag)
    return [*argv, flag]


def _disable_flag(argv: list[str], flag: str) -> list[str]:
    argv = _strip_flag(argv, flag)
    if not flag.startswith("--"):
        raise ValueError(f"boolean flag must start with --: {flag}")
    return [*argv, f"--no-{flag[2:]}"]


def _parse_set(item: str) -> tuple[str, list[str]]:
    if "=" not in item:
        raise ValueError(f"--set expects FLAG=VALUE: {item}")
    flag, raw = item.split("=", 1)
    if not flag.startswith("--"):
        raise ValueError(f"flag must start with --: {flag}")
    return flag, shlex.split(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--output-command", type=Path, required=True)
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="FLAG=VALUE")
    parser.add_argument("--enable", action="append", default=[], metavar="FLAG")
    parser.add_argument("--disable", action="append", default=[], metavar="FLAG")
    parser.add_argument("--remove", action="append", default=[], metavar="FLAG")
    args = parser.parse_args()

    argv = shlex.split(args.base_command.read_text(encoding="utf-8"))
    for flag in args.remove:
        argv = _strip_flag(argv, flag)
    for flag in args.disable:
        argv = _disable_flag(argv, flag)
    for flag in args.enable:
        argv = _enable_flag(argv, flag)
    for item in args.sets:
        flag, values = _parse_set(item)
        argv = _set_flag(argv, flag, values)

    args.output_command.parent.mkdir(parents=True, exist_ok=True)
    args.output_command.write_text(shlex.join(argv) + "\n", encoding="utf-8")
    print(args.output_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

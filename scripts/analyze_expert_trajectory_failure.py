#!/usr/bin/env python3
"""Analyze expert trajectory debug artifacts with GPT-5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.expert_generator.debug_artifacts import (  # noqa: E402
    build_gpt5_failure_prompt,
    call_gpt5_failure_analysis,
    compact_payload_with_retry,
)


def run_analysis(*, debug_dir: str | Path, model: str = "gpt-5", dry_run: bool = False) -> dict[str, Path]:
    root = Path(debug_dir)
    if root.name != "debug" and (root / "debug").is_dir():
        root = root / "debug"
    payload, period = compact_payload_with_retry(root)
    payload["effective_sample_period_sec"] = period
    prompt = build_gpt5_failure_prompt(payload)
    payload_path = root / "gpt5_failure_payload.json"
    prompt_path = root / "gpt5_failure_prompt.md"
    analysis_path = root / "gpt5_failure_analysis.md"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    if dry_run:
        analysis = (
            "# Dry Run Expert Trajectory Failure Analysis\n\n"
            f"GPT-5 was not called. Payload built at {period:.1f}s sampling.\n"
        )
    else:
        analysis = call_gpt5_failure_analysis(prompt, model=model)
        if not analysis.strip():
            raise RuntimeError("GPT-5 returned empty/unparseable output")
    analysis_path.write_text(analysis, encoding="utf-8")
    return {"payload": payload_path, "prompt": prompt_path, "analysis": analysis_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", required=True, help="Path to dataset/debug or dataset root containing debug/.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(debug_dir=args.debug_dir, model=args.model, dry_run=args.dry_run)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()

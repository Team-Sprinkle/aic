#!/usr/bin/env python3
"""Analyze bundled agentic teleop failure traces with GPT-5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.debug_recorder import (  # noqa: E402
    build_failure_analysis_payload,
    build_failure_analysis_prompt,
    load_openai_api_key,
    write_json,
)


def _extract_bundle(bundle: Path) -> Path:
    if bundle.is_dir():
        return bundle
    if bundle.suffix != ".zip":
        raise ValueError("--bundle must be a directory or .zip")
    output_dir = bundle.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "r") as archive:
        archive.extractall(output_dir)
    return output_dir


def _call_gpt5(prompt: str, *, model: str) -> tuple[str, dict[str, Any]]:
    api_key = load_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used")
    try:
        from openai import OpenAI
    except Exception as ex:
        raise RuntimeError("The openai Python package is required for GPT-5 analysis") from ex
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "You are a senior robotics failure-analysis engineer. "
            "Be direct, evidence-based, and code-level where possible."
        ),
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    text = getattr(response, "output_text", None)
    if not text:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    chunks.append(str(getattr(content, "text", "")))
        text = "\n".join(chunks)
    response_json = json.loads(response.model_dump_json()) if hasattr(response, "model_dump_json") else {}
    return str(text), response_json


def run_analysis(
    *,
    run_dir: str | Path | None,
    bundle: str | Path | None,
    model: str = "gpt-5",
    dry_run: bool = False,
) -> dict[str, Path]:
    if run_dir is None and bundle is None:
        raise ValueError("run_dir or bundle is required")
    root = Path(run_dir) if run_dir is not None else _extract_bundle(Path(bundle))
    payload = build_failure_analysis_payload(root)
    prompt = build_failure_analysis_prompt(payload)
    prompt_path = root / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    if dry_run:
        analysis = (
            "# Dry Run Failure Analysis\n\n"
            "GPT-5 was not called. The prompt and compact structured payload were built successfully.\n"
        )
        analysis_json: dict[str, Any] = {
            "dry_run": True,
            "model": model,
            "attempt_count": payload.get("attempt_count"),
            "prompt_path": str(prompt_path),
        }
    else:
        analysis, response_json = _call_gpt5(prompt, model=model)
        analysis_json = {
            "dry_run": False,
            "model": model,
            "attempt_count": payload.get("attempt_count"),
            "response": response_json,
        }
    analysis_path = root / "failure_analysis.md"
    analysis_json_path = root / "failure_analysis.json"
    analysis_path.write_text(analysis, encoding="utf-8")
    write_json(analysis_json_path, analysis_json)
    return {"analysis": analysis_path, "analysis_json": analysis_json_path, "prompt": prompt_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--bundle")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(
        run_dir=args.run_dir,
        bundle=args.bundle,
        model=args.model,
        dry_run=args.dry_run,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()

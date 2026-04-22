"""Minimal Responses API smoke test using the teacher OpenAI backend machinery."""

from __future__ import annotations

import argparse
import json

from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="Return a conservative guarded_insert plan with one waypoint near the target.",
    )
    args = parser.parse_args()

    backend = OpenAIPlannerBackend(
        OpenAIPlannerConfig(
            enabled=True,
            model=args.model,
            temperature=args.temperature,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
        )
    )
    result = backend.run_smoke_test(prompt=args.prompt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

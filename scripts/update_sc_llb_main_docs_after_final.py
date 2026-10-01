#!/usr/bin/env python3
"""Refresh three main docs after the detached LLB finalizer exits."""

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiments/2026-09-30_sc_llb_collection"


def running(pid):
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return False
    text = status.read_text()
    return "State:\tZ" not in text


def replace_line(path, prefix, replacement):
    lines = path.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {prefix!r} in {path}, got {len(matches)}")
    lines[matches[0]] = replacement
    path.write_text("\n".join(lines) + "\n")


def main():
    pid = int((BASE / "finalizer.pid").read_text())
    while running(pid):
        time.sleep(45)
    report_path = BASE / "collection_final_report.json"
    if not report_path.exists():
        raise RuntimeError("Finalizer exited without final report; main docs left unchanged")
    report = json.loads(report_path.read_text())
    n, full, bags = report["attempted"], report["full_insertions"], report["scored_bags"]
    state = "Completed" if report["complete"] else "Stopped early"
    replace_line(
        ROOT / "docs/STATUS.md", "**September 30 SC Low Lateral Bypass collection:**",
        f"**September 30 SC Low Lateral Bypass collection:** {state} after **{n}/200 "
        f"attempted settings**, with **{full} official full insertions** and **{bags} valid "
        f"scored replay bags**. The pilot passed 17/30; varied robot starts were checked "
        f"against scored bags. Only full insertions with a bag and action trace are "
        f"linked in `outputs/trajectory_datasets/sc_llb_score_verified/`. No BC training "
        f"or final-scene evaluation was started. "
        f"[Evidence](experiments/2026-09-30-sc-llb-collection.md).")
    replace_line(
        ROOT / "docs/EXPERIMENTS.md", "**September 30 SC teacher collection:**",
        f"**September 30 SC teacher collection:** {state} at {n}/200 attempts, with "
        f"{full} official full insertions. The 30-case pilot passed 17/30. This is "
        f"privileged teacher data, not autonomous policy performance; no BC training "
        f"was started. [Scores, videos and method](experiments/2026-09-30-sc-llb-collection.md).")
    datasets = ROOT / "docs/DATASETS.md"
    source = datasets.read_text()
    header = "## September 30 SC Low Lateral Bypass collection\n"
    start = source.index(header) + len(header)
    end = source.index("\n## ", start)
    paragraph = (f"The `outputs/experiments/2026-09-30_sc_llb_collection/` teacher "
                 f"collection {state.lower()} at **{n}/200 attempted settings** with "
                 f"**{full} official full insertions** and **{bags} valid scored bags**. "
                 f"Only successes with a bag and action trace are linked in "
                 f"`outputs/trajectory_datasets/sc_llb_score_verified/manifest.json`. "
                 f"This collection remains separate from canonical `expert_verified`; "
                 f"no training split or BC run was created. "
                 f"[Exact results](experiments/2026-09-30-sc-llb-collection.md).")
    datasets.write_text(source[:start] + "\n" + paragraph + "\n" + source[end:])
    print(json.dumps({"updated_main_docs": True, "attempted": n, "full_insertions": full}))


if __name__ == "__main__":
    main()

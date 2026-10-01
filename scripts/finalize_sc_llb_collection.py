#!/usr/bin/env python3
"""Finalize the detached LLB collection after its supervisor stops.

Creates a score-verified manifest and concise experiment record. It never
starts training and never includes failed or bagless trials as experts.
"""

import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiments/2026-09-30_sc_llb_collection"
STATUS = BASE / "supervisor_status.json"
SUMMARY = BASE / "summary.json"
DATASET = ROOT / "outputs/trajectory_datasets/sc_llb_score_verified"
RECORD = ROOT / "docs/experiments/2026-09-30-sc-llb-collection.md"
STATUS_DOC = ROOT / "docs/STATUS.md"
DATASETS_DOC = ROOT / "docs/DATASETS.md"
EXPERIMENTS_DOC = ROOT / "docs/EXPERIMENTS.md"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def wait_for_supervisor():
    while True:
        if STATUS.exists():
            status = json.loads(STATUS.read_text())
            if status.get("status") in {"continuation_complete", "continuation_stopped_early",
                                        "pilot_failed_stop", "pilot_runner_stopped_early"}:
                return status
        time.sleep(45)


def replace_line_start(path, prefix, replacement):
    lines = path.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one status paragraph in {path}, found {len(matches)}")
    lines[matches[0]] = replacement
    path.write_text("\n".join(lines) + "\n")


def main():
    status = wait_for_supervisor()
    command = [str(ROOT / ".pixi/envs/default/bin/python"),
               str(ROOT / "scripts/summarize_sc_llb_collection.py"),
               str(BASE / "pilot30"), str(BASE / "continuation170"),
               "--output", str(SUMMARY)]
    subprocess.run(command, cwd=ROOT, check=True)
    summary = json.loads(SUMMARY.read_text())
    complete = status["status"] == "continuation_complete" and summary["attempted"] == 200
    episodes = []
    continuation_progress_path = BASE / "continuation170/collection_progress.json"
    continuation_attempts = (json.loads(continuation_progress_path.read_text())
                             .get("attempts", {}) if continuation_progress_path.exists() else {})
    episodes_dir = DATASET / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for row in summary["rows"]:
        if not row["full_insertion"]:
            continue
        audit = Path(row["audit"])
        score_path = audit / "results/scoring.yaml"
        score = yaml.safe_load(score_path.read_text())
        trial = next(value for key, value in score.items() if key.startswith("trial_"))
        bags = sorted((audit / "results").glob("bag_*/metadata.yaml"))
        if len(bags) != 1 or not (audit / "runtime_trace.jsonl").exists():
            raise RuntimeError(f"Full-insertion case lacks one bag or action trace: {row['case']}")
        if float(trial["tier_3"]["score"]) < 75:
            raise RuntimeError(f"Score changed for {row['case']}")
        link = episodes_dir / row["case"]
        destination = audit.resolve()
        if link.is_symlink():
            if link.resolve() != destination:
                raise RuntimeError(f"Existing expert link points elsewhere: {link}")
        elif link.exists():
            raise RuntimeError(f"Expert link path occupied: {link}")
        else:
            link.symlink_to(destination, target_is_directory=True)
        episodes.append({"case": row["case"], "selected_port": row["selected_port"],
                         "card_count": row["card_count"], "score_tier3": row["tier3"],
                         "scene_sha256": row["scene_sha256"],
                         "episode": str(link), "bag": str(bags[0].parent),
                         "action_trace": str(audit / "runtime_trace.jsonl"),
                         "actual_start_contract": continuation_attempts.get(row["case"], {}).get("start_contract")})
    manifest = {"schema": "aic_sc_llb_score_verified/v1", "complete": complete,
                "collection_status": status["status"], "attempted": summary["attempted"],
                "official_full_insertions": len(episodes),
                "scored_bags": summary["scored_bags"],
                "reserved_final_scenes_opened": False,
                "bc_training_started": False,
                "source_summary": str(SUMMARY), "episodes": episodes}
    write_json(DATASET / "manifest.json", manifest)
    final = {"schema": "aic_sc_llb_collection_final/v1", "supervisor": status,
             "complete": complete, "summary": str(SUMMARY),
             "verified_manifest": str(DATASET / "manifest.json"),
             "attempted": summary["attempted"], "full_insertions": len(episodes),
             "scored_bags": summary["scored_bags"],
             "by_card_count": summary["by_card_count"],
             "by_selected_port": summary["by_selected_port"]}
    write_json(BASE / "collection_final_report.json", final)
    marker = "## Final collection result (automatic finalizer)"
    body = (f"\n{marker}\n\n"
            f"Status: **{'complete' if complete else 'stopped early'}**. "
            f"Attempted settings: **{summary['attempted']}/200**. "
            f"Official full insertions: **{len(episodes)}**. "
            f"Valid scored replay bags: **{summary['scored_bags']}**. "
            f"Final scenes opened: **no**. BC training started: **no**.\n\n"
            f"Exact scores and per-case reasons: `outputs/experiments/2026-09-30_sc_llb_collection/summary.json`. "
            f"Only score-verified successes with a bag and action trace are linked in "
            f"`outputs/trajectory_datasets/sc_llb_score_verified/manifest.json`. "
            f"Failures remain in the source collection. "
            f"If stopped early, inspect `outputs/experiments/2026-09-30_sc_llb_collection/supervisor_status.json` "
            f"and `continuation170/collection.log` before resuming.\n")
    record = RECORD.read_text()
    if marker not in record:
        RECORD.write_text(record.rstrip() + "\n" + body)
    outcome_text = (f"**September 30 SC Low Lateral Bypass collection:** "
                    f"{'Completed' if complete else 'Stopped early'} after "
                    f"**{summary['attempted']}/200 attempted settings**, with "
                    f"**{len(episodes)} official full insertions** and "
                    f"**{summary['scored_bags']} valid scored replay bags**. "
                    f"The 30-case pilot passed 17/30; varied robot starts were checked "
                    f"against scored bags. Only full insertions with a bag and action trace "
                    f"are linked in `outputs/trajectory_datasets/sc_llb_score_verified/`. "
                    f"No BC training or final-scene evaluation was started. "
                    f"[Evidence](experiments/2026-09-30-sc-llb-collection.md).")
    replace_line_start(STATUS_DOC, "**September 30 SC Low Lateral Bypass collection:**", outcome_text)
    experiment_text = (f"**September 30 SC teacher collection:** "
                       f"{'Completed' if complete else 'Stopped early'} at "
                       f"{summary['attempted']}/200 attempts, with {len(episodes)} official "
                       f"full insertions. The 30-case pilot passed 17/30. This is privileged "
                       f"teacher data, not autonomous policy performance; no BC training "
                       f"was started. [Scores, videos and method](experiments/2026-09-30-sc-llb-collection.md).")
    replace_line_start(EXPERIMENTS_DOC, "**September 30 SC teacher collection:**", experiment_text)
    dataset_text = (f"The `outputs/experiments/2026-09-30_sc_llb_collection/` teacher "
                    f"collection {'completed' if complete else 'stopped early'} at "
                    f"**{summary['attempted']}/200 attempted settings** with "
                    f"**{len(episodes)} official full insertions** and "
                    f"**{summary['scored_bags']} valid scored bags**. "
                    f"Only successes with a bag and action trace are linked in "
                    f"`outputs/trajectory_datasets/sc_llb_score_verified/manifest.json`. "
                    f"This collection remains separate from canonical `expert_verified`; "
                    f"no training split or BC run was created. "
                    f"[Exact results](experiments/2026-09-30-sc-llb-collection.md).")
    dataset_doc = DATASETS_DOC.read_text()
    section = "## September 30 SC Low Lateral Bypass collection\n"
    start = dataset_doc.index(section) + len(section)
    end = dataset_doc.index("\n## ", start)
    DATASETS_DOC.write_text(dataset_doc[:start] + "\n" + dataset_text + "\n" + dataset_doc[end:])
    print(json.dumps({"complete": complete, "attempted": summary["attempted"],
                      "full_insertions": len(episodes), "manifest": str(DATASET / "manifest.json")}))


if __name__ == "__main__":
    main()

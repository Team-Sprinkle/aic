#!/usr/bin/env python3
"""Read-only causal selected-port anchor replay on frozen SFP predictions.

Simulator labels are loaded only after decisions have been made, for scoring.
This is a diagnostic on development scenes, not a deployment verifier.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


MODELS = ("allport", "structured", "crop")


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return row["episode"], round(row["time_s"], 3)


def point(row):
    value = row.get("predicted_base_m")
    return np.asarray(value, float) if row.get("accepted") and value is not None else None


def summarize(rows, episodes, first_time):
    accepted = [r for r in rows if r["accepted"]]
    by_episode = defaultdict(list)
    for row in accepted:
        by_episode[row["episode"]].append(row)
    return {
        "episodes": len(episodes),
        "episodes_with_accept": len(by_episode),
        "episodes_early_3s": sum(
            min(r["time_s"] for r in group) - first_time[episode] <= 3
            for episode, group in by_episode.items()
        ),
        "accepted_observations": len(accepted),
        "wrong_accepted_observations": sum(r["wrong_label_only"] for r in accepted),
        "episodes_with_wrong_accept": sum(
            any(r["wrong_label_only"] for r in group) for group in by_episode.values()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scores", type=Path,
                        help="Frozen model-score directory; defaults to source/all_ten_model_scores")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve prior diagnostic")

    labels = {key(r): r for r in read_rows(args.source / "labels_all_ten/rows.jsonl")}
    scores = args.scores or args.source / "all_ten_model_scores"
    frozen = {
        name: {key(r): r for r in read_rows(scores / f"{name}.jsonl")}
        for name in ("selected", *MODELS)
    }
    scenes = defaultdict(list)
    for pair, row in labels.items():
        scenes[pair[0]].append(row)

    # Fixed before scoring: an accepted selected-only observation establishes an
    # anchor. Later accepted frozen predictions can confirm it within 5 mm. The
    # anchor expires after 150 mm of unrefreshed measured TCP travel, as in the
    # earlier SC shadow diagnostic. No truth is consulted in these decisions.
    decisions = []
    for episode, rows in sorted(scenes.items()):
        anchor = None
        travel_mm = 0.0
        last_refresh_mm = None
        previous_tcp = None
        for label in sorted(rows, key=lambda r: r["time_s"]):
            pair = key(label)
            tcp = np.asarray(label["state"][:3], float)
            if previous_tcp is not None:
                travel_mm += float(np.linalg.norm(tcp - previous_tcp) * 1000)
            previous_tcp = tcp
            if anchor is not None and travel_mm - last_refresh_mm > 150:
                anchor = None
                last_refresh_mm = None

            selected = frozen["selected"].get(pair)
            selected_point = point(selected) if selected else None
            if selected_point is not None:
                if anchor is None:
                    anchor = selected_point
                    last_refresh_mm = travel_mm
                elif np.linalg.norm(selected_point - anchor) * 1000 <= 5:
                    anchor = selected_point
                    last_refresh_mm = travel_mm
                else:
                    # Conflicting evidence must not silently move the anchor.
                    anchor = None
                    last_refresh_mm = None

            matches = []
            if anchor is not None:
                for name in MODELS:
                    candidate = frozen[name].get(pair)
                    candidate_point = point(candidate) if candidate else None
                    if candidate_point is not None and np.linalg.norm(candidate_point - anchor) * 1000 <= 5:
                        matches.append(name)
            accepted = selected_point is not None or bool(matches)
            # Choose an actually observed estimate, never a truth-derived point.
            estimate = selected_point if selected_point is not None else (
                point(frozen[matches[0]][pair]) if matches else None
            )
            decisions.append({
                "episode": episode,
                "time_s": label["time_s"],
                "requested": label["selected_identity"],
                "accepted": accepted,
                "source": "selected" if selected_point is not None else (matches[0] if matches else None),
                "matching_models": matches,
                "anchor_base_m": anchor.tolist() if anchor is not None else None,
                "predicted_base_m": estimate.tolist() if estimate is not None else None,
                "travel_since_refresh_mm": travel_mm - last_refresh_mm if last_refresh_mm is not None else None,
            })

    # Posthoc scoring pass. This cannot affect any saved decision above.
    for row in decisions:
        row["wrong_label_only"] = False
        if not row["accepted"]:
            continue
        label = labels[key(row)]
        estimate = np.asarray(row["predicted_base_m"])
        nearest = min(label["targets"], key=lambda t: np.linalg.norm(
            estimate - np.asarray(t["position_base_m"])
        ))
        row["nearest_identity_label_only"] = nearest["identity"]
        row["wrong_label_only"] = nearest["identity"] != row["requested"]
        truth = next(t for t in label["targets"] if t["identity"] == row["requested"])
        row["requested_error_mm_label_only"] = float(
            np.linalg.norm(estimate - np.asarray(truth["position_base_m"])) * 1000
        )

    first_time = {ep: min(r["time_s"] for r in rows) for ep, rows in scenes.items()}
    baseline = []
    for pair in sorted(labels):
        r = frozen["selected"].get(pair)
        baseline.append({
            "episode": pair[0], "time_s": pair[1],
            "accepted": bool(r and r["accepted"]),
            "wrong_label_only": bool(r and r["accepted"] and
                                     r["nearest_identity_label_only"] != r["requested"]),
        })
    summary = {
        "schema": "aic_sfp_selected_anchor_tracking_audit/v1",
        "source": str(args.source),
        "scores": str(scores),
        "decision_rule": "selected-only anchor; accepted current candidate within 5 mm; expire after 150 mm unrefreshed measured TCP travel",
        "no_truth_in_decision": True,
        "final_scenes_opened": False,
        "baseline_selected_only": summarize(baseline, scenes, first_time),
        "anchor_tracking": summarize(decisions, scenes, first_time),
        "accepts_without_current_selected": sum(
            r["accepted"] and r["source"] != "selected" for r in decisions
        ),
    }
    args.output.mkdir(parents=True)
    with (args.output / "decisions.jsonl").open("w") as stream:
        for row in decisions:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

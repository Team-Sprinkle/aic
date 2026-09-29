#!/usr/bin/env python3
"""Write a compact machine summary of the matched SC native-crop pilot."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run=args.run
    labels=json.loads(Path("docs/experiments/2026-09-24-sc-native-ablation-labels-matched.json").read_text())
    locator=json.loads((run/"locator/locator_metrics.json").read_text())
    low=json.loads((run/"pose_lowres_warm/metrics.json").read_text())
    crop=json.loads((run/"pose_native_crop_warm/metrics.json").read_text())
    oracle=json.loads((run/"pose_oracle_crop_warm/metrics.json").read_text())
    feature=json.loads((run/"full_crop_fusion/metrics.json").read_text())
    phases=json.loads((run/"full_crop_fusion/phase_metrics.json").read_text())
    scalar=json.loads((run/"fusion_metrics.json").read_text())
    latency=json.loads((run/"offline_latency.json").read_text())
    causal=json.loads((run/"causal_filter.json").read_text())
    def best(history):
        return min(history,key=lambda r:r["validation"]["near_port"]["lateral_mm"]["p95"])
    def probe(data):
        item=best(data["history"])
        return {"selected_epoch":item["epoch"],"near_port":item["validation"]["near_port"]}
    artifacts={key:{"path":str(path),"sha256":sha(path)} for key,path in {
        "native_labels":run/"pose_labels_native.jsonl",
        "lowres_labels":run/"pose_labels_lowres.jsonl",
        "locator":run/"locator/locator.pt",
        "lowres_pose":run/"pose_lowres_warm/best.pt",
        "crop_pose":run/"pose_native_crop_warm/best.pt",
        "oracle_pose":run/"pose_oracle_crop_warm/best.pt",
        "feature_fusion_pose":run/"full_crop_fusion/best.pt",
    }.items()}
    summary={"schema":"sc_native_pose_ablation_summary/v1",
             "status":"development_only_gate_failed",
             "training_label_only":True,"actor_or_rl_trained":False,
             "label_counts":{"admitted_episodes":sum(r.get("admitted",False) for r in labels["episodes"]),
                             "excluded_trials":[r for r in labels["episodes"] if not r.get("admitted")],
                             "train_scenes":labels["train_scene_count"],"validation_scenes":labels["validation_scene_count"],
                             "native_frames":sum(r.get("frames",0) for r in labels["episodes"]),
                             "near_port_frames":sum(r.get("near_port_frames",0) for r in labels["episodes"])},
             "locator_heldout_pixel_error":locator["heldout_error_px"]["validation"],
             "arms":{"lowres_full":probe(low),"learned_native_crop":probe(crop),
                     "oracle_geometry_crop_diagnostic_only":probe(oracle),
                     "scalar_fusion":scalar["fused"]["validation"],
                     "full_plus_native_feature_fusion":phases["validation"]},
             "feature_fusion_train_near":phases["train"],
             "model_parameters":{"feature_fusion_total":23824272,"feature_fusion_trainable":8857353},
             "offline_latency":latency,"gate":{"lateral_p95_mm":0.5,"passed":False,
             "reason":"selected feature fusion validation p95 = 0.914 mm; live end-to-end latency not checked"},
             "causal_fixed_port_filter":causal,
             "artifacts":artifacts,"reserved_final_split_opened":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"episodes":summary["label_counts"]["admitted_episodes"],
                      "selected_lateral_p95_mm":summary["arms"]["full_plus_native_feature_fusion"]["lateral_mm"]["p95"]}))


if __name__=="__main__":main()

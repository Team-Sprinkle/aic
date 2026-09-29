#!/usr/bin/env python3
"""Check scored SC camera files, metadata, and episode-grouped split provenance."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    episodes = [e for e in manifest["episodes"] if e.get("task", {}).get("task_family") == "sc_to_sc"]
    report = []
    split_scenes = {"train": set(), "validation": set()}
    issues = []
    for episode in episodes:
        folder = Path(episode["canonical_episode_dir"])
        metadata = json.loads((folder / "episode.json").read_text())
        rows = [json.loads(line) for line in (folder / "frames.jsonl").open()]
        split_scenes[episode["split"]].add(episode["scene_sha256"])
        indices = sorted({0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1})
        sizes = Counter();modes=Counter();camera_hash_collisions=0
        for index in indices:
            row=rows[index]
            if set(row["images"]) != {"left","center","right"}:
                issues.append(f"{folder}: missing camera at frame {index}")
                continue
            hashes=[]
            for camera in ("left","center","right"):
                path=folder/row["images"][camera]
                if not path.is_file():
                    issues.append(f"{path}: missing")
                    continue
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                with Image.open(path) as im:
                    sizes[str(im.size)]+=1;modes[im.mode]+=1
            if len(hashes)==3 and len(set(hashes))<3:camera_hash_collisions+=1
        if len(rows)!=episode["original_recorded_frames"]:
            issues.append(f"{folder}: frame count mismatch")
        if metadata["image_channel_order"]!="rgb":
            issues.append(f"{folder}: unexpected channel order")
        report.append({"episode_index":episode["episode_index"],"scene_sha256":episode["scene_sha256"],
                       "split":episode["split"],"frames":len(rows),"sampled_frame_indices":indices,
                       "sampled_sizes":dict(sizes),"sampled_modes":dict(modes),
                       "same-frame_camera_hash_collisions":camera_hash_collisions,
                       "timestamp_source":metadata.get("timestamp_source"),
                       "recorded_image_shape_hwc":metadata.get("image_shape_hwc"),
                       "recorded_channel_order":metadata.get("image_channel_order")})
    overlap=split_scenes["train"]&split_scenes["validation"]
    if overlap:issues.append(f"{len(overlap)} scene hashes cross splits")
    result={"schema":"aic_sc_image_provenance/v1","source_manifest":str(a.manifest),
            "episode_count":len(report),"scene_count_by_split":{k:len(v) for k,v in split_scenes.items()},
            "cross_split_scene_hash_overlap":len(overlap),"issues":issues,"episodes":report,
            "sample_scope":"five frames per episode, three cameras; file existence for all strict/held-label anchors is checked by label builders"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("episode_count","scene_count_by_split","cross_split_scene_hash_overlap","issues")}))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Cache explicitly score-verified LeRobot episodes for bounded ACT experiments."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-json", type=Path, required=True)
    parser.add_argument("--source-suffix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--recorded-image-channel-order", choices=["rgb", "bgr"], default=None,
                        help="Audited physical channel order in the stored video; decoding alone cannot establish it.")
    args = parser.parse_args()
    records = [r for r in json.loads(args.verification_json.read_text())
               if r["source"].endswith(args.source_suffix) and r["verified_success_and_lineage"]]
    if not records or len({r["dataset"] for r in records}) != 1:
        raise ValueError("Select one nonempty source with verified scores and action/state lineage")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    dataset = Path(records[0]["dataset"]).resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    frame_table = pd.concat([pd.read_parquet(p) for p in sorted((dataset / "data").rglob("*.parquet"))])
    selected = {r["episode_index"] for r in records}
    frame_table = frame_table[frame_table.episode_index.isin(selected)].sort_values(["episode_index", "frame_index"])
    states = np.stack(frame_table["observation.state"]).astype("float32")
    # Match AICRuntimeFeatureAssembler's quaternion convention exactly.
    states[states[:, 6] < 0, 3:7] *= -1
    actions = np.stack(frame_table.action).astype("float32")
    episodes = frame_table.episode_index.to_numpy(dtype="int64")
    timestamps = frame_table.timestamp.to_numpy(dtype="float32")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite training data")
    for name, data in {"states": states, "actions": actions, "episodes": episodes, "timestamps": timestamps}.items():
        np.save(root / (name + ".npy"), data)
    keys = ["observation.images." + camera + "_camera" for camera in ("center", "left", "right")]
    cameras = {key: np.lib.format.open_memmap(root / (key + ".npy"), mode="w+", dtype="uint8",
                                             shape=(len(states), *info["features"][key]["shape"])) for key in keys}
    metadata = pd.concat([pd.read_parquet(p) for p in sorted((dataset / "meta/episodes").rglob("*.parquet"))]).set_index("episode_index")
    jobs, groups, episode_meta = [], {}, []
    for record in records:
        ep = record["episode_index"]
        indices = np.flatnonzero(episodes == ep)
        row = metadata.loc[ep]
        trial_path = dataset.parent / "trials" / (record["trial_id"] + ".yaml")
        config = yaml.safe_load(trial_path.read_text())
        trial = next(iter(config["trials"].values()))
        nic_count = sum(bool(value.get("entity_present")) for key, value in trial["scene"]["task_board"].items() if key.startswith("nic_rail_"))
        groups.setdefault(nic_count, []).append(ep)
        episode_meta.append({**record, "cache_from_index": int(indices[0]), "cache_to_index": int(indices[-1] + 1),
                             "nic_count": nic_count, "trial_yaml": str(trial_path)})
        for key in keys:
            prefix = "videos/" + key
            path = dataset / info["video_path"].format(video_key=key, chunk_index=int(row[prefix + "/chunk_index"]),
                                                       file_index=int(row[prefix + "/file_index"]))
            jobs.append((key, path, indices, float(row[prefix + "/from_timestamp"])))

    def decode(job):
        key, path, indices, start = job
        # Decode sequentially after one seek; verify frame timestamps at every sample.
        targets = start + np.arange(len(indices)) / info["fps"]
        position = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            container.seek(max(0, int(start / float(stream.time_base))), stream=stream, backward=True)
            for frame in container.decode(stream):
                time = float(frame.pts * stream.time_base)
                if time < targets[position] - .025:
                    continue
                if abs(time - targets[position]) > .026:
                    raise ValueError(f"Video/state timestamp mismatch: {path}, {time} vs {targets[position]}")
                cameras[key][indices[position]] = frame.to_ndarray(format="rgb24")
                position += 1
                if position == len(indices):
                    break
        if position != len(indices):
            raise ValueError(f"Incomplete video {path}: {position}/{len(indices)}")
        return len(indices)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for count in pool.map(decode, jobs):
            completed += count
            if completed // 10000 != (completed - count) // 10000:
                print(f"Decoded {completed}/{len(states) * len(keys)} camera frames", flush=True)
    for camera in cameras.values():
        camera.flush()
    rng = np.random.default_rng(args.seed)
    splits = {}
    for count, eps in sorted(groups.items()):
        ids = rng.permutation(sorted(eps))
        validation_count = max(3, round(len(ids) * .2))
        splits[str(count)] = {"train": sorted(ids[validation_count:].tolist()), "validation": sorted(ids[:validation_count].tolist())}
    description = {"source_dataset": str(dataset), "frames": len(states), "fps": info["fps"], "camera_keys": keys,
                   "image_shape_hwc": info["features"][keys[0]]["shape"], "state_dim": states.shape[1],
                   "action_dim": actions.shape[1], "action_semantics": "Recorded full TCP-relative pose command; unchanged",
                   "state_transform": "Canonicalize quaternion sign to w >= 0, matching runtime",
                   "image_channel_order": args.recorded_image_channel_order,
                   "seed": args.seed, "splits_by_nic_count": splits, "episodes": episode_meta}
    (root / "cache.json").write_text(json.dumps(description, indent=2) + "\n")
    print(json.dumps({k: v for k, v in description.items() if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

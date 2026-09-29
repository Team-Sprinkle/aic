#!/usr/bin/env python3
"""Offline JPEG decode + three-camera pose-probe latency on one GPU.

This excludes live ROS transport, crop selection, actor, and robot command
conversion; it is not the complete <300 ms deployment latency check.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from train_sc_pose_probe import Probe, SCFrames


def quantile(values, fraction):
    ordered=sorted(values)
    return ordered[min(len(ordered)-1,int(fraction*len(ordered)))]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=100)
    a=p.parse_args()
    torch.set_num_threads(8)
    checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if checkpoint["state_only"]:raise ValueError("Visual checkpoint required")
    records=[json.loads(line) for line in a.labels.open() if '"validation"' in line]
    if not records:raise ValueError("No validation record")
    ds=SCFrames(records[:1],"/data1/chmin/yj/ws_aic/src/aic","/workspace/isaaclab/aic",
                int(checkpoint["image_size"]),True,checkpoint["state_mean"],checkpoint["state_std"],
                checkpoint["label_frame"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=Probe(images=True,pretrained_path=None).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    durations=[]
    with torch.inference_mode():
        for i in range(a.repeats+10):
            start=time.perf_counter()
            images,state,*_=ds[0]
            model(images[None].to(device),state[None].to(device))
            if device=="cuda":torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            if i>=10:durations.append(elapsed)
    result={"schema":"aic_sc_pose_probe_offline_latency/v1","checkpoint":str(a.checkpoint),
            "source_image":"one held-out Gazebo SC JPEG triplet, repeatedly decoded from file cache",
            "scope":"JPEG decode, resize, normalization, GPU transfer, shared ResNet-18 and pose head; excludes live ROS/actor/control",
            "repeats":len(durations),"device":device,"p50_ms":statistics.median(durations),
            "p95_ms":quantile(durations,.95),"p99_ms":quantile(durations,.99),
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes":a.checkpoint.stat().st_size}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()

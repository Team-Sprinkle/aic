#!/usr/bin/env python3
"""Run the frozen observation-only pose stack once for an RPDP replay."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("pose_gru_cache",HERE/"train_pose_conditioned_gru_policy.py")
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay",type=Path,required=True);p.add_argument("--dataset-manifest",type=Path,required=True)
    p.add_argument("--pose-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0");a=p.parse_args()
    cache=module.frozen_perception(SimpleNamespace(replay=a.replay,dataset_manifest=a.dataset_manifest,
                                                    pose_checkpoint=a.pose_checkpoint),torch.device(a.device))
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(cache,a.output)
    print({name:{"rows":len(value["transition_indices"]),"episodes":len(set(value["episode_ids"]))}
           for name,value in cache["splits"].items()})


if __name__=="__main__":main()

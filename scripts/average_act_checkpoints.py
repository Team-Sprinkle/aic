#!/usr/bin/env python3
"""Average compatible ACT weights; preserve normalizers and explicit provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils/lerobot_robot_aic"))
from lerobot_robot_aic.act_backbone import read_backbone_geometry


def average_weights(states):
    if len(states) < 2 or any(set(s) != set(states[0]) for s in states[1:]):
        raise ValueError("Need at least two matching state dictionaries")
    result = {}
    for key, first in states[0].items():
        values = [s[key] for s in states]
        if any(v.shape != first.shape or v.dtype != first.dtype for v in values):
            raise ValueError(f"Incompatible parameter: {key}")
        if first.is_floating_point():
            if not all(torch.isfinite(v).all() for v in values):
                raise ValueError(f"Nonfinite parameter: {key}")
            result[key] = sum(v.double() for v in values).div_(len(values)).to(first.dtype).contiguous()
        else:
            if not all(torch.equal(v, first) for v in values):
                raise ValueError(f"Nonfloating buffer changed: {key}")
            result[key] = first.clone()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = [p.resolve() for p in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints) or len(checkpoints) < 2:
        raise ValueError("Need at least two distinct checkpoints")
    if len({p.parents[2] for p in checkpoints}) != 1:
        raise ValueError("This utility averages checkpoints from one training run")
    torch.set_num_threads(4)
    first = checkpoints[0]
    geometry = read_backbone_geometry(first)
    if any(read_backbone_geometry(p) != geometry for p in checkpoints[1:]):
        raise ValueError("Checkpoint backbone geometry differs")
    if not (first / "policy_preprocessor_step_3_normalizer_processor.safetensors").is_file():
        raise ValueError("Missing ACT observation/action normalizer")
    for filename in ["config.json", "aic_action_config.json"]:
        reference = json.loads((first / filename).read_text())
        if any(json.loads((p / filename).read_text()) != reference for p in checkpoints[1:]):
            raise ValueError(f"Checkpoint semantics differ: {filename}")
    for filename in [p.name for p in first.glob("*processor*.safetensors")]:
        reference = load_file(str(first / filename))
        for checkpoint in checkpoints[1:]:
            candidate = load_file(str(checkpoint / filename))
            if set(reference) != set(candidate) or not all(torch.equal(reference[k], candidate[k]) for k in reference):
                raise ValueError(f"Normalization differs: {checkpoint / filename}")
    states = [load_file(str(p / "model.safetensors")) for p in checkpoints]
    averaged = average_weights(states)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    destination = root / "checkpoints/000000/pretrained_model"
    shutil.copytree(first, destination)
    with safe_open(str(first / "model.safetensors"), framework="pt") as source:
        metadata = source.metadata()
    save_file(averaged, str(destination / "model.safetensors"), metadata=metadata)
    manifest = {"kind": "ACT_checkpoint_weight_average", "created_utc": datetime.now(timezone.utc).isoformat(),
                "backbone_geometry": geometry,
                "optimization_updates": 0, "method": "Equal arithmetic mean of floating weights, accumulated in float64; identical nonfloating buffers and normalizers required",
                "components": [{"checkpoint": str(p), "weight": 1 / len(checkpoints),
                                "model_sha256": hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()} for p in checkpoints]}
    training = json.loads((first / "training_step.json").read_text())["training_config"]
    (destination / "training_step.json").write_text(json.dumps({"step": 0, "kind": manifest["kind"],
        "training_config": training, "averaging_provenance": manifest}, indent=2) + "\n")
    (root / "source_training_config.json").write_text(json.dumps(training, indent=2) + "\n")
    (root / "model_average.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

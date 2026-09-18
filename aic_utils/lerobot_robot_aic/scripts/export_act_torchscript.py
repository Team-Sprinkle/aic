#!/usr/bin/env python3
"""Export a LeRobot ACT checkpoint to a TorchScript module for Isaac runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES

from lerobot_robot_aic.act_warmstart import inspect_act_checkpoint, resolve_act_checkpoint_dir
from lerobot_robot_aic.act_state_contract import validate_act_state_contract
from lerobot_robot_aic.act_backbone import load_act_policy, read_backbone_geometry


class ACTTorchScriptWrapper(nn.Module):
    def __init__(self, policy: ACTPolicy):
        super().__init__()
        self.model = policy.model
        self.image_features = list(policy.config.image_features)

    def forward(
        self,
        state: torch.Tensor,
        center_image: torch.Tensor,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
    ) -> torch.Tensor:
        image_by_key = {
            "observation.images.center_camera": center_image,
            "observation.images.left_camera": left_image,
            "observation.images.right_camera": right_image,
        }
        batch = {
            "observation.state": state,
            OBS_IMAGES: [image_by_key[key] for key in self.image_features],
        }
        return self.model(batch)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-channel-order", choices=["rgb", "bgr"], default=None,
                        help="Explicit audited convention for older checkpoints without camera metadata.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = resolve_act_checkpoint_dir(args.act_checkpoint)
    metadata = inspect_act_checkpoint(checkpoint_dir)
    action_config = checkpoint_dir / "aic_action_config.json"
    if action_config.is_file():
        metadata.update(json.loads(action_config.read_text()))
    # Fail before tracing if feature order differs from ROS runtime's contract.
    metadata.update(validate_act_state_contract(metadata))
    if args.image_channel_order:
        metadata["image_channel_order"] = args.image_channel_order
    metadata["backbone_geometry"] = read_backbone_geometry(checkpoint_dir)
    policy = load_act_policy(checkpoint_dir, local_files_only=True)
    policy.eval().to(args.device)
    wrapper = ACTTorchScriptWrapper(policy).eval().to(args.device)
    state_dim = int(metadata["state_shape"][0])
    example = (
        torch.zeros(1, state_dim, device=args.device),
        torch.zeros(1, 3, 256, 288, device=args.device),
        torch.zeros(1, 3, 256, 288, device=args.device),
        torch.zeros(1, 3, 256, 288, device=args.device),
    )
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced(*example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                **metadata,
                "torchscript_path": str(args.output),
                "torchscript_export_device": str(args.device),
                "runtime_note": (
                    "TorchScript traces can specialize internal tensors to the export device. "
                    "Use a CPU export for CPU runtime evaluation and a CUDA export for Isaac GPU training."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote ACT TorchScript: {args.output}")
    print(f"Wrote metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

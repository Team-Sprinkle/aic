#!/usr/bin/env python3
"""Measure an ACT goal head on recorded observations; never controls the robot."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils/lerobot_robot_aic"))
from lerobot_robot_aic.act_backbone import load_act_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="005000")
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--reference-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = args.run_dir / "checkpoints" / args.checkpoint / "pretrained_model"
    config = json.loads((checkpoint / "aic_action_config.json").read_text())
    training = json.loads((checkpoint / "training_step.json").read_text())["training_config"]["args"]
    policy = load_act_policy(checkpoint, local_files_only=True).eval().cuda()
    head = torch.nn.Sequential(torch.nn.Linear(policy.config.dim_model, policy.config.dim_model),
                               torch.nn.ReLU(), torch.nn.Linear(policy.config.dim_model, 6)).eval().cuda()
    head.load_state_dict(torch.load(checkpoint / "auxiliary_goal_head.pt", weights_only=True))
    norm = np.load(args.run_dir / "normalization.npz")
    goal_norm = np.load(args.run_dir / "goal_normalization.npz")
    captured = {}
    policy.model.encoder.register_forward_hook(lambda module, inputs, output: captured.update(value=output))
    paths = sorted(args.rollout_root.glob("task_*/frames.jsonl"))
    if len(paths) != len(args.reference_episodes):
        raise ValueError("Need one source episode per chronologically ordered rollout")
    ref_states = np.load(args.reference_cache / "states.npy", mmap_mode="r")
    ref_eps = np.load(args.reference_cache / "episodes.npy", mmap_mode="r")
    results = []
    with torch.inference_mode():
        for path, episode in zip(paths, args.reference_episodes):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            reference = ref_states[np.flatnonzero(ref_eps == episode)[-1], :3]
            predictions = []
            for row in rows:
                state = np.array(row["tcp_position"] + row["tcp_orientation_xyzw"] +
                                 row["tcp_velocity_linear"] + row["tcp_velocity_angular"] + row["tcp_error"] +
                                 row["joint_positions"] + row["wrist_force"] + row["wrist_torque"], np.float32)
                sign_index = 3 if config["quaternion_sign"] == "x" else 6
                if state[sign_index] < 0:
                    state[3:7] *= -1
                if config["include_elapsed_sim_time"]:
                    state = np.append(state, min(config["time_clip_sec"], row["sim_time"] - rows[0]["sim_time"]))
                batch = {"observation.state": torch.as_tensor((state - norm["state_mean"]) / norm["state_std"],
                                                              dtype=torch.float32, device="cuda")[None]}
                for key in policy.config.image_features:
                    cam = key.removeprefix("observation.images.").removesuffix("_camera")
                    with Image.open(path.parent / row["images"][cam]) as im:
                        pixels = np.asarray(im.convert("RGB").resize((288, 256), Image.Resampling.BOX))
                    if config["image_channel_order"] == "bgr":
                        pixels = pixels[..., ::-1]
                    tensor = torch.as_tensor(np.ascontiguousarray(pixels.transpose(2, 0, 1)), device="cuda").float() / 255
                    mean = torch.tensor([.485, .456, .406], device="cuda")[:, None, None]
                    std = torch.tensor([.229, .224, .225], device="cuda")[:, None, None]
                    batch[key] = ((tensor - mean) / std)[None]
                action = policy.predict_action_chunk(batch)[0, 0].cpu().numpy() * norm["action_std"] + norm["action_mean"]
                if training.get("goal_images_only"):
                    from lerobot.utils.constants import OBS_IMAGES
                    policy.model({"observation.state": torch.zeros_like(batch["observation.state"]),
                                  OBS_IMAGES: [batch[key] for key in policy.config.image_features]})
                goal = head(captured["value"][1])[0].cpu().numpy() * goal_norm["std"] + goal_norm["mean"]
                goal_position = goal[:3]
                if training.get("goal_representation") == "relative_pose":
                    from scipy.spatial.transform import Rotation
                    goal_position = np.asarray(row["tcp_position"]) + Rotation.from_quat(row["tcp_orientation_xyzw"]).apply(goal[:3])
                predictions.append({"elapsed_sim_sec": row["sim_time"] - rows[0]["sim_time"],
                                    "tcp_position": row["tcp_position"], "predicted_action": action.tolist(),
                                    "predicted_goal": goal.tolist(), "predicted_goal_position_base_link": goal_position.tolist(),
                                    "goal_error_mm": float(np.linalg.norm(goal_position - reference) * 1000)})
            results.append({"rollout": str(path.resolve()), "reference_episode": episode,
                            "reference_final_tcp_position": reference.tolist(), "predictions": predictions,
                            "last_10_goal_error_mm": float(np.mean([x["goal_error_mm"] for x in predictions[-10:]]))})
    args.output.write_text(json.dumps({"kind": "offline_diagnostic_only", "trials": results}, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in trial.items() if k != "predictions"} for trial in results], indent=2))


if __name__ == "__main__":
    main()

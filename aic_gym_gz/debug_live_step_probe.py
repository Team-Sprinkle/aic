"""Short live-step probe for the gazebo gym runtime."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aic_gym_gz.env import make_live_env
from aic_gym_gz.run_cheatcode_gym import CheatCodeGymAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-backend", choices=("transport", "cli"), default="transport")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--live-timeout", type=float, default=20.0)
    parser.add_argument("--attach-ready-timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = make_live_env(
        include_images=False,
        enable_randomization=False,
        attach_to_existing=True,
        transport_backend=args.transport_backend,
        timeout=args.live_timeout,
        attach_ready_timeout=args.attach_ready_timeout,
    )
    try:
        t0 = time.time()
        print(json.dumps({"event": "reset_begin", "transport_backend": args.transport_backend}), flush=True)
        observation, info = env.reset(seed=args.seed)
        print(
            json.dumps(
                {
                    "event": "reset_done",
                    "reset_wall_s": round(time.time() - t0, 3),
                    "trial_id": info.get("trial_id"),
                    "step_count": int(observation["step_count"]),
                    "sim_time": float(observation["sim_time"]),
                    "distance_to_target": float(observation["score_geometry"]["distance_to_target"][0]),
                    "distance_to_entrance": float(observation["score_geometry"]["distance_to_entrance"][0]),
                    "insertion_progress": float(observation["score_geometry"]["insertion_progress"][0]),
                    "plug_pose": [round(float(x), 4) for x in np.asarray(observation["plug_pose"][:3], dtype=float).tolist()],
                    "tcp_pose": [round(float(x), 4) for x in np.asarray(observation["tcp_pose"][:3], dtype=float).tolist()],
                    "controller_tcp_pose": [round(float(x), 4) for x in np.asarray(observation["controller_tcp_pose"][:3], dtype=float).tolist()],
                    "target_port_pose": [round(float(x), 4) for x in np.asarray(observation["target_port_pose"][:3], dtype=float).tolist()],
                    "target_port_entrance_pose": [round(float(x), 4) for x in np.asarray(observation["target_port_entrance_pose"][:3], dtype=float).tolist()],
                }
            ),
            flush=True,
        )
        policy = CheatCodeGymAdapter()
        for step_index in range(args.steps):
            action = policy.action(observation)
            wall_start = time.time()
            print(
                json.dumps(
                    {
                        "event": "step_begin",
                        "step_index": step_index,
                        "phase": policy.phase,
                        "action": [round(float(value), 4) for value in action.tolist()],
                    }
                ),
                flush=True,
            )
            observation, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            print(
                json.dumps(
                    {
                        "event": "step_done",
                        "step_index": step_index,
                        "wall_s": round(time.time() - wall_start, 3),
                        "reward": round(float(reward), 4),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "step_count": int(observation["step_count"]),
                        "sim_time": float(observation["sim_time"]),
                        "distance_to_target": float(observation["score_geometry"]["distance_to_target"][0]),
                        "distance_to_entrance": float(observation["score_geometry"]["distance_to_entrance"][0]),
                        "insertion_progress": float(observation["score_geometry"]["insertion_progress"][0]),
                        "plug_pose": [round(float(x), 4) for x in np.asarray(observation["plug_pose"][:3], dtype=float).tolist()],
                        "tcp_pose": [round(float(x), 4) for x in np.asarray(observation["tcp_pose"][:3], dtype=float).tolist()],
                        "controller_tcp_pose": [round(float(x), 4) for x in np.asarray(observation["controller_tcp_pose"][:3], dtype=float).tolist()],
                        "target_port_pose": [round(float(x), 4) for x in np.asarray(observation["target_port_pose"][:3], dtype=float).tolist()],
                        "target_port_entrance_pose": [round(float(x), 4) for x in np.asarray(observation["target_port_entrance_pose"][:3], dtype=float).tolist()],
                    }
                ),
                flush=True,
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

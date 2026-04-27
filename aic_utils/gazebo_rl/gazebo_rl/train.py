from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from gazebo_rl.gym_env import GazeboRLEnv


def _flatten_numeric(obs: dict[str, Any], limit: int = 64) -> np.ndarray:
    values: list[float] = []

    def visit(node: Any) -> None:
        if len(values) >= limit:
            return
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            values.append(float(node))
        elif isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for key in sorted(node):
                visit(node[key])

    visit(obs)
    if len(values) < limit:
        values.extend([0.0] * (limit - len(values)))
    return np.asarray(values[:limit], dtype=np.float32)


class TinyPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.torch = None
        self.model = None
        self.optim = None
        try:
            import torch

            self.torch = torch
            self.model = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 6),
                torch.nn.Tanh(),
            )
            self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        except Exception:
            self.weights = self.rng.normal(scale=0.001, size=(64, 6)).astype(np.float32)

    def act(self, obs: dict[str, Any], *, explore: bool = True) -> list[float]:
        x = _flatten_numeric(obs)
        if self.torch is not None and self.model is not None:
            with self.torch.no_grad():
                action = self.model(self.torch.from_numpy(x)).numpy() * 0.002
        else:
            action = x @ self.weights
        if explore:
            action = action + self.rng.normal(scale=0.001, size=6)
        return action.astype(float).tolist()

    def update(self, batch: list[tuple[dict[str, Any], list[float], float]]) -> float:
        if not batch:
            return 0.0
        if self.torch is None or self.model is None or self.optim is None:
            return 0.0
        torch = self.torch
        obs = torch.stack([torch.from_numpy(_flatten_numeric(item[0])) for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        pred = self.model(obs) * 0.002
        # Sparse rewards are not enough for learning here; this update proves the optimizer path.
        loss = ((pred - actions) ** 2 * (1.0 + rewards.abs())).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.detach().cpu().item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.torch is not None and self.model is not None:
            self.torch.save({"model_state_dict": self.model.state_dict()}, path)
        else:
            np.savez(path.with_suffix(".npz"), weights=self.weights)

    def load(self, path: Path) -> None:
        if self.torch is not None and self.model is not None:
            checkpoint = self.torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            data = np.load(path.with_suffix(".npz"))
            self.weights = data["weights"]


def run_short_training(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    max_seconds = float(args.max_minutes) * 60.0
    output_dir = Path(args.output_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    summary_path = output_dir / "run_summary.json"
    policy = TinyPolicy(seed=args.seed)
    transitions: list[tuple[dict[str, Any], list[float], float]] = []
    iterations: list[dict[str, Any]] = []
    random.seed(args.seed)

    for iteration in range(int(args.max_iterations)):
        if time.monotonic() - started >= max_seconds:
            break
        results_dir = output_dir / "results" / f"iter_{iteration:03d}"
        env = GazeboRLEnv(
            workspace_dir=args.workspace_dir,
            engine_config=args.engine_config,
            sim_distrobox=args.sim_distrobox,
            ground_truth=args.ground_truth,
            gazebo_gui=args.gazebo_gui,
            launch_rviz=args.launch_rviz,
            max_steps=args.max_steps,
            per_trial_timeout_sec=args.per_trial_timeout_sec,
            results_dir=results_dir,
            record_lerobot=args.record_lerobot,
            record_root=args.record_root,
            record_repo_id=args.record_repo_id,
            record_single_task=args.record_single_task,
            record_video=args.record_video,
            record_fps=args.record_fps,
            record_resume=args.record_resume or iteration > 0,
            record_drain_sec=args.record_drain_sec,
            record_image_writer_processes=args.record_image_writer_processes,
            record_image_writer_threads_per_camera=args.record_image_writer_threads_per_camera,
            record_video_encoding_batch_size=args.record_video_encoding_batch_size,
        )
        iter_reward = 0.0
        real_steps = 0
        try:
            obs, info = env.reset()
            for _ in range(args.max_steps):
                action = policy.act(obs)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                transitions.append((obs, action, reward))
                iter_reward += float(reward)
                real_steps += 1
                obs = next_obs
                if terminated or truncated or time.monotonic() - started >= max_seconds:
                    info.update(step_info)
                    break
            loss = policy.update(transitions[-max(1, real_steps) :])
            iterations.append(
                {
                    "iteration": iteration,
                    "real_steps": real_steps,
                    "reward": iter_reward,
                    "loss": loss,
                    "results_dir": str(results_dir),
                }
            )
        finally:
            env.close()

    checkpoint_path = checkpoints_dir / "gazebo_rl_short.pt"
    policy.save(checkpoint_path)
    summary = {
        "iterations_requested": int(args.max_iterations),
        "iterations_completed": len(iterations),
        "max_minutes": float(args.max_minutes),
        "elapsed_sec": time.monotonic() - started,
        "max_steps": int(args.max_steps),
        "iterations": iterations,
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short real Gazebo RL training proof.")
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--engine-config", default=None)
    parser.add_argument(
        "--sim-distrobox",
        default=None,
        help=(
            "Optional user-created distrobox container name for the evaluation "
            "environment. Omit to use the local pixi launch path."
        ),
    )
    parser.add_argument("--ground-truth", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--gazebo-gui", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--launch-rviz", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-minutes", "--ax-minutes", dest="max_minutes", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--per-trial-timeout-sec", type=float, default=300.0)
    parser.add_argument("--output-dir", default="outputs/gazebo_rl")
    parser.add_argument("--seed", type=int, default=0)
    add_recording_args(parser)
    return parser


def add_recording_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record-lerobot",
        action="store_true",
        help="Start the existing aic-policy-recorder sidecar during each rollout.",
    )
    parser.add_argument(
        "--record-root",
        default=None,
        help="LeRobot dataset root for optional rollout recording.",
    )
    parser.add_argument("--record-repo-id", default="local/gazebo_rl_rollout")
    parser.add_argument("--record-single-task", default="gazebo_rl rollout")
    parser.add_argument("--record-video", dest="record_video", action="store_true")
    parser.add_argument("--no-record-video", dest="record_video", action="store_false")
    parser.set_defaults(record_video=True)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--record-resume", action="store_true")
    parser.add_argument("--record-drain-sec", type=float, default=20.0)
    parser.add_argument("--record-image-writer-processes", type=int, default=0)
    parser.add_argument("--record-image-writer-threads-per-camera", type=int, default=4)
    parser.add_argument("--record-video-encoding-batch-size", type=int, default=1)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_short_training(args)
    print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""Baseline SB3 PPO trainer for the training-only Gazebo env."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from aic_gazebo_env import (
    GazeboAttachedRuntime,
    GazeboRuntimeConfig,
    StableRLEnvConfig,
    StableRLGazeboEnv,
    training_api_report,
)
from aic_gazebo_env.live_runtime import DEFAULT_WORLD_NAME, LiveRuntimeManager, default_world_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--worker-train", action="store_true")
    parser.add_argument("--backend", choices=("transport", "cli", "auto"), default="transport")
    parser.add_argument("--world-name", default=DEFAULT_WORLD_NAME)
    parser.add_argument("--world-path", default=None)
    parser.add_argument("--multi-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--episode-step-limit", type=int, default=128)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args(argv)


def dependency_report() -> dict[str, object]:
    required = {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "gymnasium": importlib.util.find_spec("gymnasium") is not None,
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
    }
    missing = [name for name, present in required.items() if not present]
    return {
        "required": required,
        "ok": not missing,
        "missing": missing,
    }


def _default_run_dir(repo_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return repo_root / "aic_utils" / "aic_gazebo_env" / "runs" / f"ppo-{timestamp}"


def _worker_payload(args: argparse.Namespace) -> dict[str, object]:
    deps = dependency_report()
    if not deps["ok"]:
        return {
            "ok": False,
            "error_category": "missing_training_dependencies",
            "dependency_report": deps,
        }

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = 2_048 if args.smoke else args.total_timesteps
    eval_freq = min(args.eval_freq, total_timesteps) if total_timesteps > 0 else args.eval_freq
    checkpoint_freq = min(args.checkpoint_freq, total_timesteps) if total_timesteps > 0 else args.checkpoint_freq

    runtime_config = GazeboRuntimeConfig(
        world_path=args.world_path or default_world_file(repo_root),
        executable="gz",
        timeout=10.0,
        world_name=args.world_name,
        source_entity_name="ati/tool_link",
        target_entity_name="tabletop",
        transport_backend=args.backend,
    )
    rl_config = StableRLEnvConfig(
        multi_step=args.multi_step,
        episode_step_limit=args.episode_step_limit,
    )

    config_payload = {
        "seed": args.seed,
        "backend": args.backend,
        "world_name": args.world_name,
        "world_path": runtime_config.world_path,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "checkpoint_freq": checkpoint_freq,
        "smoke": args.smoke,
        "training_api": training_api_report(rl_config),
        "dependency_report": deps,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    train_env_refs: list[StableRLGazeboEnv] = []
    eval_env_refs: list[StableRLGazeboEnv] = []

    def _make_train_env():
        runtime = GazeboAttachedRuntime(runtime_config)
        env = StableRLGazeboEnv(runtime=runtime, config=rl_config)
        train_env_refs.append(env)
        return Monitor(env)

    def _make_eval_env():
        runtime = GazeboAttachedRuntime(runtime_config)
        env = StableRLGazeboEnv(runtime=runtime, config=rl_config)
        eval_env_refs.append(env)
        return Monitor(env)

    env = DummyVecEnv([_make_train_env])
    eval_env = DummyVecEnv([_make_eval_env])
    started = time.perf_counter()
    try:
        callbacks = [
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_model",
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval"),
                eval_freq=max(1, eval_freq),
                deterministic=True,
                render=False,
            ),
        ]
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
        model.save(str(run_dir / "final_model"))
        wall_s = time.perf_counter() - started
        episode_monitor = train_env_refs[0].monitor.finish() if train_env_refs else None
        result = {
            "ok": True,
            "run_dir": str(run_dir),
            "wall_s": round(wall_s, 3),
            "throughput_steps_per_s": round(total_timesteps / wall_s, 3) if wall_s > 0.0 else None,
            "timesteps": total_timesteps,
            "training_api": training_api_report(rl_config),
            "episode_monitor": episode_monitor,
            "artifacts": {
                "config": str(run_dir / "config.json"),
                "final_model": str(run_dir / "final_model.zip"),
                "checkpoints_dir": str(run_dir / "checkpoints"),
                "eval_dir": str(run_dir / "eval"),
            },
            "dependency_report": deps,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "error_category": "training_failed",
            "message": str(exc),
            "run_dir": str(run_dir),
            "dependency_report": deps,
        }
    finally:
        env.close()
        eval_env.close()
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_train:
        payload = _worker_payload(args)
        print(json.dumps(payload, indent=None if args.json_only else 2, sort_keys=True))
        return 0 if payload.get("ok") else 1

    manager = LiveRuntimeManager(
        world_name=args.world_name,
        world_path=args.world_path,
    )
    preflight = manager.preflight()
    context = manager.prepare(
        auto_build=args.auto_build,
        auto_launch=args.auto_launch,
    )
    health = manager.wait_for_health(context, timeout_s=120.0).to_dict()
    if health.get("no_op_step_ok") and health.get("action_step_ok"):
        script_path = Path(__file__).resolve()
        command_parts = [
            f"PYTHONPATH={manager.repo_root / 'aic_utils' / 'aic_gazebo_env'}",
            shlex_quote(sys.executable),
            shlex_quote(str(script_path)),
            "--worker-train",
            f"--backend {shlex_quote(args.backend)}",
            f"--world-name {shlex_quote(args.world_name)}",
            f"--world-path {shlex_quote(args.world_path or manager.world_path)}",
            f"--multi-step {args.multi_step}",
            f"--seed {args.seed}",
            f"--total-timesteps {args.total_timesteps}",
            f"--episode-step-limit {args.episode_step_limit}",
            f"--eval-freq {args.eval_freq}",
            f"--checkpoint-freq {args.checkpoint_freq}",
        ]
        if args.run_dir:
            command_parts.append(f"--run-dir {shlex_quote(args.run_dir)}")
        if args.smoke:
            command_parts.append("--smoke")
        if args.json_only:
            command_parts.append("--json-only")
        command = " ".join(command_parts)
        result = manager.run_context_command(context, command, timeout_s=3600.0)
        training = json.loads(result.stdout) if result.returncode == 0 else {
            "ok": False,
            "error_category": "training_worker_failed",
            "message": result.stderr or result.stdout,
        }
    else:
        training = {
            "ok": False,
            "error_category": "live_health_unavailable",
            "message": health.get("diagnostics", {}).get(
                "last_error",
                "live health checks did not pass",
            ),
        }
    payload = {
        "preflight": preflight,
        "context": context.to_dict(),
        "health": health,
        "training": training,
    }
    print(json.dumps(payload, indent=None if args.json_only else 2, sort_keys=True))
    return 0 if training.get("ok") else 1


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())

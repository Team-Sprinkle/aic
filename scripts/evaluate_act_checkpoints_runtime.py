#!/usr/bin/env python3
"""Evaluate saved ACT checkpoints with the rootless AIC runtime container.

This is intended to run as a sidecar while LeRobot training writes checkpoints.
It polls ``<run-dir>/checkpoints/*/pretrained_model`` and evaluates each new
checkpoint once through the official AIC runtime stack in the ``aic_eval``
container.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--container", default="aic_eval")
    parser.add_argument("--workspace-host", type=Path, default=Path.cwd())
    parser.add_argument("--workspace-container", default="/home/chmin/yj/ws_aic/src/aic")
    parser.add_argument(
        "--docker-host",
        default=f"unix:///run/user/{os.getuid()}/docker.sock",
        help="Rootless Docker socket.",
    )
    parser.add_argument("--checkpoint-glob", default="checkpoints/[0-9]*/pretrained_model")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once-existing", action="store_true")
    parser.add_argument("--max-runtime-sec", type=float, default=12.0)
    parser.add_argument("--start-delay-sec", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--command-mode", default="none", choices=["none", "velocity", "delta_pose"])
    parser.add_argument("--command-frame", default="base_link")
    parser.add_argument("--max-translation-delta", type=float, default=0.02)
    parser.add_argument("--max-rotation-delta", type=float, default=0.2)
    parser.add_argument("--sim-wait-sec", type=float, default=25.0)
    parser.add_argument("--readiness-timeout-sec", type=int, default=120)
    parser.add_argument("--engine-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--engine-config",
        default="/home/chmin/yj/ws_aic/src/aic/aic_engine/config/sample_config.yaml",
    )
    return parser.parse_args()


def docker_cmd(args: argparse.Namespace, *parts: str) -> list[str]:
    return ["docker", "--host", args.docker_host, *parts]


def host_to_container(path: Path, args: argparse.Namespace) -> str:
    resolved = path.resolve()
    workspace_host = args.workspace_host.resolve()
    try:
        rel = resolved.relative_to(workspace_host)
    except ValueError as exc:
        raise ValueError(f"{resolved} is not under workspace host {workspace_host}") from exc
    return str(Path(args.workspace_container) / rel)


def run_capture(cmd: list[str], log_path: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    return result


def container_bash(args: argparse.Namespace, script: str) -> list[str]:
    return docker_cmd(args, "exec", "-i", args.container, "bash", "-lc", script)


def start_long_container_bash(args: argparse.Namespace, script: str, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        container_bash(args, script),
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def restart_container(args: argparse.Namespace, log_path: Path) -> int:
    result = run_capture(docker_cmd(args, "restart", args.container), log_path=log_path)
    return result.returncode


def wait_for_policy_ready(args: argparse.Namespace, node_name: str, action_name: str, log_path: Path) -> bool:
    script = f"""
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
for i in $(seq 1 {args.readiness_timeout_sec}); do
  state=$(ros2 lifecycle get /{node_name} 2>/dev/null || true)
  actions=$(ros2 action list 2>/dev/null || true)
  if printf "%s" "$state" | grep -q "unconfigured" && printf "%s" "$actions" | grep -q "^/{action_name}$"; then
    echo ready
    echo "$state"
    ros2 action list
    exit 0
  fi
  sleep 1
done
echo not_ready
ros2 lifecycle get /{node_name} || true
ros2 action list || true
exit 1
"""
    result = run_capture(container_bash(args, script), log_path=log_path)
    return result.returncode == 0


def evaluate_checkpoint(checkpoint_path: Path, args: argparse.Namespace) -> dict[str, object]:
    step = checkpoint_path.parent.name
    eval_dir = args.run_dir.resolve() / "runtime_eval" / step
    logs_dir = eval_dir / "logs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_container = host_to_container(checkpoint_path, args)
    eval_container = host_to_container(eval_dir, args)
    node_name = f"aic_model_act_{step}"
    action_name = f"insert_cable_act_{step}"

    summary: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_container": checkpoint_container,
        "eval_dir": str(eval_dir),
        "eval_dir_container": eval_container,
        "node_name": node_name,
        "action_name": action_name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    restart_container(args, logs_dir / "container_restart_before.log")
    sim_script = f"""
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
cd {args.workspace_container}
/entrypoint.sh ground_truth:=false start_aic_engine:=false gazebo_gui:=false launch_rviz:=false
"""
    sim_proc = start_long_container_bash(args, sim_script, logs_dir / "sim.log")
    time.sleep(args.sim_wait_sec)

    policy_script = f"""
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
cd {args.workspace_container}
export PYTHONPATH={args.workspace_container}/scripts/pythonpath_bootstrap:$PYTHONPATH
export AIC_CHECKOUT_PYTHONPATH={args.workspace_container}/aic_example_policies
export AIC_ACT_POLICY_PATH={checkpoint_container}
export AIC_ACT_DEVICE=cuda
export AIC_ACT_MAX_RUNTIME_SEC={args.max_runtime_sec}
export AIC_ACT_START_DELAY_SEC={args.start_delay_sec}
export AIC_ACT_CONTROL_HZ={args.control_hz}
export AIC_ACT_COMMAND_MODE={args.command_mode}
export AIC_ACT_COMMAND_FRAME={args.command_frame}
export AIC_ACT_MAX_TRANSLATION_DELTA={args.max_translation_delta}
export AIC_ACT_MAX_ROTATION_DELTA={args.max_rotation_delta}
pixi run ros2 run aic_model aic_model --ros-args \\
  -p use_sim_time:=true \\
  -p policy:=aic_example_policies.ros.RunACT \\
  -r __node:={node_name} \\
  -r /insert_cable:=/{action_name}
"""
    policy_proc = start_long_container_bash(args, policy_script, logs_dir / "policy.log")
    ready = wait_for_policy_ready(args, node_name, action_name, logs_dir / "readiness.log")
    summary["policy_ready"] = ready
    if ready:
        engine_script = f"""
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
cd {args.workspace_container}
export AIC_RESULTS_DIR={eval_container}
mkdir -p "$AIC_RESULTS_DIR"
ros2 run aic_engine aic_engine --ros-args \\
  -r /insert_cable:=/{action_name} \\
  -p use_sim_time:=true \\
  -p config_file_path:={args.engine_config} \\
  -p model_node_name:={node_name} \\
  -p model_discovery_timeout_seconds:=60 \\
  -p model_configure_timeout_seconds:=120
"""
        try:
            engine = run_capture(
                container_bash(args, engine_script),
                log_path=logs_dir / "engine.log",
                timeout=args.engine_timeout_sec,
            )
            summary["engine_returncode"] = engine.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            (logs_dir / "engine.log").write_text(stdout, encoding="utf-8")
            summary["engine_returncode"] = None
            summary["failure_reason"] = f"aic_engine timed out after {args.engine_timeout_sec} seconds"
    else:
        summary["engine_returncode"] = None

    for proc in (policy_proc, sim_proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    restart_container(args, logs_dir / "container_restart_after.log")

    scoring = eval_dir / "scoring.yaml"
    summary["scoring_yaml"] = str(scoring) if scoring.exists() else None
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (eval_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def iter_checkpoints(args: argparse.Namespace) -> list[Path]:
    return sorted(path for path in args.run_dir.glob(args.checkpoint_glob) if (path / "model.safetensors").exists())


def main() -> int:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.workspace_host = args.workspace_host.resolve()
    evaluated: set[Path] = set()
    while True:
        for checkpoint in iter_checkpoints(args):
            if checkpoint in evaluated:
                continue
            marker = args.run_dir / "runtime_eval" / checkpoint.parent.name / "eval_summary.json"
            if marker.exists():
                evaluated.add(checkpoint)
                continue
            summary = evaluate_checkpoint(checkpoint, args)
            print(json.dumps(summary, sort_keys=True), flush=True)
            evaluated.add(checkpoint)
        if args.once_existing:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

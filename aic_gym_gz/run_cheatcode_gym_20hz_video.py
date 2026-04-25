"""Run cheatcode policy with synchronous 20 Hz Gazebo-native video capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from aic_gym_gz.env import make_live_env
from aic_gym_gz.io import CameraBridgeSidecar, _gazebo_image_text_to_array
from aic_gym_gz.official_scene import bringup_launch_args_for_scenario, export_training_world_for_scenario
from aic_gym_gz.randomizer import AicEnvRandomizer
from aic_gym_gz.run_cheatcode_gym import CheatCodeGymAdapter
from aic_gym_gz.utils import to_jsonable
from aic_gym_gz.video import build_run_name


TOPICS = {
    "camera_left": "/left_camera/image",
    "camera_center": "/center_camera/image",
    "camera_right": "/right_camera/image",
    "overview_top_down_xy": "/overview_camera/image",
    "overview_front_xz": "/overview_front_camera/image",
    "overview_side_yz": "/overview_side_camera/image",
    "overview_oblique_xy": "/overview_oblique_camera/image",
}


def _stop_existing_gazebo_processes() -> None:
    pattern = (
        "component_container|ros_gz_container|ros2 launch|gz sim|"
        "parameter_bridge|aic_gz_transport_bridge|gz topic -e"
    )
    command = (
        "ps -eo pid=,args= | "
        f"awk '/{pattern}/ && !/awk/ {{print $1}}' | "
        "xargs -r kill -TERM; "
        "sleep 2; "
        "ps -eo pid=,args= | "
        f"awk '/{pattern}/ && !/awk/ {{print $1}}' | "
        "xargs -r kill -KILL; true"
    )
    subprocess.run(["bash", "-lc", command], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_world_control(*, world_name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["gz", "service", "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if completed.returncode == 0 and f"/world/{world_name}/control" in completed.stdout:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for /world/{world_name}/control.")


def _start_official_bringup(scenario: Any, *, log_path: Path) -> subprocess.Popen[str]:
    _stop_existing_gazebo_processes()
    launch_args = bringup_launch_args_for_scenario(
        scenario,
        ground_truth=False,
        start_aic_engine=False,
        gazebo_gui=False,
        launch_rviz=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["ros2", "launch", "aic_bringup", "aic_gz_bringup.launch.py", *launch_args],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_file.close()
    _wait_for_world_control(world_name="aic_world", timeout_s=90.0)
    return process


def _stop_process_tree(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.0)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            process.wait(timeout=5.0)
        except Exception:
            pass


def _resize_camera_images_in_sdf(path: Path, *, width: int, height: int) -> None:
    text = path.read_text(encoding="utf-8")

    def prepare_sensor(match: re.Match[str]) -> str:
        block = match.group(0)
        if "<always_on>" not in block:
            block = re.sub(
                r"(<sensor\b[^>]*type=['\"]camera['\"][^>]*>\s*)",
                r"\1<always_on>true</always_on>\n",
                block,
                count=1,
            )
        return block

    def replace_image(match: re.Match[str]) -> str:
        block = match.group(0)
        block = re.sub(r"<width>\s*\d+\s*</width>", f"<width>{width}</width>", block)
        block = re.sub(r"<height>\s*\d+\s*</height>", f"<height>{height}</height>", block)
        return block

    text = re.sub(
        r"<sensor\b[^>]*type=['\"]camera['\"][\s\S]*?</sensor>",
        prepare_sensor,
        text,
    )
    text = re.sub(r"<image>\s*.*?\s*</image>", replace_image, text, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def _start_image_captures(
    topics: dict[str, str],
    *,
    capture_dir: Path,
) -> dict[str, tuple[subprocess.Popen[str], Path]]:
    env = dict(os.environ)
    env.setdefault("GZ_IP", "127.0.0.1")
    processes: dict[str, tuple[subprocess.Popen[str], Path]] = {}
    for name, topic in topics.items():
        output_path = capture_dir / f"{name}.txt"
        output = output_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["gz", "topic", "-e", "-n", "1", "-t", topic],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        output.close()
        processes[name] = (process, output_path)
    return processes


def _collect_image_captures(
    processes: dict[str, tuple[subprocess.Popen[str], Path]],
    *,
    timeout_s: float,
    image_shape: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    frames: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    deadline = time.monotonic() + timeout_s
    for name, (process, output_path) in processes.items():
        remaining = max(deadline - time.monotonic(), 0.1)
        try:
            _, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            errors[name] = "timeout"
            continue
        stdout = output_path.read_text(encoding="utf-8", errors="replace")
        if process.returncode != 0 or not stdout.strip():
            errors[name] = stderr.strip() or f"empty capture returncode={process.returncode}"
            continue
        frame = _gazebo_image_text_to_array(stdout, expected_shape=image_shape)
        if frame is None or frame.size == 0 or int(frame.sum()) == 0:
            errors[name] = "blank_or_unparseable"
            continue
        frames[name] = np.asarray(frame, dtype=np.uint8)
    return frames, errors


def _open_writers(output_dir: Path, *, fps: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: imageio.get_writer(
            output_dir / f"{name}.mp4",
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        )
        for name in TOPICS
    }


def _nonblank_frames(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(frame, dtype=np.uint8)
        for name, frame in frames.items()
        if frame is not None and np.asarray(frame).size > 0 and int(np.asarray(frame).sum()) > 0
    }


def _start_ros_file_subscriber(
    *,
    output_dir: Path,
    image_shape: tuple[int, int, int],
) -> subprocess.Popen[str]:
    script = Path(__file__).with_name("ros_image_file_subscriber.py")
    return subprocess.Popen(
        [
            "/usr/bin/python3",
            str(script),
            "--topics-json",
            json.dumps(TOPICS, sort_keys=True),
            "--output-dir",
            str(output_dir),
            "--height",
            str(int(image_shape[0])),
            "--width",
            str(int(image_shape[1])),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _wait_for_ros_file_frames(
    frame_dir: Path,
    *,
    previous_timestamps: dict[str, float],
    timeout_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, str]]:
    deadline = time.monotonic() + timeout_s
    last_frames: dict[str, np.ndarray] = {}
    last_timestamps: dict[str, float] = {}
    while time.monotonic() < deadline:
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        for name in TOPICS:
            meta_path = frame_dir / f"{name}.json"
            frame_path = frame_dir / f"{name}.npy"
            if not meta_path.exists() or not frame_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                frame = np.load(frame_path)
            except Exception:
                continue
            frames[name] = np.asarray(frame, dtype=np.uint8)
            timestamps[name] = float(meta.get("timestamp", 0.0))
        live_frames = _nonblank_frames(frames)
        ready = {
            name
            for name in TOPICS
            if name in live_frames and float(timestamps.get(name, 0.0)) > float(previous_timestamps.get(name, 0.0))
        }
        last_frames = live_frames
        last_timestamps = dict(timestamps)
        if len(ready) == len(TOPICS):
            return (
                {name: live_frames[name] for name in TOPICS},
                last_timestamps,
                {},
            )
        time.sleep(0.02)
    errors = {
        name: "stale_or_missing_ros_frame"
        for name in TOPICS
        if name not in last_frames
        or float(last_timestamps.get(name, 0.0)) <= float(previous_timestamps.get(name, 0.0))
    }
    return last_frames, last_timestamps, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--capture-timeout", type=float, default=8.0)
    parser.add_argument("--subscriber-settle-s", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--no-cleanup-existing-gazebo", action="store_true")
    parser.add_argument("--capture-backend", choices=("ros", "gz"), default="ros")
    parser.add_argument("--require-fresh-frames", action="store_true")
    args = parser.parse_args()

    if int(args.ticks_per_step) != 25:
        raise ValueError("This runner is intended for 20 Hz .step(); use --ticks-per-step 25.")
    fps = 1.0 / (int(args.ticks_per_step) * 0.002)
    image_shape = (int(args.image_size), int(args.image_size), 3)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scenario = AicEnvRandomizer(enable_randomization=False).sample(
        seed=args.seed,
        trial_id=args.trial_id,
    )
    world_path = summary_path.parent / (
        build_run_name(prefix="gazebo_training_fast_world_20hz", seed=args.seed, trial_id=args.trial_id)
        + ".sdf"
    )
    world_metadata = export_training_world_for_scenario(scenario, output_path=world_path)
    _resize_camera_images_in_sdf(world_path, width=args.image_size, height=args.image_size)
    launch_process = _start_official_bringup(
        scenario,
        log_path=summary_path.parent / "official_bringup.log",
    )

    env = make_live_env(
        include_images=False,
        enable_randomization=False,
        ticks_per_step=int(args.ticks_per_step),
        world_path=str(world_path),
        attach_to_existing=True,
        transport_backend="transport",
        timeout=20.0,
        attach_ready_timeout=90.0,
        image_observation_mode="async_training",
        observation_transport_override="persistent",
        state_observation_mode="synthetic_training",
    )
    writers = _open_writers(output_dir, fps=fps)
    ros_bridge: CameraBridgeSidecar | None = None
    ros_subscriber: subprocess.Popen[str] | None = None
    ros_frame_dir = summary_path.parent / "ros_frames"
    ros_timestamps: dict[str, float] = {name: 0.0 for name in TOPICS}
    if args.capture_backend == "ros":
        ros_bridge = CameraBridgeSidecar(topic_map=dict(TOPICS))
        ros_bridge.start()
        ros_subscriber = _start_ros_file_subscriber(output_dir=ros_frame_dir, image_shape=image_shape)
    frame_errors: list[dict[str, Any]] = []
    step_walls: list[float] = []
    sim_times: list[float] = []
    try:
        with tempfile.TemporaryDirectory(prefix="aic_gz_frames_") as capture_tmp:
            capture_dir = Path(capture_tmp)
            observation, info = env.reset(seed=args.seed, options={"trial_id": args.trial_id} if args.trial_id else {})
            for warmup_idx in range(max(int(args.warmup_steps), 0)):
                previous_ros_timestamps = dict(ros_timestamps)
                processes = (
                    {}
                    if ros_subscriber is not None
                    else _start_image_captures(TOPICS, capture_dir=capture_dir)
                )
                if ros_subscriber is None:
                    time.sleep(max(float(args.subscriber_settle_s), 0.0))
                env._state = env.runtime.step(np.zeros(6, dtype=np.float32), ticks=int(args.ticks_per_step))
                if ros_subscriber is not None:
                    frames, ros_timestamps, errors = _wait_for_ros_file_frames(
                        ros_frame_dir,
                        previous_timestamps=previous_ros_timestamps,
                        timeout_s=float(args.capture_timeout),
                    )
                else:
                    frames, errors = _collect_image_captures(
                        processes,
                        timeout_s=float(args.capture_timeout),
                        image_shape=image_shape,
                    )
                print(
                    json.dumps(
                        {
                            "stage": "warmup_capture",
                            "attempt": warmup_idx + 1,
                            "frames": sorted(frames),
                            "errors": errors,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if len(frames) == len(TOPICS):
                    break
            policy = CheatCodeGymAdapter()
            terminated = truncated = False
            step_count = 0
            total_reward = 0.0
            last_info = dict(info)
            while not (terminated or truncated) and step_count < int(args.max_steps):
                action = policy.action(observation)
                previous_ros_timestamps = (
                    dict(ros_timestamps)
                    if args.require_fresh_frames
                    else {name: 0.0 for name in TOPICS}
                )
                processes = (
                    {}
                    if ros_subscriber is not None
                    else _start_image_captures(TOPICS, capture_dir=capture_dir)
                )
                if ros_subscriber is None:
                    time.sleep(max(float(args.subscriber_settle_s), 0.0))
                start = time.perf_counter()
                observation, reward, terminated, truncated, last_info = env.step(action.astype(np.float32))
                step_walls.append(time.perf_counter() - start)
                total_reward += float(reward)
                if ros_subscriber is not None:
                    frames, ros_timestamps, errors = _wait_for_ros_file_frames(
                        ros_frame_dir,
                        previous_timestamps=previous_ros_timestamps,
                        timeout_s=float(args.capture_timeout) if args.require_fresh_frames else 0.5,
                    )
                else:
                    frames, errors = _collect_image_captures(
                        processes,
                        timeout_s=float(args.capture_timeout),
                        image_shape=image_shape,
                    )
                if len(frames) != len(TOPICS):
                    frame_errors.append({"step": step_count + 1, "errors": errors})
                    missing = sorted(set(TOPICS) - set(frames))
                    raise RuntimeError(f"Missing required Gazebo camera frames at step {step_count + 1}: {missing}")
                for name, writer in writers.items():
                    writer.append_data(frames[name])
                sim_times.append(float(observation["sim_time"]))
                step_count += 1
                print(
                    json.dumps(
                        {
                            "stage": "step_done",
                            "step": step_count,
                            "step_wall_s": step_walls[-1],
                            "sim_time": sim_times[-1],
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if step_count >= int(args.max_steps) and not (terminated or truncated):
            truncated = True
            last_info = dict(last_info)
            last_info["termination_reason"] = "max_steps_guard"
        streams = {}
        for name, writer in writers.items():
            writer.close()
            streams[name] = {
                "path": str(output_dir / f"{name}.mp4"),
                "frame_count": step_count,
                "fps": fps,
            }
        payload = {
            "ok": True,
            "seed": args.seed,
            "trial_id": info.get("trial_id"),
            "ticks_per_step": int(args.ticks_per_step),
            "sim_dt": 0.002,
            "step_hz": fps,
            "video_fps": fps,
            "capture_backend": args.capture_backend,
            "require_fresh_frames": bool(args.require_fresh_frames),
            "image_shape": list(image_shape),
            "length": step_count,
            "return": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "mean_env_step_wall_s": sum(step_walls) / len(step_walls) if step_walls else None,
            "max_env_step_wall_s": max(step_walls) if step_walls else None,
            "first_sim_time": sim_times[0] if sim_times else None,
            "last_sim_time": sim_times[-1] if sim_times else None,
            "world_path": str(world_path),
            "world_metadata": world_metadata,
            "official_bringup_log": str(summary_path.parent / "official_bringup.log"),
            "streams": streams,
            "frame_errors": frame_errors,
            "final_info": last_info,
        }
        summary_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), flush=True)
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        if ros_subscriber is not None:
            ros_subscriber.terminate()
            try:
                ros_subscriber.wait(timeout=3.0)
            except Exception:
                ros_subscriber.kill()
        if ros_bridge is not None:
            ros_bridge.close()
        env.close()
        _stop_process_tree(launch_process if "launch_process" in locals() else None)
        if not args.no_cleanup_existing_gazebo:
            _stop_existing_gazebo_processes()


if __name__ == "__main__":
    main()

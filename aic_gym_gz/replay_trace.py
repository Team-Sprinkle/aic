"""Replay a fixed action trace against the live attached aic_gym_gz backend."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aic_gym_gz.io import RosCameraSubscriber, summarize_image_batch
    from aic_gym_gz.parity import AicParityHarness
    from aic_gym_gz.randomizer import AicEnvRandomizer
    from aic_gym_gz.runtime import AicGazeboRuntime, ScenarioGymGzBackend
else:
    from .io import RosCameraSubscriber, summarize_image_batch
    from .parity import AicParityHarness
    from .randomizer import AicEnvRandomizer
    from .runtime import AicGazeboRuntime, ScenarioGymGzBackend


class CameraBridgeSidecar:
    """Dedicated non-lazy camera bridge for replay image capture."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "/left_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/center_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/right_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        time.sleep(1.0)

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3.0)
        self._process = None


def _candidate_record(*, step_idx: int, action: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_idx": step_idx,
        "action": action,
        "native": native,
    }


def replay_trace_against_attached_runtime(
    *,
    trace_report: dict[str, Any],
    include_images: bool = False,
) -> dict[str, Any]:
    randomizer = AicEnvRandomizer(enable_randomization=False)
    scenario = randomizer.sample(seed=123)
    backend = ScenarioGymGzBackend(
        attach_to_existing=True,
        world_path="/home/ubuntu/ws_aic/src/aic/aic_description/world/aic.sdf",
        transport_backend="transport",
    )
    runtime = AicGazeboRuntime(
        backend=backend,
        ticks_per_step=25,
    )
    camera_bridge = CameraBridgeSidecar() if include_images else None
    camera_subscriber = RosCameraSubscriber() if include_images else None
    try:
        if camera_bridge is not None:
            camera_bridge.start()
        del scenario
        backend.connect_existing_world()
        candidate = {
            "mode": "aic_gym_gz_attached_runtime_replay",
            "initial_native": _candidate_record(
                step_idx=-1,
                action={"linear_xyz": [0.0, 0.0, 0.0], "angular_xyz": [0.0, 0.0, 0.0], "frame_id": "base_link", "sim_steps": 0},
                native=backend.last_native_trace_fields(),
            )["native"],
            "initial_images": None,
            "records": [],
        }
        for record in trace_report.get("records", []):
            action = record["action"]
            vector = np.asarray(
                list(action["linear_xyz"]) + list(action["angular_xyz"]),
                dtype=np.float64,
            )
            runtime.step(vector, ticks=int(action["sim_steps"]))
            if camera_subscriber is not None and candidate["initial_images"] is None:
                if not camera_subscriber.wait_until_ready(timeout_s=20.0):
                    raise TimeoutError("Timed out waiting for wrist camera images for replay trace.")
                candidate["initial_images"] = summarize_image_batch(*camera_subscriber.latest_images())
            candidate["records"].append(
                _candidate_record(
                    step_idx=int(record["step_idx"]),
                    action=action,
                    native=backend.last_native_trace_fields(),
                )
            )
            if camera_subscriber is not None:
                candidate["records"][-1]["images"] = summarize_image_batch(
                    *camera_subscriber.latest_images()
                )
        return candidate
    finally:
        if camera_bridge is not None:
            camera_bridge.close()
        if camera_subscriber is not None:
            camera_subscriber.close()
        runtime.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-trace", required=True)
    parser.add_argument("--candidate-trace-output", default=None)
    parser.add_argument("--parity-output", default=None)
    parser.add_argument("--include-images", action="store_true")
    args = parser.parse_args()

    harness = AicParityHarness()
    reference = harness.load_trace_report(args.reference_trace)
    candidate = replay_trace_against_attached_runtime(
        trace_report=reference,
        include_images=args.include_images,
    )
    parity = harness.compare_trace_json(reference_report=reference, candidate_report=candidate)

    if args.candidate_trace_output:
        Path(args.candidate_trace_output).write_text(
            json.dumps(candidate, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.parity_output:
        Path(args.parity_output).write_text(
            json.dumps(parity, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps({"candidate": candidate, "parity": parity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import tempfile
import types
import unittest

import numpy as np

from aic_gym_gz.io import CameraBridgeSidecar, _ros_image_to_array
from aic_gym_gz.video import HeadlessTrajectoryVideoRecorder


class IoTest(unittest.TestCase):
    def test_camera_bridge_sidecar_builds_topic_pairs(self) -> None:
        bridge = CameraBridgeSidecar(topic_map={"overview": "/overview_camera/image"})
        self.assertEqual(
            bridge._bridge_arguments(),
            [
                "/overview_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/overview_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            ],
        )

    def test_ros_image_to_array_rgb8(self) -> None:
        message = types.SimpleNamespace(
            height=2,
            width=2,
            encoding="rgb8",
            data=bytes(
                [
                    255,
                    0,
                    0,
                    0,
                    255,
                    0,
                    0,
                    0,
                    255,
                    10,
                    20,
                    30,
                ]
            ),
        )
        array = _ros_image_to_array(message, expected_shape=(2, 2, 3))
        self.assertEqual(array.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(array[0, 0], np.array([255, 0, 0], dtype=np.uint8)))

    def test_video_recorder_rejects_blank_wrist_frames_when_real_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HeadlessTrajectoryVideoRecorder(
                output_dir=tmpdir,
                enabled=True,
                require_real_wrist_images=True,
                require_live_overview=False,
                prefer_live_overview_camera=False,
            )
            with self.assertRaises(RuntimeError):
                recorder.capture(
                    observation={
                        "sim_time": 0.0,
                        "images": {
                            "left": np.zeros((64, 64, 3), dtype=np.uint8),
                            "center": np.zeros((64, 64, 3), dtype=np.uint8),
                            "right": np.zeros((64, 64, 3), dtype=np.uint8),
                        },
                    },
                    scenario=types.SimpleNamespace(),
                    state=types.SimpleNamespace(sim_time=0.0),
                )
            recorder.close()

    def test_video_recorder_writes_all_overview_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HeadlessTrajectoryVideoRecorder(
                output_dir=tmpdir,
                enabled=True,
                require_real_wrist_images=False,
                require_live_overview=False,
                prefer_live_overview_camera=False,
            )
            recorder.capture(
                observation={
                    "sim_time": 0.0,
                    "images": {
                        "left": np.full((64, 64, 3), 20, dtype=np.uint8),
                        "center": np.full((64, 64, 3), 30, dtype=np.uint8),
                        "right": np.full((64, 64, 3), 40, dtype=np.uint8),
                    },
                },
                scenario=types.SimpleNamespace(
                    task_board=types.SimpleNamespace(
                        pose_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        nic_rails={},
                        sc_rails={},
                        mount_rails={},
                    )
                ),
                state=types.SimpleNamespace(
                    sim_time=0.0,
                    target_port_pose=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    target_port_entrance_pose=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    plug_pose=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                ),
            )
            summary = recorder.close()
            self.assertIn("overview_top_down_xy", summary["streams"])
            self.assertIn("overview_front_xz", summary["streams"])
            self.assertIn("overview_side_yz", summary["streams"])
            self.assertIn("overview_oblique_xy", summary["streams"])


if __name__ == "__main__":
    unittest.main()

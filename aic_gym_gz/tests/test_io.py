from __future__ import annotations

import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from aic_gym_gz.io import (
    CameraBridgeSidecar,
    RosCameraSidecarIO,
    _gazebo_image_text_to_array,
    _is_nonblank_image,
    _ros_topic_text_to_array,
    _ros_image_to_array,
)
from aic_gym_gz.runtime import RuntimeState
from aic_gym_gz.video import HeadlessTrajectoryVideoRecorder


class IoTest(unittest.TestCase):
    class _FakeOverviewSubscriber:
        def __init__(self, images):
            self._images = images

        def latest_images(self):
            timestamps = {name: 1.0 for name in self._images}
            return self._images, timestamps, {}

        def close(self):
            return None

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

    def test_gazebo_image_text_to_array_parses_rgb_payload(self) -> None:
        text = '\n'.join(
            [
                'width: 2',
                'height: 1',
                'data: "\\377\\000\\000\\000\\377\\000"',
                'pixel_format_type: RGB_INT8',
            ]
        )
        array = _gazebo_image_text_to_array(text, expected_shape=(1, 2, 3))
        assert array is not None
        self.assertEqual(array.shape, (1, 2, 3))
        self.assertTrue(np.array_equal(array[0, 0], np.array([255, 0, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(array[0, 1], np.array([0, 255, 0], dtype=np.uint8)))

    def test_ros_topic_text_to_array_parses_rgb_payload(self) -> None:
        text = "\n".join(
            [
                "height: 1",
                "width: 2",
                "data:",
                "- 255",
                "- 0",
                "- 0",
                "- 0",
                "- 255",
                "- 0",
            ]
        )
        array = _ros_topic_text_to_array(text, expected_shape=(1, 2, 3))
        assert array is not None
        self.assertEqual(array.shape, (1, 2, 3))
        self.assertTrue(np.array_equal(array[0, 0], np.array([255, 0, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(array[0, 1], np.array([0, 255, 0], dtype=np.uint8)))

    def test_is_nonblank_image(self) -> None:
        self.assertFalse(_is_nonblank_image(None))
        self.assertFalse(_is_nonblank_image(np.zeros((2, 2, 3), dtype=np.uint8)))
        self.assertTrue(_is_nonblank_image(np.ones((2, 2, 3), dtype=np.uint8)))

    def test_ros_camera_sidecar_io_fills_missing_cameras_from_direct_gazebo(self) -> None:
        class _FakeSubscriber:
            def __init__(self) -> None:
                self.calls = 0

            def start(self):
                return None

            def wait_until_ready(self, *, timeout_s: float = 10.0) -> bool:
                del timeout_s
                return False

            def latest_images(self):
                self.calls += 1
                images = {
                    "left": np.full((4, 4, 3), 5, dtype=np.uint8),
                    "center": np.zeros((4, 4, 3), dtype=np.uint8),
                    "right": np.zeros((4, 4, 3), dtype=np.uint8),
                }
                timestamps = {"left": 1.0, "center": 0.0, "right": 0.0}
                info = {name: {} for name in images}
                return images, timestamps, info

            def close(self):
                return None

        class _FakeBridge:
            def start(self):
                return None

            def close(self):
                return None

        io = RosCameraSidecarIO(
            camera_subscriber=_FakeSubscriber(),
            camera_bridge=_FakeBridge(),
            ready_timeout_s=0.01,
            image_shape=(4, 4, 3),
        )
        state = RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0,
            tcp_pose=np.zeros(7, dtype=np.float64),
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=np.zeros(7, dtype=np.float64),
            target_port_pose=np.zeros(7, dtype=np.float64),
            target_port_entrance_pose=np.zeros(7, dtype=np.float64),
            wrench=np.zeros(6, dtype=np.float64),
            wrench_timestamp=0.0,
            off_limit_contact=False,
            controller_state={},
            score_geometry={},
        )
        fake_direct = {
            "/center_camera/image": np.full((4, 4, 3), 7, dtype=np.uint8),
            "/right_camera/image": np.full((4, 4, 3), 9, dtype=np.uint8),
        }
        with patch("aic_gym_gz.io.fetch_gazebo_topic_image", side_effect=lambda topic, **_: fake_direct.get(topic)):
            observation = io.observation_from_state(state, include_images=True, step_count=0)
        self.assertEqual(set(observation["images"]), {"left", "center", "right"})
        self.assertTrue(np.array_equal(observation["images"]["left"], np.full((4, 4, 3), 5, dtype=np.uint8)))
        self.assertTrue(np.array_equal(observation["images"]["center"], np.full((4, 4, 3), 7, dtype=np.uint8)))
        self.assertTrue(np.array_equal(observation["images"]["right"], np.full((4, 4, 3), 9, dtype=np.uint8)))
        io.close()

    def test_ros_camera_sidecar_io_does_not_bootstrap_on_missing_wrist_frames(self) -> None:
        class _FakeSubscriber:
            def start(self):
                return None

            def wait_until_ready(self, *, timeout_s: float = 10.0) -> bool:
                del timeout_s
                return False

            def latest_images(self):
                images = {
                    "left": np.zeros((4, 4, 3), dtype=np.uint8),
                    "center": np.zeros((4, 4, 3), dtype=np.uint8),
                    "right": np.zeros((4, 4, 3), dtype=np.uint8),
                }
                timestamps = {"left": 0.0, "center": 0.0, "right": 0.0}
                return images, timestamps, {}

            def close(self):
                return None

        class _FakeBridge:
            def start(self):
                return None

            def close(self):
                return None

        io = RosCameraSidecarIO(
            camera_subscriber=_FakeSubscriber(),
            camera_bridge=_FakeBridge(),
            ready_timeout_s=0.01,
            image_shape=(4, 4, 3),
            allow_direct_fetch_fallback=False,
        )
        state = RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0,
            tcp_pose=np.zeros(7, dtype=np.float64),
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=np.zeros(7, dtype=np.float64),
            target_port_pose=np.zeros(7, dtype=np.float64),
            target_port_entrance_pose=np.zeros(7, dtype=np.float64),
            wrench=np.zeros(6, dtype=np.float64),
            wrench_timestamp=0.0,
            off_limit_contact=False,
            controller_state={},
            score_geometry={},
        )
        observation = io.observation_from_state(state, include_images=True, step_count=0)
        self.assertFalse(io._wrist_bootstrapped)
        self.assertEqual([float(value) for value in observation["image_timestamps"]], [0.0, 0.0, 0.0])
        io.close()

    def test_video_recorder_rejects_blank_wrist_frames_when_real_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HeadlessTrajectoryVideoRecorder(
                output_dir=tmpdir,
                enabled=True,
                require_real_wrist_images=True,
                require_live_overview=True,
                prefer_live_overview_camera=False,
            )
            recorder._overview_camera_subscriber = self._FakeOverviewSubscriber(
                {
                    "top_down_xy": np.full((64, 64, 3), 10, dtype=np.uint8),
                    "front_xz": np.full((64, 64, 3), 10, dtype=np.uint8),
                    "side_yz": np.full((64, 64, 3), 10, dtype=np.uint8),
                    "oblique_xy": np.full((64, 64, 3), 10, dtype=np.uint8),
                }
            )
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
            with self.assertRaises(RuntimeError):
                recorder.close()

    def test_video_recorder_writes_all_overview_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HeadlessTrajectoryVideoRecorder(
                output_dir=tmpdir,
                enabled=True,
                require_real_wrist_images=False,
                require_live_overview=True,
                prefer_live_overview_camera=False,
            )
            recorder._overview_camera_subscriber = self._FakeOverviewSubscriber(
                {
                    "top_down_xy": np.full((64, 64, 3), 50, dtype=np.uint8),
                    "front_xz": np.full((64, 64, 3), 60, dtype=np.uint8),
                    "side_yz": np.full((64, 64, 3), 70, dtype=np.uint8),
                    "oblique_xy": np.full((64, 64, 3), 80, dtype=np.uint8),
                }
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

    def test_video_recorder_uses_scene_probe_when_overview_topics_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HeadlessTrajectoryVideoRecorder(
                output_dir=tmpdir,
                enabled=True,
                require_real_wrist_images=False,
                require_live_overview=True,
                prefer_live_overview_camera=False,
            )
            recorder._overview_camera_subscriber = self._FakeOverviewSubscriber({})
            probe_images = {
                "top_down_xy": np.full((64, 64, 3), 90, dtype=np.uint8),
                "front_xz": np.full((64, 64, 3), 91, dtype=np.uint8),
                "side_yz": np.full((64, 64, 3), 92, dtype=np.uint8),
                "oblique_xy": np.full((64, 64, 3), 93, dtype=np.uint8),
            }
            with patch("aic_gym_gz.video.fetch_gazebo_topic_image", return_value=None), patch(
                "aic_gym_gz.video.capture_scene_probe_images",
                return_value=probe_images,
            ):
                recorder.capture(
                    observation={
                        "sim_time": 0.0,
                        "images": {
                            "left": np.full((64, 64, 3), 20, dtype=np.uint8),
                            "center": np.full((64, 64, 3), 30, dtype=np.uint8),
                            "right": np.full((64, 64, 3), 40, dtype=np.uint8),
                        },
                    },
                    scenario=types.SimpleNamespace(),
                    state=types.SimpleNamespace(sim_time=0.0),
                )
            summary = recorder.close()
            self.assertIn("overview_top_down_xy", summary["streams"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import os
import types
import unittest
from unittest import mock

import numpy as np

from aic_gym_gz.runtime import (
    ScenarioGymGzBackend,
    _angular_delta_to_quaternion,
    _local_target_and_entrance_for_task,
    _module_pose_in_board_frame,
    _parse_sim_time_seconds,
    _pose_dict_to_array,
    _target_module_pose_from_scenario,
)


class LiveRuntimeHelpersTest(unittest.TestCase):
    def test_parse_sim_time_seconds_from_state_text(self) -> None:
        payload = """
stats {
  sim_time {
    sec: 12
    nsec: 345000000
  }
}
"""
        self.assertAlmostEqual(_parse_sim_time_seconds(payload), 12.345)

    def test_pose_dict_to_array_extracts_yaw(self) -> None:
        pose = {
            "position": [1.0, 2.0, 3.0],
            "orientation": [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
        }
        pose_array = _pose_dict_to_array(pose)
        np.testing.assert_allclose(pose_array[:3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(pose_array[5], math.pi / 2.0, places=5)
        self.assertAlmostEqual(pose_array[6], math.cos(math.pi / 4.0), places=5)

    def test_angular_delta_to_quaternion_identity_for_zero_rotation(self) -> None:
        quaternion = _angular_delta_to_quaternion(np.zeros(3, dtype=np.float64))
        np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_module_pose_in_board_frame_for_nic_mount(self) -> None:
        scenario = types.SimpleNamespace(
            task_board=types.SimpleNamespace(
                nic_rails={
                    "nic_rail_0": types.SimpleNamespace(
                        translation=0.036,
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                    )
                },
                sc_rails={},
            )
        )
        task = types.SimpleNamespace(target_module_name="nic_card_mount_0")
        pose = _module_pose_in_board_frame(scenario=scenario, task=task)
        assert pose is not None
        np.testing.assert_allclose(pose["position"], [-0.045418, -0.1745, 0.012], atol=1e-6)

    def test_target_module_pose_from_scenario_for_board_offset(self) -> None:
        scenario = types.SimpleNamespace(
            task_board=types.SimpleNamespace(
                pose_xyz_rpy=(0.15, -0.2, 1.14, 0.0, 0.0, math.pi),
                nic_rails={
                    "nic_rail_0": types.SimpleNamespace(
                        translation=0.036,
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                    )
                },
                sc_rails={},
            )
        )
        task = types.SimpleNamespace(target_module_name="nic_card_mount_0")
        pose = _target_module_pose_from_scenario(scenario=scenario, task=task)
        assert pose is not None
        self.assertAlmostEqual(float(pose["position"][2]), 1.152, places=3)

    def test_local_target_and_entrance_for_sfp_port(self) -> None:
        task = types.SimpleNamespace(target_module_name="nic_card_mount_0", port_name="sfp_port_0")
        target_pose, entrance_pose = _local_target_and_entrance_for_task(task)
        assert target_pose is not None
        assert entrance_pose is not None
        self.assertLess(float(entrance_pose["position"][2]), 0.0)

    def test_synthesize_plug_pose_when_live_plug_pose_is_implausible(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(cable_name="cable_0")
        backend._scenario = types.SimpleNamespace(
            cables={
                "cable_0": types.SimpleNamespace(gripper_offset_xyz=(0.0, 0.015385, 0.04245))
            }
        )
        synthesized = ScenarioGymGzBackend._maybe_synthesize_plug_pose(
            backend,
            tcp_pose=np.array([0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            controller_state={},
        )
        assert synthesized is not None
        np.testing.assert_allclose(synthesized[:3], [0.1, 0.215385, 1.04245], atol=1e-6)

    def test_synthesize_plug_pose_prefers_controller_tcp_pose(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(cable_name="cable_0")
        backend._scenario = types.SimpleNamespace(
            cables={
                "cable_0": types.SimpleNamespace(gripper_offset_xyz=(0.0, 0.015385, 0.04245))
            }
        )
        synthesized = ScenarioGymGzBackend._maybe_synthesize_plug_pose(
            backend,
            tcp_pose=np.array([0.2, -0.1, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            controller_state={
                "tcp_pose": np.array([0.2, -0.1, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            },
        )
        assert synthesized is not None
        np.testing.assert_allclose(synthesized[:3], [0.2, -0.084615, 1.24245], atol=1e-6)

    def test_synthesize_plug_pose_preserves_good_live_pose_without_controller(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(cable_name="cable_0")
        backend._scenario = types.SimpleNamespace(
            cables={
                "cable_0": types.SimpleNamespace(gripper_offset_xyz=(0.0, 0.015385, 0.04245))
            }
        )
        synthesized = ScenarioGymGzBackend._maybe_synthesize_plug_pose(
            backend,
            tcp_pose=np.array([0.2, -0.1, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.2, -0.084615, 1.24245, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            controller_state={},
        )
        assert synthesized is None

    def test_synthesize_tcp_pose_when_observation_teleports(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._action = np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        previous_state = types.SimpleNamespace(
            tcp_pose=np.array([0.2, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        synthesized = ScenarioGymGzBackend._maybe_synthesize_tcp_pose(
            backend,
            observed_tcp_pose=np.array([-0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            previous_state=previous_state,
            step_tick_count=8,
        )
        assert synthesized is not None
        np.testing.assert_allclose(synthesized[:3], [0.2016, 0.0, 1.1984], atol=1e-6)

    def test_synthesize_tcp_pose_accepts_close_high_observation(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._action = np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        previous_state = types.SimpleNamespace(
            tcp_pose=np.array([0.2, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        synthesized = ScenarioGymGzBackend._maybe_synthesize_tcp_pose(
            backend,
            observed_tcp_pose=np.array([0.201, 0.0, 1.199, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            previous_state=previous_state,
            step_tick_count=8,
        )
        assert synthesized is None

    def test_state_is_sane_for_live_reset_prefers_controller_pose(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._last_state = None
        backend._action = np.zeros(6, dtype=np.float64)
        state = types.SimpleNamespace(
            controller_state={
                "tcp_pose": np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            },
            tcp_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        assert ScenarioGymGzBackend._state_is_sane_for_live_reset(backend, state) is True

    def test_state_is_sane_for_live_reset_falls_back_when_controller_pose_is_bad(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._last_state = None
        backend._action = np.zeros(6, dtype=np.float64)
        state = types.SimpleNamespace(
            controller_state={
                "tcp_pose": np.array([-0.37, 0.19, 0.32, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            },
            tcp_pose=np.array([0.17, 0.01, 1.64, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.17, 0.02, 1.68, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        assert ScenarioGymGzBackend._state_is_sane_for_live_reset(backend, state) is True

    def test_is_sane_live_tcp_pose_rejects_low_z_controller_frame_jump(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._action = np.array([0.065, -0.12, -0.12, 0.0, 0.0, 0.0], dtype=np.float64)
        previous_state = types.SimpleNamespace(
            tcp_pose=np.array([0.1715, 0.0054, 1.6414, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        sane = ScenarioGymGzBackend._is_sane_live_tcp_pose(
            backend,
            np.array([-0.3713, 0.1948, 0.3276, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            observed_tcp_pose=np.array([0.1715, 0.0054, 1.6414, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            previous_state=previous_state,
            step_tick_count=8,
        )
        assert sane is False

    def test_is_sane_live_tcp_pose_accepts_close_high_pose(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._action = np.array([0.065, -0.12, -0.12, 0.0, 0.0, 0.0], dtype=np.float64)
        previous_state = types.SimpleNamespace(
            tcp_pose=np.array([0.1715, 0.0054, 1.6414, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        sane = ScenarioGymGzBackend._is_sane_live_tcp_pose(
            backend,
            np.array([0.1725, 0.0035, 1.6395, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            observed_tcp_pose=np.array([0.1715, 0.0054, 1.6414, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            previous_state=previous_state,
            step_tick_count=8,
        )
        assert sane is True

    def test_live_insertion_event_from_geometry_marks_success(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(
            target_module_name="nic_card_mount_0",
            port_name="sfp_port_0",
        )
        event = ScenarioGymGzBackend._live_insertion_event_from_geometry(
            backend,
            {
                "distance_to_target": 0.001,
                "lateral_misalignment": 0.001,
                "insertion_progress": 0.99,
            },
        )
        assert event == "nic_card_mount_0/sfp_port_0"

    def test_live_insertion_event_from_geometry_rejects_misaligned_pose(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(
            target_module_name="nic_card_mount_0",
            port_name="sfp_port_0",
        )
        event = ScenarioGymGzBackend._live_insertion_event_from_geometry(
            backend,
            {
                "distance_to_target": 0.001,
                "lateral_misalignment": 0.02,
                "insertion_progress": 0.99,
            },
        )
        assert event is None

    def test_attach_to_existing_does_not_fabricate_missing_module_pose_from_scenario(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(
            target_module_name="nic_card_mount_0",
            port_name="sfp_port_0",
        )
        backend._scenario = types.SimpleNamespace(
            task_board=types.SimpleNamespace(
                pose_xyz_rpy=(0.15, -0.2, 1.14, 0.0, 0.0, math.pi),
                nic_rails={
                    "nic_rail_0": types.SimpleNamespace(
                        translation=0.0,
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                    )
                },
                sc_rails={},
            )
        )
        backend._attach_to_existing = True
        backend._target_entity_name = "task_board::missing_target"
        synthetic = ScenarioGymGzBackend._synthetic_target_pose(
            backend,
            entities_by_name={},
            observed_target_name="task_board_base_link",
        )
        assert synthetic is None

    def test_bootstrap_existing_world_uses_short_cli_timeout(self) -> None:
        recorded: dict[str, float] = {}

        class _FakeClient:
            def __init__(self, config):
                recorded["timeout"] = float(config.timeout)

            def get_observation(self, request):
                raise RuntimeError("no sample")

            def close(self):
                return None

        backend = object.__new__(ScenarioGymGzBackend)
        backend._cli_client_type = _FakeClient
        backend._cli_config_type = types.SimpleNamespace
        backend._get_observation_request_type = dict
        backend._world_path = "aic_description/world/aic.sdf"
        backend._world_name = "aic_world"
        backend._source_entity_name = "ati/tool_link"
        backend._target_entity_name = "tabletop"
        backend._timeout = 30.0
        backend._action = np.zeros(6, dtype=np.float64)
        backend._last_observation = None
        backend._last_info = None
        backend._last_state = None
        backend._tick_existing_world_for_sample = lambda: None

        result = ScenarioGymGzBackend._bootstrap_existing_world_state(backend, timeout_s=20.0)
        assert result is None
        self.assertEqual(recorded["timeout"], 4.0)

    def test_connect_existing_world_skips_cli_bootstrap_for_transport_backend(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._runtime = types.SimpleNamespace(
            get_observation=lambda: (
                {"sim_time": 0.0, "sim_tick": 0},
                {},
            )
        )
        backend._transport_backend = "transport"
        backend._attach_ready_timeout = 0.01
        backend._action = np.zeros(6, dtype=np.float64)
        backend._last_observation = None
        backend._last_info = None
        backend._last_state = None
        backend._bootstrap_existing_world_state = lambda timeout_s: (_ for _ in ()).throw(
            AssertionError("bootstrap should not be called")
        )
        backend._runtime_state_from_observation = lambda observation, info: types.SimpleNamespace(
            controller_state={"tcp_pose": np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)},
            tcp_pose=np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        backend._state_is_sane_for_live_reset = lambda state: True
        result = ScenarioGymGzBackend.connect_existing_world(backend)
        assert result is not None

    def test_connect_existing_world_skips_cli_bootstrap_by_default(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._runtime = types.SimpleNamespace(
            get_observation=lambda: (
                {"sim_time": 0.0, "sim_tick": 0},
                {},
            )
        )
        backend._transport_backend = "cli"
        backend._attach_ready_timeout = 0.01
        backend._action = np.zeros(6, dtype=np.float64)
        backend._last_observation = None
        backend._last_info = None
        backend._last_state = None
        backend._bootstrap_existing_world_state = lambda timeout_s: (_ for _ in ()).throw(
            AssertionError("bootstrap should not be called without explicit opt-in")
        )
        backend._runtime_state_from_observation = lambda observation, info: types.SimpleNamespace(
            controller_state={"tcp_pose": np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)},
            tcp_pose=np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        backend._state_is_sane_for_live_reset = lambda state: True
        with mock.patch.dict(os.environ, {}, clear=False):
            result = ScenarioGymGzBackend.connect_existing_world(backend)
        assert result is not None

    def test_resolve_live_insertion_event_prefers_official_topic(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(
            target_module_name="nic_card_mount_0",
            port_name="sfp_port_0",
        )
        event, source = ScenarioGymGzBackend._resolve_live_insertion_event(
            backend,
            ros_sample={"official_insertion_event": "nic_card_mount_0/sfp_port_0"},
            geometry={
                "distance_to_target": 0.001,
                "lateral_misalignment": 0.001,
                "insertion_progress": 0.99,
            },
        )
        assert event == "nic_card_mount_0/sfp_port_0"
        assert source == "official_topic"

    def test_resolve_live_insertion_event_does_not_use_geometry_fallback(self) -> None:
        backend = object.__new__(ScenarioGymGzBackend)
        backend._task = types.SimpleNamespace(
            target_module_name="nic_card_mount_0",
            port_name="sfp_port_0",
        )
        event, source = ScenarioGymGzBackend._resolve_live_insertion_event(
            backend,
            ros_sample={},
            geometry={
                "distance_to_target": 0.001,
                "lateral_misalignment": 0.001,
                "insertion_progress": 0.99,
            },
        )
        assert event is None
        assert source == "none"


if __name__ == "__main__":
    unittest.main()

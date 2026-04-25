from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from aic_gym_gz.official_scene import build_official_launch_spec, sanitize_training_world_sdf
from aic_gym_gz.randomizer import AicEnvRandomizer


class OfficialSceneLaunchSpecTest(unittest.TestCase):
    def test_container_spec_uses_entrypoint_and_zenoh_env(self) -> None:
        scenario = AicEnvRandomizer(enable_randomization=False).sample(seed=123, trial_id="trial_1")
        with patch("aic_gym_gz.official_scene.Path.exists", return_value=True):
            spec = build_official_launch_spec(
                scenario,
                setup_script="/ws_aic/install/setup.bash",
                ground_truth=True,
                start_aic_engine=False,
            )
        self.assertEqual(spec.launch_mode, "entrypoint")
        self.assertEqual(spec.shell_environment["RMW_IMPLEMENTATION"], "rmw_zenoh_cpp")
        self.assertIn("transport/shared_memory/enabled=false", spec.shell_environment["ZENOH_CONFIG_OVERRIDE"])
        self.assertIn("/entrypoint.sh", spec.shell_command)
        self.assertIsNone(spec.router_command_prefix)

    def test_host_spec_skips_eval_session_and_router_scripts(self) -> None:
        scenario = AicEnvRandomizer(enable_randomization=False).sample(seed=123, trial_id="trial_1")
        with (
            patch("aic_gym_gz.official_scene._should_use_container_entrypoint", return_value=False),
            patch("aic_gym_gz.official_scene._should_enable_official_eval_middleware", return_value=False),
            patch("aic_gym_gz.official_scene._eval_session_script", return_value=Path("/tmp/zenoh_eval_session.sh")),
            patch("aic_gym_gz.official_scene._router_script", return_value=Path("/tmp/zenoh_router.sh")),
        ):
            spec = build_official_launch_spec(
                scenario,
                setup_script="/home/ubuntu/ws_aic/src/aic/install/setup.bash",
                ground_truth=False,
                start_aic_engine=False,
            )
        self.assertEqual(spec.launch_mode, "ros2_launch")
        self.assertEqual(spec.shell_environment["RMW_IMPLEMENTATION"], "rmw_cyclonedds_cpp")
        self.assertIn("GZ_CONFIG_PATH", spec.shell_environment)
        self.assertIn("GZ_SIM_RESOURCE_PATH", spec.shell_environment)
        self.assertIn("GZ_SIM_SYSTEM_PLUGIN_PATH", spec.shell_environment)
        self.assertIn("export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp", spec.shell_command)
        self.assertIn("source /home/ubuntu/ws_aic/src/aic/install/setup.bash", spec.shell_command)
        self.assertNotIn("source /tmp/zenoh_eval_session.sh", spec.shell_command)
        self.assertNotIn("source /tmp/zenoh_router.sh", spec.shell_command)
        self.assertNotIn("ros2 run rmw_zenoh_cpp rmw_zenohd", spec.shell_command)
        self.assertIn("ros2 launch aic_bringup aic_gz_bringup.launch.py", spec.shell_command)

    def test_sanitize_training_world_injects_scene_probe_camera(self) -> None:
        world_sdf = "<sdf version='1.9'><world name='aic_world'></world></sdf>"
        sanitized = sanitize_training_world_sdf(
            world_sdf,
            overview_models="\n<model name='overview_camera_array'></model>\n",
            scene_probe_model="\n<model name='scene_probe_camera'></model>\n",
        )
        self.assertIn("scene_probe_camera", sanitized)

    def test_sanitize_training_world_injects_joint_target_plugin(self) -> None:
        world_sdf = (
            "<sdf version='1.9'><world name='aic_world'><model name='ur5e'>"
            "<plugin name='aic_gazebo::ResetJointsPlugin' filename='ResetJointsPlugin'/>"
            "</model></world></sdf>"
        )
        sanitized = sanitize_training_world_sdf(
            world_sdf,
            scene_probe_model="\n<model name='scene_probe_camera'></model>\n",
        )
        self.assertIn("JointTargetPlugin", sanitized)
        self.assertIn("<world_name>aic_world</world_name>", sanitized)

    def test_sanitize_training_world_strips_ros2_control_plugin(self) -> None:
        world_sdf = (
            "<sdf version='1.9'><world name='aic_world'><model name='ur5e'>"
            "<plugin name='gz_ros2_control::GazeboSimROS2ControlPlugin' "
            "filename='gz_ros2_control-system'><parameters>ignored</parameters></plugin>"
            "</model></world></sdf>"
        )
        sanitized = sanitize_training_world_sdf(
            world_sdf,
            scene_probe_model="\n<model name='scene_probe_camera'></model>\n",
        )
        self.assertNotIn("gz_ros2_control-system", sanitized)
        self.assertNotIn("GazeboSimROS2ControlPlugin", sanitized)


if __name__ == "__main__":
    unittest.main()

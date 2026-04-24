from __future__ import annotations

import unittest
from unittest.mock import patch

from aic_gym_gz.official_scene import build_official_launch_spec
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

    def test_host_spec_keeps_plain_ros_launch_when_entrypoint_is_missing(self) -> None:
        scenario = AicEnvRandomizer(enable_randomization=False).sample(seed=123, trial_id="trial_1")
        with patch("aic_gym_gz.official_scene.Path.exists", return_value=False):
            spec = build_official_launch_spec(
                scenario,
                setup_script="/home/ubuntu/ws_aic/src/aic/install/setup.bash",
                ground_truth=False,
                start_aic_engine=False,
            )
        self.assertEqual(spec.launch_mode, "ros2_launch")
        self.assertEqual(spec.shell_environment, {})
        self.assertIn("source /home/ubuntu/ws_aic/src/aic/install/setup.bash", spec.shell_command)
        self.assertIn("ros2 launch aic_bringup aic_gz_bringup.launch.py", spec.shell_command)


if __name__ == "__main__":
    unittest.main()

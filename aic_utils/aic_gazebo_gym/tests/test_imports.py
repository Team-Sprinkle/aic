"""Import-only tests for the Gazebo gym skeleton."""

import unittest


class ImportTests(unittest.TestCase):
    def test_top_level_imports(self) -> None:
        from aic_gazebo_gym import (  # noqa: F401
            GazeboBackend,
            GazeboEnv,
            GazeboRuntime,
            ResetResult,
            StepResult,
        )

    def test_module_imports(self) -> None:
        import aic_gazebo_gym.backend  # noqa: F401
        import aic_gazebo_gym.env  # noqa: F401
        import aic_gazebo_gym.runtime  # noqa: F401
        import aic_gazebo_gym.types  # noqa: F401


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("geometry_probe", Path(__file__).resolve().parents[1] / "scripts/probe_task_geometry_reward.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_quaternion_rotation_preserves_small_translation_and_zero():
    q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    np.testing.assert_allclose(probe._quat_apply(q, [0.001, 0, 0]), [0, 0.001, 0], atol=1e-12)
    np.testing.assert_allclose(probe._quat_apply(q, [0, 0, 0]), [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(probe._quat_apply([0, 0, 0, 1], [0.003, -0.002, 0.001]), [0.003, -0.002, 0.001])

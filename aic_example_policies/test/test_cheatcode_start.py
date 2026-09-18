import importlib.util
import math
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/cheatcode_start.py"
spec = importlib.util.spec_from_file_location("cheatcode_start", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
choose = module.initial_insertion_z_offset


@pytest.mark.parametrize("depth", [-0.02, -0.01, 0.0, 0.004])
def test_aligned_near_or_inserted_start_does_not_retract(depth):
    offset = choose((0, 0, 0), (0, 0, depth), (1, 0, 0, 0), (1, 0, 0, 0))
    assert offset == depth
    assert min(0.005, offset) <= depth


@pytest.mark.parametrize("position", [(0.05, 0, -0.01), (0, 0, 0.2), (0, 0, -0.1)])
def test_unrelated_start_keeps_original_approach(position):
    assert choose((0, 0, 0), position, (1, 0, 0, 0), (1, 0, 0, 0)) == 0.2


def test_misaligned_plug_does_not_enter_final_descent():
    assert choose((0, 0, 0), (0, 0, -0.01), (1, 0, 0, 0), (0, 1, 0, 0)) == 0.2


def test_quaternion_sign_and_world_translation_do_not_change_handoff():
    assert choose((1, 2, 3), (1, 2, 2.99), (1, 0, 0, 0), (-1, 0, 0, 0)) == pytest.approx(-0.01)


def test_invalid_tf_fails_instead_of_commanding_motion():
    with pytest.raises(ValueError):
        choose((0, 0, 0), (math.nan, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0))

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.contact_recovery_features import (  # noqa: E402
    CONTACT_RECOVERY_FEATURE_DIM,
    CONTACT_RECOVERY_FEATURE_NAMES,
    ContactRecoveryFeatureComputer,
    ContactRecoveryFeatureConfig,
)


def _feature_map(features: np.ndarray) -> dict[str, float]:
    return dict(zip(CONTACT_RECOVERY_FEATURE_NAMES, features.astype(float), strict=True))


def test_force_threshold_memory_uses_one_step_delta_and_seconds() -> None:
    computer = ContactRecoveryFeatureComputer(ContactRecoveryFeatureConfig())

    features = computer.update(
        time_sec=0.0,
        tcp_position_base=[0.0, 0.0, 0.0],
        tcp_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        force=[0.0, 0.0, 0.0],
        torque=[0.0, 0.0, 0.0],
    )
    assert features.shape == (CONTACT_RECOVERY_FEATURE_DIM,)
    values = _feature_map(features)
    assert values["force_thresh_1.time_since_first_sec"] == -1.0
    assert values["force_thresh_1.time_since_latest_sec"] == -1.0

    features = computer.update(
        time_sec=0.1,
        tcp_position_base=[0.0, 0.0, 0.0],
        tcp_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        force=[1.5, 0.0, 0.0],
        torque=[0.0, 0.0, 0.0],
    )
    values = _feature_map(features)
    assert values["force_thresh_1.time_since_first_sec"] == 0.0
    assert values["force_thresh_1.time_since_latest_sec"] == 0.0
    assert values["force_thresh_1.first_delta.x"] == 1.5
    assert values["force_thresh_1.first_delta_norm"] == 1.5
    assert values["force_thresh_3.time_since_first_sec"] == -1.0

    features = computer.update(
        time_sec=0.4,
        tcp_position_base=[0.0, 0.0, 0.0],
        tcp_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        force=[5.0, 0.0, 0.0],
        torque=[0.0, 0.0, 0.0],
    )
    values = _feature_map(features)
    assert np.isclose(values["force_thresh_1.time_since_first_sec"], 0.3)
    assert values["force_thresh_1.time_since_latest_sec"] == 0.0
    assert values["force_thresh_1.first_delta_norm"] == 1.5
    assert values["force_thresh_1.latest_delta_norm"] == 3.5
    assert values["force_thresh_3.time_since_first_sec"] == 0.0
    assert values["force_thresh_3.first_delta_norm"] == 3.5


def test_threshold_feature_names_are_dot_style() -> None:
    assert "force_thresh_1.time_since_latest_sec" in CONTACT_RECOVERY_FEATURE_NAMES
    assert "force_thresh_7.latest_delta_norm" in CONTACT_RECOVERY_FEATURE_NAMES
    assert CONTACT_RECOVERY_FEATURE_DIM == 40

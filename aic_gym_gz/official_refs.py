"""Stable references into the official AIC stack."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

OFFICIAL_SAMPLE_CONFIG = REPO_ROOT / "aic_engine" / "config" / "sample_config.yaml"
OFFICIAL_WORLD_SDF = REPO_ROOT / "aic_description" / "world" / "aic.sdf"
OFFICIAL_TASK_BOARD_XACRO = (
    REPO_ROOT / "aic_description" / "urdf" / "task_board.urdf.xacro"
)
OFFICIAL_CABLE_XACRO = REPO_ROOT / "aic_description" / "urdf" / "cable.sdf.xacro"
OFFICIAL_CONTROLLER_CONFIG = (
    REPO_ROOT / "aic_bringup" / "config" / "aic_ros2_controllers.yaml"
)
OFFICIAL_SCORING_CPP = REPO_ROOT / "aic_scoring" / "src" / "ScoringTier2.cc"

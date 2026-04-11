"""Standalone Gym-style AIC training path.

This package is intentionally separate from the official ROS-first evaluation
stack. It reuses official scene and scoring definitions where possible while
keeping the inner RL loop free of ROS-specific orchestration.
"""

from .env import AicInsertionEnv, make_default_env
from .io import AicGazeboIO, GazeboNativeIOPlaceholder, MockGazeboIO
from .parity import AicParityHarness
from .randomizer import AicEnvRandomizer
from .reward import AicEvaluationSummary, AicRewardBreakdown, AicScoreCalculator
from .runtime import (
    AicGazeboRuntime,
    MockStepperBackend,
    RuntimeBackend,
    RuntimeState,
    ScenarioGymGzBackend,
)
from .task import AicInsertionTask

__all__ = [
    "AicEvaluationSummary",
    "AicEnvRandomizer",
    "AicGazeboIO",
    "AicGazeboRuntime",
    "AicInsertionEnv",
    "AicInsertionTask",
    "AicParityHarness",
    "AicRewardBreakdown",
    "AicScoreCalculator",
    "GazeboNativeIOPlaceholder",
    "MockGazeboIO",
    "MockStepperBackend",
    "RuntimeBackend",
    "RuntimeState",
    "ScenarioGymGzBackend",
    "make_default_env",
]

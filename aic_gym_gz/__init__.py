"""Standalone Gym-style AIC training path.

This package is intentionally separate from the official ROS-first evaluation
stack. It reuses official scene and scoring definitions where possible while
keeping the inner RL loop free of ROS-specific orchestration.
"""

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

try:
    from .env import AicInsertionEnv, make_default_env, make_live_env
    from .task import AicInsertionTask
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
    AicInsertionEnv = None
    AicInsertionTask = None
    make_default_env = None
    make_live_env = None

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
    "make_live_env",
]

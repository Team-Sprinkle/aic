"""ROS-free Gazebo training package skeleton for AIC."""

from .action import Action
from .alignment import AlignmentSpec, build_alignment_report, check_alignment_invariants
from .backend import Backend, StubBackend
from .env import GazeboEnv, MinimalTaskEnv
from .gymnasium_env import GYMNASIUM_AVAILABLE, GymnasiumGazeboEnv
from .observation import Observation, ObservationDict
from .official_scoring import OfficialTier3TrackedPairScorer
from .protocol import (
    GetObservationRequest,
    GetObservationResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from .reward import Reward
from .rl import StableRLEnvConfig, StableRLGazeboEnv, training_api_report
from .runtime import FakeRuntime, GazeboAttachedRuntime, GazeboRuntime, GazeboRuntimeConfig, Runtime
from .task import MinimalTask, MinimalTaskSamplerConfig, MinimalTaskState, PoseRandomizationConfig
from .termination import Termination

__all__ = [
    "Action",
    "AlignmentSpec",
    "Backend",
    "build_alignment_report",
    "check_alignment_invariants",
    "FakeRuntime",
    "GazeboEnv",
    "GazeboAttachedRuntime",
    "GazeboRuntime",
    "GazeboRuntimeConfig",
    "GYMNASIUM_AVAILABLE",
    "GetObservationRequest",
    "GetObservationResponse",
    "GymnasiumGazeboEnv",
    "MinimalTask",
    "MinimalTaskSamplerConfig",
    "MinimalTaskEnv",
    "MinimalTaskState",
    "Observation",
    "ObservationDict",
    "OfficialTier3TrackedPairScorer",
    "PoseRandomizationConfig",
    "ResetRequest",
    "ResetResponse",
    "Reward",
    "StableRLEnvConfig",
    "StableRLGazeboEnv",
    "Runtime",
    "StepRequest",
    "StepResponse",
    "StubBackend",
    "Termination",
    "training_api_report",
]

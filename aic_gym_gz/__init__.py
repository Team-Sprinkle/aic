"""Standalone Gym-style AIC training path.

This package is intentionally separate from the official ROS-first evaluation
stack. It reuses official scene and scoring definitions where possible while
keeping the inner RL loop free of ROS-specific orchestration.
"""

from .io import AicGazeboIO, GazeboNativeIOPlaceholder, MockGazeboIO
from .parity import AicParityHarness
from .randomizer import AicEnvRandomizer
from .reward import (
    AicEvaluationSummary,
    AicRewardMetrics,
    AicRlRewardBreakdown,
    AicRlRewardCalculator,
    AicRlRewardWeights,
    AicScoreCalculator,
)
from .runtime import (
    AicGazeboRuntime,
    AuxiliaryForceContactSummary,
    MockStepperBackend,
    MockTransientContactConfig,
    RuntimeBackend,
    RuntimeCheckpoint,
    RuntimeState,
    ScenarioGymGzBackend,
)

# Base env exports.
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

# Teacher-layer exports.
try:
    from .teacher import (
        AgentTeacherController,
        OfficialStyleScore,
        OfficialStyleScoreEvaluator,
        TeacherCandidateSearch,
        TeacherConfig,
        TeacherContextExtractor,
        TeacherReplayArtifact,
        TeacherReplayComparator,
        TeacherReplayRunner,
        TeacherRolloutResult,
        TeacherSearchConfig,
        TeacherSearchResult,
        TemporalObservationBuffer,
        export_selected_candidate_to_replay,
        load_teacher_replay,
        run_teacher_rollout,
        save_teacher_replay,
    )
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
    AgentTeacherController = None
    OfficialStyleScore = None
    OfficialStyleScoreEvaluator = None
    TeacherCandidateSearch = None
    TeacherConfig = None
    TeacherContextExtractor = None
    TeacherReplayArtifact = None
    TeacherReplayComparator = None
    TeacherReplayRunner = None
    TeacherRolloutResult = None
    TeacherSearchConfig = None
    TeacherSearchResult = None
    TemporalObservationBuffer = None
    export_selected_candidate_to_replay = None
    load_teacher_replay = None
    run_teacher_rollout = None
    save_teacher_replay = None

__all__ = [
    "AgentTeacherController",
    "AicEvaluationSummary",
    "AicEnvRandomizer",
    "AicGazeboIO",
    "AicGazeboRuntime",
    "AuxiliaryForceContactSummary",
    "AicInsertionEnv",
    "AicInsertionTask",
    "AicParityHarness",
    "AicRewardMetrics",
    "AicRlRewardBreakdown",
    "AicRlRewardCalculator",
    "AicRlRewardWeights",
    "AicScoreCalculator",
    "GazeboNativeIOPlaceholder",
    "MockGazeboIO",
    "MockStepperBackend",
    "MockTransientContactConfig",
    "RuntimeCheckpoint",
    "RuntimeBackend",
    "RuntimeState",
    "ScenarioGymGzBackend",
    "OfficialStyleScore",
    "OfficialStyleScoreEvaluator",
    "TeacherCandidateSearch",
    "TeacherConfig",
    "TeacherContextExtractor",
    "TeacherReplayArtifact",
    "TeacherReplayComparator",
    "TeacherReplayRunner",
    "TeacherRolloutResult",
    "TeacherSearchConfig",
    "TeacherSearchResult",
    "TemporalObservationBuffer",
    "export_selected_candidate_to_replay",
    "load_teacher_replay",
    "make_default_env",
    "make_live_env",
    "run_teacher_rollout",
    "save_teacher_replay",
]

"""Expert trajectory generator for AIC cable insertion.

This package keeps VLM output at the strategy/scene-understanding layer. Rigid
free-space motion must come from MoveIt, and final insertion remains
CheatCode-style task geometry.
"""

from aic_teacher_official.expert_generator.scene_snapshot import (
    ObjectGeometry,
    SceneSnapshot,
    SerializablePose,
)
from aic_teacher_official.expert_generator.vlm_strategy import (
    CableRisk,
    ExpertMode,
    VLMStrategy,
    parse_vlm_strategy,
)
from aic_teacher_official.expert_generator.moveit_planner import (
    MoveItPlanner,
    MoveItUnavailableError,
    PlannedApproach,
    PlanningFailure,
)
from aic_teacher_official.expert_generator.moveit_py_backend import (
    MoveItPyBackendConfig,
    MoveItPyPlanningBackend,
)
from aic_teacher_official.expert_generator.replay_runner import (
    OfficialRecordingReplayRunner,
    OfficialReplayConfig,
)

__all__ = [
    "CableRisk",
    "ExpertMode",
    "MoveItPlanner",
    "MoveItPyBackendConfig",
    "MoveItPyPlanningBackend",
    "MoveItUnavailableError",
    "ObjectGeometry",
    "OfficialRecordingReplayRunner",
    "OfficialReplayConfig",
    "PlannedApproach",
    "PlanningFailure",
    "SceneSnapshot",
    "SerializablePose",
    "VLMStrategy",
    "parse_vlm_strategy",
]

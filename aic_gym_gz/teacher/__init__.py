"""Teacher-policy utilities layered on top of the base AIC gym runtime."""

try:
    from .context import TeacherContextExtractor
    from .history import TemporalObservationBuffer
    from .policy import AgentTeacherController, TeacherConfig
    from .replay import (
        TeacherReplayArtifact,
        TeacherReplayComparator,
        TeacherReplayRunner,
        load_teacher_replay,
        save_teacher_replay,
    )
    from .runner import TeacherRolloutResult, run_teacher_rollout
    from .scoring import OfficialStyleScore, OfficialStyleScoreEvaluator
    from .search import (
        TeacherCandidateSearch,
        TeacherSearchConfig,
        TeacherSearchResult,
        export_selected_candidate_to_replay,
    )
except ModuleNotFoundError:
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
    "OfficialStyleScore",
    "OfficialStyleScoreEvaluator",
    "TeacherConfig",
    "TeacherCandidateSearch",
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
    "run_teacher_rollout",
    "save_teacher_replay",
]

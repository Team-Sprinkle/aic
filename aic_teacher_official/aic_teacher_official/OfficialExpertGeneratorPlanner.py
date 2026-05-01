"""Official policy entry point for the new expert generator architecture.

This policy captures a live scene snapshot, asks GPT-5-mini for symbolic
strategy/cable-risk JSON, requires MoveIt for free-space approach planning, and
writes a piecewise trajectory for a separate replay/validation pass. It never
asks the VLM for executable waypoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task

from aic_teacher_official.expert_generator.candidate_generation import generate_approach_candidates
from aic_teacher_official.expert_generator.ft_guard import FTGuardConfig
from aic_teacher_official.expert_generator.moveit_planner import MoveItPlanner, MoveItUnavailableError, PlanningFailure
from aic_teacher_official.expert_generator.moveit_py_backend import MoveItPyBackendConfig, MoveItPyPlanningBackend
from aic_teacher_official.expert_generator.nominal_expert import NominalExpert
from aic_teacher_official.expert_generator.recovery_expert import RecoveryExpert
from aic_teacher_official.expert_generator.ros_scene_snapshot import LiveSceneSnapshotProvider
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode
from aic_teacher_official.expert_generator.vlm_strategy_client import OpenAIVLMStrategyProvider


class OfficialExpertGeneratorPlanner(Policy):
    """Generate one expert candidate trajectory in a live official sim."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._mode = ExpertMode(os.environ.get("AIC_EXPERT_MODE", "nominal"))
        self._output = Path(
            os.environ.get(
                "AIC_EXPERT_PIECEWISE_OUTPUT",
                "artifacts/expert_piecewise_trajectory.json",
            )
        )
        self._debug_dir = Path(
            os.environ.get(
                "AIC_EXPERT_DEBUG_OUTPUT_DIR",
                str(self._output.with_suffix("") / "debug"),
            )
        )
        self._engine_config = os.environ.get("AIC_EXPERT_ENGINE_CONFIG", "")
        self._run_id = os.environ.get("AIC_EXPERT_RUN_ID", "expert_generation")
        self._seed = int(os.environ.get("AIC_EXPERT_SEED", "0"))
        self._candidate_index = int(os.environ.get("AIC_EXPERT_CANDIDATE_INDEX", "0"))
        self._candidates_per_scene = int(os.environ.get("AIC_EXPERT_CANDIDATES_PER_SCENE", "5"))
        self._strategy_model = os.environ.get("AIC_EXPERT_STRATEGY_MODEL", "gpt-5-mini")
        self._image_sample_period_sec = float(os.environ.get("AIC_EXPERT_IMAGE_SAMPLE_PERIOD_SEC", "0.5"))
        self._image_capture_duration_sec = float(os.environ.get("AIC_EXPERT_IMAGE_CAPTURE_DURATION_SEC", "2.0"))
        self._max_images = int(os.environ.get("AIC_EXPERT_MAX_IMAGES", "8"))
        self._ft_config = FTGuardConfig(
            soft_threshold_n=float(os.environ.get("AIC_EXPERT_FT_SOFT_THRESHOLD_N", "1.0")),
            hard_threshold_n=float(os.environ.get("AIC_EXPERT_FT_HARD_THRESHOLD_N", "3.0")),
            backup_distance_m=float(os.environ.get("AIC_EXPERT_BACKUP_DISTANCE_M", "0.006")),
            max_retries=int(os.environ.get("AIC_EXPERT_MAX_RETRIES", "3")),
            probe_pattern=os.environ.get("AIC_EXPERT_PROBE_PATTERN", "small_cross"),
        )
        self._moveit_backend_config = MoveItPyBackendConfig(
            planning_group=os.environ.get("AIC_EXPERT_MOVEIT_PLANNING_GROUP", "ur_manipulator"),
            end_effector_link=os.environ.get("AIC_EXPERT_MOVEIT_EE_LINK", "gripper/tcp"),
            workspace_dir=os.environ.get("AIC_WORKSPACE_DIR"),
            ur_type=os.environ.get("AIC_EXPERT_MOVEIT_UR_TYPE", "ur5e"),
            description_file=os.environ.get("AIC_EXPERT_MOVEIT_DESCRIPTION_FILE") or None,
            controllers_file=os.environ.get("AIC_EXPERT_MOVEIT_CONTROLLERS_FILE") or None,
            planning_time=float(os.environ.get("AIC_EXPERT_MOVEIT_PLANNING_TIME", "5.0")),
            planning_attempts=int(os.environ.get("AIC_EXPERT_MOVEIT_PLANNING_ATTEMPTS", "5")),
            max_velocity_scaling_factor=float(os.environ.get("AIC_EXPERT_MOVEIT_MAX_VEL_SCALE", "0.1")),
            max_acceleration_scaling_factor=float(os.environ.get("AIC_EXPERT_MOVEIT_MAX_ACCEL_SCALE", "0.1")),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._debug_dir.mkdir(parents=True, exist_ok=True)

        def safe_feedback(message: str) -> None:
            try:
                send_feedback(message)
            except Exception as ex:
                self._parent_node.get_logger().warning(
                    f"Unable to publish expert-generator feedback '{message}': {type(ex).__name__}: {ex}"
                )

        safe_feedback("official_expert_generator_started")
        snapshot_provider = LiveSceneSnapshotProvider(
            self._parent_node,
            output_dir=self._debug_dir,
            run_id=self._run_id,
            seed=self._seed,
            engine_config=self._engine_config or None,
            image_sample_period_sec=self._image_sample_period_sec,
            image_capture_duration_sec=self._image_capture_duration_sec,
            max_images=self._max_images,
        )
        try:
            snapshot = snapshot_provider.capture_from_policy(
                task=task,
                get_observation=get_observation,
                mode=self._mode.value,
                scene_id=str(getattr(task, "id", "task")),
            )
            strategy = OpenAIVLMStrategyProvider(model=self._strategy_model).strategy_for_scene(
                snapshot,
                mode=self._mode,
                output_dir=self._debug_dir,
            )
            candidates = generate_approach_candidates(
                snapshot,
                strategy,
                count=self._candidates_per_scene,
            )
            candidate = candidates[min(max(0, self._candidate_index), len(candidates) - 1)]
            planner = MoveItPlanner(
                required=True,
                backend=MoveItPyPlanningBackend(self._moveit_backend_config),
            )
            expert = (
                NominalExpert(moveit_planner=planner, candidates_per_scene=self._candidates_per_scene)
                if self._mode == ExpertMode.NOMINAL
                else RecoveryExpert(
                    moveit_planner=planner,
                    ft_config=self._ft_config,
                    candidates_per_scene=self._candidates_per_scene,
                )
            )
            result = expert.generate_candidate(snapshot, strategy, candidate=candidate)
            self._write_result(result.metadata)
            if isinstance(result.planning_result, PlanningFailure) or result.trajectory is None:
                safe_feedback("official_expert_generator_moveit_planning_failed")
                return False
            result.trajectory.save_json(self._output)
            safe_feedback(f"official_expert_generator_piecewise_written:{self._output}")
            return False
        except MoveItUnavailableError as ex:
            self._write_result({"success": False, "reason": "moveit_unavailable", "error": str(ex)})
            safe_feedback("official_expert_generator_moveit_unavailable")
            return False
        except Exception as ex:
            self._write_result({"success": False, "reason": "exception", "type": type(ex).__name__, "error": str(ex)})
            safe_feedback("official_expert_generator_failed")
            return False

    def _write_result(self, payload: dict) -> None:
        path = self._debug_dir / "expert_generation_result.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

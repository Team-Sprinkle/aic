from __future__ import annotations

import os
from io import BytesIO
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from aic_gym_gz.planners.openai_backend import (
    OpenAIPlannerAPIError,
    OpenAIPlannerBackend,
    OpenAIPlannerConfig,
    sanitize_payload,
)
from aic_gym_gz.teacher.types import TeacherPlanningState


class _MockOpenAIPlannerBackend(OpenAIPlannerBackend):
    def __init__(self, response_payload: dict[str, object], config: OpenAIPlannerConfig) -> None:
        super().__init__(config=config)
        self._response_payload = response_payload

    def _post_responses_request(self, payload):  # type: ignore[override]
        self.last_payload = payload
        return self._response_payload


def _planning_state() -> TeacherPlanningState:
    return TeacherPlanningState(
        trial_id="trial_0",
        task_id="task_0",
        goal_summary="goal",
        current_phase="free_space_approach",
        policy_context={
            "plug_pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "target_port_pose": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
            "target_port_entrance_pose": [0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 1.0],
            "off_limit_contact": False,
            "distance_to_target": 0.1,
            "distance_to_entrance": 0.12,
            "lateral_misalignment": 0.02,
            "orientation_error": 0.1,
            "insertion_progress": 0.0,
            "score_geometry": {
                "distance_to_entrance": 0.12,
                "lateral_misalignment": 0.02,
                "orientation_error": 0.1,
                "insertion_progress": 0.0,
            },
        },
        obstacle_summary=[],
        dynamics_summary={
            "quasi_static": True,
            "cable_settling_score": 0.9,
        },
        image_refs=[],
        image_timestamps={},
        image_summaries={},
        recent_probe_results=[],
        temporal_context={
            "phase_guidance": {"recommended_phase": "pre_insert_align"},
            "geometry_progress_summary": {"net_distance_to_entrance_progress": 0.0},
            "auxiliary_history_summary": {"hidden_contact_recent": False},
        },
        data_quality={"wrench": {"is_real": False}},
        oracle_context={
            "task_board_pose_xyz_rpy": [0.1, -0.2, 1.1, 0.0, 0.0, 3.14],
            "target_port_pose": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
            "target_port_entrance_pose": [0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 1.0],
            "plug_pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "clearance_summary": {"present_obstacle_count": 3.0},
            "scene_layout_summary": {
                "present_obstacles": [
                    {
                        "name": "rail_0",
                        "approximate_world_xyz": [0.1, -0.2, 1.1],
                    }
                ]
            },
            "cable": {"name": "cable_0", "type": "sfp_sc"},
        },
    )


class TeacherOpenAIPlannerBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        if self._old_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._old_api_key

    def test_parses_valid_structured_response(self) -> None:
        backend = _MockOpenAIPlannerBackend(
            response_payload={
                "output_text": """{
                    "next_phase": "pre_insert_align",
                    "waypoints": [{"position_xyz": [0.0, 0.0, 0.95], "yaw": 0.0, "speed_scale": 0.5, "clearance_hint": 0.01}],
                    "motion_mode": "fine_cartesian",
                    "caution_flag": false,
                    "should_probe": true,
                    "segment_horizon_steps": 6,
                    "segment_granularity": "fine",
                    "rationale_summary": "Use a conservative align segment."
                }"""
            },
            config=OpenAIPlannerConfig(enabled=True, max_retries=0),
        )
        plan = backend.plan(_planning_state())
        self.assertEqual(plan.next_phase, "pre_insert_align")
        self.assertEqual(len(plan.waypoints), 1)
        self.assertEqual(backend.last_payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(
            backend.last_payload["input"][0]["content"][0]["type"],
            "input_text",
        )
        self.assertIn("compact_planning_brief", backend.last_payload["input"][1]["content"][0]["text"])

    def test_candidate_family_changes_with_candidate_index(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True))
        first = backend.build_debug_payload(_planning_state(), candidate_index=0)
        second = backend.build_debug_payload(_planning_state(), candidate_index=3)
        first_text = first["input"][1]["content"][0]["text"]
        second_text = second["input"][1]["content"][0]["text"]
        self.assertIn("baseline_safe", first_text)
        self.assertIn("guarded_insert", second_text)

    def test_payload_includes_scene_geometry_context(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True))
        payload = backend.build_debug_payload(_planning_state())
        user_text = payload["input"][1]["content"][0]["text"]
        self.assertIn("scene_geometry_context", user_text)
        self.assertIn("board_pose_xyz_rpy", user_text)
        self.assertIn("present_obstacles", user_text)

    def test_visual_context_is_included_when_present(self) -> None:
        state = _planning_state()
        state.recent_visual_observations.append(
            {
                "label": "recent_left",
                "camera_name": "left",
                "sim_tick": 12,
                "sim_time": 0.24,
                "timestamp": 0.24,
                "source": "official_wrist_camera_history",
                "image_data_url": "data:image/png;base64,AAA",
            }
        )
        state.scene_overview_images.append(
            {
                "label": "scene_top",
                "view_name": "top_down_xy",
                "source": "teacher_schematic_scene_overview",
                "image_data_url": "data:image/png;base64,BBB",
            }
        )
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True))
        payload = backend.build_debug_payload(state)
        content = payload["input"][1]["content"]
        self.assertTrue(any(item.get("type") == "input_image" for item in content))
        self.assertTrue(any(item.get("text", "").startswith("Recent official wrist-camera") for item in content))
        self.assertTrue(any(item.get("text", "").startswith("Teacher-side scene overview") for item in content))
        self.assertTrue(any(item.get("image_url") == "<image_data_url>" for item in content if item.get("type") == "input_image"))

    def test_temperature_is_omitted_when_configured_none(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True, temperature=None, model="gpt-5-nano"))
        payload = backend.build_debug_payload(_planning_state())
        self.assertNotIn("temperature", payload)

    def test_reasoning_and_low_verbosity_are_included_by_default(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True))
        payload = backend.build_debug_payload(_planning_state())
        self.assertEqual(payload["reasoning"]["effort"], "low")
        self.assertEqual(payload["text"]["verbosity"], "low")

    def test_rejects_invalid_structured_response(self) -> None:
        backend = _MockOpenAIPlannerBackend(
            response_payload={
                "output_text": """{
                    "next_phase": "not_a_phase",
                    "waypoints": [],
                    "motion_mode": "fine_cartesian",
                    "caution_flag": false,
                    "should_probe": false,
                    "segment_horizon_steps": 0,
                    "segment_granularity": "fine",
                    "rationale_summary": ""
                }"""
            },
            config=OpenAIPlannerConfig(enabled=True, max_retries=0),
        )
        with self.assertRaises(RuntimeError):
            backend.plan(_planning_state())

    def test_build_debug_payload_is_sanitized_and_schema_is_handcrafted(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True))
        payload = backend.build_debug_payload(_planning_state())
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertNotIn("$defs", payload["text"]["format"]["schema"])
        self.assertEqual(
            payload["text"]["format"]["schema"]["properties"]["waypoints"]["items"]["properties"]["position_xyz"]["maxItems"],
            3,
        )
        self.assertEqual(sanitize_payload({"Authorization": "secret"})["Authorization"], "<redacted>")
        self.assertEqual(sanitize_payload({"image_url": "data:image/png;base64,AAA"})["image_url"], "<image_data_url>")

    def test_http_error_message_includes_status_and_body(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True, max_retries=0))
        error = HTTPError(
            url=backend.config.base_url,
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"error":{"message":"invalid schema"}}'),
        )
        with patch("aic_gym_gz.planners.openai_backend.urlopen", side_effect=error):
            with self.assertRaises(OpenAIPlannerAPIError) as ctx:
                backend._post_responses_request(backend.build_smoke_test_payload())
        message = str(ctx.exception)
        self.assertIn("status=400", message)
        self.assertIn("invalid schema", message)
        self.assertIn("structured_output_requested=True", message)

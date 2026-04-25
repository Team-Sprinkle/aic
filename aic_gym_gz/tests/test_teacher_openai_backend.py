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
        task_definition={
            "scenario_task_definition": {
                "task_id": "task_0",
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": "nic_card_mount_0",
                "time_limit_s": 180.0,
            },
            "task_msg": {
                "id": "task_0",
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": "nic_card_mount_0",
                "time_limit": 180,
            },
        },
        current_phase="free_space_approach",
        policy_context={
            "tcp_pose": [0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 1.0],
            "tcp_velocity": [0.01, 0.0, -0.02, 0.0, 0.0, 0.1],
            "plug_pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "target_port_pose": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
            "target_port_entrance_pose": [0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 1.0],
            "wrench": [0.0, 0.0, 1.5, 0.0, 0.0, 0.1],
            "wrench_timestamp": 12.5,
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
            "relative_geometry": {
                "plug_to_entrance_xyz": [0.0, 0.0, -0.08],
                "insertion_axis_world_xyz": [0.0, 0.0, -1.0],
            },
            "frame_context": {
                "runtime_pose_frame": "world",
                "runtime_action_command_frame": "world",
                "official_policy_reference_frame": "base_link",
            },
            "geometry_tool_outputs": {
                "frame_transform_queries": {
                    "official_policy_base_link_to_runtime_world": {
                        "ok": False,
                        "transform_available": False,
                    }
                },
                "distance_and_alignment_queries": {
                    "plug_to_target_port": {
                        "distance_to_target_m": 0.1,
                        "distance_to_entrance_m": 0.12,
                        "lateral_offset_m": 0.02,
                        "axial_depth_m": -0.08,
                        "insertion_progress": 0.0,
                    }
                },
                "clearance_distance_queries": {
                    "plug_to_entrance_segment": {
                        "nearest_obstacle_distance_m": 0.031,
                        "approach_segment_min_clearance_m": 0.015,
                    }
                },
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
            "wrench_contact_trend_summary": {"current_force_l2": 0.0},
            "compact_signal_samples": {
                "wrench_force_samples": [[0.0, 0.0, 0.0]],
                "tcp_speed_samples": [0.0],
            },
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
        controller_context={
            "reference_tcp_pose": [0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 1.0],
            "tcp_error": [0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
            "controller_target_mode": 1,
        },
        planning_metadata={
            "planner_output_mode": "absolute_cartesian_waypoint",
            "scene_overview_sources": {"top_down_xy": "teacher_schematic_scene_overview"},
            "scene_overview_live_source_used": False,
            "prefer_live_scene_overview": False,
            "overlay_metadata": {
                "xyz_axis_overlay_targets": [{"name": "plug"}],
                "insertion_axis_overlay": {"label": "port_insertion_axis"},
                "zoomed_interaction_crop": {"radius_m": 0.05},
            },
            "signal_reliability_summary": {
                "real_signals": ["controller_state"],
                "approximate_signals": ["wrench"],
                "missing_signals": [],
            },
            "available_helper_tool_outputs": [
                "frame_transform_query",
                "distance_and_alignment_query",
                "clearance_distance_query",
                "signal_reliability_summary",
                "xyz_axis_overlay_metadata",
            ],
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
                    "rationale_summary": "Use a conservative align segment.",
                    "decision_diagnostics": {
                        "confidence_level": "medium",
                        "blocking_gaps": ["No explicit transform from base_link to world was provided."],
                        "ambiguous_frames": ["Controller pose appears to use a different frame from world entities."],
                        "requested_tools": ["frame_transform_query", "distance_and_alignment_query"],
                        "requested_visual_aids": ["xyz_axis_arrow_overlay", "port_axis_or_insertion_axis_overlay"],
                        "assumptions_used": ["Assumed target and plug poses are both expressed in the runtime world frame."]
                    }
                }"""
            },
            config=OpenAIPlannerConfig(enabled=True, max_retries=0),
        )
        plan = backend.plan(_planning_state())
        self.assertEqual(plan.next_phase, "pre_insert_align")
        self.assertEqual(len(plan.waypoints), 1)
        self.assertEqual(plan.decision_diagnostics["confidence_level"], "medium")
        self.assertIn("frame_transform_query", plan.decision_diagnostics["requested_tools"])
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
        self.assertIn("relative_geometry", user_text)
        self.assertIn("reference_tcp_pose", user_text)
        self.assertIn("wrench_contact_trend_summary", user_text)
        self.assertIn("compact_signal_samples", user_text)
        self.assertIn("tcp_velocity", user_text)
        self.assertIn("wrench_timestamp", user_text)
        self.assertIn("target_yaw", user_text)
        self.assertIn("yaw_error_to_target", user_text)
        self.assertIn("scene_overview_sources", user_text)
        self.assertIn("runtime_pose_frame", user_text)
        self.assertIn("task_definition", user_text)
        self.assertIn("available_helper_tools", user_text)
        self.assertIn("geometry_tool_outputs", user_text)
        self.assertIn("overlay_metadata", user_text)
        self.assertIn("signal_reliability_summary", user_text)
        self.assertIn("distance_and_alignment_query", user_text)
        self.assertIn("clearance_distance_query", user_text)
        self.assertIn("xyz_axis_arrow_overlay", user_text)
        self.assertIn("\"target_module_name\": \"nic_card_mount_0\"", user_text)

    def test_build_global_guidance_payload_uses_global_schema(self) -> None:
        backend = OpenAIPlannerBackend(
            OpenAIPlannerConfig(
                enabled=True,
                enable_global_guidance=True,
                global_model="gpt-5.4-mini",
            )
        )
        payload = backend.build_global_guidance_request_payload(_planning_state())
        self.assertEqual(payload["model"], "gpt-5.4-mini")
        self.assertEqual(payload["text"]["format"]["name"], "teacher_global_guidance")
        self.assertEqual(payload["text"]["format"]["schema"]["required"][0], "strategy_summary")
        self.assertIn("decision_diagnostics", payload["text"]["format"]["schema"]["required"])

    def test_visual_context_is_included_when_present(self) -> None:
        state = _planning_state()
        state.recent_visual_observations.append(
            {
                "label": "recent_left",
                "camera_name": "left",
                "sim_tick": 12,
                "sim_time": 0.24,
                "timestamp": 0.24,
                "age_from_latest_s": 0.0,
                "age_from_latest_steps": 0,
                "timepoint_label": "current",
                "source": "official_wrist_camera_history",
                "image_data_url": "data:image/png;base64,AAA",
            }
        )
        state.scene_overview_images.append(
            {
                "label": "scene_top",
                "view_name": "top_down_xy",
                "source": "teacher_schematic_scene_overview",
                "timestamp": None,
                "image_data_url": "data:image/png;base64,BBB",
            }
        )
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True))
        payload = backend.build_debug_payload(state)
        content = payload["input"][1]["content"]
        self.assertTrue(any(item.get("type") == "input_image" for item in content))
        self.assertTrue(any(item.get("text", "").startswith("Recent official wrist-camera") for item in content))
        self.assertTrue(any(item.get("text", "").startswith("Teacher-side scene overview") for item in content))
        self.assertTrue(any("timepoint_label=current" in item.get("text", "") for item in content))
        self.assertTrue(any(item.get("image_url") == "<image_data_url>" for item in content if item.get("type") == "input_image"))

    def test_temperature_is_omitted_when_configured_none(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True, temperature=None, model="gpt-5-nano"))
        payload = backend.build_debug_payload(_planning_state())
        self.assertNotIn("temperature", payload)

    def test_temperature_is_omitted_for_gpt5_models(self) -> None:
        backend = OpenAIPlannerBackend(OpenAIPlannerConfig(enabled=True, temperature=0.1, model="gpt-5.4-mini"))
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
                    "rationale_summary": "",
                    "decision_diagnostics": {
                        "confidence_level": "low",
                        "blocking_gaps": [],
                        "ambiguous_frames": [],
                        "requested_tools": [],
                        "requested_visual_aids": [],
                        "assumptions_used": []
                    }
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

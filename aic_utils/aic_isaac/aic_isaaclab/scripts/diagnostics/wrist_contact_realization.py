#!/usr/bin/env python3
"""Standalone wrist IK contact-realization diagnostic for SFP insertion.

This bypasses ACT/SERL completely.  It creates the Isaac Lab AIC task, uses the
configured Differential IK wrist action path, drives the semantic tip to a
shallow positive insertion depth, then applies direct wrist translation or
rotation probes while logging realized ``sfp_tip_link`` and ``sfp_module_link``
motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="aic/outputs/agentic_reward_curriculum_20260524_wrist_contact/runs")
parser.add_argument("--run_name", default="wrist_contact_realization")
parser.add_argument("--episode_config_dir", type=str, default=None)
parser.add_argument(
    "--override_start_signed_depth_m",
    type=float,
    default=float("nan"),
    help=(
        "If finite, generate a temporary episode-config copy whose semantic tip reset starts "
        "at this signed depth relative to the entrance. Positive values start inside the cage."
    ),
)
parser.add_argument("--override_start_lateral_m", type=float, default=0.0003)
parser.add_argument(
    "--override_start_orientation_wxyz",
    type=float,
    nargs=4,
    default=None,
    help="Optional reset-body orientation to write into generated shallow/depth override episode configs.",
)
parser.add_argument(
    "--override_start_orientation_rotvec_world",
    type=float,
    nargs=3,
    default=None,
    help="Optional world-frame rotation vector composed onto the generated reset-body orientation.",
)
parser.add_argument(
    "--override_start_tip_orientation_wxyz",
    type=float,
    nargs=4,
    default=None,
    help=(
        "Desired semantic tip orientation for generated resets. With "
        "--derive_reset_orientation_from_tip, the reset-body orientation is derived by preserving "
        "the source episode's gripper-to-tip orientation relationship."
    ),
)
parser.add_argument(
    "--derive_reset_orientation_from_tip",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--derive_reset_position_from_orientation",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "When generating an override reset with a changed reset-body orientation, recompute the "
        "reset-body position from the source local reset-body-to-tip offset instead of reusing "
        "the source world offset. This prevents rotation-induced semantic tip lateral sweep."
    ),
)
parser.add_argument("--episode_length_s", type=float, default=12.0)
parser.add_argument("--near_gate_reset_max_iterations", type=int, default=0)
parser.add_argument("--near_gate_reset_position_tolerance", type=float, default=0.0)
parser.add_argument("--near_gate_reset_orientation_tolerance", type=float, default=0.0)
parser.add_argument("--isaac_action_scale", type=float, default=1.0)
parser.add_argument(
    "--absolute_ik_target_pose",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: switch Differential IK to absolute pose mode and send root-frame target poses "
        "instead of relative delta poses. This mirrors the official Gazebo teacher's exact-position insertion path."
    ),
)
parser.add_argument(
    "--absolute_ik_pin_reset_orientation",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="With --absolute_ik_target_pose, hold the post-reset wrist orientation as the absolute IK target orientation.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--target_reward_body", default="sfp_tip_link")
parser.add_argument("--target_reward_consistency_body", default="sfp_module_link")
parser.add_argument("--target_reward_orientation_error_mode", choices=["axis", "quat"], default="axis")
parser.add_argument("--target_reward_orientation_axis_local", type=float, nargs=3, default=(0.0, 0.0, 1.0))
parser.add_argument("--target_reward_consistency_axial_std", type=float, default=0.0025)
parser.add_argument("--target_reward_consistency_lateral_sigma", type=float, default=0.0020)
parser.add_argument("--approach_steps", type=int, default=95)
parser.add_argument("--probe_steps", type=int, default=30)
parser.add_argument(
    "--stop_on_near_success_capture",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: stop rollout after capturing the first frame that satisfies configured near-success "
        "teacher thresholds. This preserves useful near-seated trajectories before retreat. It does not alter "
        "strict_success or success thresholds."
    ),
)
parser.add_argument("--near_success_capture_min_s_m", type=float, default=0.0450)
parser.add_argument("--near_success_capture_max_r_m", type=float, default=0.0005)
parser.add_argument("--near_success_capture_max_theta_rad", type=float, default=0.0360)
parser.add_argument("--near_success_capture_min_module_consistency", type=float, default=0.85)
parser.add_argument("--shallow_depth_m", type=float, default=0.0050)
parser.add_argument("--approach_lateral_step_m", type=float, default=0.00018)
parser.add_argument("--approach_axial_step_m", type=float, default=0.00012)
parser.add_argument("--approach_lateral_gate_m", type=float, default=0.00035)
parser.add_argument("--approach_lateral_sign", type=float, default=1.0)
parser.add_argument("--approach_axial_sign", type=float, default=1.0)
parser.add_argument(
    "--probe",
    choices=[
        "axis_backout",
        "zero_action",
        "axis_forward",
        "rotation_axes",
        "rotation_axis",
        "orientation_servo_best",
        "pose_hold",
        "pose_hold_orientation_servo_best",
        "pose_hold_rotation_axis",
        "pose_hold_constrained_rotation_axis",
        "target_tip_stabilize",
        "target_module_stabilize",
        "target_tip_then_module_stabilize",
        "target_tip_fixed_rotation_axis",
        "pinned_wrist_axis_descent",
    ],
    default="axis_backout",
)
parser.add_argument("--probe_translation_step_m", type=float, default=0.00020)
parser.add_argument("--probe_rotation_step_rad", type=float, default=0.004)
parser.add_argument("--probe_rotation_axis", choices=["x", "y", "z"], default="y")
parser.add_argument("--probe_rotation_sign", type=float, default=1.0)
parser.add_argument("--probe_lateral_correction_step_m", type=float, default=0.0)
parser.add_argument("--probe_axial_step_m", type=float, default=0.0)
parser.add_argument("--probe_axial_lateral_gate_m", type=float, default=0.0005)
parser.add_argument(
    "--target_tip_stabilize_axial_step_m",
    type=float,
    default=float("nan"),
    help="Axial correction clip for target_tip_stabilize; defaults to --probe_translation_step_m.",
)
parser.add_argument(
    "--target_tip_stabilize_lateral_step_m",
    type=float,
    default=float("nan"),
    help="Lateral correction clip for target_tip_stabilize; defaults to --probe_translation_step_m.",
)
parser.add_argument(
    "--target_tip_stabilize_rotation_compensation_clip_m",
    type=float,
    default=0.0010,
    help="Clip for translation compensation that counteracts rotation-induced semantic tip sweep.",
)
parser.add_argument(
    "--target_tip_stabilize_orientation_gate_lateral_m",
    type=float,
    default=0.0007,
    help="Only apply semantic orientation correction when current lateral error is below this threshold.",
)
parser.add_argument(
    "--target_tip_stabilize_orientation_gate_depth_m",
    type=float,
    default=0.0060,
    help="Only apply semantic orientation correction when current signed depth is above this threshold.",
)
parser.add_argument(
    "--target_tip_stabilize_orientation_error_threshold_rad",
    type=float,
    default=0.0,
    help=(
        "Only apply semantic orientation correction when theta is above this threshold. "
        "Use the strict theta threshold to avoid rotation-induced lateral sweep after alignment is already good."
    ),
)
parser.add_argument(
    "--target_tip_stabilize_inward_bias_m",
    type=float,
    default=0.0,
    help="Small positive axis bias added while near the target; useful to preload contact at reset.",
)
parser.add_argument(
    "--target_tip_stabilize_goal_depth_m",
    type=float,
    default=float("nan"),
    help=(
        "If finite, target the semantic tip to entrance + this signed depth along the insertion "
        "axis instead of holding the post-reset tip position. Defaults to holding the post-reset tip."
    ),
)
parser.add_argument(
    "--target_module_stabilize_body",
    default="sfp_module_link",
    help="Semantic body targeted by the target_module_stabilize probe.",
)
parser.add_argument(
    "--target_module_stabilize_goal_depth_m",
    type=float,
    default=float("nan"),
    help=(
        "If finite, target --target_module_stabilize_body to entrance + this signed depth along the "
        "insertion axis. Defaults to holding the post-reset module/body position."
    ),
)
parser.add_argument(
    "--target_module_stabilize_axial_step_m",
    type=float,
    default=float("nan"),
    help="Axial correction clip for target_module_stabilize; defaults to --target_tip_stabilize_axial_step_m.",
)
parser.add_argument(
    "--target_module_stabilize_lateral_step_m",
    type=float,
    default=float("nan"),
    help="Lateral correction clip for target_module_stabilize; defaults to --target_tip_stabilize_lateral_step_m.",
)
parser.add_argument(
    "--target_module_stabilize_tip_lateral_step_m",
    type=float,
    default=float("nan"),
    help=(
        "Optional semantic-tip lateral correction clip for target_module_stabilize. "
        "When finite, module axial motion is combined with tip centerline correction instead of relying only on "
        "module lateral error. Defaults disabled to preserve historical behavior."
    ),
)
parser.add_argument(
    "--target_module_stabilize_secondary_module_lateral_step_m",
    type=float,
    default=0.0,
    help=(
        "Optional additional module-body lateral trim blended into target_module_stabilize after the semantic tip "
        "is deep. Defaults disabled; use tiny values to avoid overriding tip centering."
    ),
)
parser.add_argument(
    "--target_module_stabilize_secondary_module_lateral_activation_depth_m",
    type=float,
    default=0.030,
)
parser.add_argument(
    "--target_module_stabilize_secondary_module_lateral_threshold_m",
    type=float,
    default=0.0010,
)
parser.add_argument(
    "--target_module_stabilize_tip_lateral_gate_m",
    type=float,
    default=float("nan"),
    help=(
        "Optional gate for module axial motion in target_module_stabilize. "
        "When finite, positive axial module commands are blocked while semantic-tip lateral error exceeds this value."
    ),
)
parser.add_argument(
    "--target_module_stabilize_tip_theta_gate_rad",
    type=float,
    default=float("nan"),
    help=(
        "Optional gate for module axial motion in target_module_stabilize. "
        "When finite, positive axial module commands are blocked while semantic-tip theta exceeds this value."
    ),
)
parser.add_argument(
    "--target_module_stabilize_orientation_step_rad",
    type=float,
    default=0.0,
    help=(
        "Optional semantic-tip orientation trim step for target_module_stabilize. "
        "Defaults disabled; when positive it uses the same bounded orientation probe as target_tip_stabilize."
    ),
)
parser.add_argument(
    "--target_module_stabilize_orientation_lateral_gate_m",
    type=float,
    default=0.00065,
)
parser.add_argument(
    "--target_module_stabilize_orientation_activation_depth_m",
    type=float,
    default=0.030,
)
parser.add_argument(
    "--target_module_stabilize_orientation_error_threshold_rad",
    type=float,
    default=0.033,
)
parser.add_argument(
    "--target_module_stabilize_orientation_start_probe_step",
    type=int,
    default=0,
    help="Diagnostic-only: earliest target_module_stabilize probe step where orientation trim may activate.",
)
parser.add_argument(
    "--target_module_stabilize_use_current_orientation_when_no_trim",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: in absolute IK mode, hold the current wrist orientation when target_module_stabilize "
        "orientation trim is inactive instead of falling back to the pinned reset orientation."
    ),
)
parser.add_argument(
    "--target_module_stabilize_orientation_cross_axis_candidate",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: include the direct semantic axis-alignment rotation "
        "(tip_axis x insertion_axis) when selecting target_module_stabilize orientation trim."
    ),
)
parser.add_argument(
    "--target_module_stabilize_orientation_min_module_consistency",
    type=float,
    default=float("nan"),
    help=(
        "Optional module-consistency gate for target_module_stabilize orientation trim. "
        "When finite, semantic-tip rotation is disabled unless the current module consistency "
        "score is at least this value. This prevents final orientation trim from destroying a "
        "near-seated module-following state."
    ),
)
parser.add_argument(
    "--target_module_stabilize_orientation_module_lateral_penalty",
    type=float,
    default=0.0,
    help=(
        "Optional predicted module-lateral-sweep penalty used when selecting the "
        "target_module_stabilize orientation trim axis. Defaults disabled. Positive "
        "values make final orientation trim prefer rotations that do not push the "
        "module body farther off the port centerline."
    ),
)
parser.add_argument(
    "--target_module_stabilize_orientation_module_lateral_margin_m",
    type=float,
    default=0.0,
    help=(
        "Allowed predicted module lateral-error increase before "
        "--target_module_stabilize_orientation_module_lateral_penalty is applied."
    ),
)
parser.add_argument(
    "--target_module_stabilize_rotation_compensation_clip_m",
    type=float,
    default=0.0005,
)
parser.add_argument(
    "--target_module_stabilize_polish_after_near_success",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: once the tip is deep/centered and module consistency is high, "
        "suppress further positive module axial motion so orientation/lateral trim can polish "
        "the near-seated state instead of over-driving the module."
    ),
)
parser.add_argument("--target_module_stabilize_polish_min_tip_depth_m", type=float, default=0.044)
parser.add_argument("--target_module_stabilize_polish_max_tip_lateral_m", type=float, default=0.0005)
parser.add_argument("--target_module_stabilize_polish_max_tip_theta_rad", type=float, default=float("inf"))
parser.add_argument("--target_module_stabilize_polish_min_module_consistency", type=float, default=0.80)
parser.add_argument("--target_module_stabilize_polish_axial_step_m", type=float, default=0.0)
parser.add_argument(
    "--target_module_stabilize_polish_latch",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Diagnostic-only: once target_module_stabilize polish activates, keep its axial limit active until reset.",
)
parser.add_argument(
    "--target_tip_then_module_switch_depth_m",
    type=float,
    default=0.024,
    help="For target_tip_then_module_stabilize, latch to target_module_stabilize after tip signed depth reaches this value.",
)
parser.add_argument(
    "--target_tip_then_module_switch_lateral_m",
    type=float,
    default=0.00075,
    help="For target_tip_then_module_stabilize, require tip lateral error below this value before latching to module mode.",
)
parser.add_argument(
    "--target_tip_then_module_switch_theta_rad",
    type=float,
    default=0.060,
    help="For target_tip_then_module_stabilize, require tip orientation error below this value before latching to module mode.",
)
parser.add_argument(
    "--pinned_wrist_axis_descent_distance_m",
    type=float,
    default=0.0360,
    help="Total positive insertion-axis distance for pinned_wrist_axis_descent.",
)
parser.add_argument(
    "--pinned_wrist_axis_descent_step_m",
    type=float,
    default=float("nan"),
    help="Per-step path increment for pinned_wrist_axis_descent; defaults to --probe_translation_step_m.",
)
parser.add_argument(
    "--probe_orientation_lateral_penalty",
    type=float,
    default=0.0,
    help="Tie-break penalty on predicted lateral sweep for orientation_servo_best.",
)
parser.add_argument(
    "--pose_hold_body",
    default="gripper_tcp",
    help="Robot body whose post-reset world pose is held by the pose_hold probe.",
)
parser.add_argument(
    "--pose_hold_position_gain",
    type=float,
    default=1.0,
    help="Gain applied to pose_hold translational error before clipping by --probe_translation_step_m.",
)
parser.add_argument(
    "--pose_hold_rotation_gain",
    type=float,
    default=1.0,
    help="Gain applied to pose_hold rotational error before clipping by --probe_rotation_step_rad.",
)
parser.add_argument("--pose_hold_orientation_activation_depth_m", type=float, default=0.0075)
parser.add_argument("--pose_hold_orientation_activation_lateral_m", type=float, default=0.0005)
parser.add_argument("--pose_hold_orientation_activation_consistency", type=float, default=0.80)
parser.add_argument(
    "--pose_hold_orientation_step_rad",
    type=float,
    default=float("nan"),
    help="Semantic orientation trim step for pose_hold_orientation_servo_best; defaults to --probe_rotation_step_rad.",
)
parser.add_argument(
    "--pose_hold_fixed_rotation_step_rad",
    type=float,
    default=float("nan"),
    help="Fixed world-axis trim step for pose_hold_rotation_axis; defaults to --probe_rotation_step_rad.",
)
parser.add_argument(
    "--pose_hold_constrained_tip_weight",
    type=float,
    default=1.0,
    help=(
        "Weight on preserving sfp_tip_link position for pose_hold_constrained_rotation_axis. "
        "The probe computes a shared translation compensation for rotation-induced body sweep."
    ),
)
parser.add_argument(
    "--pose_hold_constrained_module_weight",
    type=float,
    default=1.0,
    help="Weight on preserving sfp_module_link position for pose_hold_constrained_rotation_axis.",
)
parser.add_argument(
    "--pose_hold_constrained_compensation_clip_m",
    type=float,
    default=0.0005,
    help="Clip for the shared tip/module translation compensation in pose_hold_constrained_rotation_axis.",
)
parser.add_argument(
    "--pose_hold_orientation_start_probe_step",
    type=int,
    default=1,
    help="Earliest zero-based probe step where pose_hold_orientation_servo_best may apply semantic orientation trim.",
)
parser.add_argument("--fix_isaac_ik_xy_sign", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--fix_isaac_ik_z_sign",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Diagnostic-only: also flip the root-frame Z translation sent to the Isaac wrist IK action.",
)
parser.add_argument(
    "--fix_isaac_ik_rot_xy_sign",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Diagnostic-only: flip root-frame X/Y rotation commands sent to the Isaac wrist IK action.",
)
parser.add_argument(
    "--fix_isaac_ik_rot_z_sign",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Diagnostic-only: flip root-frame Z rotation commands sent to the Isaac wrist IK action.",
)
parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--image_log_every", type=int, default=1)
parser.add_argument("--max_logged_image_steps", type=int, default=140)
parser.add_argument(
    "--save_initial_images",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save step_000000 camera images immediately after reset, before the first command.",
)
parser.add_argument(
    "--save_initial_metrics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Write a step_000000 metrics row immediately after reset, before the first command.",
)
parser.add_argument(
    "--diagnostic_camera_width",
    type=int,
    default=448,
    help="Override diagnostic camera width without changing the task default. Non-positive keeps the config default.",
)
parser.add_argument(
    "--diagnostic_camera_height",
    type=int,
    default=448,
    help="Override diagnostic camera height without changing the task default. Non-positive keeps the config default.",
)
parser.add_argument(
    "--diagnostic_camera_focal_length",
    type=float,
    default=12.0,
    help="Diagnostic-only camera focal length override. Smaller values zoom out for full-scene diagnostic videos.",
)
parser.add_argument(
    "--diagnostic_camera_horizontal_aperture",
    type=float,
    default=float("nan"),
    help="Diagnostic-only camera horizontal aperture override. Finite larger values widen FOV.",
)
parser.add_argument(
    "--diagnostic_camera_vertical_aperture",
    type=float,
    default=float("nan"),
    help="Diagnostic-only camera vertical aperture override. Finite larger values widen FOV.",
)
parser.add_argument(
    "--diagnostic_camera_offset_pos",
    type=float,
    nargs=3,
    default=None,
    help="Diagnostic-only camera offset position override applied to center/left/right camera configs.",
)
parser.add_argument(
    "--diagnostic_camera_offset_rot",
    type=float,
    nargs=4,
    default=None,
    help="Diagnostic-only camera offset quaternion override applied to center/left/right camera configs.",
)
parser.add_argument(
    "--diagnostic_fixed_camera_pos",
    type=float,
    nargs=3,
    default=(0.18, 0.52, 0.22),
    help=(
        "Diagnostic-only fixed scene camera position for the center camera in env-local coordinates. "
        "Defaults to a fixed full-scene video view; pass --diagnostic_wrist_cameras to keep the robot-mounted cameras."
    ),
)
parser.add_argument(
    "--diagnostic_fixed_camera_target",
    type=float,
    nargs=3,
    default=(0.236, 0.371, 0.135),
    help=(
        "Diagnostic-only fixed scene camera look-at target in env-local coordinates. "
        "Defaults to the SFP/NIC insertion region."
    ),
)
parser.add_argument(
    "--diagnostic_wrist_cameras",
    action="store_true",
    help="Use the task's robot-mounted cameras instead of the fixed scene cameras for diagnostic images/videos.",
)
parser.add_argument(
    "--diagnostic_fixed_camera_side_offset_m",
    type=float,
    default=0.040,
    help="Left/right fixed diagnostic camera lateral offset from the center camera.",
)
parser.add_argument(
    "--diagnostic_fixed_camera_convention",
    choices=["ros", "opengl"],
    default="opengl",
    help="Coordinate convention for the diagnostic fixed-camera look-at quaternion.",
)
parser.add_argument(
    "--diagnostic_disable_command_visuals",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Hide command/debug visualization markers in diagnostic camera images and videos.",
)
parser.add_argument(
    "--save_videos",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Encode separate per-camera MP4 videos from saved step images at the end of the run.",
)
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--video_final_hold_s", type=float, default=2.0)
parser.add_argument("--video_crf", type=int, default=16)
parser.add_argument(
    "--disable_collision_prim_regex",
    action="append",
    default=[],
    help=(
        "Diagnostic-only regex matched against USD prim paths after env creation. "
        "Matched collision prims have physics collision disabled and are logged. "
        "Repeatable. Does not modify source USD assets."
    ),
)
parser.add_argument(
    "--collision_contact_tune_prim_regex",
    action="append",
    default=[],
    help=(
        "Diagnostic-only regex matched against USD collision prim paths after env creation. Matching prims receive "
        "PhysX contact/rest offset overrides when the corresponding offset flags are finite. Repeatable."
    ),
)
parser.add_argument("--collision_contact_offset_m", type=float, default=float("nan"))
parser.add_argument("--collision_rest_offset_m", type=float, default=float("nan"))
parser.add_argument(
    "--collision_material_tune_prim_regex",
    action="append",
    default=[],
    help=(
        "Diagnostic-only regex matched against USD collision prim paths after env creation. Matching prims are "
        "bound to a runtime physics material when one or more material flags are finite. Repeatable."
    ),
)
parser.add_argument("--collision_static_friction", type=float, default=float("nan"))
parser.add_argument("--collision_dynamic_friction", type=float, default=float("nan"))
parser.add_argument("--collision_restitution", type=float, default=float("nan"))
parser.add_argument(
    "--replace_sfp_body_sdf_collision_with_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: disable converted SFP module body_sdf_collision mesh prims and add runtime cube "
        "colliders matching the four Gazebo body_collider_box entries in aic_assets/models/SFP Module/model.sdf. "
        "Does not modify source assets or defaults."
    ),
)
parser.add_argument(
    "--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: like --replace_sfp_body_sdf_collision_with_sdf_boxes, but shrinks each Gazebo body box "
        "by configurable per-axis margins. This tests a physically registered multi-box shell between the converted "
        "mesh and the permissive single clearance box. Defaults remain unchanged."
    ),
)
parser.add_argument(
    "--replace_sfp_module_sdf_collision_with_active_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: disable converted SFP module body_sdf_collision mesh prims and add runtime cube "
        "colliders for the sfp_module_link boxes that remain active in aic_assets/models/sfp_sc_cable/model.sdf."
    ),
)
parser.add_argument(
    "--replace_sfp_body_sdf_collision_with_clearance_box",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: disable converted SFP module body_sdf_collision mesh prims and add one configurable "
        "box collider approximating the SFP module envelope. This isolates whether the converted mesh complexity "
        "or thickness is the blocker without changing default assets."
    ),
)
parser.add_argument("--sfp_clearance_box_size_m", type=float, nargs=3, default=(0.0135, 0.0470, 0.0082))
parser.add_argument("--sfp_clearance_box_translation_m", type=float, nargs=3, default=(0.0, 0.001, 0.0))
parser.add_argument("--sfp_shrunk_box_margin_m", type=float, nargs=3, default=(0.00015, 0.0, 0.00015))
parser.add_argument(
    "--replace_nic_cage_p0_with_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: disable Isaac NIC cage_p0_* mesh collisions and add runtime cube colliders from "
        "the first SFP cage in aic_assets/models/NIC Card/model.sdf. Defaults remain unchanged."
    ),
)
parser.add_argument(
    "--replace_nic_cage_p0_with_aligned_cubes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Diagnostic-only: disable Isaac NIC cage_p0_* mesh collisions and add runtime USD cubes using each "
        "original cage prim's local transform. This preserves registration while testing mesh-vs-box contact."
    ),
)
parser.add_argument(
    "--nic_card_sdf",
    default="aic/aic_assets/models/NIC Card/model.sdf",
    help="SDF source for --replace_nic_cage_p0_with_sdf_boxes.",
)
parser.add_argument(
    "--sfp_module_sdf",
    default="aic/aic_assets/models/SFP Module/model.sdf",
    help="SDF source for SFP collision replacement flags.",
)
parser.add_argument(
    "--sfp_cable_sdf",
    default="aic/aic_assets/models/sfp_sc_cable/model.sdf",
    help="SDF wrapper source used to exclude Gazebo-removed SFP module colliders.",
)
parser.add_argument(
    "--save_image_env_indices",
    type=str,
    default="",
    help="Optional comma-separated env indices to save images for. Empty saves every env.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _quat_normalize_list(q: list[float] | tuple[float, float, float, float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    return [float(v) / norm for v in q]


def _quat_conj_list(q: list[float] | tuple[float, float, float, float]) -> list[float]:
    qn = _quat_normalize_list(q)
    return [qn[0], -qn[1], -qn[2], -qn[3]]


def _quat_mul_list(
    lhs: list[float] | tuple[float, float, float, float],
    rhs: list[float] | tuple[float, float, float, float],
) -> list[float]:
    lw, lx, ly, lz = _quat_normalize_list(lhs)
    rw, rx, ry, rz = _quat_normalize_list(rhs)
    return _quat_normalize_list(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_from_rotvec_list(rotvec: list[float] | tuple[float, float, float]) -> list[float]:
    angle = math.sqrt(sum(float(v) * float(v) for v in rotvec))
    if angle < 1.0e-12:
        return [1.0, 0.0, 0.0, 0.0]
    axis = [float(v) / angle for v in rotvec]
    half = 0.5 * angle
    return _quat_normalize_list([math.cos(half), *(math.sin(half) * v for v in axis)])


def _quat_apply_list(
    quat: list[float] | tuple[float, float, float, float],
    vec: list[float] | tuple[float, float, float],
) -> list[float]:
    def raw_mul(lhs: list[float], rhs: list[float]) -> list[float]:
        lw, lx, ly, lz = lhs
        rw, rx, ry, rz = rhs
        return [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]

    qn = _quat_normalize_list(quat)
    rotated = raw_mul(raw_mul(qn, [0.0, *[float(v) for v in vec]]), _quat_conj_list(qn))
    return [float(v) for v in rotated[1:]]


def _prepare_episode_config_dir() -> str | None:
    if not args_cli.episode_config_dir:
        return None
    source = Path(args_cli.episode_config_dir)
    if not math.isfinite(float(args_cli.override_start_signed_depth_m)):
        return str(source)
    source_episodes = source / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(f"episode_config_dir has no episodes directory: {source_episodes}")
    out_root = (
        Path("aic/outputs/agentic_reward_curriculum_20260524_wrist_contact/generated_episode_configs")
        / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_s{int(round(float(args_cli.override_start_signed_depth_m) * 1000.0))}mm"
    )
    episodes_out = out_root / "episodes"
    episodes_out.mkdir(parents=True, exist_ok=True)
    for idx, episode_path in enumerate(sorted(source_episodes.glob("episode_*.yaml"))[: max(int(args_cli.num_envs), 1)], start=1):
        data = yaml.safe_load(episode_path.read_text(encoding="utf-8"))
        scene = data.get("scene") or {}
        target = scene.get("target") or {}
        start = target.get("start_near_gate")
        if not isinstance(start, dict):
            start = scene.get("start_near_gate")
        if not isinstance(start, dict):
            start = data.setdefault("start_near_gate", {})
        entrance = target.get("entrance_pose_world", {}).get("position")
        axis = target.get("insertion_axis_world")
        lateral_dir = start.get("lateral_direction_world")
        offset = start.get("reset_body_offset_from_reference_world")
        source_body_start = start.get("body_start_position_world") or start.get("tcp_start_position_world")
        source_reference = (
            start.get("reference_reward_body_start_position_world")
            or start.get("reference_tip_center_position_world")
            or start.get("reference_body_position")
            or start.get("reference_tcp_position")
        )
        source_reset_orientation = start.get("body_start_orientation_wxyz") or start.get("reset_body_orientation_wxyz")
        source_tip_orientation = start.get("reference_reward_body_start_orientation_wxyz") or target.get(
            "body_start_orientation_wxyz"
        )
        desired_tip_orientation = (
            [float(x) for x in args_cli.override_start_tip_orientation_wxyz]
            if args_cli.override_start_tip_orientation_wxyz is not None
            else source_tip_orientation
        )
        if bool(args_cli.derive_reset_orientation_from_tip):
            if source_reset_orientation is None or desired_tip_orientation is None or source_tip_orientation is None:
                raise ValueError(
                    f"{episode_path} cannot derive reset orientation without source reset/tip orientations"
                )
            tip_to_reset = _quat_mul_list(_quat_conj_list(source_tip_orientation), source_reset_orientation)
            orientation = _quat_mul_list(desired_tip_orientation, tip_to_reset)
        else:
            orientation = (
                [float(x) for x in args_cli.override_start_orientation_wxyz]
                if args_cli.override_start_orientation_wxyz is not None
                else source_reset_orientation or target.get("body_start_orientation_wxyz")
            )
        if orientation is not None and args_cli.override_start_orientation_rotvec_world is not None:
            orientation = _quat_mul_list(
                _quat_from_rotvec_list([float(x) for x in args_cli.override_start_orientation_rotvec_world]),
                orientation,
            )
        if entrance is None or axis is None or lateral_dir is None:
            raise ValueError(f"{episode_path} is missing entrance/axis/lateral fields")
        if (
            isinstance(source_body_start, (list, tuple))
            and len(source_body_start) == 3
            and isinstance(source_reference, (list, tuple))
            and len(source_reference) == 3
        ):
            # Some generated episode configs carry stale reset_body_offset_from_reference_world.
            # The actual body/reference poses are authoritative for preserving semantic tip placement.
            offset = [float(source_body_start[i]) - float(source_reference[i]) for i in range(3)]
        elif offset is None:
            raise ValueError(f"{episode_path} is missing reset body/reference pose fields")
        signed_depth = float(args_cli.override_start_signed_depth_m)
        lateral_m = float(args_cli.override_start_lateral_m)
        reference = [
            float(entrance[i]) + signed_depth * float(axis[i]) + lateral_m * float(lateral_dir[i])
            for i in range(3)
        ]
        if bool(args_cli.derive_reset_position_from_orientation):
            if source_reset_orientation is None or orientation is None:
                raise ValueError(f"{episode_path} cannot derive reset position without source/current orientation")
            local_reset_to_tip = _quat_apply_list(
                _quat_conj_list(source_reset_orientation),
                [-float(offset[i]) for i in range(3)],
            )
            rotated_reset_to_tip = _quat_apply_list(orientation, local_reset_to_tip)
            body_start = [reference[i] - rotated_reset_to_tip[i] for i in range(3)]
        else:
            body_start = [reference[i] + float(offset[i]) for i in range(3)]
        start["axial_distance_m"] = -signed_depth
        start["achieved_axial_distance_m"] = -signed_depth
        start["lateral_distance_m"] = lateral_m
        start["achieved_lateral_distance_m"] = lateral_m
        start["reference_tip_center_position_world"] = reference
        start["reference_reward_body_start_position_world"] = reference
        start["reference_body_position"] = reference
        start["reference_tcp_position"] = reference
        start["body_start_position_world"] = body_start
        start["tcp_start_position_world"] = body_start
        if orientation is not None:
            orientation = [round(float(v), 6) for v in _quat_normalize_list(orientation)]
            start["body_start_orientation_wxyz"] = orientation
            start["reset_body_orientation_wxyz"] = orientation
            start["tcp_start_orientation_world"] = orientation
        if desired_tip_orientation is not None:
            start["reference_reward_body_start_orientation_wxyz"] = [
                round(float(v), 6) for v in _quat_normalize_list(desired_tip_orientation)
            ]
        data["episode_id"] = f"episode_{idx:06d}"
        data["episode_index"] = idx
        (episodes_out / f"episode_{idx:06d}.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
    (out_root / "source.txt").write_text(str(source) + "\n", encoding="utf-8")
    return str(out_root)


prepared_episode_config_dir = _prepare_episode_config_dir()
if prepared_episode_config_dir:
    os.environ["AIC_ISAAC_EPISODE_CONFIG_DIR"] = prepared_episode_config_dir
os.environ["AIC_ISAAC_ENABLE_CONTACT_SENSOR"] = "1"
os.environ["AIC_ISAAC_POLICY_HZ"] = os.environ.get("AIC_ISAAC_POLICY_HZ", "20.0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402
import omni.usd  # noqa: E402

import aic_task.tasks  # noqa: F401,E402
from aic_task.tasks.manager_based.aic_task.mdp.insertion_geometry import compute_insertion_geometry  # noqa: E402


STRICT = {
    "max_depth_error_m": 0.0005,
    "max_lateral_m": 0.0005,
    "max_theta_rad": 0.030,
    "min_module_consistency": 0.80,
}


def _run_git(args: list[str]) -> str:
    cwd = _repo_root()
    try:
        return subprocess.run(
            ["git", *args],
            cwd=None if cwd is None else cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except Exception as exc:
        return f"<git failed: {exc}>"


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tensor_list(tensor: torch.Tensor | None) -> list[float] | None:
    if tensor is None:
        return None
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_rows(tensor: torch.Tensor | None) -> list[list[float]] | None:
    if tensor is None:
        return None
    return [[float(v) for v in row] for row in tensor.detach().cpu().reshape(tensor.shape[0], -1).tolist()]


def _normalize3(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector {vec}")
    return tuple(v / norm for v in vec)


def _normalize4(vec: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1.0e-12:
        raise ValueError(f"cannot normalize near-zero quaternion {vec}")
    return tuple(v / norm for v in vec)


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quat_from_matrix3(matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return (
        (m10 - m01) / scale,
        (m02 + m20) / scale,
        (m12 + m21) / scale,
        0.25 * scale,
    )


def _ros_camera_look_at_quat_wxyz(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    forward = _normalize3((target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]))
    up_hint = (0.0, 0.0, 1.0)
    if abs(sum(forward[i] * up_hint[i] for i in range(3))) > 0.95:
        up_hint = (0.0, 1.0, 0.0)
    left = _normalize3(_cross3(up_hint, forward))
    up = _normalize3(_cross3(forward, left))
    # ROS camera frame: x forward, y left, z up.
    matrix = (
        (forward[0], left[0], up[0]),
        (forward[1], left[1], up[1]),
        (forward[2], left[2], up[2]),
    )
    return _normalize4(_quat_from_matrix3(matrix))


def _opengl_camera_look_at_quat_wxyz(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    forward = _normalize3((target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]))
    up_hint = (0.0, 0.0, 1.0)
    if abs(sum(forward[i] * up_hint[i] for i in range(3))) > 0.95:
        up_hint = (0.0, 1.0, 0.0)
    right = _normalize3(_cross3(forward, up_hint))
    up = _normalize3(_cross3(right, forward))
    # OpenGL camera frame: -z forward, y up, x right.
    matrix = (
        (right[0], up[0], -forward[0]),
        (right[1], up[1], -forward[1]),
        (right[2], up[2], -forward[2]),
    )
    return _normalize4(_quat_from_matrix3(matrix))


def _refresh_cameras(env) -> None:
    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "render"):
        sim.render()
    step_dt = float(getattr(env.unwrapped, "step_dt", 0.0) or 0.0)
    for sensor_name in ("center_camera", "left_camera", "right_camera"):
        sensor = env.unwrapped.scene.sensors.get(sensor_name)
        if sensor is None or not hasattr(sensor, "update"):
            continue
        try:
            sensor.update(step_dt, force_recompute=True)
        except TypeError:
            sensor.update(step_dt)


def _camera_rgb_uint8(env, sensor_name: str) -> torch.Tensor:
    sensor = env.unwrapped.scene.sensors.get(sensor_name)
    if sensor is None:
        raise RuntimeError(f"Camera sensor {sensor_name!r} is not present in the scene")
    output = sensor.data.output
    if "rgb" not in output:
        raise RuntimeError(f"Camera sensor {sensor_name!r} does not expose rgb output keys: {sorted(output)}")
    image = output["rgb"].detach()
    if image.ndim != 4:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.shape[1] in (3, 4):
        image = image[:, :3].permute(0, 2, 3, 1).contiguous()
    else:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        return image.cpu()
    image_f = image.float()
    if image_f.numel() and float(image_f.max().detach().cpu()) <= 2.0:
        image_f = image_f * 255.0
    return image_f.clamp(0.0, 255.0).to(torch.uint8).cpu()


def _save_step_images(env, *, run_dir: Path, step: int, record: dict[str, Any]) -> list[str]:
    from PIL import Image, ImageDraw

    _refresh_cameras(env)
    image_dir = run_dir / "step_images" / f"step_{step:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    geom = record.get("post_step_insertion_geometry") or {}
    module_geom = record.get("post_step_module_geometry") or {}
    saved: list[str] = []
    selected_envs: set[int] | None = None
    if str(args_cli.save_image_env_indices).strip():
        selected_envs = {
            int(item)
            for item in str(args_cli.save_image_env_indices).split(",")
            if item.strip()
        }

    def value(values: list[Any], env_idx: int, default: float = float("nan")) -> float:
        return float(values[env_idx]) if env_idx < len(values) else default

    for camera in ("center_camera", "left_camera", "right_camera"):
        images = _camera_rgb_uint8(env, camera)
        for env_idx in range(images.shape[0]):
            if selected_envs is not None and env_idx not in selected_envs:
                continue
            image = Image.fromarray(images[env_idx].numpy())
            draw = ImageDraw.Draw(image)
            s_vals = geom.get("signed_depth_m_by_env") or []
            r_vals = geom.get("lateral_error_m_by_env") or []
            theta_vals = geom.get("orientation_error_rad_by_env") or []
            strict_vals = geom.get("strict_success_by_env") or []
            cons_vals = geom.get("consistency_gate_by_env") or module_geom.get("consistency_gate_by_env") or []
            label = (
                f"step={step} env={env_idx} {camera} "
                f"s={value(s_vals, env_idx) * 1000.0:+.3f}mm "
                f"r={value(r_vals, env_idx) * 1000.0:.3f}mm "
                f"theta={value(theta_vals, env_idx):.5f} "
                f"cons={value(cons_vals, env_idx):.3f} "
                f"strict={bool(strict_vals[env_idx]) if env_idx < len(strict_vals) else False}"
            )
            draw.rectangle([0, 0, min(image.size[0], 900), 24], fill=(255, 255, 255))
            draw.text((4, 4), label, fill=(0, 0, 0))
            out_path = image_dir / f"env_{env_idx:04d}_{camera}.png"
            image.save(out_path)
            saved.append(str(out_path))
    return saved


def _zero_action(env) -> torch.Tensor:
    action_dim = int(getattr(env.unwrapped.action_manager, "total_action_dim", 0))
    return torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)


def _encode_step_videos(run_dir: Path) -> dict[str, Any]:
    image_root = run_dir / "step_images"
    if not image_root.exists():
        return {"enabled": True, "videos": [], "warnings": ["no step_images directory"]}

    cameras = ("center_camera", "left_camera", "right_camera")
    env_indices: set[str] = set()
    for path in image_root.glob("step_*/env_*_center_camera.png"):
        name = path.name
        if not name.startswith("env_") or not name.endswith("_center_camera.png"):
            continue
        env_indices.add(name[len("env_") : len("env_0000")])

    videos: list[str] = []
    warnings: list[str] = []
    fps = max(int(args_cli.video_fps), 1)
    hold_s = max(float(args_cli.video_final_hold_s), 0.0)
    crf = min(max(int(args_cli.video_crf), 0), 51)
    for env_id in sorted(env_indices):
        for camera in cameras:
            image_files = sorted(image_root.glob(f"step_*/env_{env_id}_{camera}.png"))
            if not image_files:
                continue
            camera_short = camera.removesuffix("_camera")
            out_path = run_dir / f"env{env_id}_{camera_short}_full_episode_{fps}fps_quality448.mp4"
            sequence_pattern = image_root / "step_%06d" / f"env_{env_id}_{camera}.png"
            glob_pattern = image_root / "step_*" / f"env_{env_id}_{camera}.png"
            input_args = ["-i", str(sequence_pattern)]
            if not (image_root / "step_000000" / f"env_{env_id}_{camera}.png").exists():
                input_args = ["-pattern_type", "glob", "-i", str(glob_pattern)]
            vf = f"tpad=stop_mode=clone:stop_duration={hold_s:g}" if hold_s > 0.0 else "null"
            ffmpeg_exe = _ffmpeg_executable()
            cmd = [
                ffmpeg_exe,
                "-y",
                "-framerate",
                str(fps),
                *input_args,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-level",
                "3.1",
                "-crf",
                str(crf),
                "-preset",
                "slow",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            ffmpeg_error = ""
            try:
                result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                ffmpeg_error = result.stdout[-1000:] if result.returncode else ""
            except FileNotFoundError as exc:
                result = None
                ffmpeg_error = str(exc)
            if result is not None and result.returncode == 0:
                videos.append(str(out_path))
                first_frame_path = out_path.with_name(f"{out_path.stem}_first_frame.png")
                first_frame_result = _extract_video_first_frame(out_path, first_frame_path)
                if first_frame_result:
                    warnings.append(first_frame_result)
                continue
            try:
                _encode_step_video_cv2(image_files, out_path, fps=fps, final_hold_s=hold_s)
                videos.append(str(out_path))
                first_frame_path = out_path.with_name(f"{out_path.stem}_first_frame.png")
                first_frame_result = _extract_video_first_frame(out_path, first_frame_path)
                if first_frame_result:
                    warnings.append(first_frame_result)
                warnings.append(f"ffmpeg unavailable/failed for env {env_id} {camera}; encoded with OpenCV fallback.")
            except Exception as exc:
                warnings.append(f"video encode failed for env {env_id} {camera}: ffmpeg={ffmpeg_error}; cv2={type(exc).__name__}: {exc}")
    return {"enabled": True, "videos": videos, "warnings": warnings, "fps": fps, "final_hold_s": hold_s, "crf": crf}


def _ffmpeg_executable() -> str:
    from shutil import which

    ffmpeg = which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return "ffmpeg"


def _extract_video_first_frame(video_path: Path, out_path: Path) -> str:
    ffmpeg_exe = _ffmpeg_executable()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return f"first-frame extraction failed for {video_path.name}: {exc}"
    if result.returncode != 0:
        return f"first-frame extraction failed for {video_path.name}: {result.stdout[-1000:]}"
    return ""


def _encode_step_video_cv2(image_files: list[Path], out_path: Path, *, fps: int, final_hold_s: float) -> None:
    import cv2

    if not image_files:
        raise ValueError("no images to encode")
    first = cv2.imread(str(image_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read {image_files[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {out_path}")
    last = first
    try:
        for image_file in image_files:
            frame = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read {image_file}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            last = frame
        for _ in range(int(round(max(final_hold_s, 0.0) * max(fps, 1)))):
            writer.write(last)
    finally:
        writer.release()


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotvec, dim=1, keepdim=True)
    half_angle = 0.5 * angle
    axis = rotvec / angle.clamp(min=1.0e-9)
    quat = torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=1)
    small_quat = torch.cat([torch.ones_like(angle), 0.5 * rotvec], dim=1)
    quat = torch.where((angle < 1.0e-8).expand_as(quat), small_quat, quat)
    return quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp(min=1.0e-9)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 0:1], -quat[:, 1:4]], dim=1)


def _current_episode_by_env(env) -> dict[int, dict[str, Any]]:
    return dict(getattr(env.unwrapped, "_aic_current_episode_by_env", {}) or {})


def _episode_target_position(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        pos = ((target.get("target_pose_world") or {}).get("position"))
        if pos is None:
            return None
        rows.append(torch.tensor(pos, dtype=origins.dtype, device=origins.device) + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_entrance_position(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        pos = ((target.get("entrance_pose_world") or {}).get("position"))
        if pos is None:
            return None
        rows.append(torch.tensor(pos, dtype=origins.dtype, device=origins.device) + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_axis(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        axis = target.get("insertion_axis_world")
        if axis is None:
            return None
        axis_t = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
        rows.append(axis_t / torch.linalg.norm(axis_t).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _episode_target_body_orientation(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    device = env.unwrapped.scene.env_origins.device
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        episode = episodes.get(env_id) or {}
        scene = episode.get("scene") or {}
        target = scene.get("target") or {}
        start = scene.get("start_near_gate") or episode.get("start_near_gate") or {}
        quat = start.get("reference_reward_body_start_orientation_wxyz") or target.get("body_start_orientation_wxyz")
        if quat is None:
            quat = (target.get("target_pose_world") or {}).get("orientation_wxyz")
        if quat is None:
            return None
        quat_t = torch.tensor(quat, dtype=torch.float32, device=device)
        rows.append(quat_t / torch.linalg.norm(quat_t).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _body_index(env, body_name: str) -> int | None:
    names = list(getattr(env.unwrapped.scene["robot"], "body_names", []))
    return names.index(body_name) if body_name in names else None


def _body_position(env, body_name: str) -> torch.Tensor | None:
    idx = _body_index(env, body_name)
    if idx is None:
        return None
    return env.unwrapped.scene["robot"].data.body_pos_w[:, idx]


def _body_orientation(env, body_name: str) -> torch.Tensor | None:
    idx = _body_index(env, body_name)
    if idx is None:
        return None
    return env.unwrapped.scene["robot"].data.body_quat_w[:, idx]


def _orientation_error(env, body_name: str, axis_w: torch.Tensor) -> torch.Tensor | None:
    quat = _body_orientation(env, body_name)
    if quat is None:
        return None
    if str(args_cli.target_reward_orientation_error_mode).lower() == "quat":
        target_quat = _episode_target_body_orientation(env)
        if target_quat is None:
            return None
        target_quat = target_quat.to(device=quat.device, dtype=quat.dtype)
        return math_utils.quat_error_magnitude(quat, target_quat)
    local_axis = torch.tensor(
        args_cli.target_reward_orientation_axis_local,
        dtype=quat.dtype,
        device=quat.device,
    ).view(1, 3)
    if body_name == "sfp_tip_link":
        # Match the existing strict/evaluation semantics: sfp_tip_link uses a
        # pi body-orientation offset, which flips the local semantic tip axis.
        local_axis = -local_axis
    local_axis = local_axis / torch.linalg.norm(local_axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    body_axis = math_utils.quat_apply(quat, local_axis.expand(quat.shape[0], -1))
    return torch.acos(torch.sum(body_axis * axis_w.to(device=quat.device, dtype=quat.dtype), dim=1).clamp(-1.0, 1.0))


def _semantic_body_axis(env, body_name: str, axis_w: torch.Tensor) -> torch.Tensor | None:
    quat = _body_orientation(env, body_name)
    if quat is None:
        return None
    local_axis = torch.tensor(
        args_cli.target_reward_orientation_axis_local,
        dtype=quat.dtype,
        device=quat.device,
    ).view(1, 3)
    if body_name == "sfp_tip_link":
        local_axis = -local_axis
    local_axis = local_axis / torch.linalg.norm(local_axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    body_axis = math_utils.quat_apply(quat, local_axis.expand(quat.shape[0], -1))
    return body_axis / torch.linalg.norm(body_axis, dim=1, keepdim=True).clamp(min=1.0e-9)


def _geometry(env, body_name: str = "sfp_tip_link") -> dict[str, Any]:
    target = _episode_target_position(env)
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    body = _body_position(env, body_name)
    out: dict[str, Any] = {
        "body_name": body_name,
        "has_target": target is not None,
        "has_entrance": entrance is not None,
        "has_axis": axis is not None,
        "has_body": body is not None,
    }
    if target is None or entrance is None or axis is None or body is None:
        return out
    geom = compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    theta = _orientation_error(env, body_name, geom.axis)
    consistency = _module_consistency(env, geom, entrance, target, axis)
    strict = (
        (geom.axial_depth >= geom.target_depth - STRICT["max_depth_error_m"])
        & (geom.lateral_error <= STRICT["max_lateral_m"])
        & (torch.zeros_like(geom.axial_depth, dtype=torch.bool) if theta is None else theta <= STRICT["max_theta_rad"])
        & (
            torch.zeros_like(geom.axial_depth, dtype=torch.bool)
            if consistency is None
            else consistency >= STRICT["min_module_consistency"]
        )
    )
    out.update(
        {
            "signed_depth_m_by_env": _tensor_list(geom.axial_depth),
            "lateral_error_m_by_env": _tensor_list(geom.lateral_error),
            "target_depth_m_by_env": _tensor_list(geom.target_depth),
            "depth_fraction_by_env": _tensor_list(geom.depth_fraction),
            "orientation_error_rad_by_env": _tensor_list(theta),
            "consistency_gate_by_env": _tensor_list(consistency),
            "strict_success_by_env": [bool(x) for x in strict.detach().cpu().tolist()],
            "axis_world_env0": _tensor_list(axis[0]),
            "entrance_world_env0": _tensor_list(entrance[0]),
            "body_world_env0": _tensor_list(body[0]),
        }
    )
    return out


def _module_consistency(
    env,
    tip_geom,
    entrance: torch.Tensor,
    target: torch.Tensor,
    axis: torch.Tensor,
) -> torch.Tensor | None:
    module = _body_position(env, str(args_cli.target_reward_consistency_body))
    if module is None:
        return None
    module_geom = compute_insertion_geometry(
        body_pos_w=module,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=max(float(args_cli.target_reward_consistency_lateral_sigma), 1.0e-9),
    )
    gap = tip_geom.axial_depth - module_geom.axial_depth
    attr = "_aic_wrist_contact_reference_gap"
    reference = getattr(env.unwrapped, attr, None)
    if reference is None or reference.shape != gap.shape:
        reference = gap.detach().clone()
        setattr(env.unwrapped, attr, reference)
    expected_module_depth = tip_geom.target_depth - reference.to(gap.device)
    axial_gate = torch.exp(
        -torch.square(
            (module_geom.axial_depth - expected_module_depth)
            / max(float(args_cli.target_reward_consistency_axial_std), 1.0e-9)
        )
    )
    return axial_gate * module_geom.lateral_gate


def _contact_summary(env) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sensor = getattr(env.unwrapped.scene, "sensors", {}).get("contact_forces")
    if sensor is None:
        return {"available": False}
    net = getattr(sensor.data, "net_forces_w", None)
    names = list(getattr(sensor, "body_names", []) or [])
    out["available"] = net is not None
    out["body_names"] = names
    if net is not None and net.numel() > 0:
        norms = torch.linalg.norm(net[:, :, :3], dim=-1)
        out["force_norm_mean_by_env"] = _tensor_list(norms.mean(dim=1))
        out["force_norm_max_by_env"] = _tensor_list(norms.max(dim=1).values)
        if names:
            out["env0_by_body"] = {name: float(norms[0, idx].detach().cpu()) for idx, name in enumerate(names)}
    return out


def _root_action_from_world_delta(env, world_delta: torch.Tensor, rotvec_world: torch.Tensor | None = None) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    root_quat = robot.data.root_quat_w.to(device=world_delta.device, dtype=world_delta.dtype)
    if bool(args_cli.absolute_ik_target_pose):
        wrist_pos = _body_position(env, "wrist_3_link")
        wrist_quat = _body_orientation(env, "wrist_3_link")
        if wrist_pos is None or wrist_quat is None:
            return _zero_action(env)
        root_pos = robot.data.root_pos_w.to(device=world_delta.device, dtype=world_delta.dtype)
        target_pos_w = wrist_pos + world_delta
        if rotvec_world is None:
            pinned_quat = getattr(env.unwrapped, "_aic_absolute_ik_pinned_wrist_quat_w", None)
            target_quat_w = (
                pinned_quat.to(device=wrist_quat.device, dtype=wrist_quat.dtype)
                if bool(args_cli.absolute_ik_pin_reset_orientation) and pinned_quat is not None
                else wrist_quat
            )
        else:
            target_quat_w = math_utils.quat_mul(_quat_from_rotvec(rotvec_world), wrist_quat)
        target_pos_root = math_utils.quat_apply_inverse(root_quat, target_pos_w - root_pos)
        target_quat_root = math_utils.quat_mul(math_utils.quat_inv(root_quat), target_quat_w)
        return torch.cat([target_pos_root, target_quat_root], dim=1)
    root_delta = math_utils.quat_apply_inverse(root_quat, world_delta)
    if bool(args_cli.fix_isaac_ik_xy_sign):
        root_delta = root_delta.clone()
        root_delta[:, 0:2] = -root_delta[:, 0:2]
    if bool(args_cli.fix_isaac_ik_z_sign):
        root_delta = root_delta.clone()
        root_delta[:, 2] = -root_delta[:, 2]
    if rotvec_world is None:
        root_rot = torch.zeros_like(root_delta)
    else:
        root_rot = math_utils.quat_apply_inverse(root_quat, rotvec_world.to(device=world_delta.device, dtype=world_delta.dtype))
        if bool(args_cli.fix_isaac_ik_rot_xy_sign):
            root_rot = root_rot.clone()
            root_rot[:, 0:2] = -root_rot[:, 0:2]
        if bool(args_cli.fix_isaac_ik_rot_z_sign):
            root_rot = root_rot.clone()
            root_rot[:, 2] = -root_rot[:, 2]
    return torch.cat([root_delta, root_rot], dim=1)


def _zero_action(env) -> torch.Tensor:
    if bool(args_cli.absolute_ik_target_pose):
        robot = env.unwrapped.scene["robot"]
        wrist_pos = _body_position(env, "wrist_3_link")
        wrist_quat = _body_orientation(env, "wrist_3_link")
        if wrist_pos is not None and wrist_quat is not None:
            root_pos = robot.data.root_pos_w.to(device=wrist_pos.device, dtype=wrist_pos.dtype)
            root_quat = robot.data.root_quat_w.to(device=wrist_pos.device, dtype=wrist_pos.dtype)
            wrist_pos_root = math_utils.quat_apply_inverse(root_quat, wrist_pos - root_pos)
            wrist_quat_root = math_utils.quat_mul(math_utils.quat_inv(root_quat), wrist_quat)
            return torch.cat([wrist_pos_root, wrist_quat_root], dim=1)
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)


def _approach_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    if tip is None or entrance is None or axis is None:
        return _zero_action(env), {"reason": "missing_geometry"}
    rel = tip - entrance
    depth = torch.sum(rel * axis, dim=1, keepdim=True)
    centerline = entrance + depth * axis
    lateral_vec = centerline - tip
    lateral_error = torch.linalg.norm(lateral_vec, dim=1, keepdim=True)
    lateral_step = torch.minimum(
        lateral_error,
        torch.full_like(lateral_error, max(float(args_cli.approach_lateral_step_m), 0.0)),
    )
    lateral_dir = lateral_vec / lateral_error.clamp(min=1.0e-9)
    lateral_delta = lateral_dir * lateral_step * (1.0 if float(args_cli.approach_lateral_sign) >= 0.0 else -1.0)
    can_advance = lateral_error <= max(float(args_cli.approach_lateral_gate_m), 0.0)
    below_shallow = depth < float(args_cli.shallow_depth_m)
    axial_step = torch.full_like(depth, max(float(args_cli.approach_axial_step_m), 0.0))
    axial_delta = torch.where(
        can_advance & below_shallow,
        axial_step * axis * (1.0 if float(args_cli.approach_axial_sign) >= 0.0 else -1.0),
        torch.zeros_like(lateral_delta),
    )
    world_delta = lateral_delta + axial_delta
    return _root_action_from_world_delta(env, world_delta), {
        "depth_m_by_env": _tensor_list(depth),
        "lateral_error_m_by_env": _tensor_list(lateral_error),
        "advanced_fraction": float((can_advance & below_shallow).float().mean().detach().cpu()),
    }


def _probe_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    axis = _episode_axis(env)
    if axis is None:
        return _zero_action(env), {"reason": "missing_axis"}
    if args_cli.probe == "axis_backout":
        world_delta = -axis * float(args_cli.probe_translation_step_m)
        return _root_action_from_world_delta(env, world_delta), {"probe_label": "axis_backout"}
    if args_cli.probe == "zero_action":
        return _zero_action(env), {"probe_label": "zero_action"}
    if args_cli.probe == "axis_forward":
        world_delta = axis * float(args_cli.probe_translation_step_m)
        return _root_action_from_world_delta(env, world_delta), {"probe_label": "axis_forward"}
    if args_cli.probe == "rotation_axis":
        axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
        rot = torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.unwrapped.device)
        sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
        rot[:, axis_idx] = sign * float(args_cli.probe_rotation_step_rad)
        world_delta, correction_info = _probe_retention_delta(env)
        return _root_action_from_world_delta(env, world_delta, rotvec_world=rot), {
            "probe_label": f"{'+' if sign >= 0.0 else '-'}r{args_cli.probe_rotation_axis}",
            **correction_info,
        }
    if args_cli.probe == "orientation_servo_best":
        return _orientation_servo_best_action(env)
    if args_cli.probe == "pose_hold":
        return _pose_hold_action(env)
    if args_cli.probe == "pose_hold_orientation_servo_best":
        return _pose_hold_orientation_servo_best_action(env, probe_step)
    if args_cli.probe == "pose_hold_rotation_axis":
        return _pose_hold_rotation_axis_action(env, probe_step)
    if args_cli.probe == "pose_hold_constrained_rotation_axis":
        return _pose_hold_constrained_rotation_axis_action(env, probe_step)
    if args_cli.probe == "target_tip_stabilize":
        return _target_tip_stabilize_action(env)
    if args_cli.probe == "target_module_stabilize":
        return _target_module_stabilize_action(env, probe_step)
    if args_cli.probe == "target_tip_then_module_stabilize":
        return _target_tip_then_module_stabilize_action(env, probe_step)
    if args_cli.probe == "target_tip_fixed_rotation_axis":
        return _target_tip_fixed_rotation_axis_action(env)
    if args_cli.probe == "pinned_wrist_axis_descent":
        return _pinned_wrist_axis_descent_action(env, probe_step)
    labels = ("+rx", "+ry", "+rz", "-rx", "-ry", "-rz")
    idx = probe_step % len(labels)
    sign = 1.0 if idx < 3 else -1.0
    axis_idx = idx % 3
    rot = torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.unwrapped.device)
    rot[:, axis_idx] = sign * float(args_cli.probe_rotation_step_rad)
    world_delta = torch.zeros_like(rot)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rot), {"probe_label": labels[idx]}


def _clip_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(vector)
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norm), torch.full_like(norm, max_norm) / norm.clamp(min=1.0e-9))
    return vector * scale


def _clip_scalar(value: torch.Tensor, max_abs: float) -> torch.Tensor:
    if max_abs <= 0.0:
        return torch.zeros_like(value)
    return value.clamp(min=-max_abs, max=max_abs)


def _target_tip_stabilize_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    stabilized = _target_tip_stabilize_delta(env)
    if stabilized is None:
        return _zero_action(env), {"probe_label": "target_tip_stabilize", "reason": "missing_target_tip_geometry"}
    world_delta, geom, info = stabilized

    rotvec, orient_info = _best_semantic_orientation_rotvec(env, world_delta)
    if rotvec is None:
        rotvec = torch.zeros_like(world_delta)
        orientation_active = torch.zeros((world_delta.shape[0], 1), dtype=torch.bool, device=world_delta.device)
    else:
        theta = _orientation_error(env, "sfp_tip_link", geom.axis)
        if theta is None:
            theta = torch.full(
                (world_delta.shape[0],),
                float("inf"),
                dtype=world_delta.dtype,
                device=world_delta.device,
            )
        orientation_active = (
            (geom.lateral_error.view(-1, 1) <= float(args_cli.target_tip_stabilize_orientation_gate_lateral_m))
            & (geom.axial_depth.view(-1, 1) >= float(args_cli.target_tip_stabilize_orientation_gate_depth_m))
            & (theta.view(-1, 1) > float(args_cli.target_tip_stabilize_orientation_error_threshold_rad))
        )
        rotvec = torch.where(orientation_active.expand_as(rotvec), rotvec, torch.zeros_like(rotvec))

    compensation = _rotation_tip_sweep_compensation(env, rotvec)
    compensation = _clip_vector_norm(compensation, max(float(args_cli.target_tip_stabilize_rotation_compensation_clip_m), 0.0))
    world_delta = world_delta + compensation
    orientation_is_active = bool(torch.any(orientation_active).item())
    rotvec_action = (
        rotvec
        if orientation_is_active or bool(args_cli.target_module_stabilize_use_current_orientation_when_no_trim)
        else None
    )
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec_action), {
        "probe_label": "target_tip_stabilize",
        "target_tip_rotation_compensation_m_by_env": _tensor_rows(compensation),
        "target_tip_orientation_active_by_env": [bool(x) for x in orientation_active.detach().cpu().reshape(-1).tolist()],
        **info,
        **orient_info,
    }


def _target_tip_fixed_rotation_axis_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    stabilized = _target_tip_stabilize_delta(env)
    if stabilized is None:
        return _zero_action(env), {
            "probe_label": "target_tip_fixed_rotation_axis",
            "reason": "missing_target_tip_geometry",
        }
    world_delta, geom, info = stabilized
    axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
    sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
    rotvec = torch.zeros_like(world_delta)
    rotvec[:, axis_idx] = sign * max(float(args_cli.probe_rotation_step_rad), 0.0)
    theta = _orientation_error(env, "sfp_tip_link", geom.axis)
    if theta is None:
        theta = torch.full(
            (world_delta.shape[0],),
            float("inf"),
            dtype=world_delta.dtype,
            device=world_delta.device,
        )
    orientation_active = (
        (geom.lateral_error.view(-1, 1) <= float(args_cli.target_tip_stabilize_orientation_gate_lateral_m))
        & (geom.axial_depth.view(-1, 1) >= float(args_cli.target_tip_stabilize_orientation_gate_depth_m))
        & (theta.view(-1, 1) > float(args_cli.target_tip_stabilize_orientation_error_threshold_rad))
    )
    rotvec = torch.where(orientation_active.expand_as(rotvec), rotvec, torch.zeros_like(rotvec))
    compensation = _rotation_tip_sweep_compensation(env, rotvec)
    compensation = _clip_vector_norm(compensation, max(float(args_cli.target_tip_stabilize_rotation_compensation_clip_m), 0.0))
    world_delta = world_delta + compensation
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "target_tip_fixed_rotation_axis",
        "target_tip_fixed_rotation_axis": str(args_cli.probe_rotation_axis),
        "target_tip_fixed_rotation_sign": sign,
        "target_tip_fixed_rotation_active_by_env": [
            bool(x) for x in orientation_active.detach().cpu().reshape(-1).tolist()
        ],
        "target_tip_rotation_compensation_m_by_env": _tensor_rows(compensation),
        **info,
    }


def _target_tip_stabilize_delta(env) -> tuple[torch.Tensor, Any, dict[str, Any]] | None:
    target_tip_pos = getattr(env.unwrapped, "_aic_target_tip_stabilize_pos_w", None)
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    target = _episode_target_position(env)
    axis = _episode_axis(env)
    if target_tip_pos is None or tip is None or entrance is None or target is None or axis is None:
        return None
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    if math.isfinite(float(args_cli.target_tip_stabilize_goal_depth_m)):
        target_tip_pos = entrance + float(args_cli.target_tip_stabilize_goal_depth_m) * axis
    else:
        target_tip_pos = target_tip_pos.to(device=tip.device, dtype=tip.dtype)

    error = target_tip_pos - tip
    axial_error = torch.sum(error * axis, dim=1, keepdim=True)
    lateral_error_vec = error - axial_error * axis
    axial_clip = (
        float(args_cli.target_tip_stabilize_axial_step_m)
        if math.isfinite(float(args_cli.target_tip_stabilize_axial_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    lateral_clip = (
        float(args_cli.target_tip_stabilize_lateral_step_m)
        if math.isfinite(float(args_cli.target_tip_stabilize_lateral_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    axial_delta = _clip_scalar(axial_error, max(axial_clip, 0.0)) * axis
    lateral_delta = _clip_vector_norm(lateral_error_vec, max(lateral_clip, 0.0))

    geom = compute_insertion_geometry(
        body_pos_w=tip,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    near_target = (
        (torch.linalg.norm(lateral_error_vec, dim=1, keepdim=True) <= max(lateral_clip, 1.0e-9))
        & (torch.abs(axial_error) <= max(axial_clip, 1.0e-9))
    )
    inward_bias = torch.where(
        near_target,
        torch.full_like(axial_error, max(float(args_cli.target_tip_stabilize_inward_bias_m), 0.0)),
        torch.zeros_like(axial_error),
    ) * axis
    world_delta = lateral_delta + axial_delta + inward_bias

    return world_delta, geom, {
        "target_tip_position_error_m_by_env": _tensor_list(torch.linalg.norm(error, dim=1, keepdim=True)),
        "target_tip_axial_error_m_by_env": _tensor_list(axial_error),
        "target_tip_lateral_error_m_by_env": _tensor_list(torch.linalg.norm(lateral_error_vec, dim=1, keepdim=True)),
        "target_tip_world_delta_m_by_env": _tensor_rows(world_delta),
        "target_tip_pre_step_s_m_by_env": _tensor_list(geom.axial_depth),
        "target_tip_pre_step_r_m_by_env": _tensor_list(geom.lateral_error),
    }


def _target_module_stabilize_action(env, probe_step: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
    stabilized = _target_module_stabilize_delta(env)
    if stabilized is None:
        return _zero_action(env), {"probe_label": "target_module_stabilize", "reason": "missing_target_body_geometry"}
    world_delta, _, info = stabilized
    orientation_step = max(float(args_cli.target_module_stabilize_orientation_step_rad), 0.0)
    rotvec = torch.zeros_like(world_delta)
    orientation_active = torch.zeros((world_delta.shape[0], 1), dtype=torch.bool, device=world_delta.device)
    orient_info: dict[str, Any] = {}
    compensation = torch.zeros_like(world_delta)
    module_consistency_for_orientation = None
    if orientation_step > 0.0 and int(probe_step) >= int(args_cli.target_module_stabilize_orientation_start_probe_step):
        tip = _body_position(env, "sfp_tip_link")
        entrance = _episode_entrance_position(env)
        target = _episode_target_position(env)
        axis = _episode_axis(env)
        theta = None if axis is None else _orientation_error(env, "sfp_tip_link", axis)
        if tip is not None and entrance is not None and target is not None and axis is not None and theta is not None:
            axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
            tip_geom = compute_insertion_geometry(
                body_pos_w=tip,
                entrance_pos_w=entrance,
                target_pos_w=target,
                axis_w=axis,
                lateral_gate_sigma=0.0025,
            )
            orientation_active = (
                (tip_geom.lateral_error.view(-1, 1) <= float(args_cli.target_module_stabilize_orientation_lateral_gate_m))
                & (tip_geom.axial_depth.view(-1, 1) >= float(args_cli.target_module_stabilize_orientation_activation_depth_m))
                & (theta.view(-1, 1) > float(args_cli.target_module_stabilize_orientation_error_threshold_rad))
            )
            if torch.any(orientation_active).item():
                candidate_rotvec, orient_info = _best_semantic_orientation_rotvec(
                    env,
                    world_delta,
                    rotation_step_rad=orientation_step,
                    module_body_name=str(args_cli.target_module_stabilize_body),
                    module_lateral_penalty=float(args_cli.target_module_stabilize_orientation_module_lateral_penalty),
                    module_lateral_margin_m=float(args_cli.target_module_stabilize_orientation_module_lateral_margin_m),
                )
                if candidate_rotvec is not None:
                    if math.isfinite(float(args_cli.target_module_stabilize_orientation_min_module_consistency)):
                        module_consistency_for_orientation = _module_consistency(env, tip_geom, entrance, target, axis)
                        if module_consistency_for_orientation is None:
                            orientation_active = torch.zeros_like(orientation_active)
                        else:
                            orientation_active = orientation_active & (
                                module_consistency_for_orientation.view(-1, 1)
                                >= float(args_cli.target_module_stabilize_orientation_min_module_consistency)
                            )
                    rotvec = torch.where(orientation_active.expand_as(candidate_rotvec), candidate_rotvec, rotvec)
                    compensation = _rotation_tip_sweep_compensation(env, rotvec)
                    compensation = _clip_vector_norm(
                        compensation,
                        max(float(args_cli.target_module_stabilize_rotation_compensation_clip_m), 0.0),
                    )
                    world_delta = world_delta + compensation
    rotvec_action = rotvec if bool(torch.any(orientation_active).item()) else None
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec_action), {
        "probe_label": "target_module_stabilize",
        "target_module_orientation_active_by_env": [
            bool(x) for x in orientation_active.detach().cpu().reshape(-1).tolist()
        ],
        "target_module_rotation_compensation_m_by_env": _tensor_rows(compensation),
        "target_module_orientation_min_module_consistency": (
            None
            if not math.isfinite(float(args_cli.target_module_stabilize_orientation_min_module_consistency))
            else float(args_cli.target_module_stabilize_orientation_min_module_consistency)
        ),
        "target_module_orientation_module_consistency_by_env": (
            None if module_consistency_for_orientation is None else _tensor_list(module_consistency_for_orientation)
        ),
        **info,
        **orient_info,
    }


def _target_tip_then_module_stabilize_action(env, probe_step: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    target = _episode_target_position(env)
    axis = _episode_axis(env)
    if tip is None or entrance is None or target is None or axis is None:
        return _target_tip_stabilize_action(env)
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    tip_geom = compute_insertion_geometry(
        body_pos_w=tip,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    theta = _orientation_error(env, "sfp_tip_link", axis)
    theta_ok = torch.ones_like(tip_geom.axial_depth.view(-1, 1), dtype=torch.bool)
    theta_gate = float(args_cli.target_tip_then_module_switch_theta_rad)
    if theta is not None and math.isfinite(theta_gate):
        theta_ok = theta.view(-1, 1) <= max(theta_gate, 0.0)
    switch_now = (
        (tip_geom.axial_depth.view(-1, 1) >= float(args_cli.target_tip_then_module_switch_depth_m))
        & (tip_geom.lateral_error.view(-1, 1) <= max(float(args_cli.target_tip_then_module_switch_lateral_m), 0.0))
        & theta_ok
    )
    latch = getattr(env.unwrapped, "_aic_target_tip_then_module_latch", None)
    if latch is None or latch.shape != switch_now.shape:
        latch = torch.zeros_like(switch_now, dtype=torch.bool)
    else:
        latch = latch.to(device=switch_now.device, dtype=torch.bool)
    latch = latch | switch_now
    setattr(env.unwrapped, "_aic_target_tip_then_module_latch", latch.detach().clone())
    if bool(latch.all()):
        action, info = _target_module_stabilize_action(env, probe_step)
    else:
        action, info = _target_tip_stabilize_action(env)
    info = dict(info)
    info.update(
        {
            "probe_label": "target_tip_then_module_stabilize",
            "target_tip_then_module_selected_mode_by_env": [
                "module" if bool(x) else "tip" for x in latch.detach().cpu().reshape(-1).tolist()
            ],
            "target_tip_then_module_switch_now_by_env": [
                bool(x) for x in switch_now.detach().cpu().reshape(-1).tolist()
            ],
            "target_tip_then_module_tip_s_m_by_env": _tensor_list(tip_geom.axial_depth),
            "target_tip_then_module_tip_r_m_by_env": _tensor_list(tip_geom.lateral_error),
            "target_tip_then_module_tip_theta_rad_by_env": None if theta is None else _tensor_list(theta),
            "target_tip_then_module_switch_depth_m": float(args_cli.target_tip_then_module_switch_depth_m),
            "target_tip_then_module_switch_lateral_m": float(args_cli.target_tip_then_module_switch_lateral_m),
            "target_tip_then_module_switch_theta_rad": float(args_cli.target_tip_then_module_switch_theta_rad),
        }
    )
    return action, info


def _target_module_stabilize_delta(env) -> tuple[torch.Tensor, Any, dict[str, Any]] | None:
    body_name = str(args_cli.target_module_stabilize_body)
    target_body_pos = getattr(env.unwrapped, "_aic_target_module_stabilize_pos_w", None)
    body = _body_position(env, body_name)
    entrance = _episode_entrance_position(env)
    target = _episode_target_position(env)
    axis = _episode_axis(env)
    if target_body_pos is None or body is None or entrance is None or target is None or axis is None:
        return None
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    if math.isfinite(float(args_cli.target_module_stabilize_goal_depth_m)):
        target_body_pos = entrance + float(args_cli.target_module_stabilize_goal_depth_m) * axis
    else:
        target_body_pos = target_body_pos.to(device=body.device, dtype=body.dtype)

    error = target_body_pos - body
    axial_error = torch.sum(error * axis, dim=1, keepdim=True)
    lateral_error_vec = error - axial_error * axis
    tip = _body_position(env, "sfp_tip_link")
    tip_geom = None
    tip_lateral_delta = None
    tip_theta = None
    if tip is not None:
        tip_geom = compute_insertion_geometry(
            body_pos_w=tip,
            entrance_pos_w=entrance,
            target_pos_w=target,
            axis_w=axis,
            lateral_gate_sigma=0.0025,
        )
        tip_centerline = entrance + tip_geom.axial_depth.view(-1, 1) * axis
        tip_lateral_error_vec = tip_centerline - tip
        tip_lateral_delta = _clip_vector_norm(
            tip_lateral_error_vec,
            max(float(args_cli.target_module_stabilize_tip_lateral_step_m), 0.0)
            if math.isfinite(float(args_cli.target_module_stabilize_tip_lateral_step_m))
            else 0.0,
        )
        tip_theta = _orientation_error(env, "sfp_tip_link", axis)
    tip_axial_default = (
        float(args_cli.target_tip_stabilize_axial_step_m)
        if math.isfinite(float(args_cli.target_tip_stabilize_axial_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    tip_lateral_default = (
        float(args_cli.target_tip_stabilize_lateral_step_m)
        if math.isfinite(float(args_cli.target_tip_stabilize_lateral_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    axial_clip = (
        float(args_cli.target_module_stabilize_axial_step_m)
        if math.isfinite(float(args_cli.target_module_stabilize_axial_step_m))
        else tip_axial_default
    )
    lateral_clip = (
        float(args_cli.target_module_stabilize_lateral_step_m)
        if math.isfinite(float(args_cli.target_module_stabilize_lateral_step_m))
        else tip_lateral_default
    )
    axial_scalar = _clip_scalar(axial_error, max(axial_clip, 0.0))
    axial_gate_active = torch.ones_like(axial_scalar, dtype=torch.bool)
    if tip_geom is not None and math.isfinite(float(args_cli.target_module_stabilize_tip_lateral_gate_m)):
        axial_gate_active = axial_gate_active & (
            tip_geom.lateral_error.view(-1, 1) <= float(args_cli.target_module_stabilize_tip_lateral_gate_m)
        )
    if tip_theta is not None and math.isfinite(float(args_cli.target_module_stabilize_tip_theta_gate_rad)):
        axial_gate_active = axial_gate_active & (
            tip_theta.view(-1, 1) <= float(args_cli.target_module_stabilize_tip_theta_gate_rad)
        )
    polish_active = torch.zeros_like(axial_gate_active)
    polish_consistency = None
    if bool(args_cli.target_module_stabilize_polish_after_near_success) and tip_geom is not None:
        polish_consistency = _module_consistency(env, tip_geom, entrance, target, axis)
        if polish_consistency is not None:
            polish_theta_ok = torch.ones_like(polish_consistency.view(-1, 1), dtype=torch.bool)
            if math.isfinite(float(args_cli.target_module_stabilize_polish_max_tip_theta_rad)):
                if tip_theta is None:
                    polish_theta_ok = torch.zeros_like(polish_theta_ok)
                else:
                    polish_theta_ok = (
                        tip_theta.view(-1, 1) <= float(args_cli.target_module_stabilize_polish_max_tip_theta_rad)
                    )
            polish_active = (
                (tip_geom.axial_depth.view(-1, 1) >= float(args_cli.target_module_stabilize_polish_min_tip_depth_m))
                & (tip_geom.lateral_error.view(-1, 1) <= float(args_cli.target_module_stabilize_polish_max_tip_lateral_m))
                & polish_theta_ok
                & (
                    polish_consistency.view(-1, 1)
                    >= float(args_cli.target_module_stabilize_polish_min_module_consistency)
                )
            )
            if bool(args_cli.target_module_stabilize_polish_latch):
                latch = getattr(env.unwrapped, "_aic_target_module_polish_latch", None)
                if latch is None or latch.shape != polish_active.shape:
                    latch = torch.zeros_like(polish_active, dtype=torch.bool)
                else:
                    latch = latch.to(device=polish_active.device, dtype=torch.bool)
                polish_active = polish_active | latch
                setattr(env.unwrapped, "_aic_target_module_polish_latch", polish_active.detach().clone())
            polish_axial = torch.full_like(axial_scalar, float(args_cli.target_module_stabilize_polish_axial_step_m))
            axial_scalar = torch.where((axial_scalar > polish_axial) & polish_active, polish_axial, axial_scalar)
    # Only gate inward module progress; allow negative/backoff correction to reduce contact.
    axial_scalar = torch.where((axial_scalar > 0.0) & (~axial_gate_active), torch.zeros_like(axial_scalar), axial_scalar)
    axial_delta = axial_scalar * axis
    module_lateral_delta = _clip_vector_norm(lateral_error_vec, max(lateral_clip, 0.0))
    lateral_delta = tip_lateral_delta if tip_lateral_delta is not None else module_lateral_delta
    secondary_module_lateral_delta = torch.zeros_like(module_lateral_delta)
    secondary_module_lateral_active = torch.zeros_like(axial_gate_active)
    secondary_step = max(float(args_cli.target_module_stabilize_secondary_module_lateral_step_m), 0.0)
    if secondary_step > 0.0 and tip_geom is not None:
        module_lateral_norm = torch.linalg.norm(lateral_error_vec, dim=1, keepdim=True)
        secondary_module_lateral_active = (
            (tip_geom.axial_depth.view(-1, 1) >= float(args_cli.target_module_stabilize_secondary_module_lateral_activation_depth_m))
            & (module_lateral_norm >= float(args_cli.target_module_stabilize_secondary_module_lateral_threshold_m))
        )
        secondary_module_lateral_delta = _clip_vector_norm(lateral_error_vec, secondary_step)
        secondary_module_lateral_delta = torch.where(
            secondary_module_lateral_active.expand_as(secondary_module_lateral_delta),
            secondary_module_lateral_delta,
            torch.zeros_like(secondary_module_lateral_delta),
        )
        lateral_delta = lateral_delta + secondary_module_lateral_delta
    world_delta = lateral_delta + axial_delta

    geom = compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    return world_delta, geom, {
        "target_module_body_name": body_name,
        "target_module_position_error_m_by_env": _tensor_list(torch.linalg.norm(error, dim=1, keepdim=True)),
        "target_module_axial_error_m_by_env": _tensor_list(axial_error),
        "target_module_lateral_error_m_by_env": _tensor_list(torch.linalg.norm(lateral_error_vec, dim=1, keepdim=True)),
        "target_module_axial_gate_active_by_env": [
            bool(x) for x in axial_gate_active.detach().cpu().reshape(-1).tolist()
        ],
        "target_module_polish_active_by_env": [
            bool(x) for x in polish_active.detach().cpu().reshape(-1).tolist()
        ],
        "target_module_polish_consistency_by_env": (
            None if polish_consistency is None else _tensor_list(polish_consistency)
        ),
        "target_module_module_lateral_delta_m_by_env": _tensor_rows(module_lateral_delta),
        "target_module_secondary_module_lateral_active_by_env": [
            bool(x) for x in secondary_module_lateral_active.detach().cpu().reshape(-1).tolist()
        ],
        "target_module_secondary_module_lateral_delta_m_by_env": _tensor_rows(secondary_module_lateral_delta),
        "target_module_tip_lateral_delta_m_by_env": None
        if tip_lateral_delta is None
        else _tensor_rows(tip_lateral_delta),
        "target_module_tip_pre_step_s_m_by_env": None
        if tip_geom is None
        else _tensor_list(tip_geom.axial_depth),
        "target_module_tip_pre_step_r_m_by_env": None
        if tip_geom is None
        else _tensor_list(tip_geom.lateral_error),
        "target_module_tip_pre_step_theta_rad_by_env": None
        if tip_theta is None
        else _tensor_list(tip_theta),
        "target_module_world_delta_m_by_env": _tensor_rows(world_delta),
        "target_module_pre_step_s_m_by_env": _tensor_list(geom.axial_depth),
        "target_module_pre_step_r_m_by_env": _tensor_list(geom.lateral_error),
    }


def _pinned_wrist_axis_descent_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    pinned_wrist_pos = getattr(env.unwrapped, "_aic_pinned_wrist_descent_pos_w", None)
    wrist = _body_position(env, "wrist_3_link")
    axis = _episode_axis(env)
    if pinned_wrist_pos is None or wrist is None or axis is None:
        return _zero_action(env), {"probe_label": "pinned_wrist_axis_descent", "reason": "missing_wrist_or_axis"}
    pinned_wrist_pos = pinned_wrist_pos.to(device=wrist.device, dtype=wrist.dtype)
    axis = axis.to(device=wrist.device, dtype=wrist.dtype)
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    step_m = (
        float(args_cli.pinned_wrist_axis_descent_step_m)
        if math.isfinite(float(args_cli.pinned_wrist_axis_descent_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    progress = min(
        max(float(args_cli.pinned_wrist_axis_descent_distance_m), 0.0),
        max(probe_step + 1, 0) * max(step_m, 0.0),
    )
    desired_wrist = pinned_wrist_pos + progress * axis
    world_delta = desired_wrist - wrist
    return _root_action_from_world_delta(env, world_delta), {
        "probe_label": "pinned_wrist_axis_descent",
        "pinned_wrist_axis_descent_progress_m": progress,
        "pinned_wrist_axis_descent_distance_m": float(args_cli.pinned_wrist_axis_descent_distance_m),
        "pinned_wrist_axis_descent_step_m": step_m,
        "pinned_wrist_axis_descent_world_delta_m_by_env": _tensor_rows(world_delta),
        "pinned_wrist_axis_descent_world_delta_norm_m_by_env": _tensor_list(
            torch.linalg.norm(world_delta, dim=1, keepdim=True)
        ),
    }


def _rotation_tip_sweep_compensation(env, rotvec_world: torch.Tensor) -> torch.Tensor:
    tip = _body_position(env, "sfp_tip_link")
    wrist = _body_position(env, "wrist_3_link")
    if tip is None or wrist is None:
        return torch.zeros_like(rotvec_world)
    q_step = _quat_from_rotvec(rotvec_world)
    lever = tip - wrist
    predicted_tip = wrist + math_utils.quat_apply(q_step, lever)
    return tip - predicted_tip


def _rotation_body_sweep_compensation(env, rotvec_world: torch.Tensor, body_name: str) -> torch.Tensor | None:
    body = _body_position(env, body_name)
    wrist = _body_position(env, "wrist_3_link")
    if body is None or wrist is None:
        return None
    q_step = _quat_from_rotvec(rotvec_world)
    lever = body - wrist
    predicted_body = wrist + math_utils.quat_apply(q_step, lever)
    return body - predicted_body


def _rotation_tip_module_sweep_compensation(
    env,
    rotvec_world: torch.Tensor,
    *,
    tip_weight: float,
    module_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tip_comp = _rotation_body_sweep_compensation(env, rotvec_world, "sfp_tip_link")
    module_comp = _rotation_body_sweep_compensation(env, rotvec_world, "sfp_module_link")
    total = torch.zeros_like(rotvec_world)
    total_weight = torch.zeros((rotvec_world.shape[0], 1), dtype=rotvec_world.dtype, device=rotvec_world.device)
    if tip_comp is not None and tip_weight > 0.0:
        total = total + float(tip_weight) * tip_comp
        total_weight = total_weight + float(tip_weight)
    if module_comp is not None and module_weight > 0.0:
        total = total + float(module_weight) * module_comp
        total_weight = total_weight + float(module_weight)
    compensation = torch.where(total_weight > 0.0, total / total_weight.clamp(min=1.0e-9), torch.zeros_like(total))
    return compensation, {
        "constrained_tip_sweep_compensation_m_by_env": None if tip_comp is None else _tensor_rows(tip_comp),
        "constrained_module_sweep_compensation_m_by_env": None if module_comp is None else _tensor_rows(module_comp),
        "constrained_tip_weight": float(tip_weight),
        "constrained_module_weight": float(module_weight),
    }


def _pose_hold_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold", "reason": "missing_pose_hold_target"}
    world_delta, rotvec_world, info = hold
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec_world), {
        "probe_label": "pose_hold",
        **info,
    }


def _pose_hold_delta(env) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
    body_name = str(args_cli.pose_hold_body)
    target_pos = getattr(env.unwrapped, "_aic_pose_hold_target_pos_w", None)
    target_quat = getattr(env.unwrapped, "_aic_pose_hold_target_quat_w", None)
    current_pos = _body_position(env, body_name)
    current_quat = _body_orientation(env, body_name)
    if target_pos is None or target_quat is None or current_pos is None or current_quat is None:
        return None
    target_pos = target_pos.to(device=current_pos.device, dtype=current_pos.dtype)
    target_quat = target_quat.to(device=current_quat.device, dtype=current_quat.dtype)
    pos_error = (target_pos - current_pos) * float(args_cli.pose_hold_position_gain)
    world_delta = _clip_vector_norm(pos_error, max(float(args_cli.probe_translation_step_m), 0.0))
    delta_quat_w = math_utils.quat_mul(target_quat, math_utils.quat_inv(current_quat))
    rotvec_world = math_utils.axis_angle_from_quat(delta_quat_w) * float(args_cli.pose_hold_rotation_gain)
    rotvec_world = _clip_vector_norm(rotvec_world, max(float(args_cli.probe_rotation_step_rad), 0.0))
    return world_delta, rotvec_world, {
        "pose_hold_body": body_name,
        "pose_hold_position_error_m_by_env": _tensor_list(torch.linalg.norm(pos_error, dim=1, keepdim=True)),
        "pose_hold_rotation_error_rad_by_env": _tensor_list(
            torch.linalg.norm(math_utils.axis_angle_from_quat(delta_quat_w), dim=1, keepdim=True)
        ),
        "pose_hold_world_delta_m_by_env": _tensor_rows(world_delta),
        "pose_hold_rotvec_world_rad_by_env": _tensor_rows(rotvec_world),
    }


def _pose_hold_orientation_servo_best_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold_orientation_servo_best", "reason": "missing_pose_hold_target"}
    world_delta, hold_rotvec, hold_info = hold
    rot_step = (
        float(args_cli.pose_hold_orientation_step_rad)
        if math.isfinite(float(args_cli.pose_hold_orientation_step_rad))
        else float(args_cli.probe_rotation_step_rad)
    )
    rotvec, orient_info = _best_semantic_orientation_rotvec(env, world_delta, rotation_step_rad=rot_step)
    if rotvec is None:
        rotvec = hold_rotvec
        active = torch.zeros((world_delta.shape[0], 1), dtype=torch.bool, device=world_delta.device)
    else:
        active = _pose_hold_orientation_active(env)
        if probe_step < int(args_cli.pose_hold_orientation_start_probe_step):
            active = torch.zeros_like(active)
        rotvec = torch.where(active.expand_as(rotvec), rotvec, hold_rotvec)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "pose_hold_orientation_servo_best",
        "orientation_servo_active_by_env": [bool(x) for x in active.detach().cpu().reshape(-1).tolist()],
        **hold_info,
        **orient_info,
    }


def _pose_hold_rotation_axis_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold_rotation_axis", "reason": "missing_pose_hold_target"}
    world_delta, hold_rotvec, hold_info = hold
    axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
    sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
    fixed_step = (
        float(args_cli.pose_hold_fixed_rotation_step_rad)
        if math.isfinite(float(args_cli.pose_hold_fixed_rotation_step_rad))
        else float(args_cli.probe_rotation_step_rad)
    )
    fixed_rot = torch.zeros_like(hold_rotvec)
    fixed_rot[:, axis_idx] = sign * max(fixed_step, 0.0)
    active = _pose_hold_orientation_active(env)
    if probe_step < int(args_cli.pose_hold_orientation_start_probe_step):
        active = torch.zeros_like(active)
    rotvec = torch.where(active.expand_as(fixed_rot), fixed_rot, hold_rotvec)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "pose_hold_rotation_axis",
        "pose_hold_fixed_rotation_active_by_env": [bool(x) for x in active.detach().cpu().reshape(-1).tolist()],
        "pose_hold_fixed_rotation_axis": str(args_cli.probe_rotation_axis),
        "pose_hold_fixed_rotation_sign": sign,
        **hold_info,
    }


def _pose_hold_constrained_rotation_axis_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold_constrained_rotation_axis", "reason": "missing_pose_hold_target"}
    hold_delta, hold_rotvec, hold_info = hold
    axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
    sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
    fixed_step = (
        float(args_cli.pose_hold_fixed_rotation_step_rad)
        if math.isfinite(float(args_cli.pose_hold_fixed_rotation_step_rad))
        else float(args_cli.probe_rotation_step_rad)
    )
    fixed_rot = torch.zeros_like(hold_rotvec)
    fixed_rot[:, axis_idx] = sign * max(fixed_step, 0.0)
    active = _pose_hold_orientation_active(env)
    if probe_step < int(args_cli.pose_hold_orientation_start_probe_step):
        active = torch.zeros_like(active)
    rotvec = torch.where(active.expand_as(fixed_rot), fixed_rot, hold_rotvec)
    compensation, comp_info = _rotation_tip_module_sweep_compensation(
        env,
        rotvec,
        tip_weight=max(float(args_cli.pose_hold_constrained_tip_weight), 0.0),
        module_weight=max(float(args_cli.pose_hold_constrained_module_weight), 0.0),
    )
    compensation = _clip_vector_norm(compensation, max(float(args_cli.pose_hold_constrained_compensation_clip_m), 0.0))
    # The constrained probe isolates rotation realization: use the shared
    # tip/module compensation instead of the pose-hold body's translational
    # correction when the fixed rotation is active.
    world_delta = torch.where(active.expand_as(compensation), compensation, hold_delta)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "pose_hold_constrained_rotation_axis",
        "pose_hold_constrained_rotation_active_by_env": [bool(x) for x in active.detach().cpu().reshape(-1).tolist()],
        "pose_hold_constrained_rotation_axis": str(args_cli.probe_rotation_axis),
        "pose_hold_constrained_rotation_sign": sign,
        "pose_hold_constrained_compensation_clip_m": float(args_cli.pose_hold_constrained_compensation_clip_m),
        "pose_hold_constrained_world_delta_m_by_env": _tensor_rows(world_delta),
        "pose_hold_constrained_rotvec_world_rad_by_env": _tensor_rows(rotvec),
        **hold_info,
        **comp_info,
    }


def _pose_hold_orientation_active(env) -> torch.Tensor:
    target = _episode_target_position(env)
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    tip = _body_position(env, "sfp_tip_link")
    if target is None or entrance is None or axis is None or tip is None:
        return torch.zeros((env.unwrapped.num_envs, 1), dtype=torch.bool, device=env.unwrapped.device)
    geom = compute_insertion_geometry(
        body_pos_w=tip,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    consistency = _module_consistency(env, geom, entrance, target, axis)
    if consistency is None:
        consistency = torch.zeros_like(geom.axial_depth)
    return (
        (geom.axial_depth.view(-1, 1) >= float(args_cli.pose_hold_orientation_activation_depth_m))
        & (geom.lateral_error.view(-1, 1) <= float(args_cli.pose_hold_orientation_activation_lateral_m))
        & (consistency.view(-1, 1) >= float(args_cli.pose_hold_orientation_activation_consistency))
    )


def _best_semantic_orientation_rotvec(
    env,
    world_delta: torch.Tensor,
    *,
    rotation_step_rad: float | None = None,
    module_body_name: str | None = None,
    module_lateral_penalty: float = 0.0,
    module_lateral_margin_m: float = 0.0,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    axis = _episode_axis(env)
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    body_quat = _body_orientation(env, "sfp_tip_link")
    body_axis = None if axis is None else _semantic_body_axis(env, "sfp_tip_link", axis)
    use_quat = str(args_cli.target_reward_orientation_error_mode).lower() == "quat"
    target_quat = _episode_target_body_orientation(env) if use_quat else None
    if axis is None or tip is None or entrance is None or body_quat is None or (not use_quat and body_axis is None):
        return None, {"orientation_servo_reason": "missing_orientation_servo_geometry"}
    if use_quat and target_quat is None:
        return None, {"orientation_servo_reason": "missing_target_quat"}
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    if use_quat:
        target_quat = target_quat.to(device=body_quat.device, dtype=body_quat.dtype)
        current_theta = math_utils.quat_error_magnitude(body_quat, target_quat).view(-1, 1)
    else:
        current_theta = torch.acos(torch.sum(body_axis * axis, dim=1, keepdim=True).clamp(-1.0, 1.0))
    base_candidate_axes = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=body_axis.dtype,
        device=body_axis.device,
    )
    labels = ["+rx", "-rx", "+ry", "-ry", "+rz", "-rz"]
    candidate_axes = base_candidate_axes.view(1, 6, 3).expand(body_axis.shape[0], -1, -1)
    if bool(args_cli.target_module_stabilize_orientation_cross_axis_candidate) and not use_quat:
        cross_axis = torch.linalg.cross(body_axis, axis, dim=1)
        cross_norm = torch.linalg.norm(cross_axis, dim=1, keepdim=True)
        cross_axis = cross_axis / cross_norm.clamp(min=1.0e-9)
        cross_axis = torch.where(cross_norm > 1.0e-9, cross_axis, torch.zeros_like(cross_axis))
        candidate_axes = torch.cat([candidate_axes, cross_axis[:, None, :]], dim=1)
        labels.append("axis_cross")
    rot_step = max(float(args_cli.probe_rotation_step_rad if rotation_step_rad is None else rotation_step_rad), 0.0)
    batch_size = int(body_quat.shape[0])
    num_candidates = int(candidate_axes.shape[1])
    candidate_rotvec = candidate_axes * rot_step
    q_candidate = _quat_from_rotvec(candidate_rotvec.reshape(-1, 3))
    if use_quat:
        predicted_quat = math_utils.quat_mul(
            q_candidate,
            body_quat[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 4),
        ).reshape(body_quat.shape[0], num_candidates, 4)
        target_quat_flat = target_quat[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 4)
        predicted_theta_raw = math_utils.quat_error_magnitude(
            predicted_quat.reshape(-1, 4),
            target_quat_flat,
        ).reshape(body_quat.shape[0], num_candidates)
        predicted_theta = predicted_theta_raw
    else:
        predicted_axis = math_utils.quat_apply(
            q_candidate,
            body_axis[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 3),
        ).reshape(body_axis.shape[0], num_candidates, 3)
        predicted_theta_raw = torch.acos(torch.sum(predicted_axis * axis[:, None, :], dim=2).clamp(-1.0, 1.0))
        predicted_theta = predicted_theta_raw
    if float(args_cli.probe_orientation_lateral_penalty) > 0.0:
        wrist = _body_position(env, "wrist_3_link")
        if wrist is not None:
            current_rel = tip - entrance
            current_depth = torch.sum(current_rel * axis, dim=1, keepdim=True)
            current_lateral = torch.linalg.norm(current_rel - current_depth * axis, dim=1, keepdim=True)
            lever = tip - wrist
            predicted_tip = (
                math_utils.quat_apply(q_candidate, lever[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 3))
                .reshape(tip.shape[0], num_candidates, 3)
                + wrist[:, None, :]
                + world_delta[:, None, :]
            )
            pred_rel = predicted_tip - entrance[:, None, :]
            pred_depth = torch.sum(pred_rel * axis[:, None, :], dim=2, keepdim=True)
            pred_lateral = torch.linalg.norm(pred_rel - pred_depth * axis[:, None, :], dim=2)
            lateral_increase = (pred_lateral - current_lateral).clamp(min=0.0)
            predicted_theta = predicted_theta + float(args_cli.probe_orientation_lateral_penalty) * lateral_increase
    selected_pred_module_lateral = None
    selected_module_lateral_increase = None
    module_penalty = max(float(module_lateral_penalty), 0.0)
    if module_penalty > 0.0 and module_body_name:
        wrist = _body_position(env, "wrist_3_link")
        module = _body_position(env, module_body_name)
        if wrist is not None and module is not None:
            current_rel = module - entrance
            current_depth = torch.sum(current_rel * axis, dim=1, keepdim=True)
            current_lateral = torch.linalg.norm(current_rel - current_depth * axis, dim=1, keepdim=True)
            lever = module - wrist
            predicted_module = (
                math_utils.quat_apply(q_candidate, lever[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 3))
                .reshape(module.shape[0], num_candidates, 3)
                + wrist[:, None, :]
                + world_delta[:, None, :]
            )
            pred_rel = predicted_module - entrance[:, None, :]
            pred_depth = torch.sum(pred_rel * axis[:, None, :], dim=2, keepdim=True)
            pred_lateral = torch.linalg.norm(pred_rel - pred_depth * axis[:, None, :], dim=2)
            margin = max(float(module_lateral_margin_m), 0.0)
            module_lateral_increase = (pred_lateral - current_lateral - margin).clamp(min=0.0)
            predicted_theta = predicted_theta + module_penalty * module_lateral_increase
            selected_pred_module_lateral = pred_lateral
            selected_module_lateral_increase = module_lateral_increase
    _, best_idx = torch.min(predicted_theta, dim=1, keepdim=True)
    selected_rotvec = torch.gather(candidate_rotvec, 1, best_idx[:, :, None].expand(-1, -1, 3)).squeeze(1)
    selected_module_lateral = None
    selected_module_increase = None
    if selected_pred_module_lateral is not None and selected_module_lateral_increase is not None:
        selected_module_lateral = torch.gather(selected_pred_module_lateral, 1, best_idx).view(-1, 1)
        selected_module_increase = torch.gather(selected_module_lateral_increase, 1, best_idx).view(-1, 1)
    if use_quat:
        predicted_raw_theta = math_utils.quat_error_magnitude(
            math_utils.quat_mul(_quat_from_rotvec(selected_rotvec), body_quat),
            target_quat,
        ).view(-1, 1)
    else:
        predicted_raw_theta = torch.acos(
            torch.sum(math_utils.quat_apply(_quat_from_rotvec(selected_rotvec), body_axis) * axis, dim=1, keepdim=True).clamp(
                -1.0, 1.0
            )
        )
    improves = predicted_raw_theta < current_theta
    selected_rotvec = torch.where(improves.expand_as(selected_rotvec), selected_rotvec, torch.zeros_like(selected_rotvec))
    return selected_rotvec, {
        "selected_axis_by_env": [labels[int(i)] for i in best_idx.detach().cpu().reshape(-1).tolist()],
        "orientation_error_mode": str(args_cli.target_reward_orientation_error_mode),
        "current_theta_rad_by_env": _tensor_list(current_theta),
        "predicted_theta_rad_by_env": _tensor_list(predicted_raw_theta),
        "improves_by_env": [bool(x) for x in improves.detach().cpu().reshape(-1).tolist()],
        "orientation_module_lateral_penalty": module_penalty,
        "orientation_module_lateral_margin_m": max(float(module_lateral_margin_m), 0.0),
        "predicted_module_lateral_m_by_env": (
            None if selected_module_lateral is None else _tensor_list(selected_module_lateral)
        ),
        "predicted_module_lateral_penalized_increase_m_by_env": (
            None if selected_module_increase is None else _tensor_list(selected_module_increase)
        ),
    }


def _orientation_servo_best_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    axis = _episode_axis(env)
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    body_axis = None if axis is None else _semantic_body_axis(env, "sfp_tip_link", axis)
    if axis is None or tip is None or entrance is None or body_axis is None:
        return _zero_action(env), {"reason": "missing_orientation_servo_geometry"}
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    current_theta = torch.acos(torch.sum(body_axis * axis, dim=1, keepdim=True).clamp(-1.0, 1.0))
    candidate_axes = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=body_axis.dtype,
        device=body_axis.device,
    )
    rot_step = max(float(args_cli.probe_rotation_step_rad), 0.0)
    candidate_rotvec = candidate_axes.view(1, 6, 3).expand(body_axis.shape[0], -1, -1) * rot_step
    q_candidate = _quat_from_rotvec(candidate_rotvec.reshape(-1, 3))
    predicted_axis = math_utils.quat_apply(
        q_candidate,
        body_axis[:, None, :].expand(-1, 6, -1).reshape(-1, 3),
    ).reshape(body_axis.shape[0], 6, 3)
    predicted_theta = torch.acos(
        torch.sum(predicted_axis * axis[:, None, :], dim=2).clamp(-1.0, 1.0)
    )
    rel = tip - entrance
    depth = torch.sum(rel * axis, dim=1, keepdim=True)
    centerline = entrance + depth * axis
    lateral_vec = centerline - tip
    lateral_error = torch.linalg.norm(lateral_vec, dim=1, keepdim=True)
    lateral_step = torch.minimum(
        lateral_error,
        torch.full_like(lateral_error, max(float(args_cli.probe_lateral_correction_step_m), 0.0)),
    )
    lateral_delta = lateral_vec / lateral_error.clamp(min=1.0e-9) * lateral_step
    axial_step = torch.full_like(depth, float(args_cli.probe_axial_step_m))
    can_advance = lateral_error <= max(float(args_cli.probe_axial_lateral_gate_m), 0.0)
    axial_delta = torch.where(can_advance, axial_step * axis, torch.zeros_like(lateral_delta))
    world_delta = lateral_delta + axial_delta
    if float(args_cli.probe_orientation_lateral_penalty) > 0.0:
        # Approximate one-step sweep from rotating the semantic tip around the wrist/root action frame.
        wrist = _body_position(env, "wrist_3_link")
        if wrist is not None:
            current_rel = tip - entrance
            current_depth = torch.sum(current_rel * axis, dim=1, keepdim=True)
            current_lateral = torch.linalg.norm(current_rel - current_depth * axis, dim=1, keepdim=True)
            lever = tip - wrist
            predicted_tip = (
                math_utils.quat_apply(q_candidate, lever[:, None, :].expand(-1, 6, -1).reshape(-1, 3))
                .reshape(tip.shape[0], 6, 3)
                + wrist[:, None, :]
                + world_delta[:, None, :]
            )
            pred_rel = predicted_tip - entrance[:, None, :]
            pred_depth = torch.sum(pred_rel * axis[:, None, :], dim=2, keepdim=True)
            pred_lateral = torch.linalg.norm(pred_rel - pred_depth * axis[:, None, :], dim=2)
            lateral_increase = (pred_lateral - current_lateral).clamp(min=0.0)
            predicted_theta = predicted_theta + float(args_cli.probe_orientation_lateral_penalty) * lateral_increase
    best_score, best_idx = torch.min(predicted_theta, dim=1, keepdim=True)
    selected_rotvec = torch.gather(
        candidate_rotvec,
        1,
        best_idx[:, :, None].expand(-1, -1, 3),
    ).squeeze(1)
    predicted_raw_theta = torch.acos(
        torch.sum(
            math_utils.quat_apply(_quat_from_rotvec(selected_rotvec), body_axis) * axis,
            dim=1,
            keepdim=True,
        ).clamp(-1.0, 1.0)
    )
    improves = predicted_raw_theta < current_theta
    selected_rotvec = torch.where(improves.expand_as(selected_rotvec), selected_rotvec, torch.zeros_like(selected_rotvec))
    labels = ["+rx", "-rx", "+ry", "-ry", "+rz", "-rz"]
    return _root_action_from_world_delta(env, world_delta, rotvec_world=selected_rotvec), {
        "probe_label": "orientation_servo_best",
        "selected_axis_by_env": [labels[int(i)] for i in best_idx.detach().cpu().reshape(-1).tolist()],
        "current_theta_rad_by_env": _tensor_list(current_theta),
        "predicted_theta_rad_by_env": _tensor_list(predicted_raw_theta),
        "improves_by_env": [bool(x) for x in improves.detach().cpu().reshape(-1).tolist()],
        "retention_lateral_error_m_by_env": _tensor_list(lateral_error),
        "retention_depth_m_by_env": _tensor_list(depth),
        "retention_advanced_fraction": float(can_advance.float().mean().detach().cpu()),
    }


def _probe_retention_delta(env) -> tuple[torch.Tensor, dict[str, Any]]:
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    if tip is None or entrance is None or axis is None:
        return torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.unwrapped.device), {
            "retention_reason": "missing_geometry"
        }
    rel = tip - entrance
    depth = torch.sum(rel * axis, dim=1, keepdim=True)
    centerline = entrance + depth * axis
    lateral_vec = centerline - tip
    lateral_error = torch.linalg.norm(lateral_vec, dim=1, keepdim=True)
    lateral_step = torch.minimum(
        lateral_error,
        torch.full_like(lateral_error, max(float(args_cli.probe_lateral_correction_step_m), 0.0)),
    )
    lateral_delta = lateral_vec / lateral_error.clamp(min=1.0e-9) * lateral_step
    axial_step = torch.full_like(depth, float(args_cli.probe_axial_step_m))
    can_advance = lateral_error <= max(float(args_cli.probe_axial_lateral_gate_m), 0.0)
    axial_delta = torch.where(can_advance, axial_step * axis, torch.zeros_like(lateral_delta))
    return lateral_delta + axial_delta, {
        "retention_lateral_error_m_by_env": _tensor_list(lateral_error),
        "retention_depth_m_by_env": _tensor_list(depth),
        "retention_advanced_fraction": float(can_advance.float().mean().detach().cpu()),
    }


def _body_positions(env) -> dict[str, list[list[float]] | None]:
    out: dict[str, list[list[float]] | None] = {}
    for name in ("wrist_3_link", "gripper_tcp", "sfp_module_link", "sfp_tip_link"):
        pos = _body_position(env, name)
        out[name] = None if pos is None else [[float(v) for v in row] for row in pos.detach().cpu().tolist()]
    return out


def _body_orientations(env) -> dict[str, list[list[float]] | None]:
    out: dict[str, list[list[float]] | None] = {}
    for name in ("wrist_3_link", "gripper_tcp", "sfp_module_link", "sfp_tip_link"):
        quat = _body_orientation(env, name)
        out[name] = None if quat is None else [[float(v) for v in row] for row in quat.detach().cpu().tolist()]
    return out


def _relative_body_transforms(env) -> dict[str, Any]:
    pairs = (
        ("gripper_tcp", "sfp_module_link"),
        ("gripper_tcp", "sfp_tip_link"),
        ("sfp_module_link", "sfp_tip_link"),
    )
    out: dict[str, Any] = {}
    for parent, child in pairs:
        parent_pos = _body_position(env, parent)
        child_pos = _body_position(env, child)
        parent_quat = _body_orientation(env, parent)
        child_quat = _body_orientation(env, child)
        key = f"{parent}_to_{child}"
        if parent_pos is None or child_pos is None or parent_quat is None or child_quat is None:
            out[key] = None
            continue
        parent_inv = _quat_conjugate(parent_quat)
        rel_pos = math_utils.quat_apply(parent_inv, child_pos - parent_pos)
        rel_quat = math_utils.quat_mul(parent_inv, child_quat)
        out[key] = {
            "position_parent_frame_m_by_env": _tensor_rows(rel_pos),
            "orientation_parent_child_wxyz_by_env": _tensor_rows(rel_quat),
        }
    return out


def _reset_diagnostic(env) -> dict[str, Any]:
    episodes = _current_episode_by_env(env)
    axis = _episode_axis(env)
    out: dict[str, Any] = {
        "episode_start_by_env": {},
        "actual_body_position_world_by_env": _body_positions(env),
        "actual_body_orientation_wxyz_by_env": _body_orientations(env),
        "relative_body_transforms": _relative_body_transforms(env),
        "tip_geometry": _geometry(env, "sfp_tip_link"),
        "module_geometry": _geometry(env, "sfp_module_link"),
        "contact": _contact_summary(env),
    }
    if axis is not None:
        out["axis_world_by_env"] = [[float(v) for v in row] for row in axis.detach().cpu().tolist()]
    for env_id, episode in episodes.items():
        scene = (episode or {}).get("scene") or {}
        target = scene.get("target") or {}
        start = target.get("start_near_gate")
        if not isinstance(start, dict):
            start = scene.get("start_near_gate")
        if not isinstance(start, dict):
            start = (episode or {}).get("start_near_gate") or {}
        out["episode_start_by_env"][str(env_id)] = {
            "body_start_position_world": start.get("body_start_position_world"),
            "body_start_orientation_wxyz": start.get("body_start_orientation_wxyz")
            or start.get("reset_body_orientation_wxyz"),
            "reference_reward_body_start_position_world": start.get("reference_reward_body_start_position_world")
            or start.get("reference_tip_center_position_world"),
            "reference_reward_body_start_orientation_wxyz": start.get("reference_reward_body_start_orientation_wxyz")
            or target.get("body_start_orientation_wxyz"),
            "axial_distance_m": start.get("axial_distance_m"),
            "lateral_distance_m": start.get("lateral_distance_m"),
        }
    return out


def _robot_state(env) -> dict[str, Any] | None:
    try:
        robot = env.unwrapped.scene["robot"]
    except Exception:
        return None
    data = getattr(robot, "data", None)
    joint_pos = getattr(data, "joint_pos", None)
    if joint_pos is None:
        return None
    joint_vel = getattr(data, "joint_vel", None)
    joint_names = list(getattr(robot, "joint_names", []) or getattr(data, "joint_names", []) or [])
    return {
        "joint_names": [str(name) for name in joint_names],
        "joint_positions_by_env": _tensor_rows(joint_pos),
        "joint_velocities_by_env": None if joint_vel is None else _tensor_rows(joint_vel),
    }


def _step_record(
    env,
    *,
    step: int,
    phase: str,
    action: torch.Tensor,
    action_info: dict[str, Any],
    before_positions: dict[str, Any],
) -> dict[str, Any]:
    after_positions = _body_positions(env)
    after_orientations = _body_orientations(env)
    geom = _geometry(env, "sfp_tip_link")
    module_geom = _geometry(env, "sfp_module_link")
    realized: dict[str, Any] = {}
    for name, after in after_positions.items():
        before = before_positions.get(name)
        if after is None or before is None:
            realized[name] = None
            continue
        delta = torch.tensor(after, dtype=torch.float32) - torch.tensor(before, dtype=torch.float32)
        realized[name] = {
            "delta_norm_m_by_env": _tensor_list(torch.linalg.norm(delta, dim=1)),
            "delta_world_by_env": [[float(v) for v in row] for row in delta.tolist()],
        }
    return {
        "step": step,
        "phase": phase,
        "action_info": action_info,
        "action_env0": _tensor_list(action[0]),
        "post_step_insertion_geometry": geom,
        "post_step_module_geometry": module_geom,
        "contact": _contact_summary(env),
        "post_step_robot_state": _robot_state(env),
        "actual_body_position_world_by_env": after_positions,
        "actual_body_orientation_wxyz_by_env": after_orientations,
        "realized_body_motion": realized,
        "relative_body_transforms": _relative_body_transforms(env),
    }


def _strict_any(row: dict[str, Any]) -> bool:
    geom = row.get("post_step_insertion_geometry") or {}
    return any(bool(x) for x in geom.get("strict_success_by_env") or [])


def _near_success_capture_any(row: dict[str, Any]) -> bool:
    if not bool(args_cli.stop_on_near_success_capture):
        return False
    geom = row.get("post_step_insertion_geometry") or {}
    s_vals = geom.get("signed_depth_m_by_env") or []
    r_vals = geom.get("lateral_error_m_by_env") or []
    theta_vals = geom.get("orientation_error_rad_by_env") or []
    cons_vals = geom.get("consistency_gate_by_env") or []
    count = max(len(s_vals), len(r_vals), len(theta_vals), len(cons_vals), 0)
    for idx in range(count):
        s = float(s_vals[min(idx, len(s_vals) - 1)]) if s_vals else -float("inf")
        r = float(r_vals[min(idx, len(r_vals) - 1)]) if r_vals else float("inf")
        theta = float(theta_vals[min(idx, len(theta_vals) - 1)]) if theta_vals else float("inf")
        cons = float(cons_vals[min(idx, len(cons_vals) - 1)]) if cons_vals else 0.0
        if (
            s >= float(args_cli.near_success_capture_min_s_m)
            and r <= float(args_cli.near_success_capture_max_r_m)
            and theta <= float(args_cli.near_success_capture_max_theta_rad)
            and cons >= float(args_cli.near_success_capture_min_module_consistency)
        ):
            return True
    return False


def _configure_semantic_reward_terms(env_cfg) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    body_names = [str(args_cli.target_reward_body)]
    for name in (
        "target_distance_tanh",
        "target_distance_exp",
        "target_distance_progress",
        "target_orientation_tanh",
        "target_orientation_gated_exp",
        "target_reaching_bonus",
        "target_success_once_bonus",
        "target_lateral_error",
        "target_motion_projection",
        "target_lateral_progress",
        "target_axial_progress",
        "target_insertion_corridor",
        "target_cheatcode_phase_reward",
    ):
        term = getattr(rewards, name, None)
        params = None if term is None else getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        body_cfg = params.get("body_cfg")
        if body_cfg is not None:
            body_cfg.body_names = body_names
        target_cfg = params.get("target_cfg")
        if target_cfg is not None:
            target_cfg.name = "nic_card"
        if "orientation_error_mode" in params:
            params["orientation_error_mode"] = str(args_cli.target_reward_orientation_error_mode)
        if "orientation_axis_local" in params:
            params["orientation_axis_local"] = tuple(float(x) for x in args_cli.target_reward_orientation_axis_local)
        if "consistency_body_name" in params:
            params["consistency_body_name"] = str(args_cli.target_reward_consistency_body)
        if "consistency_axial_std" in params:
            params["consistency_axial_std"] = float(args_cli.target_reward_consistency_axial_std)
        if "consistency_lateral_sigma" in params:
            params["consistency_lateral_sigma"] = float(args_cli.target_reward_consistency_lateral_sigma)
    terminations = getattr(env_cfg, "terminations", None)
    target_success = None if terminations is None else getattr(terminations, "target_success", None)
    params = None if target_success is None else getattr(target_success, "params", None)
    if isinstance(params, dict):
        body_cfg = params.get("body_cfg")
        if body_cfg is not None:
            body_cfg.body_names = body_names
        target_cfg = params.get("target_cfg")
        if target_cfg is not None:
            target_cfg.name = "nic_card"
        if "orientation_error_mode" in params:
            params["orientation_error_mode"] = str(args_cli.target_reward_orientation_error_mode)
        if "orientation_axis_local" in params:
            params["orientation_axis_local"] = tuple(float(x) for x in args_cli.target_reward_orientation_axis_local)
        if "consistency_body_name" in params:
            params["consistency_body_name"] = str(args_cli.target_reward_consistency_body)


def _configure_near_gate_reset(env_cfg) -> dict[str, Any]:
    events = getattr(env_cfg, "events", None)
    term = None if events is None else getattr(events, "reset_robot_tcp_to_episode_start", None)
    params = None if term is None else getattr(term, "params", None)
    if not isinstance(params, dict):
        return {"configured": False}
    before = dict(params)
    if int(args_cli.near_gate_reset_max_iterations) > 0:
        params["max_iterations"] = int(args_cli.near_gate_reset_max_iterations)
    if float(args_cli.near_gate_reset_position_tolerance) > 0.0:
        params["position_tolerance"] = float(args_cli.near_gate_reset_position_tolerance)
    if float(args_cli.near_gate_reset_orientation_tolerance) > 0.0:
        params["orientation_tolerance"] = float(args_cli.near_gate_reset_orientation_tolerance)
    return {"configured": True, "before": before, "after": dict(params)}


def _disable_matching_collision_prims(run_dir: Path) -> dict[str, Any]:
    patterns = [str(pattern) for pattern in (args_cli.disable_collision_prim_regex or []) if str(pattern).strip()]
    if not patterns:
        return {"enabled": False, "patterns": [], "matched": []}
    try:
        compiled = [re.compile(pattern) for pattern in patterns]
    except re.error as exc:
        raise ValueError(f"invalid --disable_collision_prim_regex: {exc}") from exc
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable for collision toggle")
    matched: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not any(pattern.search(path) for pattern in compiled):
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        previous = attr.Get() if attr and attr.HasValue() else None
        if attr:
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        matched.append({"path": path, "type": prim.GetTypeName(), "previous_collision_enabled": previous})
    report = {"enabled": True, "patterns": patterns, "matched": matched, "matched_count": len(matched)}
    (run_dir / "collision_toggle_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_sdf_body_boxes(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    boxes: list[dict[str, Any]] = []
    for link in root.findall(".//link"):
        if link.attrib.get("name") != "sfp_module_link":
            continue
        collisions = link.findall("./collision")
        break
    else:
        collisions = []
    for collision in collisions:
        name = str(collision.attrib.get("name", ""))
        if not name.startswith("body_collider_box"):
            continue
        box = collision.find("./geometry/box")
        size_elem = None if box is None else box.find("./size")
        if size_elem is None:
            continue
        pose_elem = collision.find("./pose")
        pose = [float(v) for v in (pose_elem.text or "0 0 0 0 0 0").split()] if pose_elem is not None else [0.0] * 6
        size = [float(v) for v in (size_elem.text or "").split()]
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected SDF body box pose/size for {name}: {pose} {size}")
        boxes.append(
            {
                "name": re.sub(r"[^A-Za-z0-9_]", "_", name),
                "translation_m": pose[:3],
                "rotation_rpy_rad": pose[3:],
                "size_m": size,
            }
        )
    return boxes


def _parse_removed_sfp_module_colliders(path: Path) -> set[str]:
    removed: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r'<collision\s+element_id="sfp_module_link::([^"]+)"\s+action="remove"\s*/>', text):
        removed.add(match.group(1).replace(".", "_"))
    return removed


def _parse_sdf_module_boxes(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    boxes: list[dict[str, Any]] = []
    for link in root.findall(".//link"):
        if link.attrib.get("name") != "sfp_module_link":
            continue
        collisions = link.findall("./collision")
        break
    else:
        collisions = []
    for collision in collisions:
        name = str(collision.attrib.get("name", ""))
        box = collision.find("./geometry/box")
        size_elem = None if box is None else box.find("./size")
        if size_elem is None:
            continue
        pose_elem = collision.find("./pose")
        pose = [float(v) for v in (pose_elem.text or "0 0 0 0 0 0").split()] if pose_elem is not None else [0.0] * 6
        size = [float(v) for v in (size_elem.text or "").split()]
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected SDF module box pose/size for {name}: {pose} {size}")
        boxes.append(
            {
                "name": re.sub(r"[^A-Za-z0-9_]", "_", name),
                "translation_m": pose[:3],
                "rotation_rpy_rad": pose[3:],
                "size_m": size,
            }
        )
    return boxes


def _parse_nic_cage_p0_sdf_boxes(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    boxes: list[dict[str, Any]] = []
    cage_names = {
        "10099100-011lfc001_collider_box",
        "10099100-011lfc001_collider_box.001",
        "10099100-011lfc001_collider_box.002",
        "10099100-011lfc001_collider_box.003",
        "10099100-011lfc001_collider_box.004",
    }
    for link in root.findall(".//link"):
        if link.attrib.get("name") != "nic_card_link":
            continue
        collisions = link.findall("./collision")
        break
    else:
        collisions = []
    for collision in collisions:
        name = str(collision.attrib.get("name", ""))
        if name not in cage_names:
            continue
        box = collision.find("./geometry/box")
        size_elem = None if box is None else box.find("./size")
        if size_elem is None:
            continue
        pose_elem = collision.find("./pose")
        pose = [float(v) for v in (pose_elem.text or "0 0 0 0 0 0").split()] if pose_elem is not None else [0.0] * 6
        size = [float(v) for v in (size_elem.text or "").split()]
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected NIC cage box pose/size for {name}: {pose} {size}")
        boxes.append(
            {
                "name": re.sub(r"[^A-Za-z0-9_]", "_", name),
                "translation_m": pose[:3],
                "rotation_rpy_rad": pose[3:],
                "size_m": size,
            }
        )
    return boxes


def _replace_sfp_body_sdf_collision_with_sdf_boxes(run_dir: Path) -> dict[str, Any]:
    body_replacement = bool(args_cli.replace_sfp_body_sdf_collision_with_sdf_boxes)
    shrunk_body_replacement = bool(args_cli.replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes)
    active_replacement = bool(args_cli.replace_sfp_module_sdf_collision_with_active_sdf_boxes)
    clearance_replacement = bool(args_cli.replace_sfp_body_sdf_collision_with_clearance_box)
    if not body_replacement and not shrunk_body_replacement and not active_replacement and not clearance_replacement:
        return {"enabled": False, "matched": [], "matched_count": 0, "created": [], "created_count": 0}
    if sum(bool(v) for v in (body_replacement, shrunk_body_replacement, active_replacement, clearance_replacement)) > 1:
        raise ValueError(
            "SFP collision replacement modes are mutually exclusive: choose only one of "
            "--replace_sfp_body_sdf_collision_with_sdf_boxes, "
            "--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes, "
            "--replace_sfp_module_sdf_collision_with_active_sdf_boxes, or "
            "--replace_sfp_body_sdf_collision_with_clearance_box"
        )
    if active_replacement:
        removed = _parse_removed_sfp_module_colliders(Path(args_cli.sfp_cable_sdf))
        boxes = [box for box in _parse_sdf_module_boxes(Path(args_cli.sfp_module_sdf)) if str(box["name"]) not in removed]
        mode = "gazebo_active_sfp_module_link_boxes"
    elif clearance_replacement:
        boxes = [
            {
                "name": "clearance_box",
                "translation_m": [float(v) for v in args_cli.sfp_clearance_box_translation_m],
                "rotation_rpy_rad": [0.0, 0.0, 0.0],
                "size_m": [float(v) for v in args_cli.sfp_clearance_box_size_m],
            }
        ]
        mode = "single_clearance_box"
    else:
        boxes = _parse_sdf_body_boxes(Path(args_cli.sfp_module_sdf))
        if shrunk_body_replacement:
            margins = [max(float(v), 0.0) for v in args_cli.sfp_shrunk_box_margin_m]
            shrunk_boxes: list[dict[str, Any]] = []
            for box in boxes:
                size = [float(v) for v in box["size_m"]]
                shrunk_size = [max(size[i] - 2.0 * margins[i], 1.0e-6) for i in range(3)]
                shrunk = dict(box)
                shrunk["name"] = f"{box['name']}_shrunk"
                shrunk["size_m"] = shrunk_size
                shrunk["original_size_m"] = size
                shrunk["shrink_margin_m"] = margins
                shrunk_boxes.append(shrunk)
            boxes = shrunk_boxes
            mode = "shrunk_body_boxes"
        else:
            mode = "body_boxes_only"
    if not boxes:
        raise ValueError(f"no SFP replacement boxes found in {args_cli.sfp_module_sdf}")
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable for SFP collision replacement")
    matched: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not path.endswith("/body_sdf_collision") or "/sfp_module/sfp_module_link/collisions/" not in path:
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        previous = attr.Get() if attr and attr.HasValue() else None
        if attr:
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        matched.append({"path": path, "type": prim.GetTypeName(), "previous_collision_enabled": previous})
        parent_path = str(prim.GetParent().GetPath())
        for box in boxes:
            box_path = f"{parent_path}/runtime_sdf_{box['name']}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.ClearXformOpOrder()
            tx, ty, tz = box["translation_m"]
            rx, ry, rz = box["rotation_rpy_rad"]
            sx, sy, sz = box["size_m"]
            xform.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
            xform.AddRotateXYZOp().Set(
                Gf.Vec3f(float(math.degrees(rx)), float(math.degrees(ry)), float(math.degrees(rz)))
            )
            xform.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            created.append(
                {
                    "path": box_path,
                    "source_sdf_collision": box["name"],
                    "translation_m": list(box["translation_m"]),
                    "rotation_rpy_rad": list(box["rotation_rpy_rad"]),
                    "size_m": list(box["size_m"]),
                    "original_size_m": list(box["original_size_m"]) if "original_size_m" in box else None,
                    "shrink_margin_m": list(box["shrink_margin_m"]) if "shrink_margin_m" in box else None,
                }
            )
    report = {
        "enabled": True,
        "mode": mode,
        "matched": matched,
        "matched_count": len(matched),
        "created": created,
        "created_count": len(created),
        "source": str(args_cli.sfp_module_sdf),
    }
    (run_dir / "sfp_body_collision_replacement_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _replace_nic_cage_p0_with_sdf_boxes(run_dir: Path) -> dict[str, Any]:
    sdf_replacement = bool(args_cli.replace_nic_cage_p0_with_sdf_boxes)
    aligned_replacement = bool(args_cli.replace_nic_cage_p0_with_aligned_cubes)
    if not sdf_replacement and not aligned_replacement:
        return {"enabled": False, "matched": [], "matched_count": 0, "created": [], "created_count": 0}
    if sdf_replacement and aligned_replacement:
        raise ValueError("NIC cage replacement modes are mutually exclusive")
    boxes = _parse_nic_cage_p0_sdf_boxes(Path(args_cli.nic_card_sdf)) if sdf_replacement else []
    if sdf_replacement and not boxes:
        raise ValueError(f"no NIC cage p0 replacement boxes found in {args_cli.nic_card_sdf}")
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable for NIC cage replacement")
    matched: list[dict[str, Any]] = []
    parent_paths: set[str] = set()
    aligned_sources: list[dict[str, Any]] = []
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if "/nic_card/collisions/cage_p0_" not in path:
            continue
        xform = UsdGeom.Xformable(prim)
        local_matrix = xform.GetLocalTransformation() if xform else Gf.Matrix4d(1.0)
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        previous = attr.Get() if attr and attr.HasValue() else None
        if attr:
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        matched.append({"path": path, "type": prim.GetTypeName(), "previous_collision_enabled": previous})
        parent_paths.add(str(prim.GetParent().GetPath()))
        aligned_sources.append({"name": prim.GetName(), "parent_path": str(prim.GetParent().GetPath()), "local_matrix": local_matrix})
    created: list[dict[str, Any]] = []
    if sdf_replacement:
        for parent_path in sorted(parent_paths):
            for box in boxes:
                box_path = f"{parent_path}/runtime_sdf_nic_p0_{box['name']}"
                cube = UsdGeom.Cube.Define(stage, box_path)
                cube.CreateSizeAttr(1.0)
                xform = UsdGeom.Xformable(cube.GetPrim())
                xform.ClearXformOpOrder()
                tx, ty, tz = box["translation_m"]
                rx, ry, rz = box["rotation_rpy_rad"]
                sx, sy, sz = box["size_m"]
                xform.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
                xform.AddRotateXYZOp().Set(
                    Gf.Vec3f(float(math.degrees(rx)), float(math.degrees(ry)), float(math.degrees(rz)))
                )
                xform.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
                created.append(
                    {
                        "path": box_path,
                        "source_sdf_collision": box["name"],
                        "translation_m": list(box["translation_m"]),
                        "rotation_rpy_rad": list(box["rotation_rpy_rad"]),
                        "size_m": list(box["size_m"]),
                    }
                )
        mode = "nic_cage_p0_sdf_boxes"
    else:
        for source in aligned_sources:
            box_path = f"{source['parent_path']}/runtime_aligned_cube_{source['name']}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(source["local_matrix"])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            created.append({"path": box_path, "source_isaac_collision": source["name"]})
        mode = "nic_cage_p0_aligned_cubes"
    report = {
        "enabled": True,
        "mode": mode,
        "matched": matched,
        "matched_count": len(matched),
        "created": created,
        "created_count": len(created),
        "source": str(args_cli.nic_card_sdf),
    }
    (run_dir / "nic_cage_p0_replacement_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _tune_matching_collision_contact_offsets(run_dir: Path) -> dict[str, Any]:
    patterns = [str(pattern) for pattern in (args_cli.collision_contact_tune_prim_regex or []) if str(pattern).strip()]
    contact_offset = float(args_cli.collision_contact_offset_m)
    rest_offset = float(args_cli.collision_rest_offset_m)
    if not patterns or (math.isnan(contact_offset) and math.isnan(rest_offset)):
        return {"enabled": False, "patterns": patterns, "matched": [], "matched_count": 0}
    try:
        compiled = [re.compile(pattern) for pattern in patterns]
    except re.error as exc:
        raise ValueError(f"invalid --collision_contact_tune_prim_regex: {exc}") from exc
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable for collision contact tuning")
    matched: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not any(pattern.search(path) for pattern in compiled):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact_attr = physx_api.GetContactOffsetAttr()
        rest_attr = physx_api.GetRestOffsetAttr()
        previous_contact = contact_attr.Get() if contact_attr and contact_attr.HasValue() else None
        previous_rest = rest_attr.Get() if rest_attr and rest_attr.HasValue() else None
        if not math.isnan(contact_offset):
            if contact_attr:
                contact_attr.Set(contact_offset)
            else:
                physx_api.CreateContactOffsetAttr(contact_offset)
        if not math.isnan(rest_offset):
            if rest_attr:
                rest_attr.Set(rest_offset)
            else:
                physx_api.CreateRestOffsetAttr(rest_offset)
        matched.append(
            {
                "path": path,
                "type": prim.GetTypeName(),
                "previous_contact_offset_m": previous_contact,
                "previous_rest_offset_m": previous_rest,
                "new_contact_offset_m": None if math.isnan(contact_offset) else contact_offset,
                "new_rest_offset_m": None if math.isnan(rest_offset) else rest_offset,
            }
        )
    report = {
        "enabled": True,
        "patterns": patterns,
        "contact_offset_m": None if math.isnan(contact_offset) else contact_offset,
        "rest_offset_m": None if math.isnan(rest_offset) else rest_offset,
        "matched": matched,
        "matched_count": len(matched),
    }
    (run_dir / "collision_contact_tuning_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _tune_matching_collision_materials(run_dir: Path) -> dict[str, Any]:
    patterns = [str(pattern) for pattern in (args_cli.collision_material_tune_prim_regex or []) if str(pattern).strip()]
    static_friction = float(args_cli.collision_static_friction)
    dynamic_friction = float(args_cli.collision_dynamic_friction)
    restitution = float(args_cli.collision_restitution)
    if not patterns or (math.isnan(static_friction) and math.isnan(dynamic_friction) and math.isnan(restitution)):
        return {"enabled": False, "patterns": patterns, "matched": [], "matched_count": 0}
    try:
        compiled = [re.compile(pattern) for pattern in patterns]
    except re.error as exc:
        raise ValueError(f"invalid --collision_material_tune_prim_regex: {exc}") from exc
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable for collision material tuning")

    material_path = "/World/aic_runtime_collision_material"
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    if not math.isnan(static_friction):
        material_api.CreateStaticFrictionAttr(static_friction).Set(static_friction)
    if not math.isnan(dynamic_friction):
        material_api.CreateDynamicFrictionAttr(dynamic_friction).Set(dynamic_friction)
    if not math.isnan(restitution):
        material_api.CreateRestitutionAttr(restitution).Set(restitution)

    matched: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not any(pattern.search(path) for pattern in compiled):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        previous_binding = None
        try:
            previous_bound = binding_api.ComputeBoundMaterial()[0]
            if previous_bound:
                previous_binding = str(previous_bound.GetPath())
        except Exception:
            previous_binding = None
        binding_api.Bind(material)
        matched.append(
            {
                "path": path,
                "type": prim.GetTypeName(),
                "previous_material": previous_binding,
                "new_material": material_path,
            }
        )

    report = {
        "enabled": True,
        "patterns": patterns,
        "static_friction": None if math.isnan(static_friction) else static_friction,
        "dynamic_friction": None if math.isnan(dynamic_friction) else dynamic_friction,
        "restitution": None if math.isnan(restitution) else restitution,
        "matched": matched,
        "matched_count": len(matched),
    }
    (run_dir / "collision_material_tuning_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    best_score = -1.0e9
    for row in rows:
        geom = row.get("post_step_insertion_geometry") or {}
        s_vals = geom.get("signed_depth_m_by_env") or []
        r_vals = geom.get("lateral_error_m_by_env") or []
        theta_vals = geom.get("orientation_error_rad_by_env") or []
        cons_vals = geom.get("consistency_gate_by_env") or []
        for idx in range(max(len(s_vals), len(r_vals), len(theta_vals), len(cons_vals), 0)):
            s = float(s_vals[min(idx, len(s_vals) - 1)]) if s_vals else -1.0
            r = float(r_vals[min(idx, len(r_vals) - 1)]) if r_vals else 1.0
            theta = float(theta_vals[min(idx, len(theta_vals) - 1)]) if theta_vals else 1.0
            cons = float(cons_vals[min(idx, len(cons_vals) - 1)]) if cons_vals else 0.0
            target_depth_vals = geom.get("target_depth_m_by_env") or [0.0458]
            target_depth = float(target_depth_vals[min(idx, len(target_depth_vals) - 1)])
            score = (
                10.0 * min(max(s, 0.0) / max(target_depth, 1.0e-9), 1.0)
                - 1000.0 * max(r - STRICT["max_lateral_m"], 0.0)
                - 20.0 * max(theta - STRICT["max_theta_rad"], 0.0)
                + cons
            )
            if score > best_score:
                best_score = score
                best = {"step": row["step"], "env": idx, "s_m": s, "r_m": r, "theta_rad": theta, "module_consistency": cons, "score": score}
    return best


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    output_root = Path(args_cli.output_dir)
    run_dir = output_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps({"stage": "created_run_dir"}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(str(x)) for x in sys.argv) + "\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text(_run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (run_dir / "git_diff.patch").write_text(_run_git(["diff", "--", "."]), encoding="utf-8")
    run_config = dict(vars(args_cli))
    run_config["prepared_episode_config_dir"] = prepared_episode_config_dir
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[wrist-contact] run_dir={run_dir}", flush=True)

    env = None
    try:
        status_path.write_text(json.dumps({"stage": "parse_env_cfg"}, indent=2) + "\n", encoding="utf-8")
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env_cfg.seed = int(args_cli.seed)
        if float(args_cli.episode_length_s) > 0.0:
            env_cfg.episode_length_s = float(args_cli.episode_length_s)
        if bool(args_cli.diagnostic_disable_command_visuals):
            command_cfg = getattr(getattr(env_cfg, "commands", None), "ee_pose", None)
            if command_cfg is not None and hasattr(command_cfg, "debug_vis"):
                command_cfg.debug_vis = False
        camera_width = int(args_cli.diagnostic_camera_width)
        camera_height = int(args_cli.diagnostic_camera_height)
        camera_focal_length = float(args_cli.diagnostic_camera_focal_length)
        camera_horizontal_aperture = float(args_cli.diagnostic_camera_horizontal_aperture)
        camera_vertical_aperture = float(args_cli.diagnostic_camera_vertical_aperture)
        camera_offset_pos = args_cli.diagnostic_camera_offset_pos
        camera_offset_rot = args_cli.diagnostic_camera_offset_rot
        fixed_camera_pos = None if bool(args_cli.diagnostic_wrist_cameras) else args_cli.diagnostic_fixed_camera_pos
        fixed_camera_target = None if bool(args_cli.diagnostic_wrist_cameras) else args_cli.diagnostic_fixed_camera_target
        if (
            camera_width > 0
            or camera_height > 0
            or not math.isnan(camera_focal_length)
            or not math.isnan(camera_horizontal_aperture)
            or not math.isnan(camera_vertical_aperture)
            or camera_offset_pos is not None
            or camera_offset_rot is not None
            or fixed_camera_pos is not None
        ):
            fixed_camera_positions: dict[str, tuple[float, float, float]] = {}
            fixed_camera_rot: tuple[float, float, float, float] | None = None
            if fixed_camera_pos is not None:
                if fixed_camera_target is None:
                    raise ValueError("--diagnostic_fixed_camera_target is required with --diagnostic_fixed_camera_pos")
                center_pos = tuple(float(v) for v in fixed_camera_pos)
                target_pos = tuple(float(v) for v in fixed_camera_target)
                fixed_camera_convention = str(args_cli.diagnostic_fixed_camera_convention)
                fixed_camera_rot = (
                    _ros_camera_look_at_quat_wxyz(center_pos, target_pos)
                    if fixed_camera_convention == "ros"
                    else _opengl_camera_look_at_quat_wxyz(center_pos, target_pos)
                )
                forward = _normalize3(
                    (
                        target_pos[0] - center_pos[0],
                        target_pos[1] - center_pos[1],
                        target_pos[2] - center_pos[2],
                    )
                )
                side = _normalize3(_cross3((0.0, 0.0, 1.0), forward))
                side_offset = float(args_cli.diagnostic_fixed_camera_side_offset_m)
                fixed_camera_positions = {
                    "center_camera": center_pos,
                    "left_camera": tuple(center_pos[i] + side_offset * side[i] for i in range(3)),
                    "right_camera": tuple(center_pos[i] - side_offset * side[i] for i in range(3)),
                }
            for camera_name in ("center_camera", "left_camera", "right_camera"):
                camera_cfg = getattr(env_cfg.scene, camera_name, None)
                if camera_cfg is None:
                    continue
                if fixed_camera_pos is not None:
                    camera_cfg.prim_path = f"{{ENV_REGEX_NS}}/diagnostic_{camera_name}"
                    camera_cfg.offset.pos = fixed_camera_positions[camera_name]
                    camera_cfg.offset.rot = fixed_camera_rot
                    camera_cfg.offset.convention = fixed_camera_convention
                if camera_width > 0:
                    camera_cfg.width = camera_width
                if camera_height > 0:
                    camera_cfg.height = camera_height
                if not math.isnan(camera_focal_length):
                    camera_cfg.spawn.focal_length = camera_focal_length
                if not math.isnan(camera_horizontal_aperture):
                    camera_cfg.spawn.horizontal_aperture = camera_horizontal_aperture
                if not math.isnan(camera_vertical_aperture):
                    camera_cfg.spawn.vertical_aperture = camera_vertical_aperture
                if camera_offset_pos is not None:
                    camera_cfg.offset.pos = tuple(float(v) for v in camera_offset_pos)
                if camera_offset_rot is not None:
                    camera_cfg.offset.rot = tuple(float(v) for v in camera_offset_rot)
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.center_rgb = None
            env_cfg.observations.policy.left_rgb = None
            env_cfg.observations.policy.right_rgb = None
        if bool(args_cli.absolute_ik_target_pose):
            env_cfg.actions.arm_action.controller.use_relative_mode = False
            env_cfg.actions.arm_action.scale = 1.0
        else:
            env_cfg.actions.arm_action.scale = float(args_cli.isaac_action_scale)
        _configure_semantic_reward_terms(env_cfg)
        reset_config = _configure_near_gate_reset(env_cfg)

        status_path.write_text(json.dumps({"stage": "gym_make"}, indent=2) + "\n", encoding="utf-8")
        env = gym.make(args_cli.task, cfg=env_cfg)
        sfp_body_collision_replacement_report = _replace_sfp_body_sdf_collision_with_sdf_boxes(run_dir)
        nic_cage_p0_replacement_report = _replace_nic_cage_p0_with_sdf_boxes(run_dir)
        collision_toggle_report = _disable_matching_collision_prims(run_dir)
        collision_contact_tuning_report = _tune_matching_collision_contact_offsets(run_dir)
        collision_material_tuning_report = _tune_matching_collision_materials(run_dir)
        rows: list[dict[str, Any]] = []
        status_path.write_text(json.dumps({"stage": "reset"}, indent=2) + "\n", encoding="utf-8")
        env.reset(seed=int(args_cli.seed))
        pose_hold_pos = _body_position(env, str(args_cli.pose_hold_body))
        pose_hold_quat = _body_orientation(env, str(args_cli.pose_hold_body))
        if pose_hold_pos is not None and pose_hold_quat is not None:
            setattr(env.unwrapped, "_aic_pose_hold_target_pos_w", pose_hold_pos.detach().clone())
            setattr(env.unwrapped, "_aic_pose_hold_target_quat_w", pose_hold_quat.detach().clone())
        pinned_wrist_quat = _body_orientation(env, "wrist_3_link")
        if pinned_wrist_quat is not None:
            setattr(env.unwrapped, "_aic_absolute_ik_pinned_wrist_quat_w", pinned_wrist_quat.detach().clone())
        pinned_wrist_pos = _body_position(env, "wrist_3_link")
        if pinned_wrist_pos is not None:
            setattr(env.unwrapped, "_aic_pinned_wrist_descent_pos_w", pinned_wrist_pos.detach().clone())
        target_tip_pos = _body_position(env, "sfp_tip_link")
        if target_tip_pos is not None:
            setattr(env.unwrapped, "_aic_target_tip_stabilize_pos_w", target_tip_pos.detach().clone())
        target_module_pos = _body_position(env, str(args_cli.target_module_stabilize_body))
        if target_module_pos is not None:
            setattr(env.unwrapped, "_aic_target_module_stabilize_pos_w", target_module_pos.detach().clone())
        reset_diagnostic = _reset_diagnostic(env)
        (run_dir / "reset_diagnostic.json").write_text(
            json.dumps(_jsonable(reset_diagnostic), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_path = run_dir / "metrics.jsonl"
        stopped_early_reason = None
        if bool(args_cli.save_initial_metrics):
            initial_record = _step_record(
                env,
                step=0,
                phase="initial",
                action=_zero_action(env),
                action_info={"probe": "initial_after_reset"},
                before_positions=_body_positions(env),
            )
            if args_cli.save_images and bool(args_cli.save_initial_images):
                try:
                    initial_record["saved_images"] = _save_step_images(
                        env,
                        run_dir=run_dir,
                        step=0,
                        record=initial_record,
                    )
                except Exception as exc:
                    initial_record["image_save_error"] = f"{type(exc).__name__}: {exc}"
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(initial_record), sort_keys=True) + "\n")
        total_steps = int(args_cli.approach_steps) + int(args_cli.probe_steps)
        for step in range(1, total_steps + 1):
            status_path.write_text(
                json.dumps({"stage": "step", "step": step, "total_steps": total_steps}, indent=2) + "\n",
                encoding="utf-8",
            )
            phase = "approach" if step <= int(args_cli.approach_steps) else "probe"
            probe_step = max(0, step - int(args_cli.approach_steps) - 1)
            before_positions = _body_positions(env)
            if phase == "approach":
                action, action_info = _approach_action(env)
            else:
                action, action_info = _probe_action(env, probe_step)
            env.step(action)
            record = _step_record(
                env,
                step=step,
                phase=phase,
                action=action,
                action_info=action_info,
                before_positions=before_positions,
            )
            rows.append(record)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            if args_cli.save_images and step % max(int(args_cli.image_log_every), 1) == 0 and step <= int(args_cli.max_logged_image_steps):
                try:
                    record["saved_images"] = _save_step_images(env, run_dir=run_dir, step=step, record=record)
                except Exception as exc:
                    record["image_save_error"] = f"{type(exc).__name__}: {exc}"
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(_jsonable({"step": step, "image_save_error": record["image_save_error"]}), sort_keys=True) + "\n")
            if _near_success_capture_any(record):
                stopped_early_reason = "near_success_capture"
                status_path.write_text(
                    json.dumps(
                        {
                            "stage": "stopped_early",
                            "reason": stopped_early_reason,
                            "step": step,
                            "total_steps": total_steps,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                break
        video_summary = {"enabled": False, "videos": [], "warnings": []}
        if args_cli.save_images and bool(args_cli.save_videos):
            video_summary = _encode_step_videos(run_dir)
            (run_dir / "video_summary.json").write_text(
                json.dumps(_jsonable(video_summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        summary = {
            "run_dir": str(run_dir),
            "strict_success": any(_strict_any(row) for row in rows),
            "best_row": _best_row(rows),
            "final_row": rows[-1] if rows else None,
            "reset_diagnostic": reset_diagnostic,
            "sfp_body_collision_replacement_report": sfp_body_collision_replacement_report,
            "nic_cage_p0_replacement_report": nic_cage_p0_replacement_report,
            "collision_toggle_report": collision_toggle_report,
            "collision_contact_tuning_report": collision_contact_tuning_report,
            "collision_material_tuning_report": collision_material_tuning_report,
            "near_gate_reset_config": reset_config,
            "video_summary": video_summary,
            "phase": "complete",
            "stopped_early_reason": stopped_early_reason,
            "near_success_capture_thresholds": {
                "enabled": bool(args_cli.stop_on_near_success_capture),
                "min_s_m": float(args_cli.near_success_capture_min_s_m),
                "max_r_m": float(args_cli.near_success_capture_max_r_m),
                "max_theta_rad": float(args_cli.near_success_capture_max_theta_rad),
                "min_module_consistency": float(args_cli.near_success_capture_min_module_consistency),
            },
            "interpretation": "direct_wrist_ik_contact_realization_probe",
        }
        (run_dir / "wrist_contact_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary_md(run_dir, summary)
        status_path.write_text(json.dumps({"stage": "complete"}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "strict_success": summary["strict_success"], "best_row": summary["best_row"]}, indent=2))
    except Exception as exc:
        error = {
            "stage": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        status_path.write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        (run_dir / "error.json").write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(error, indent=2), flush=True)
        return 1
    finally:
        if env is not None:
            env.close()
    return 0


def _write_summary_md(run_dir: Path, summary: dict[str, Any]) -> None:
    best = summary.get("best_row") or {}
    lines = [
        "# Wrist/contact realization diagnostic",
        "",
        f"Run: `{run_dir}`",
        f"Strict success: `{str(bool(summary.get('strict_success'))).lower()}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| best step | {best.get('step')} |",
        f"| best env | {best.get('env')} |",
        f"| best s mm | {1000.0 * float(best.get('s_m', float('nan'))):.3f} |",
        f"| best r mm | {1000.0 * float(best.get('r_m', float('nan'))):.3f} |",
        f"| best theta rad | {float(best.get('theta_rad', float('nan'))):.5f} |",
        f"| best consistency | {float(best.get('module_consistency', float('nan'))):.3f} |",
        "",
        "## Videos",
        "",
    ]
    video_summary = summary.get("video_summary") or {}
    videos = list(video_summary.get("videos") or [])
    if videos:
        lines.extend(f"- `{path}`" for path in videos)
    elif video_summary.get("enabled"):
        lines.append("- Video encoding was enabled, but no MP4 files were produced.")
    else:
        lines.append("- Video encoding disabled.")
    warnings = list(video_summary.get("warnings") or [])
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend([
        "",
        "This run bypasses ACT/SERL and sends scripted wrist IK commands through `env.step`.",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()

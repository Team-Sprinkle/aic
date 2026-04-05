"""Alignment helpers for comparing the training env to evaluation assumptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentSpec:
    """Minimal set of invariants compared against official evaluation docs."""

    world_frame: str = "world"
    robot_name: str = "ur5e"
    robot_base_frame: str = "base_link"
    tcp_frame: str = "gripper/tcp"
    training_action_semantics: str = "joint_position_delta"
    eval_action_semantics: str = "joint_or_cartesian_target"
    success_mode: str = "object_to_target_distance"


def build_alignment_report() -> dict[str, object]:
    """Build a package-local report describing alignment and known gaps."""
    spec = AlignmentSpec()
    return {
        "world_frame": spec.world_frame,
        "robot_name": spec.robot_name,
        "robot_base_frame": spec.robot_base_frame,
        "tcp_frame": spec.tcp_frame,
        "training_action_semantics": spec.training_action_semantics,
        "eval_action_semantics": spec.eval_action_semantics,
        "success_mode": spec.success_mode,
        "differences_vs_eval": [
            "training env uses privileged observations; eval policies receive ROS observations assembled by aic_adapter",
            "training env action space is joint_position_delta; eval controller accepts joint or cartesian targets",
            "training env success is simplified to object-to-target distance; eval task is cable insertion with scoring logic",
            "training env omits cameras and force-torque sensing in the observation path",
        ],
    }


def check_alignment_invariants() -> dict[str, object]:
    """Return key invariants that should remain stable for policy transfer."""
    report = build_alignment_report()
    return {
        "world_frame_matches_eval": report["world_frame"] == "world",
        "robot_name_matches_eval": report["robot_name"] == "ur5e",
        "tcp_frame_matches_eval_docs": report["tcp_frame"] == "gripper/tcp",
        "base_frame_matches_eval_docs": report["robot_base_frame"] == "base_link",
        "action_semantics_documented_difference": (
            report["training_action_semantics"] != report["eval_action_semantics"]
        ),
        "success_criterion_documented": report["success_mode"] == "object_to_target_distance",
    }

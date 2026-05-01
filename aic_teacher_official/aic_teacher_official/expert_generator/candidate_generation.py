"""Candidate staging pose generation for MoveIt free-space planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot, SerializablePose
from aic_teacher_official.expert_generator.vlm_strategy import VLMStrategy
from transforms3d._gohlketransforms import quaternion_multiply


@dataclass(frozen=True)
class ApproachCandidate:
    index: int
    name: str
    safe_lift_pose: SerializablePose
    approach_standoff_pose: SerializablePose
    pre_insert_pose: SerializablePose
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "safe_lift_pose": self.safe_lift_pose.to_dict(),
            "approach_standoff_pose": self.approach_standoff_pose.to_dict(),
            "pre_insert_pose": self.pre_insert_pose.to_dict(),
            "metadata": dict(self.metadata),
        }


DEFAULT_CANDIDATE_ORDER = [
    "above",
    "above_left",
    "above_right",
    "high_clearance_vertical",
    "front",
]
SIDE_OFFSETS = {
    "above": (0.0, 0.0),
    "above_left": (-0.04, 0.04),
    "above_right": (0.04, 0.04),
    "high_clearance_vertical": (0.0, 0.0),
    "front": (0.0, -0.05),
    "back": (0.0, 0.05),
}


def _unique_candidate_names(preferred: str, count: int) -> list[str]:
    ordered = [preferred] + [name for name in DEFAULT_CANDIDATE_ORDER if name != preferred]
    return ordered[: max(0, count)]


def _pose_at(base: SerializablePose, *, x: float, y: float, z: float, orientation_xyzw: Iterable[float]) -> SerializablePose:
    return SerializablePose(
        position=[float(x), float(y), float(z)],
        orientation_xyzw=[float(v) for v in orientation_xyzw],
        frame_id=base.frame_id,
    )


def generate_approach_candidates(
    snapshot: SceneSnapshot,
    strategy: VLMStrategy,
    *,
    count: int = 5,
    pre_insert_z_offset_m: float = 0.20,
) -> list[ApproachCandidate]:
    """Generate symbolic candidate poses.

    Position candidates are deterministic and conservative. Orientation comes
    from the target-port pose, which mirrors the CheatCode-style "align plug to
    port geometry" contract without asking the VLM for orientation.
    """

    if snapshot.target_port_pose is None:
        raise ValueError("target_port_pose is required to generate approach candidates")
    if snapshot.tcp_pose is None:
        raise ValueError("tcp_pose is required to generate approach candidates")
    target = snapshot.target_port_pose
    current = snapshot.tcp_pose
    cheatcode_pre_insert = _cheatcode_gripper_target(snapshot, z_offset=pre_insert_z_offset_m)
    orientation = list(cheatcode_pre_insert.orientation_xyzw)
    clearance = float(strategy.preferred_clearance_m)
    names = _unique_candidate_names(strategy.approach_side, count)
    candidates: list[ApproachCandidate] = []
    for idx, name in enumerate(names):
        dx, dy = SIDE_OFFSETS.get(name, (0.0, 0.0))
        extra_clearance = 0.06 if name == "high_clearance_vertical" else 0.0
        staging_z = cheatcode_pre_insert.position[2]
        lift_z = max(current.position[2], staging_z + extra_clearance)
        safe_lift = _pose_at(
            cheatcode_pre_insert,
            x=current.position[0],
            y=current.position[1],
            z=lift_z,
            orientation_xyzw=orientation,
        )
        approach = _pose_at(
            cheatcode_pre_insert,
            x=cheatcode_pre_insert.position[0] + dx,
            y=cheatcode_pre_insert.position[1] + dy,
            z=staging_z + clearance + extra_clearance,
            orientation_xyzw=orientation,
        )
        pre_insert = _pose_at(
            cheatcode_pre_insert,
            x=cheatcode_pre_insert.position[0],
            y=cheatcode_pre_insert.position[1],
            z=staging_z,
            orientation_xyzw=orientation,
        )
        candidates.append(
            ApproachCandidate(
                index=idx,
                name=name,
                safe_lift_pose=safe_lift,
                approach_standoff_pose=approach,
                pre_insert_pose=pre_insert,
                metadata={
                    "orientation_source": "cheatcode_style_target_port_geometry",
                    "vlm_preferred": name == strategy.approach_side,
                    "pre_insert_z_offset_m": pre_insert_z_offset_m,
                    "pre_insert_pose_source": "cheatcode_calc_gripper_pose",
                },
            )
        )
    return candidates


def _cheatcode_gripper_target(snapshot: SceneSnapshot, *, z_offset: float) -> SerializablePose:
    target = snapshot.target_port_pose
    current = snapshot.tcp_pose
    if target is None or current is None:
        raise ValueError("target_port_pose and tcp_pose are required")
    if snapshot.plug_pose is None:
        return _pose_at(
            target,
            x=target.position[0],
            y=target.position[1],
            z=target.position[2] + z_offset,
            orientation_xyzw=target.orientation_xyzw,
        )

    q_port = _wxyz(target.orientation_xyzw)
    q_plug = _wxyz(snapshot.plug_pose.orientation_xyzw)
    q_gripper = _wxyz(current.orientation_xyzw)
    q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
    q_diff = quaternion_multiply(q_port, q_plug_inv)
    q_gripper_target = quaternion_multiply(q_diff, q_gripper)
    plug_tip_gripper_offset_z = current.position[2] - snapshot.plug_pose.position[2]
    return _pose_at(
        target,
        x=target.position[0],
        y=target.position[1],
        z=target.position[2] + z_offset - plug_tip_gripper_offset_z,
        orientation_xyzw=_xyzw(q_gripper_target),
    )


def _wxyz(q_xyzw: Iterable[float]) -> tuple[float, float, float, float]:
    q = [float(v) for v in q_xyzw]
    return (q[3], q[0], q[1], q[2])


def _xyzw(q_wxyz: Iterable[float]) -> list[float]:
    q = [float(v) for v in q_wxyz]
    return [q[1], q[2], q[3], q[0]]

"""Micro-probe library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..runtime import RuntimeState
from ..teacher.history import TemporalObservationBuffer
from ..teacher.types import ProbeObservationSummary, ProbeResult


@dataclass(frozen=True)
class ProbePrimitive:
    name: str
    actions: tuple[np.ndarray, ...]


class ProbeLibrary:
    def __init__(self) -> None:
        self._probes = {
            "hold_settle": ProbePrimitive(
                name="hold_settle",
                actions=tuple(np.zeros(6, dtype=np.float32) for _ in range(3)),
            ),
            "micro_sweep_xy": ProbePrimitive(
                name="micro_sweep_xy",
                actions=(
                    np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([-0.015, 0.015, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, -0.015, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ),
            ),
            "yaw_wiggle": ProbePrimitive(
                name="yaw_wiggle",
                actions=(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.45], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.45], dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                ),
            ),
            "lift_and_hold": ProbePrimitive(
                name="lift_and_hold",
                actions=(
                    np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                ),
            ),
        }

    def list_probe_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._probes.keys()))

    def actions_for(self, probe_name: str) -> tuple[np.ndarray, ...]:
        if probe_name not in self._probes:
            raise KeyError(f"Unknown probe {probe_name!r}.")
        return tuple(action.copy() for action in self._probes[probe_name].actions)

    def summarize_result(
        self,
        *,
        probe_name: str,
        before_state: RuntimeState,
        after_state: RuntimeState,
        before_summary: TemporalObservationBuffer,
        after_summary: TemporalObservationBuffer,
        action_count: int,
    ) -> ProbeResult:
        before_dynamics = before_summary.dynamics_summary()
        after_dynamics = after_summary.dynamics_summary()
        return ProbeResult(
            probe_name=probe_name,
            before=self._observation_summary(before_state),
            after=self._observation_summary(after_state),
            duration_s=float(after_state.sim_time - before_state.sim_time),
            action_count=action_count,
            plug_relative_motion=float(np.linalg.norm(after_state.plug_pose[:3] - before_state.plug_pose[:3])),
            peak_force=float(max(np.linalg.norm(before_state.wrench[:3]), np.linalg.norm(after_state.wrench[:3]))),
            settling_delta=float(after_dynamics.cable_settling_score - before_dynamics.cable_settling_score),
            notes=(
                f"quasi_static_before={before_dynamics.quasi_static} "
                f"quasi_static_after={after_dynamics.quasi_static}"
            ),
        )

    def _observation_summary(self, state: RuntimeState) -> ProbeObservationSummary:
        return ProbeObservationSummary(
            sim_time=float(state.sim_time),
            tcp_position_xyz=tuple(float(value) for value in state.tcp_pose[:3]),
            plug_position_xyz=tuple(float(value) for value in state.plug_pose[:3]),
            wrench_force_xyz=tuple(float(value) for value in state.wrench[:3]),
            contact=bool(state.off_limit_contact),
        )

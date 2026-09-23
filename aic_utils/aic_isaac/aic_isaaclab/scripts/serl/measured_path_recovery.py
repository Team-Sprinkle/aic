"""Observation-side measured-path backtracking for contact recovery.

The controller owns only the immediate escape from a detected obstruction.
After clearance, the learned policy regains control and its translational
proposal is projected into a cone near the plane perpendicular to the blocked
direction.  No port or object geometry is consumed here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import torch


class RecoveryMode(IntEnum):
    POLICY = 0
    BACKTRACK = 1
    LATERAL = 2
    ABORT = 3


@dataclass(frozen=True)
class MeasuredPathRecoveryConfig:
    history_steps: int = 40
    history_spacing_m: float = 0.00005
    trigger_force_n: float = 8.0
    clear_force_n: float = 4.0
    trigger_consecutive_steps: int = 2
    command_motion_min_m: float = 0.00005
    realized_motion_max_m: float = 0.00015
    min_clearance_m: float = 0.0010
    max_clearance_m: float = 0.0100
    backtrack_step_m: float = 0.00025
    force_increase_abort_n: float = 4.0
    abort_hold_steps: int = 4
    lateral_policy_steps: int = 12
    lateral_toward_fraction: float = 0.10
    lateral_away_fraction: float = 0.50

    def validate(self) -> None:
        if self.history_steps < 2:
            raise ValueError("history_steps must be at least 2")
        if self.trigger_force_n <= self.clear_force_n:
            raise ValueError("trigger_force_n must exceed clear_force_n for hysteresis")
        if self.trigger_consecutive_steps < 1:
            raise ValueError("trigger_consecutive_steps must be positive")
        if not 0.0 < self.min_clearance_m <= self.max_clearance_m:
            raise ValueError("clearance distances must satisfy 0 < min <= max")
        if self.backtrack_step_m <= 0.0:
            raise ValueError("backtrack_step_m must be positive")
        if not 0.0 <= self.lateral_toward_fraction <= 1.0:
            raise ValueError("lateral_toward_fraction must be in [0, 1]")
        if not 0.0 <= self.lateral_away_fraction <= 1.0:
            raise ValueError("lateral_away_fraction must be in [0, 1]")


def _quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with unit wxyz quaternions."""
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def project_near_lateral(
    world_delta: torch.Tensor,
    blocked_direction_world: torch.Tensor,
    *,
    toward_fraction: float,
    away_fraction: float,
) -> torch.Tensor:
    """Keep lateral motion and bounded axial motion relative to obstruction."""
    direction = blocked_direction_world / blocked_direction_world.norm().clamp_min(1.0e-9)
    axial = torch.dot(world_delta, direction)
    lateral = world_delta - axial * direction
    # Positive axial repeats the blocked direction; negative axial moves away.
    if float(axial) >= 0.0:
        kept_axial = axial * float(toward_fraction)
    else:
        kept_axial = axial * float(away_fraction)
    return lateral + kept_axial * direction


class _SingleRecovery:
    def __init__(self, config: MeasuredPathRecoveryConfig):
        self.config = config
        self.history: deque[torch.Tensor] = deque(maxlen=config.history_steps)
        self.mode = RecoveryMode.POLICY
        self.force_trigger_count = 0
        self.lateral_steps_remaining = 0
        self.retreat_origin: torch.Tensor | None = None
        self.blocked_direction_world: torch.Tensor | None = None
        self.previous_position: torch.Tensor | None = None
        self.previous_quaternion: torch.Tensor | None = None
        self.previous_command_body: torch.Tensor | None = None
        self.trigger_force = 0.0
        self.previous_force = 0.0
        self.backtrack_events = 0
        self.clear_events = 0
        self.abort_events = 0
        self.abort_steps_remaining = 0

    def reset(self) -> None:
        self.__init__(self.config)

    def _record_position(self, position: torch.Tensor) -> None:
        point = position.detach().clone()
        if not self.history or torch.linalg.norm(point - self.history[-1]) >= self.config.history_spacing_m:
            self.history.append(point)

    def _previous_command_world(self) -> torch.Tensor | None:
        if self.previous_command_body is None or self.previous_quaternion is None:
            return None
        return _quat_apply_wxyz(self.previous_quaternion, self.previous_command_body[:3])

    def _blocked(self, position: torch.Tensor, force_n: float) -> bool:
        stalled = False
        command_world = self._previous_command_world()
        if command_world is not None and self.previous_position is not None:
            command_norm = float(torch.linalg.norm(command_world))
            realized_norm = float(torch.linalg.norm(position - self.previous_position))
            stalled = (
                command_norm >= self.config.command_motion_min_m
                and realized_norm <= self.config.realized_motion_max_m
                and force_n >= self.config.trigger_force_n
            )
        # Force by itself is ambiguous during insertion: a correctly aligned
        # connector can briefly carry high load while still moving. Recovery
        # requires the conjunction of load and missing measured progress.
        if stalled:
            self.force_trigger_count += 1
        else:
            self.force_trigger_count = 0
        return self.force_trigger_count >= self.config.trigger_consecutive_steps

    def _start_backtrack(self, position: torch.Tensor, force_n: float) -> None:
        command_world = self._previous_command_world()
        if command_world is None or float(torch.linalg.norm(command_world)) < 1.0e-9:
            if len(self.history) >= 2:
                command_world = self.history[-1] - self.history[-2]
            else:
                command_world = torch.tensor([0.0, 0.0, 1.0], device=position.device, dtype=position.dtype)
        self.blocked_direction_world = command_world / command_world.norm().clamp_min(1.0e-9)
        self.retreat_origin = position.detach().clone()
        self.trigger_force = float(force_n)
        self.previous_force = float(force_n)
        self.mode = RecoveryMode.BACKTRACK
        self.force_trigger_count = 0
        self.backtrack_events += 1

    def _backtrack_world_delta(self, position: torch.Tensor) -> torch.Tensor:
        while self.history and torch.linalg.norm(self.history[-1] - position) < self.config.history_spacing_m:
            self.history.pop()
        if self.history:
            desired = self.history[-1] - position
            if torch.dot(desired, self.blocked_direction_world) > 0.0:
                self.history.pop()
                return self._backtrack_world_delta(position)
        else:
            desired = -self.blocked_direction_world * self.config.backtrack_step_m
        norm = desired.norm().clamp_min(1.0e-9)
        return desired * min(1.0, self.config.backtrack_step_m / float(norm))

    def apply(
        self,
        *,
        position_world: torch.Tensor,
        quaternion_world_wxyz: torch.Tensor,
        force_n: float,
        proposed_action_body: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        position = position_world.detach()
        quaternion = quaternion_world_wxyz.detach()
        self._record_position(position)

        if self.mode == RecoveryMode.POLICY and self._blocked(position, float(force_n)):
            self._start_backtrack(position, float(force_n))

        actor_owned = self.mode in {RecoveryMode.POLICY, RecoveryMode.LATERAL}
        action = proposed_action_body.clone()
        retreat_distance = 0.0
        if self.retreat_origin is not None:
            retreat_distance = float(torch.linalg.norm(position - self.retreat_origin))

        if self.mode == RecoveryMode.BACKTRACK:
            if float(force_n) > self.previous_force + self.config.force_increase_abort_n:
                self.mode = RecoveryMode.ABORT
                self.abort_events += 1
                self.abort_steps_remaining = self.config.abort_hold_steps
                action.zero_()
                actor_owned = False
            elif retreat_distance >= self.config.max_clearance_m:
                self.mode = RecoveryMode.ABORT
                self.abort_events += 1
                self.abort_steps_remaining = self.config.abort_hold_steps
                action.zero_()
                actor_owned = False
            elif retreat_distance >= self.config.min_clearance_m and float(force_n) <= self.config.clear_force_n:
                self.mode = RecoveryMode.LATERAL
                self.lateral_steps_remaining = self.config.lateral_policy_steps
                self.clear_events += 1
                actor_owned = True
            else:
                world_delta = self._backtrack_world_delta(position)
                action.zero_()
                action[:3] = _quat_apply_wxyz(_quat_conjugate_wxyz(quaternion), world_delta)
                actor_owned = False

        if self.mode == RecoveryMode.LATERAL:
            world_delta = _quat_apply_wxyz(quaternion, proposed_action_body[:3])
            world_delta = project_near_lateral(
                world_delta,
                self.blocked_direction_world,
                toward_fraction=self.config.lateral_toward_fraction,
                away_fraction=self.config.lateral_away_fraction,
            )
            action[:3] = _quat_apply_wxyz(_quat_conjugate_wxyz(quaternion), world_delta)
            self.lateral_steps_remaining -= 1
            if self.lateral_steps_remaining <= 0:
                self.mode = RecoveryMode.POLICY

        if self.mode == RecoveryMode.ABORT:
            action.zero_()
            actor_owned = False
            self.abort_steps_remaining -= 1
            if self.abort_steps_remaining <= 0:
                # A safety stop is temporary. Continuing to hold until timeout
                # made recovery impossible when cable preload kept force above
                # the nominal clear threshold.
                self.mode = RecoveryMode.LATERAL
                self.lateral_steps_remaining = self.config.lateral_policy_steps

        self.previous_position = position.clone()
        self.previous_quaternion = quaternion.clone()
        self.previous_command_body = action.detach().clone()
        self.previous_force = float(force_n)
        metrics = {
            "mode": int(self.mode),
            "mode_name": self.mode.name.lower(),
            "actor_owned": bool(actor_owned),
            "retreat_distance_m": retreat_distance,
            "force_n": float(force_n),
            "backtrack_events": self.backtrack_events,
            "clear_events": self.clear_events,
            "abort_events": self.abort_events,
            "blocked_direction_world": None
            if self.blocked_direction_world is None
            else [float(v) for v in self.blocked_direction_world.detach().cpu()],
        }
        return action, actor_owned, metrics


class MeasuredPathRecovery:
    """Independent measured-path recovery state for every vectorized env."""

    def __init__(self, num_envs: int, config: MeasuredPathRecoveryConfig):
        config.validate()
        self.config = config
        self.controllers = [_SingleRecovery(config) for _ in range(int(num_envs))]

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        if reset_mask is None:
            for controller in self.controllers:
                controller.reset()
            return
        flat = reset_mask.detach().cpu().bool().reshape(-1)
        for index, reset in enumerate(flat):
            if bool(reset):
                self.controllers[index].reset()

    def apply(
        self,
        *,
        position_world: torch.Tensor,
        quaternion_world_wxyz: torch.Tensor,
        force_n: torch.Tensor,
        proposed_action_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        if position_world.shape[0] != len(self.controllers):
            raise ValueError("position batch does not match recovery environment count")
        output = proposed_action_body.clone()
        owner = torch.ones((len(self.controllers), 1), dtype=torch.bool, device=output.device)
        metrics = []
        for index, controller in enumerate(self.controllers):
            output[index], actor_owned, row = controller.apply(
                position_world=position_world[index],
                quaternion_world_wxyz=quaternion_world_wxyz[index],
                force_n=float(force_n[index].reshape(-1)[0]),
                proposed_action_body=proposed_action_body[index],
            )
            owner[index, 0] = actor_owned
            metrics.append(row)
        return output, owner, metrics

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

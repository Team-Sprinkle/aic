"""Reset-time randomization aligned to the official AIC scene schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .scenario import (
    AicScenario,
    CableScenario,
    RailEntity,
    TaskBoardScenario,
    TaskDefinition,
    load_trials,
)


@dataclass(frozen=True)
class TranslationLimits:
    nic_rail: tuple[float, float] = (-0.0215, 0.0234)
    sc_rail: tuple[float, float] = (-0.06, 0.055)
    mount_rail: tuple[float, float] = (-0.09425, 0.09425)


@dataclass(frozen=True)
class EvalLikeRandomizationProfile:
    board_x: tuple[float, float] = (0.12, 0.18)
    board_y: tuple[float, float] = (-0.24, -0.12)
    board_yaw: tuple[float, float] = (2.95, 3.20)
    max_board_offset_xy: float = 0.10
    nic_roll: tuple[float, float] = (0.0, 0.0)
    nic_pitch: tuple[float, float] = (0.0, 0.0)
    nic_yaw: tuple[float, float] = (-math.radians(10.0), math.radians(10.0))
    nic_extra_present_prob: float = 0.25
    sc_roll: tuple[float, float] = (0.0, 0.0)
    sc_pitch: tuple[float, float] = (0.0, 0.0)
    sc_yaw: tuple[float, float] = (0.0, 0.0)
    sc_extra_present_prob: float = 0.25
    mount_roll: tuple[float, float] = (0.0, 0.0)
    mount_pitch: tuple[float, float] = (0.0, 0.0)
    mount_yaw: tuple[float, float] = (0.0, 0.0)
    mount_present_prob: float = 0.75
    cable_roll_jitter: float = 0.04
    cable_pitch_jitter: float = 0.04
    cable_yaw_jitter: float = 0.04
    cable_offset_x_jitter: float = 0.002
    cable_offset_y_jitter: float = 0.002
    cable_offset_z_jitter: float = 0.002


class AicEnvRandomizer:
    """Samples scenario variants while preserving official scene semantics."""

    def __init__(
        self,
        *,
        enable_randomization: bool = True,
        translation_limits: TranslationLimits | None = None,
        profile: EvalLikeRandomizationProfile | None = None,
    ) -> None:
        self._trials = load_trials()
        self._trial_ids = tuple(sorted(self._trials))
        self._enable_randomization = enable_randomization
        self._limits = translation_limits or TranslationLimits()
        self._profile = profile or EvalLikeRandomizationProfile()

    @property
    def trial_ids(self) -> tuple[str, ...]:
        return self._trial_ids

    def sample(self, *, seed: int | None = None, trial_id: str | None = None) -> AicScenario:
        rng = random.Random(seed)
        selected_trial_id = trial_id or rng.choice(self._trial_ids)
        source = self._trials[selected_trial_id]
        if not self._enable_randomization:
            return source
        return self._sample_randomized_scenario(source=source, rng=rng, seed=seed)

    def _sample_randomized_scenario(
        self,
        *,
        source: AicScenario,
        rng: random.Random,
        seed: int | None,
    ) -> AicScenario:
        source_task = next(iter(source.tasks.values()))
        is_sfp_task = source_task.port_type == "sfp"
        target_index = rng.randint(0, 4 if is_sfp_task else 1)
        if is_sfp_task:
            task_definition = TaskDefinition(
                task_id=source_task.task_id,
                cable_type=source_task.cable_type,
                cable_name=source_task.cable_name,
                plug_type=source_task.plug_type,
                plug_name=source_task.plug_name,
                port_type=source_task.port_type,
                port_name=rng.choice(("sfp_port_0", "sfp_port_1")),
                target_module_name=f"nic_card_mount_{target_index}",
                time_limit_s=source_task.time_limit_s,
            )
        else:
            task_definition = TaskDefinition(
                task_id=source_task.task_id,
                cable_type=source_task.cable_type,
                cable_name=source_task.cable_name,
                plug_type=source_task.plug_type,
                plug_name=source_task.plug_name,
                port_type=source_task.port_type,
                port_name=source_task.port_name,
                target_module_name=f"sc_port_{target_index}",
                time_limit_s=source_task.time_limit_s,
            )
        return AicScenario(
            trial_id=source.trial_id,
            task_board=TaskBoardScenario(
                pose_xyz_rpy=self._randomize_board_pose(rng, nominal_pose=source.task_board.pose_xyz_rpy),
                nic_rails=self._randomize_nic_rails(
                    source=source.task_board.nic_rails,
                    rng=rng,
                    target_index=target_index if is_sfp_task else None,
                ),
                sc_rails=self._randomize_sc_rails(
                    source=source.task_board.sc_rails,
                    rng=rng,
                    target_index=target_index if not is_sfp_task else None,
                ),
                mount_rails=self._randomize_mount_rails(source=source.task_board.mount_rails, rng=rng),
            ),
            cables={name: self._randomize_cable(cable, rng=rng) for name, cable in source.cables.items()},
            tasks={task_definition.task_id: task_definition},
            metadata={
                **source.metadata,
                "randomized": True,
                "seed": seed,
                "source_trial_id": source.trial_id,
                "randomization_profile": "qualification_eval_like",
            },
        )

    def _randomize_board_pose(
        self,
        rng: random.Random,
        *,
        nominal_pose: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        nominal_x, nominal_y, nominal_z, nominal_roll, nominal_pitch, _ = nominal_pose
        board_x = nominal_x
        board_y = nominal_y
        for _ in range(128):
            candidate_x = rng.uniform(*self._profile.board_x)
            candidate_y = rng.uniform(*self._profile.board_y)
            if math.hypot(candidate_x - nominal_x, candidate_y - nominal_y) <= self._profile.max_board_offset_xy:
                board_x = candidate_x
                board_y = candidate_y
                break
        return (
            round(board_x, 6),
            round(board_y, 6),
            float(nominal_z),
            float(nominal_roll),
            float(nominal_pitch),
            round(rng.uniform(*self._profile.board_yaw), 6),
        )

    def _randomize_nic_rails(
        self,
        *,
        source: dict[str, RailEntity],
        rng: random.Random,
        target_index: int | None,
    ) -> dict[str, RailEntity]:
        randomized: dict[str, RailEntity] = {}
        for key, entity in source.items():
            index = self._index_from_key(key)
            must_present = target_index is not None and index == target_index
            randomized[key] = self._randomize_entity(
                entity,
                rng,
                self._limits.nic_rail,
                fallback_name=f"nic_card_{index}" if index >= 0 else entity.name,
                present=must_present or rng.random() < self._profile.nic_extra_present_prob,
                roll_range=self._profile.nic_roll,
                pitch_range=self._profile.nic_pitch,
                yaw_range=self._profile.nic_yaw,
            )
        return randomized

    def _randomize_sc_rails(
        self,
        *,
        source: dict[str, RailEntity],
        rng: random.Random,
        target_index: int | None,
    ) -> dict[str, RailEntity]:
        randomized: dict[str, RailEntity] = {}
        for key, entity in source.items():
            index = self._index_from_key(key)
            must_present = target_index is not None and index == target_index
            randomized[key] = self._randomize_entity(
                entity,
                rng,
                self._limits.sc_rail,
                fallback_name=f"sc_mount_{index}" if index >= 0 else entity.name,
                present=must_present or rng.random() < self._profile.sc_extra_present_prob,
                roll_range=self._profile.sc_roll,
                pitch_range=self._profile.sc_pitch,
                yaw_range=self._profile.sc_yaw,
            )
        return randomized

    def _randomize_mount_rails(
        self,
        *,
        source: dict[str, RailEntity],
        rng: random.Random,
    ) -> dict[str, RailEntity]:
        return {
            key: self._randomize_entity(
                entity,
                rng,
                self._limits.mount_rail,
                fallback_name=self._fallback_mount_name(key),
                present=rng.random() < self._profile.mount_present_prob,
                roll_range=self._profile.mount_roll,
                pitch_range=self._profile.mount_pitch,
                yaw_range=self._profile.mount_yaw,
            )
            for key, entity in source.items()
        }

    def _randomize_cable(self, cable: CableScenario, *, rng: random.Random) -> CableScenario:
        offset_x, offset_y, offset_z = cable.gripper_offset_xyz
        roll, pitch, yaw = cable.rpy
        return CableScenario(
            cable_name=cable.cable_name,
            cable_type=cable.cable_type,
            attach_to_gripper=cable.attach_to_gripper,
            spawn_pose_xyz=tuple(float(value) for value in cable.spawn_pose_xyz),
            gripper_offset_xyz=(
                round(offset_x + rng.uniform(-self._profile.cable_offset_x_jitter, self._profile.cable_offset_x_jitter), 6),
                round(offset_y + rng.uniform(-self._profile.cable_offset_y_jitter, self._profile.cable_offset_y_jitter), 6),
                round(offset_z + rng.uniform(-self._profile.cable_offset_z_jitter, self._profile.cable_offset_z_jitter), 6),
            ),
            rpy=(
                round(roll + rng.uniform(-self._profile.cable_roll_jitter, self._profile.cable_roll_jitter), 6),
                round(pitch + rng.uniform(-self._profile.cable_pitch_jitter, self._profile.cable_pitch_jitter), 6),
                round(yaw + rng.uniform(-self._profile.cable_yaw_jitter, self._profile.cable_yaw_jitter), 6),
            ),
        )

    @staticmethod
    def _index_from_key(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    @staticmethod
    def _fallback_mount_name(key: str) -> str | None:
        index = AicEnvRandomizer._index_from_key(key)
        if key.startswith("lc_mount_rail_") and index >= 0:
            return f"lc_mount_{index}"
        if key.startswith("sfp_mount_rail_") and index >= 0:
            return f"sfp_mount_{index}"
        if key.startswith("sc_mount_rail_") and index >= 0:
            return f"sc_mount_{index}"
        return None

    @staticmethod
    def _randomize_entity(
        entity: RailEntity,
        rng: random.Random,
        limits: tuple[float, float],
        *,
        fallback_name: str | None,
        present: bool,
        roll_range: tuple[float, float],
        pitch_range: tuple[float, float],
        yaw_range: tuple[float, float],
    ) -> RailEntity:
        if not present:
            return RailEntity(present=False, name=None)
        return RailEntity(
            present=True,
            name=entity.name or fallback_name,
            translation=round(rng.uniform(*limits), 6),
            roll=round(rng.uniform(*roll_range), 6),
            pitch=round(rng.uniform(*pitch_range), 6),
            yaw=round(rng.uniform(*yaw_range), 6),
        )

"""Reset-time randomization aligned to the official AIC scene schema."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .scenario import AicScenario, RailEntity, load_trials


@dataclass(frozen=True)
class TranslationLimits:
    nic_rail: tuple[float, float] = (-0.0215, 0.0234)
    sc_rail: tuple[float, float] = (-0.06, 0.055)
    mount_rail: tuple[float, float] = (-0.09425, 0.09425)


class AicEnvRandomizer:
    """Samples scenario variants while preserving official scene semantics."""

    def __init__(
        self,
        *,
        enable_randomization: bool = True,
        translation_limits: TranslationLimits | None = None,
    ) -> None:
        self._trials = load_trials()
        self._trial_ids = tuple(sorted(self._trials))
        self._enable_randomization = enable_randomization
        self._limits = translation_limits or TranslationLimits()

    @property
    def trial_ids(self) -> tuple[str, ...]:
        return self._trial_ids

    def sample(self, *, seed: int | None = None, trial_id: str | None = None) -> AicScenario:
        rng = random.Random(seed)
        source = self._trials[trial_id or self._trial_ids[0]]
        if not self._enable_randomization:
            return source
        return AicScenario(
            trial_id=source.trial_id,
            task_board=source.task_board.__class__(
                pose_xyz_rpy=source.task_board.pose_xyz_rpy,
                nic_rails={
                    key: self._randomize_entity(value, rng, self._limits.nic_rail)
                    for key, value in source.task_board.nic_rails.items()
                },
                sc_rails={
                    key: self._randomize_entity(value, rng, self._limits.sc_rail)
                    for key, value in source.task_board.sc_rails.items()
                },
                mount_rails={
                    key: self._randomize_entity(value, rng, self._limits.mount_rail)
                    for key, value in source.task_board.mount_rails.items()
                },
            ),
            cables=source.cables,
            tasks=source.tasks,
            metadata={
                **source.metadata,
                "randomized": True,
                "seed": seed,
            },
        )

    @staticmethod
    def _randomize_entity(
        entity: RailEntity,
        rng: random.Random,
        limits: tuple[float, float],
    ) -> RailEntity:
        if not entity.present:
            return entity
        return RailEntity(
            present=entity.present,
            name=entity.name,
            translation=round(rng.uniform(*limits), 6),
            roll=entity.roll,
            pitch=entity.pitch,
            yaw=entity.yaw,
        )

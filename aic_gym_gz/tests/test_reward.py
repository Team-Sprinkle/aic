from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.reward import AicRlRewardCalculator, AicScoreCalculator
from aic_gym_gz.runtime import RuntimeState


def _state(
    *,
    plug_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    entrance_xyz: tuple[float, float, float] = (0.0, 0.0, -0.05),
    action_like_velocity: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    force_z: float = 0.0,
    off_limit_contact: bool = False,
    orientation_error: float = 0.0,
    insertion_progress: float = 0.0,
    lateral_misalignment: float = 0.0,
    partial_insertion: bool = False,
) -> RuntimeState:
    plug_pose = np.array([*plug_xyz, 0.0, 0.0, orientation_error, 1.0], dtype=np.float64)
    target_pose = np.array([*target_xyz, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    entrance_pose = np.array([*entrance_xyz, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return RuntimeState(
        sim_tick=0,
        sim_time=0.0,
        joint_positions=np.zeros(6, dtype=np.float64),
        joint_velocities=np.zeros(6, dtype=np.float64),
        gripper_position=0.0,
        tcp_pose=np.zeros(7, dtype=np.float64),
        tcp_velocity=np.asarray(action_like_velocity, dtype=np.float64),
        plug_pose=plug_pose,
        target_port_pose=target_pose,
        target_port_entrance_pose=entrance_pose,
        wrench=np.array([0.0, 0.0, force_z, 0.0, 0.0, 0.0], dtype=np.float64),
        wrench_timestamp=0.0,
        off_limit_contact=off_limit_contact,
        insertion_event=None,
        controller_state={},
        score_geometry={
            "distance_to_target": float(np.linalg.norm(plug_pose[:3] - target_pose[:3])),
            "distance_to_entrance": float(np.linalg.norm(plug_pose[:3] - entrance_pose[:3])),
            "orientation_error": orientation_error,
            "insertion_progress": insertion_progress,
            "lateral_misalignment": lateral_misalignment,
            "partial_insertion": partial_insertion,
            "plug_to_port_depth": float(plug_pose[2] - target_pose[2]),
            "port_to_entrance_depth": float(entrance_pose[2] - target_pose[2]),
        },
    )


class RewardTest(unittest.TestCase):
    def test_potential_progress_rewards_are_positive_when_state_improves(self) -> None:
        calculator = AicRlRewardCalculator()
        previous_state = _state(
            plug_xyz=(0.05, 0.02, -0.08),
            orientation_error=0.3,
            insertion_progress=0.1,
            lateral_misalignment=0.02,
        )
        current_state = _state(
            plug_xyz=(0.02, 0.01, -0.03),
            action_like_velocity=(0.01, 0.0, 0.02, 0.0, 0.0, 0.0),
            orientation_error=0.1,
            insertion_progress=0.55,
            lateral_misalignment=0.005,
            partial_insertion=True,
        )
        previous_metrics = calculator.metrics_from_state(previous_state)
        current_metrics = calculator.metrics_from_state(current_state)
        breakdown = calculator.evaluate_step(
            previous_state=previous_state,
            current_state=current_state,
            action=np.array([0.01, 0.0, 0.02, 0.0, 0.0, 0.0], dtype=np.float64),
            previous_action=np.zeros(6, dtype=np.float64),
            previous_metrics=previous_metrics,
            current_metrics=current_metrics,
            success=False,
            wrong_port=False,
            distance_history=[
                previous_metrics.target_distance + 0.01,
                previous_metrics.target_distance,
                current_metrics.target_distance,
            ],
        )
        self.assertGreater(breakdown.target_progress_reward, 0.0)
        self.assertGreater(breakdown.entrance_progress_reward, 0.0)
        self.assertGreater(breakdown.corridor_progress_reward, 0.0)
        self.assertGreater(breakdown.orientation_progress_reward, 0.0)
        self.assertGreater(breakdown.total, 0.0)

    def test_local_smoothness_and_force_penalties_are_applied(self) -> None:
        calculator = AicRlRewardCalculator()
        previous_state = _state(plug_xyz=(0.02, 0.0, -0.02))
        current_state = _state(
            plug_xyz=(0.021, 0.0, -0.021),
            action_like_velocity=(0.4, 0.0, 0.0, 0.0, 0.0, 0.0),
            force_z=18.0,
        )
        previous_metrics = calculator.metrics_from_state(previous_state)
        current_metrics = calculator.metrics_from_state(current_state)
        breakdown = calculator.evaluate_step(
            previous_state=previous_state,
            current_state=current_state,
            action=np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            previous_action=np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            previous_metrics=previous_metrics,
            current_metrics=current_metrics,
            success=False,
            wrong_port=False,
            distance_history=[0.02, 0.019, 0.0205],
        )
        self.assertLess(breakdown.action_delta_penalty, 0.0)
        self.assertLess(breakdown.tcp_velocity_delta_penalty, 0.0)
        self.assertLess(breakdown.force_penalty, 0.0)
        self.assertLess(breakdown.oscillation_penalty, 0.0)

    def test_terminal_terms_are_explicit(self) -> None:
        calculator = AicRlRewardCalculator()
        previous_state = _state(plug_xyz=(0.01, 0.0, -0.01))
        current_state = _state(plug_xyz=(0.0, 0.0, 0.0), off_limit_contact=True)
        previous_metrics = calculator.metrics_from_state(previous_state)
        current_metrics = calculator.metrics_from_state(current_state)
        breakdown = calculator.evaluate_step(
            previous_state=previous_state,
            current_state=current_state,
            action=np.zeros(6, dtype=np.float64),
            previous_action=np.zeros(6, dtype=np.float64),
            previous_metrics=previous_metrics,
            current_metrics=current_metrics,
            success=True,
            wrong_port=False,
        )
        self.assertEqual(breakdown.success_bonus, calculator.weights.success_bonus)
        self.assertEqual(
            breakdown.off_limit_contact_penalty,
            calculator.weights.off_limit_contact_penalty,
        )

    def test_official_like_summary_has_expected_keys(self) -> None:
        calculator = AicScoreCalculator()
        episode = {
            "initial_distance": 0.2,
            "sim_time": [0.0, 0.1, 0.2],
            "tcp_positions": [np.zeros(3), np.array([0.01, 0.0, 0.0]), np.array([0.02, 0.0, 0.0])],
            "tcp_linear_velocity": [np.zeros(3), np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])],
            "distances": [0.2, 0.15, 0.1],
            "plug_positions": [np.array([0.2, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])],
            "target_port_pose": np.zeros(7),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]),
            "force_magnitudes": [0.0, 0.0, 0.0],
            "wrench_time": [0.0, 0.1, 0.2],
            "wrench_samples": [np.zeros(6), np.zeros(6), np.zeros(6)],
            "off_limit_contacts": [False, False, False],
            "success": False,
            "wrong_port": False,
        }
        summary = calculator.evaluate(episode)
        self.assertIn("duration", summary.tier2)
        self.assertIn("score", summary.tier3)
        self.assertIn("parity_notes", summary.__dict__)

    def test_partial_insertion_score_uses_port_entrance_geometry(self) -> None:
        calculator = AicScoreCalculator()
        episode = {
            "initial_distance": 0.15,
            "sim_time": [0.0, 0.1, 0.2, 0.3, 0.4],
            "tcp_positions": [np.zeros(3) for _ in range(5)],
            "tcp_linear_velocity": [np.array([0.02, 0.0, -0.01]) for _ in range(5)],
            "distances": [0.15, 0.10, 0.05, 0.02, 0.005],
            "plug_positions": [
                np.array([0.03, 0.0, 0.03]),
                np.array([0.02, 0.0, 0.02]),
                np.array([0.01, 0.0, 0.015]),
                np.array([0.002, 0.001, 0.008]),
                np.array([0.001, 0.001, 0.005]),
            ],
            "target_port_pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "target_port_entrance_pose": np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]),
            "wrench_time": [0.0, 0.1, 0.2, 0.3, 0.4],
            "wrench_samples": [np.zeros(6) for _ in range(5)],
            "off_limit_contacts": [False] * 5,
            "success": False,
            "wrong_port": False,
        }
        summary = calculator.evaluate(episode)
        self.assertGreater(float(summary.tier3["score"]), 38.0)
        self.assertIn("Partial insertion detected", summary.message)


if __name__ == "__main__":
    unittest.main()

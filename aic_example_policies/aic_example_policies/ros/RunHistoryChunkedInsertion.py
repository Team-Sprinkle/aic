"""Preserve causal measured transport history into the chunked insertion actor.

The current model consumes robot/relative-pose history, not historical RGB.
Pending action chunks are still invalidated at every ownership transfer.
"""
import os
import numpy as np

from .RunSharedPortTCP import RunSharedPortTCP
from .RunRuleThenSharedRecovery import SupervisedLockedActor, ObservationContractError
from .RunSample25mmThenInsertion import RunSample25mmThenInsertion
from .RunRuleTransport import RunRuleTransport
from chunked_port_actor import ChunkedPortActor
from stable_pose_window import StablePoseWindow
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from scipy.spatial.transform import Rotation


class HistoryChunkedLockedActor(SupervisedLockedActor):
    @staticmethod
    def _make_actor(checkpoint, stats, **kwargs):
        return ChunkedPortActor(stats, action_horizon=checkpoint.get('action_horizon', 4), **kwargs)

    def reset_for_task(self, task):
        super().reset_for_task(task)
        provider = getattr(self, 'history_provider', None)
        if provider is None:
            return
        record = provider()
        if record['task_id'] != str(task.id):
            raise ValueError('History belongs to another task/reset')
        self.robot_history = [(stamp, state.copy()) for stamp, state in record['rows']]
        self._trace(event='insertion_history_seeded', task_id=str(task.id),
            samples=len(self.robot_history),
            first_sim_time=self.robot_history[0][0] if self.robot_history else None,
            last_sim_time=self.robot_history[-1][0] if self.robot_history else None,
            pending_actions_restored=False, geometry_input=False)


class RunHistoryChunkedInsertion(RunSample25mmThenInsertion):
    def _make_local_actor(self, checkpoint):
        actor = HistoryChunkedLockedActor(self.parent, checkpoint)
        actor.history_provider = lambda: dict(task_id=self.history_task_id, rows=self.observation_history)
        return actor

    def __init__(self, parent_node):
        if not hasattr(RunSharedPortTCP, '_make_actor'):
            raise ValueError('The backwards-compatible actor factory hook must be installed after the active pilot finishes')
        self.history_task_id = None
        self.observation_history = []
        self.pose_consensus = StablePoseWindow()
        super().__init__(parent_node)
        if not isinstance(self.local, HistoryChunkedLockedActor):
            raise ValueError('Shared wrapper did not use its local-actor factory hook')
        steps = int(os.environ.get('AIC_INSERTION_EXECUTE_STEPS', '4'))
        if not 1 <= steps <= self.local.actor.action_horizon:
            raise ValueError('Execution exceeds trained prediction horizon')
        self.local.execute_steps = steps

    def _fresh_report(self, observation, task_vector):
        report, _, state, sim = super()._fresh_report(observation, task_vector)
        # Port refinement already made its JPEG86 copy. Give the grasp helper
        # raw RGB, so ITS training codec is applied once, not a second time.
        raw = {c: RunRuleTransport.native(getattr(observation, c+'_image')) for c in ('center', 'left', 'right')}
        if report['valid']:
            pose = matrix_from_xyz_xyzw([*report['predicted_opening_base_m'], *report['predicted_port_quat_xyzw']])
            accepted = self.pose_consensus.update(pose, sim)
            if accepted is None:
                report = dict(report, valid=False, reason='awaiting_fresh_pose_window_consensus')
            else:
                pose = accepted.pop('matrix')
                report = dict(report, predicted_opening_base_m=pose[:3, 3].tolist(),
                    predicted_port_quat_xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
                    acquisition_consensus=accepted)
        else:
            self.pose_consensus.clear()
        return report, raw, state, sim

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.pose_consensus.clear()
        return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.history_task_id = str(task.id)
        self.observation_history = []

        def observed():
            value = get_observation()
            if value is not None:
                stamp = self.local._control_sim_time(value)
                if self.observation_history and stamp < self.observation_history[-1][0]-1e-6:
                    raise ObservationContractError('Clock reset inside the active history window')
                if not self.observation_history or stamp > self.observation_history[-1][0]:
                    state = self.local.assembler.assemble_ros(value)
                    if len(state) != 32 or not np.isfinite(state).all():
                        raise ObservationContractError('Invalid public robot state')
                    self.observation_history.append((stamp, state.copy()))
                    span = max(self.local.history_offsets_s or [1.2]) + .5
                    while len(self.observation_history) > 1 and self.observation_history[1][0] < stamp-span:
                        self.observation_history.pop(0)
            return value

        return super().insert_cable(task, observed, move_robot, send_feedback, **kwargs)

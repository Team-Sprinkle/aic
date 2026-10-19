"""Frozen full-target BC with the existing measured-path recovery supervisor.

No RGB estimator, actor weights, gains, or privileged inputs are changed.
The parent no-recovery diagnostic stays intact for its active comparison.
"""
import os
import numpy as np
from .RunConsistentConnectorInsertion import RunConsistentConnectorInsertion
from .RunRuleThenSharedRecovery import measured_pose


class RunConsistentConnectorBackoff(RunConsistentConnectorInsertion):
    def __init__(self, parent_node):
        requested = os.environ.get('AIC_SHARED_RECOVERY_MODE', 'none')
        if requested not in ('none', 'local', 'hierarchical'):
            raise ValueError('Unknown recovery comparison arm')
        # The inherited 25mm diagnostic intentionally rejects recovery. This
        # explicit new class opts into it without editing that frozen class.
        previous = os.environ.get('AIC_SHARED_RECOVERY_MODE')
        os.environ['AIC_SHARED_RECOVERY_MODE'] = 'none'
        try:
            super().__init__(parent_node)
        finally:
            if previous is None:
                os.environ.pop('AIC_SHARED_RECOVERY_MODE', None)
            else:
                os.environ['AIC_SHARED_RECOVERY_MODE'] = previous
        self.mode = requested
        self.recovery_seeded_supervisor = None
        self.approach_measured_history = []
        self.recovery_command = None
        self.recovery_command_stamp = None

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        # The ordinary actor window only retains ~1.7s. A deeper retreat must
        # retain the measured path across the entire 150->25mm approach.
        history = []
        def observed():
            value = get_observation()
            if value is not None:
                stamp = self.local._control_sim_time(value)
                if not history or stamp > history[-1][0]:
                    history.append((stamp, self.local.assembler.assemble_ros(value).copy()))
            return value
        accepted = super()._lock(task_vector, observed, move_robot, fresh_context=fresh_context)
        if accepted:
            self.approach_measured_history = history
        return accepted

    def _supervise(self, state, absolute):
        if self.recovery_seeded_supervisor is not self.supervisor:
            for stamp, measured in self.approach_measured_history:
                self.supervisor._append_history(measured_pose(measured), self._force(measured),
                    'approach', self.route)
            self.recovery_seeded_supervisor = self.supervisor
            self._trace(event='backoff_approach_history_seeded',
                observed_samples=len(self.approach_measured_history),
                retained_waypoints=len(self.supervisor.history), geometry_input=False)
        return super()._supervise(state, absolute)

    def _command(self, move_robot, matrix, *, owner):
        super()._command(move_robot, matrix, owner=owner)
        if owner in ('measured_path_backtrack', 'supervisor_lateral_exploration'):
            self.recovery_command = np.asarray(matrix).copy()
            self.recovery_command_stamp = self.local.last_observation_stamp

    def _lateral_retry(self, get_observation, move_robot, *, route_level):
        self.recovery_command = None
        accepted = super()._lateral_retry(get_observation, move_robot, route_level=route_level)
        if not accepted:
            return False
        if self.recovery_command is None or self.recovery_command_stamp is None:
            raise ValueError('Recovery completed without a causal issued reference')
        self.local.clear_pending_actions()
        self.local.governor = None
        self.local.governor_seed = (self.recovery_command.copy(), self.recovery_command_stamp)
        self.local.seated_since = None
        self._trace(event='backoff_reference_reseeded',
            sim_time=self.recovery_command_stamp, issued_tcp_target=self.recovery_command.tolist(),
            geometry_input=False, old_pending_actions_restored=False)
        return True

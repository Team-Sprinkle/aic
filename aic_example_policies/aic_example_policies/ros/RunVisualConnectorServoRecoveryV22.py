"""Bounded SC near-contact rotational-compliance diagnostic, not an RCC claim."""
import numpy as np
from .RunVisualConnectorServoRecoveryV19 import RunVisualConnectorServoRecoveryV19
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunVisualConnectorServoRecoveryV22(RunVisualConnectorServoRecoveryV19):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.compliance_fraction = 0.

    def _execute(self, *args, **kwargs):
        self.compliance_fraction = 0.
        return super()._execute(*args, **kwargs)

    def _supervise(self, state, absolute):
        tip = (np.linalg.inv(self.local.visual_port) @ matrix_from_xyz_xyzw(state[:7]) @
               np.r_[self.local.visual_body_tip, 1.])[:3]
        target = 1. if tip[2] >= -.005 else 0.
        self.compliance_fraction += np.clip(target-self.compliance_fraction, -.1, .1)
        return super()._supervise(state, absolute)

    def _command(self, move_robot, matrix, *, owner):
        if owner != 'image_only_insertion_servo_diagnostic' or self.compliance_fraction <= 0:
            return super()._command(move_robot, matrix, owner=owner)
        original_k = self.local.cartesian_stiffness
        original_d = self.local.cartesian_damping
        stiffness = list(original_k)
        damping = list(original_d)
        alpha = self.compliance_fraction
        stiffness[3:] = [(1-alpha)*k+alpha*5. for k in stiffness[3:]]
        damping[3:] = [(1-alpha)*d+alpha*2. for d in damping[3:]]
        self.local.cartesian_stiffness = stiffness
        self.local.cartesian_damping = damping
        try:
            super()._command(move_robot, matrix, owner=owner)
        finally:
            self.local.cartesian_stiffness = original_k
            self.local.cartesian_damping = original_d
        self._trace(event='SC_contact_compliance_command', fraction=alpha,
                    stiffness=stiffness, damping=damping, unchanged_force_guards=True,
                    geometry_input=False, remote_center_compliance_claim=False)

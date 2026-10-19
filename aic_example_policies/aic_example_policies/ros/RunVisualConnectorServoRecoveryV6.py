"""Same SC recovery controller, with a bounded earlier public stall trigger.

The old1.5s detector can fire after the official one-second contact latch has
already frozen the cable. No simulator completion event is read at runtime.
"""
import numpy as np
from .RunVisualConnectorServoRecoveryV5 import RunVisualConnectorServoRecoveryV5
from .RunRuleThenSharedRecovery import measured_pose,RecoveryTransfer


class RunVisualConnectorServoRecoveryV6(RunVisualConnectorServoRecoveryV5):
    def _supervise_servo_stall(self,state,issued,point,stamp,force,depth):
        self.servo_stall_history.append((stamp,float(point[2,3])))
        self.servo_stall_history=[x for x in self.servo_stall_history if stamp-x[0]<=.25]
        if len(self.servo_stall_history)<2:return
        elapsed=stamp-self.servo_stall_history[0][0]
        gain=point[2,3]-self.servo_stall_history[0][1]
        demand=float(np.linalg.norm(issued[:3,3]-state[:3]))
        if elapsed>=.19 and gain<=.00015 and demand>=.003 and force>=3. and point[2,3]<depth-.0005:
            phase='contact' if point[2,3]>-.005 else 'alignment'
            self.supervisor._trigger(measured_pose(state),force,phase,self.route)
            decision=self.supervisor.step(pose=measured_pose(state),force_n=force,
                prior_command_m=self.last_command_m,tracking_error_m=demand,phase=phase,route=self.route)
            self._trace(event='loaded_servo_stall_recovery',sim_time=stamp,load_change_n=force,
                progress_window_s=elapsed,axial_gain_mm=float(gain*1000),tracking_error_mm=demand*1000,
                reason='early_loaded_stall_before_estimated_seating',
                synthetic_force_used=False,geometry_input=False,level=decision.level)
            self.local.clear_pending_actions();raise RecoveryTransfer(decision)


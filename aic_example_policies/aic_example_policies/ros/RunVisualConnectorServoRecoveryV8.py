"""Recover on early *excess* tracking pressure, before the official contact latch.

Public unloaded tracking compensation separates ordinary gravity-loaded holds
from increasing contact demand. Simulator completion data remain unavailable.
"""
import numpy as np
from .RunVisualConnectorServoRecoveryV7 import RunVisualConnectorServoRecoveryV7
from .RunRuleThenSharedRecovery import measured_pose,RecoveryTransfer


class RunVisualConnectorServoRecoveryV8(RunVisualConnectorServoRecoveryV7):
    def _supervise_servo_stall(self,state,issued,point,stamp,force,depth):
        self.servo_stall_history.append((stamp,float(point[2,3])))
        self.servo_stall_history=[x for x in self.servo_stall_history if stamp-x[0]<=.25]
        if len(self.servo_stall_history)<2:return
        elapsed=stamp-self.servo_stall_history[0][0]
        gain=point[2,3]-self.servo_stall_history[0][1]
        excess=issued[:3,3]-state[:3]-self.unloaded_reference_bias_base_m
        axial_pressure=float(self.local.visual_port[:3,2]@excess)
        if elapsed>=.19 and gain<=.00015 and axial_pressure>=.001 and force>=1.5 and point[2,3]<depth-.0005:
            phase='contact' if point[2,3]>-.005 else 'alignment'
            self.supervisor._trigger(measured_pose(state),force,phase,self.route)
            decision=self.supervisor.step(pose=measured_pose(state),force_n=force,
                prior_command_m=self.last_command_m,tracking_error_m=float(np.linalg.norm(issued[:3,3]-state[:3])),phase=phase,route=self.route)
            self._trace(event='loaded_servo_stall_recovery',sim_time=stamp,load_change_n=force,
                progress_window_s=elapsed,axial_gain_mm=float(gain*1000),
                excess_axial_tracking_pressure_mm=axial_pressure*1000,
                reason='early_actual_load_and_excess_forward_tracking_pressure',
                synthetic_force_used=False,geometry_input=False,level=decision.level)
            self.local.clear_pending_actions();raise RecoveryTransfer(decision)

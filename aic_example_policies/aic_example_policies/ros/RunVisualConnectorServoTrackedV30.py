"""V29 alignment gates with faster lateral-only tracking; axial loop unchanged."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoAlignedV29 import RunVisualConnectorServoAlignedV29
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from insertion_reference_governor_v2 import InsertionReferenceGovernor,ReferenceLimits


class RunVisualConnectorServoTrackedV30(RunVisualConnectorServoAlignedV29):
    def _close_pose_servo(self,task,get_observation,move_robot,send_feedback,**kwargs):
        vector=self.local.task_vector(task)
        if vector[0]==1:return super()._close_pose_servo(task,get_observation,move_robot,send_feedback,**kwargs)
        port=self.local.visual_port.copy();inverse=np.linalg.inv(port)
        grasp=np.eye(4);grasp[:3,3]=self.local.visual_body_tip;grasp[:3,:3]=self.local.visual_body_connector_rotation
        inverse_grasp=np.linalg.inv(grasp);depth=.0465 if vector[0]==1 else .0156
        depth+=float(self.servo_goal_translation[2])
        # Command margin and estimated-seat acceptance are separate. V18
        # incorrectly raised both; accurate SC cannot physically reach15.8mm.
        command_depth=depth+(0.0004 if vector[0]!=1 else 0.)
        seed,seed_stamp=self.local.governor_seed
        started=time.monotonic();first=None;last=None;stable=None;best=-float('inf');last_progress=None;reference=None;integral=None;governor=None;reason='visual_servo_wall_budget'
        alignment_streak=0;alignment_latched=False
        while time.monotonic()-started<60 and time.monotonic()<self.episode_deadline-2:
            observation=get_observation()
            if observation is None:time.sleep(.002);continue
            stamp=self.local._control_sim_time(observation)
            if last is not None and stamp-last<.049:time.sleep(.002);continue
            control_started=time.perf_counter()
            dt=.05 if last is None else min(.1,stamp-last);last=stamp
            state=self.local.assembler.assemble_ros(observation);measured=matrix_from_xyz_xyzw(state[:7]);point=inverse@measured@grasp
            angle=Rotation.from_matrix(self.servo_goal_rotation.T@point[:3,:3]).magnitude();force=self._force(state)
            if first is None:
                first=stamp;last_progress=stamp;reference=np.array([self.servo_goal_translation[0],self.servo_goal_translation[1],point[2,3]])
                integral=(inverse@seed@grasp)[:3,3]-reference
                governor=InsertionReferenceGovernor(port,grasp,seed,measured,seed_stamp,
                    ReferenceLimits(forward_lead_m=.020,alignment_lateral_m=1.,alignment_angle_rad=np.pi,contact_load_n=float('inf')))
                self._trace(event='visual_servo_started',sim_time=stamp,geometry_input=False,actor_owned=False,
                    learned_actor_inference=False,required_seating_dwell_s=2.0,nominal_seating_depth_m=depth,command_depth_m=command_depth,source='frozen RGB port/grasp and public robot state only')
            if point[2,3]>best+.0005:best=float(point[2,3]);last_progress=stamp
            if np.linalg.norm(state[26:29])>40 or (force>=8 and self.mode=='none'):
                reason='visual_servo_force_guard';break
            if stamp-first>35:reason='visual_servo_simulation_budget';break
            if stamp-last_progress>8:reason='visual_servo_no_measured_progress';break
            lateral=float(np.linalg.norm(point[:2,3]-self.servo_goal_translation[:2]))
            alignment_streak=alignment_streak+1 if lateral<=.0001 and angle<=np.deg2rad(.5) else 0
            if lateral>.00015 or angle>np.deg2rad(.5):alignment_latched=False
            elif alignment_streak>=3:alignment_latched=True
            aligned=alignment_latched

            if aligned:reference[2]=min(command_depth,reference[2]+.006*dt)
            integral=np.clip(integral+np.array([1.8,1.8,.6])*dt*(reference-point[:3,3]),-.015,.015)
            desired_point=np.eye(4);desired_point[:3,3]=reference+integral;desired_point[:3,:3]=self.servo_goal_rotation
            desired=port@desired_point@inverse_grasp
            issued,diagnostic=governor.step(desired,measured,stamp,force)
            if self.mode!='none':
                self._supervise(state,issued)
                self._supervise_servo_stall(state,issued,point,stamp,force,depth)
            self._command(move_robot,issued,owner='image_only_insertion_servo_diagnostic')
            self._trace(event='visual_servo_step',sim_time=stamp,estimated_tip_port_m=point[:3,3].tolist(),
                estimated_orientation_error_deg=float(np.degrees(angle)),reference_tip_port_m=reference.tolist(),
                integral_port_m=integral.tolist(),load_change_n=force,aligned=aligned,
                advance_gate_entry_lateral_mm=.1,advance_gate_exit_lateral_mm=.15,alignment_streak=alignment_streak,
                persistent_goal_translation_m=self.servo_goal_translation.tolist(),goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(),
                complete_control_step_ms=(time.perf_counter()-control_started)*1000,
                geometry_input=False,actor_owned=False,learned_actor_inference=False,governor=diagnostic)
            seated=(point[2,3]>=depth-.0002 and lateral<=.0005 and angle<=np.deg2rad(.75) and np.linalg.norm(state[7:10])<=.002)
            stable=(stamp if stable is None else stable) if seated else None
            if stable is not None and stamp-stable>=2.0:
                reason='visual_servo_public_estimated_seating';self._trace(event='visual_servo_finished',reason=reason,sim_time=stamp,geometry_input=False,physical_insertion_asserted=False);return True
        self._trace(event='visual_servo_finished',reason=reason,sim_time=last,geometry_input=False,physical_insertion_asserted=False)
        return False

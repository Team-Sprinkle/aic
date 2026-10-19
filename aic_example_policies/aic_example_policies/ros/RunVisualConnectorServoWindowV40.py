"""Observe the scene before choosing its route; keep the last API hold fixed."""
import time
import numpy as np
from .RunVisualConnectorServoOldPoseEfficientV39 import RunVisualConnectorServoOldPoseEfficientV39
from .RunVisualConnectorServoRecoveryFastV34 import RunVisualConnectorServoRecoveryFastV34
from .RunObservedEmptyBoardWindowV40 import RunObservedEmptyBoardWindowV40
from .RunRuleTransport import RunRuleTransport
from requested_native_port_pose import RequestedNativePortPose
from empty_scene_receipt import EmptySceneConsensus


class RunVisualConnectorServoWindowV40(RunVisualConnectorServoOldPoseEfficientV39):
    def __init__(self,parent_node):
        RunVisualConnectorServoRecoveryFastV34.__init__(self,parent_node)
        self.cached_transport=RunObservedEmptyBoardWindowV40(self.parent)
        bundle=self.cached_transport.spec['bundle']
        for family in ('sfp','sc'):
            self.cached_transport.components[family]=RequestedNativePortPose(
                bundle['backbone'],bundle[family+'_coarse'],bundle[family+'_native_refiner'],bundle['mount_priors'],
                family=family,device=str(self.local.device),maximum_shift_px=64.,
                maximum_reprojection_px=2.5 if family=='sc' else 2.,minimum_projection_probability=.9 if family=='sc' else .95)
        self.scene_task_code=None;self.initial_hold_reference=None

    def _command(self,move_robot,matrix,*,owner):
        if owner=='initial_free_hold':self.initial_hold_reference=np.asarray(matrix).copy()
        return super()._command(move_robot,matrix,owner=owner)

    def _initial_force_reference(self,get_observation,move_robot):
        if not super()._initial_force_reference(get_observation,move_robot):return False
        if self.scene_task_code[1]!=1:return True
        if self.initial_hold_reference is None:return False
        consensus=EmptySceneConsensus(packet_camera_timing=getattr(self,'use_packet_camera_timing',False));started=time.monotonic();last=-float('inf');first=None
        while time.monotonic()-started<3:
            observation=get_observation()
            if observation is None:time.sleep(.002);continue
            stamp=self.local._control_sim_time(observation)
            if stamp-last<.099:time.sleep(.002);continue
            last=stamp;first=stamp if first is None else first
            if stamp-first>2:return True
            tick=time.perf_counter()
            state=self.local.assembler.assemble_ros(observation)
            self._command(move_robot,self.initial_hold_reference,owner='empty_row_initial_observation_hold')
            if self._force(state)>=8 or np.linalg.norm(state[26:29])>40:
                self._trace(event='empty_row_scene_window_rejected',sim_time=stamp,reason='loaded_hold',geometry_input=False);return False
            images={c:RunRuleTransport.native(getattr(observation,c+'_image')) for c in ('center','left','right')}
            stamps=[getattr(observation,c+'_image').header.stamp.sec*1000000000+getattr(observation,c+'_image').header.stamp.nanosec for c in ('center','left','right')]
            prediction=self.cached_transport.empty_row.predict(images)
            receipt=consensus.add(stamp_s=stamp,image_stamps_ns=stamps,tcp_pose=state[:7],linear_speed=np.linalg.norm(state[7:10]),angular_speed=np.linalg.norm(state[10:13]),clear=prediction['clear'],task_code=self.scene_task_code)
            self._trace(event='empty_row_stationary_scene_check',sim_time=stamp,**prediction,consistent_observations=consensus.count,image_ages_s=(stamp-np.array(stamps)*1e-9).tolist(),complete_scene_check_ms=(time.perf_counter()-tick)*1000,geometry_input=False)
            if receipt:
                self.cached_transport.pending_initial_receipt=receipt;return True
            if prediction['score']<.2:return True
        return True

    def _execute(self,task,get_observation,move_robot,send_feedback,**kwargs):
        self.scene_task_code=self.local.task_vector(task);self.cached_transport.pending_initial_receipt=None
        try:return super()._execute(task,get_observation,move_robot,send_feedback,**kwargs)
        finally:self.cached_transport.pending_initial_receipt=None;self.scene_task_code=None

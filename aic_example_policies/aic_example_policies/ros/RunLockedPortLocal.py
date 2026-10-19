"""Insertion BC with a static port frame supplied by approved RGB handoff."""
import time
import numpy as np
from PIL import Image
import torch

from .RunSharedPortTCP import RunSharedPortTCP
from .RunPredictedPortLocal import RunPredictedPortLocal
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from local_causal_history import sample_robot_history
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation
from native_actor_images import actor_image,CONTRACT


class RunLockedPortLocal(RunSharedPortTCP):
    supports_sc_tip_target_reference=True
    def __init__(self,parent_node,checkpoint_path):
        super().__init__(parent_node,checkpoint_path)
        metadata=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        self.physical_tip_families=metadata.get('physical_tip_families',['sc'])
        self.tip_action_families=metadata.get('tip_action_families',['sc'])
        self.actor_image_preprocess_contract=metadata.get('actor_image_preprocess_contract','legacy_mixed')
        if self.actor_image_preprocess_contract not in ('legacy_mixed',CONTRACT):raise ValueError('Unknown actor image preprocessing contract')
        declared=self.physical_tip_families if self.actor.use_tip else []
        if self.sc_action_reference!='tcp':declared=declared+self.tip_action_families
        if not set(declared)<=set(self.trained_families):
            raise ValueError('Declared plug-control-point family lacks training coverage')
        if self.target_frame!='estimated_port':
            raise ValueError('Expected predicted-port complete target checkpoint')
        self.visual_port=None
        self.visual_body_tip=None
        self.visual_body_connector_rotation=None
        self.local_terminal_reason=None
        self.robot_history=[]

    def reset_for_task(self,task):
        super().reset_for_task(task)
        self.robot_history=[]

    def retain_history_observation(self,observation):
        if observation is None or not self.actor.use_local_history:
            return
        now=self._control_sim_time(observation)
        if self.robot_history and now < self.robot_history[-1][0]:
            raise ValueError('Observation clock reset inside local attempt')
        if not self.robot_history or now > self.robot_history[-1][0]:
            self.robot_history.append((now,base_state_from_ros_observation(observation).copy()))
        span=max(self.history_offsets_s)+.5
        while len(self.robot_history)>1 and self.robot_history[1][0]<now-span:
            self.robot_history.pop(0)

    def _trace(self, **fields):
        if fields.get('event') == 'terminal_before_return':
            self.local_terminal_reason=fields.get('reason')
        return super()._trace(**fields)

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.local_terminal_reason=None
        def observed_with_history():
            observation=get_observation()
            self.retain_history_observation(observation)
            return observation
        super().insert_cable(task,observed_with_history,move_robot,send_feedback,**kwargs)
        # Acknowledge a normally completed bounded attempt, as the official
        # example policies do. The independent scorer decides insertion.
        # No scoring topic or simulator completion/freeze signal is consumed.
        completed=self.local_terminal_reason == 'bounded_runtime'
        self._trace(event='local_attempt_acknowledged',
                    attempt_completed=completed,reason=self.local_terminal_reason,
                    physical_insertion_asserted=False,scoring_input=False)
        return completed

    def bind_visual_handoff(self,report,sim_time):
        if not report['valid'] or report.get('runtime_true_geometry',True):
            raise ValueError('Only approved observation-only handoff can supply the frame')
        from scipy.spatial.transform import Rotation
        self.visual_port=np.eye(4)
        self.visual_port[:3,3]=report['predicted_opening_base_m']
        self.visual_port[:3,:3]=Rotation.from_quat(report['predicted_port_quat_xyzw']).as_matrix()
        self.visual_lock_sim_time=sim_time

    def bind_predicted_body_tip(self,position):
        value=np.asarray(position,float)
        if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)>.10:
            raise ValueError('Invalid camera-estimated gripper-to-tip position')
        self.visual_body_tip=value

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        started=time.perf_counter()
        if self.visual_port is None:raise ValueError('Visual handoff must precede insertion')
        family='sfp' if task_vector[0]==1 else 'sc'
        if family not in self.trained_families:
            raise ValueError(f'Checkpoint has no declared {family} training contract')
        if family=='sfp' and self.recorded_controller_contract is None:
            raise ValueError('Shared SFP insertion requires audited controller metadata')
        if self.recorded_controller_contract is not None:
            stiffness,damping=self._cartesian_gains()
            if (not np.allclose(stiffness,self.recorded_controller_contract['stiffness']) or
                    not np.allclose(damping,self.recorded_controller_contract['damping'])):
                raise ValueError('Runtime insertion gains differ from the recorded expert')
        if self.actor.use_local_history and self.history_offsets_s is None:
            raise ValueError('History actor requires explicit causal sampling metadata')
        state=self.assembler.assemble_ros(observation)
        sim=self._control_sim_time(observation)
        if self.first_sim_time is None:self.first_sim_time=sim
        self.last_elapsed_sim_sec=sim-self.first_sim_time
        relative=np.linalg.inv(self.visual_port)@matrix_from_xyz_xyzw(state[:7])
        pose=np.r_[relative[:3,3],relative[:3,:2].T.reshape(-1)].astype(np.float32)
        history=None
        if self.actor.use_local_history:
            self.retain_history_observation(observation)
            causal=[r for r in self.robot_history if r[0]<=sim]
            values=sample_robot_history([r[0] for r in causal],[r[1] for r in causal],
                sim,self.visual_port,self.history_offsets_s,self.history_masked)
            history=torch.as_tensor(values,device=self.device)[None]
        tip=tip_mask=None
        if self.actor.use_tip:
            enabled=family in self.physical_tip_families and self.tip_input_mode in ('translation','constant_translation','full_pose')
            if enabled and self.tip_input_mode in ('translation','full_pose') and self.visual_body_tip is None:
                raise ValueError('Camera-estimated body tip must be bound before insertion')
            if enabled and self.tip_input_mode=='constant_translation' and self.constant_body_tip is None:
                raise ValueError('Constant control requires its training-only mean')
            body=((np.asarray(self.constant_body_tip) if self.tip_input_mode=='constant_translation'
                  else self.visual_body_tip) if enabled else None)
            point=(relative[:3,:3]@body+relative[:3,3] if body is not None else np.zeros(3))
            tip_rotation=np.eye(3)
            if enabled and self.tip_input_mode=='full_pose':
                if self.visual_body_connector_rotation is None:
                    raise ValueError('Full connector actor requires an RGB orientation lock')
                tip_rotation=relative[:3,:3]@self.visual_body_connector_rotation
            tip=torch.tensor([*point,*tip_rotation[:,:2].T.reshape(-1)],device=self.device,dtype=torch.float32)[None]
            tip_mask=torch.tensor([enabled],device=self.device,dtype=torch.bool)
        images=[]
        for cam in ('center','left','right'):
            native=RunPredictedPortLocal.raw_rgb(getattr(observation,cam+'_image'))
            if self.actor_image_preprocess_contract==CONTRACT:
                images.append(torch.from_numpy(actor_image(native,family)))
                continue
            _,small=RunPredictedPortLocal.rgb_for_training(native,family)
            rgb=np.asarray(Image.fromarray(small).resize((224,224),Image.Resampling.BILINEAR)).copy()
            images.append(torch.from_numpy(rgb).permute(2,0,1))
        with torch.inference_mode(),torch.autocast(self.device.type,dtype=torch.bfloat16,
                                                  enabled=self.device.type=='cuda'):
            target,_=self.actor(torch.stack(images)[None].to(self.device),
                torch.as_tensor(state,device=self.device)[None],
                torch.as_tensor(task_vector,device=self.device)[None],
                external_pose=torch.as_tensor(pose,device=self.device)[None],
                external_mask=torch.ones(1,device=self.device,dtype=torch.bool),
                physical_tip=tip,tip_mask=tip_mask,local_history=history)
        # Include GPU completion and the result transfer in inference timing.
        result=target[0].float().cpu().numpy()
        control_point_targets=result.copy()
        if family in self.tip_action_families and self.sc_action_reference=='plug_tip_translation_tcp_orientation':
            if self.visual_body_tip is None:raise ValueError('Missing observation-only body-tip lock')
            from shared_port_tcp_geometry import matrix_from_pose9
            for value in result:
                value[:3]-=matrix_from_pose9(value)[:3,:3]@self.visual_body_tip
        elif family in self.tip_action_families and self.sc_action_reference=='full_connector_pose':
            from shared_port_tcp_geometry import matrix_from_pose9
            from full_connector_geometry import connector_target_to_tcp
            if self.visual_body_connector_rotation is None or self.visual_body_tip is None:
                raise ValueError('Missing full RGB grasp lock')
            grasp=np.eye(4);grasp[:3,:3]=self.visual_body_connector_rotation;grasp[:3,3]=self.visual_body_tip
            for value in result:
                tcp=connector_target_to_tcp(matrix_from_pose9(value),grasp)
                value[:]=np.r_[tcp[:3,3],tcp[:3,:2].T.reshape(-1)]
        self._inference_latency_ms.append((time.perf_counter()-started)*1000)
        self._trace(event='locked_local_perception',sim_time=sim,state=state.tolist(),
                    predicted_tcp_pose_in_port=pose.tolist(),
                    visual_lock_age_s=sim-self.visual_lock_sim_time,
                    predicted_tip_port_position_m=(point.tolist() if self.actor.use_tip else None),
                    tip_input_mode=self.tip_input_mode,tip_orientation_input=self.tip_input_mode=='full_pose',
                    estimated_connector_rotation_in_tcp=(self.visual_body_connector_rotation.tolist()
                        if self.tip_input_mode=='full_pose' else None),
                    sc_action_reference=self.sc_action_reference,
                    tip_action_families=self.tip_action_families,
                    actor_control_point_targets_port=control_point_targets.tolist(),
                    decoded_tcp_targets_port=result.tolist(),
                    geometry_input=False,static_visual_lock=True)
        if history is not None:
            self._trace(event='local_causal_history',sim_time=sim,
                        history=values.tolist(),offsets_s=self.history_offsets_s,
                        masked=self.history_masked,geometry_input=False)
        return result,pose,state

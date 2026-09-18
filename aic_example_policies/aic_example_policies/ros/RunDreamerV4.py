"""AIC Dreamer pilot: observed-history policy with four absolute commands per decision.

Reuses the reviewed ACT execution loop and bounds. Model history receives the
actual bounded targets sent through set_pose_target, never unexecuted BC labels.
"""
from __future__ import annotations
import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from aic_model.policy import (Policy,pose_to_position_motion_update,DEFAULT_CARTESIAN_STIFFNESS,DEFAULT_CARTESIAN_DAMPING)
from aic_example_policies.ros.RunACTTorchScript import RunACTTorchScript
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation


class RunDreamerV4(RunACTTorchScript):
    def __init__(self,parent_node):
        Policy.__init__(self,parent_node)
        source=Path(os.environ['AIC_DREAMER_SOURCE_ROOT'])
        sys.path.insert(0,str(source))
        from dreamer4.aic.models import AICController
        from dreamer4.aic.train import load
        from dreamer4.aic.data import absolute6,letterbox_array
        self._absolute6=absolute6;self._letterbox=letterbox_array
        from dreamer4.aic.data import PhysicalActionMap
        self._action_map_class=PhysicalActionMap
        torch.set_num_threads(2)
        self.device=torch.device(os.environ.get('AIC_ACT_DEVICE','cuda:0'))
        checkpoint=Path(os.environ['AIC_ACT_POLICY_PATH'])
        self.model,saved=load(checkpoint,self.device,control_only=True)
        if saved.get('type')!='aic_dreamer_control' or saved.get('stage') not in {'agent','imagine'}:
            raise ValueError('A frozen trained Dreamer control export is required')
        self.controller=AICController(self.model,saved['normalization'],self.device,'bf16')
        self._physical_map=self._action_map_class(**{k:saved['normalization']['action_map'][k] for k in ['center','scale']})
        self.control_hz=float(os.environ.get('AIC_ACT_CONTROL_HZ','20'))
        self.control_clock=os.environ.get('AIC_ACT_CONTROL_CLOCK','simulation')
        self.n_action_steps=int(os.environ.get('AIC_ACT_N_ACTION_STEPS','4'))
        self.command_mode=os.environ.get('AIC_ACT_RUNTIME_COMMAND_MODE','absolute_pose')
        self.command_frame=os.environ.get('AIC_ACT_COMMAND_FRAME','base_link')
        if (self.control_hz,self.control_clock,self.n_action_steps,self.command_mode,self.command_frame)!=(20.,'simulation',4,'absolute_pose','base_link'):
            raise ValueError('Dreamer pilot requires20Hz simulation-clock,4 commands, absolute_pose/base_link')
        self.max_runtime_sec=float(os.environ.get('AIC_ACT_MAX_RUNTIME_SEC','180'))
        self.max_simulation_sec=float(os.environ.get('AIC_ACT_MAX_SIMULATION_SEC','90'))
        self.start_delay_sec=0.
        self.max_translation_delta=float(os.environ.get('AIC_ACT_MAX_TRANSLATION_DELTA','.1'))
        self.max_rotation_delta=float(os.environ.get('AIC_ACT_MAX_ROTATION_DELTA','.2'))
        self.translation_limit_mode='norm';self.translation_deadband=0.;self.rotation_deadband=0.
        self.log_every_n_commands=20;self._action_queue=[];self._executed=[];self._current_task=None
        self._episode_first_image_time=None;self._task_values=None
        self.inference_latencies=[];self._previous_decision_time=None;self._decision_started_at=None;self.command_latencies=[]
        if self.max_runtime_sec<=0 or self.max_simulation_sec<=0:raise ValueError('Positive watchdog and simulation duration required')
        # All kernels warm before engine task start, then clear every history cache.
        dummy=np.zeros((1,3,3,self.model.config.image_size,self.model.config.image_size),np.uint8)
        state=np.array(saved['normalization']['state_mean'],np.float32)[None];state[0,3:7]=[1,0,0,0]
        task=np.array(saved['normalization']['task_vector'],np.float32)[None];previous=None;start=time.monotonic()
        for i in range(10):previous=self.controller.act(dummy,state,task,[i*.2],previous)
        self.controller.reset()
        self.get_logger().info(f'DreamerV4 control export loaded: {checkpoint};43D observation contract;27.13M full/22.57M acting parameters;warmup{time.monotonic()-start:.3f}s;no imagined-video generation during acting')

    def reset_for_task(self,task):
        self._current_task=task;values=np.asarray(self._task_vector(task),dtype=np.float32)
        expected=np.array([1,0,0,1,1,0,0,0,0,1],np.float32)
        if not np.array_equal(values,expected):raise ValueError('This strict60 pilot supports only SFP card0/port1')
        self._task_values=values;self.controller.reset();self._action_queue=[];self._executed=[]
        self._episode_first_image_time=None;self._previous_decision_time=None;self.inference_latencies=[];self.command_latencies=[];self._decision_started_at=None

    @staticmethod
    def _rgb(message):
        channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(message.encoding.lower())
        if channels is None or message.height!=256 or message.width!=288:raise ValueError('Expected RGB/BGR256x288 camera')
        array=np.frombuffer(bytes(message.data),np.uint8).reshape(message.height,message.step)
        array=array[:,:message.width*channels].reshape(message.height,message.width,channels)[...,:3]
        return array[...,::-1].copy() if message.encoding.lower().startswith('bgr') else array.copy()

    def select_delta_action(self,observation_msg):
        if not self._action_queue:
            began=time.perf_counter();self._decision_started_at=began;stamp=observation_msg.center_image.header.stamp
            now=float(stamp.sec)+float(stamp.nanosec)*1e-9
            if self._episode_first_image_time is None:self._episode_first_image_time=now
            if self._previous_decision_time is not None:
                interval=now-self._previous_decision_time
                if abs(interval-.2)>.051:
                    self.get_logger().warning(f'Dreamer decision interval differs from trained200ms transition: {interval:.6f}s')
            state=base_state_from_ros_observation(observation_msg)
            if state.shape!=(32,) or not np.isfinite(state).all():raise ValueError('Invalid physical state')
            images=np.stack([self._letterbox(self._rgb(getattr(observation_msg,cam+'_image')),self.model.config.image_size) for cam in ['center','left','right']])[None]
            previous=None if self.controller.offset==0 else np.array(self._executed,np.float32)[None]
            if previous is not None:
                if previous.shape!=(1,4,6):raise ValueError('Missing actual executed command history')
                self._physical_map.encode(previous)
            prediction=self.controller.act(images,state[None],self._task_values[None],[now-self._episode_first_image_time],previous)[0]
            if prediction.shape!=(4,6) or not np.isfinite(prediction).all():raise ValueError('Nonfinite Dreamer output')
            self._action_queue=list(prediction);self._executed=[];self._previous_decision_time=now
            elapsed=(time.perf_counter()-began)*1000;self.inference_latencies.append(elapsed)
            if len(self.inference_latencies)==1 or len(self.inference_latencies)%25==0:
                self.get_logger().info('DREAMER_DECISION '+json.dumps({'decision':len(self.inference_latencies),'sim_time':now,'latency_ms':elapsed,'context_offset':self.controller.offset,'executed_previous_commands':0 if previous is None else 4}))
        return self._action_queue.pop(0)

    def set_pose_target(self,move_robot,pose,frame_id='base_link'):
        if frame_id!='base_link':raise ValueError('Dreamer executes absolute base_link poses')
        motion=pose_to_position_motion_update(pose,stamp=self.get_clock().now().to_msg(),frame_id=frame_id,stiffness=DEFAULT_CARTESIAN_STIFFNESS,damping=DEFAULT_CARTESIAN_DAMPING)
        result=move_robot(motion_update=motion)
        if result is not True:raise RuntimeError('Command publication was not accepted; refusing fabricated executed history')
        if self._decision_started_at is not None:
            latency=(time.perf_counter()-self._decision_started_at)*1000;self.command_latencies.append(latency);self._decision_started_at=None
            self.get_logger().info('DREAMER_COMMAND_LATENCY '+json.dumps({'decision':len(self.command_latencies),'observation_to_command_ms':latency,'includes':'RGB parse+letterbox+transfer+physical state+causal model+output transfer+pose conversion+bounds+ROS publication callback'}))
        raw=[pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
        self._executed.append(self._absolute6(np.array(raw)))
        return result

"""AIC Dreamer pilot: four observation-relative TCP delta commands per decision.

Reuses the reviewed ACT execution loop and bounds. Model history receives the
actual bounded TCP deltas sent through set_pose_target, never unexecuted BC labels.
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
        from dreamer4.aic.tcp_delta import body_delta
        self._body_delta=body_delta
        from dreamer4.aic.data import PhysicalActionMap
        self._action_map_class=PhysicalActionMap
        import cv2
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        self.get_logger().info('DREAMER_CPU_THREADS '+json.dumps({'opencv':cv2.getNumThreads(),'torch_intraop':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads(),'environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}))
        self.device=torch.device(os.environ.get('AIC_ACT_DEVICE','cuda:0'))
        checkpoint=Path(os.environ['AIC_ACT_POLICY_PATH'])
        # Keep Torch inference outside ROS callback/GIL contention. The parent
        # retains CPU model metadata only; the child owns CUDA and causal caches.
        self.model,saved=load(checkpoint,'cpu',control_only=True)
        if saved.get('type')!='aic_dreamer_control' or saved.get('stage') not in {'agent','imagine'}:
            raise ValueError('A frozen trained Dreamer control export is required')
        if saved.get('options',{}).get('action_representation')!='tcp_delta_observation':
            raise ValueError('Only the corrected observation-relative TCP delta policy is supported')
        from aic_example_policies.ros.dreamer_inference_worker import RemoteAICController
        self.controller=RemoteAICController(checkpoint,source,self.device,'bf16')
        self.get_logger().info('DREAMER_INFERENCE_WORKER '+json.dumps(self.controller.worker_metadata))
        self._physical_map=self._action_map_class(**{k:saved['normalization']['action_map'][k] for k in ['center','scale']})
        self.control_hz=float(os.environ.get('AIC_ACT_CONTROL_HZ','20'))
        self.control_clock=os.environ.get('AIC_ACT_CONTROL_CLOCK','simulation')
        self.n_action_steps=int(os.environ.get('AIC_ACT_N_ACTION_STEPS','4'))
        self.command_mode=os.environ.get('AIC_ACT_RUNTIME_COMMAND_MODE','delta_pose')
        self.command_frame=os.environ.get('AIC_ACT_COMMAND_FRAME','gripper/tcp')
        self.delta_pose_reference=os.environ.get('AIC_ACT_DELTA_POSE_REFERENCE','observation')
        if (self.control_hz,self.control_clock,self.n_action_steps,self.command_mode,self.command_frame,self.delta_pose_reference)!=(20.,'simulation',4,'delta_pose','gripper/tcp','observation'):
            raise ValueError('Dreamer requires20Hz simulation-clock,4 observation-relative TCP delta commands')
        self.max_runtime_sec=float(os.environ.get('AIC_ACT_MAX_RUNTIME_SEC','180'))
        self.max_simulation_sec=float(os.environ.get('AIC_ACT_MAX_SIMULATION_SEC','90'))
        self.start_delay_sec=0.
        self.max_translation_delta=float(os.environ.get('AIC_ACT_MAX_TRANSLATION_DELTA','.1'))
        self.max_rotation_delta=float(os.environ.get('AIC_ACT_MAX_ROTATION_DELTA','.2'))
        self.translation_limit_mode='norm';self.translation_deadband=0.;self.rotation_deadband=0.
        self.log_every_n_commands=20;self._action_queue=[];self._executed=[];self._current_task=None
        self._episode_first_image_time=None;self._task_values=None
        self._command_observed_pose=None
        self.inference_latencies=[];self._previous_decision_time=None;self._decision_started_at=None;self.command_latencies=[]
        if self.max_runtime_sec<=0 or self.max_simulation_sec<=0:raise ValueError('Positive watchdog and simulation duration required')
        # All kernels warm before engine task start, then clear every history cache.
        dummy=np.zeros((1,self.model.config.aic_views,3,self.model.config.image_size,self.model.config.image_size),np.uint8)
        state=np.array(saved['normalization']['state_mean'],np.float32)[None];state[0,3:7]=[1,0,0,0]
        task=np.array(saved['normalization']['task_vector'],np.float32)[None];previous=None;start=time.monotonic()
        for i in range(10):previous=self.controller.act(dummy,state,task,[i*.2],previous)
        self.controller.reset()
        self.get_logger().info(f'DreamerV4 control export loaded: {checkpoint};43D observation contract;{self.model.config.aic_views} observed views;{sum(p.numel() for p in self.model.parameters())} acting parameters;warmup{time.monotonic()-start:.3f}s;no imagined-video generation during acting')

    def reset_for_task(self,task):
        self._current_task=task;values=np.asarray(self._task_vector(task),dtype=np.float32)
        expected=np.array([1,0,0,1,1,0,0,0,0,1],np.float32)
        if not np.array_equal(values,expected):raise ValueError('This strict60 pilot supports only SFP card0/port1')
        self._task_values=values;self.controller.reset();self._action_queue=[];self._executed=[]
        self._command_observed_pose=None
        self._episode_first_image_time=None;self._previous_decision_time=None;self.inference_latencies=[];self.command_latencies=[];self._decision_started_at=None

    @staticmethod
    def _rgb(message):
        channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(message.encoding.lower())
        if channels is None or message.height<=0 or message.width<=0:raise ValueError('Expected a nonempty RGB/BGR camera')
        array=np.frombuffer(bytes(message.data),np.uint8).reshape(message.height,message.step)
        array=array[:,:message.width*channels].reshape(message.height,message.width,channels)[...,:3]
        rgb=array[...,::-1].copy() if message.encoding.lower().startswith('bgr') else array.copy()
        # The expert recorder stores288x256 JPEGs after this exact area resize.
        # Simulator messages are1152x1024; crop coordinates belong to the saved
        # observation grid, not the full-resolution camera message.
        if rgb.shape[:2]!=(256,288):
            import cv2
            rgb=cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA)
        return rgb

    def select_delta_action(self,observation_msg):
        decision_began=time.perf_counter() if not self._action_queue else None
        state=base_state_from_ros_observation(observation_msg)
        if state.shape!=(32,) or not np.isfinite(state).all():raise ValueError('Invalid physical state')
        self._command_observed_pose=state[:7].copy()
        state_ready=time.perf_counter() if decision_began is not None else None
        if not self._action_queue:
            began=decision_began;self._decision_started_at=began;stamp=observation_msg.center_image.header.stamp
            now=float(stamp.sec)+float(stamp.nanosec)*1e-9
            if self._episode_first_image_time is None:self._episode_first_image_time=now
            if self._previous_decision_time is not None:
                interval=now-self._previous_decision_time
                if abs(interval-.2)>.051:
                    self.get_logger().warning(f'Dreamer decision interval differs from trained200ms transition: {interval:.6f}s')
            from dreamer4.aic.data import camera_views
            images=camera_views([self._rgb(getattr(observation_msg,cam+'_image')) for cam in ['center','left','right']],self.model.config.image_size,self.model.config.aic_views)[None]
            previous=None if self.controller.offset==0 else np.array(self._executed,np.float32)[None]
            if previous is not None:
                if previous.shape!=(1,4,6):raise ValueError('Missing actual executed command history')
                self._physical_map.encode(previous)
            images_ready=time.perf_counter()
            prediction=self.controller.act(images,state[None],self._task_values[None],[now-self._episode_first_image_time],previous)[0]
            model_ready=time.perf_counter()
            self._decision_phase_ms={'state_ms':(state_ready-began)*1000,'camera_and_history_ms':(images_ready-state_ready)*1000,'model_ms':(model_ready-images_ready)*1000}
            if prediction.shape!=(4,6) or not np.isfinite(prediction).all():raise ValueError('Nonfinite Dreamer output')
            self._action_queue=list(prediction);self._executed=[];self._previous_decision_time=now
            elapsed=(time.perf_counter()-began)*1000;self.inference_latencies.append(elapsed)
            if len(self.inference_latencies)==1 or len(self.inference_latencies)%25==0:
                self.get_logger().info('DREAMER_DECISION '+json.dumps({'decision':len(self.inference_latencies),'sim_time':now,'latency_ms':elapsed,'context_offset':self.controller.offset,'executed_previous_commands':0 if previous is None else 4}))
        return self._action_queue.pop(0)

    def set_pose_target(self,move_robot,pose,frame_id='base_link'):
        if frame_id!='base_link' or self._command_observed_pose is None:raise ValueError('TCP delta must be composed from its observed pose at the physical API boundary')
        motion=pose_to_position_motion_update(pose,stamp=self.get_clock().now().to_msg(),frame_id=frame_id,stiffness=DEFAULT_CARTESIAN_STIFFNESS,damping=DEFAULT_CARTESIAN_DAMPING)
        result=move_robot(motion_update=motion)
        if result is not True:raise RuntimeError('Command publication was not accepted; refusing fabricated executed history')
        if self._decision_started_at is not None:
            latency=(time.perf_counter()-self._decision_started_at)*1000;self.command_latencies.append(latency);self._decision_started_at=None
            self.get_logger().info('DREAMER_COMMAND_LATENCY '+json.dumps({'decision':len(self.command_latencies),'observation_to_command_ms':latency,'phase_ms':dict(getattr(self,'_decision_phase_ms',{}),command_tail_ms=latency-sum(getattr(self,'_decision_phase_ms',{}).values())),'includes':'RGB parse+letterbox+transfer+physical state+causal model+output transfer+pose conversion+bounds+ROS publication callback'}))
        raw=[pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
        self._executed.append(self._body_delta(self._command_observed_pose,np.array(raw)))
        return result

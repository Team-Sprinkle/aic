"""Observation-only causal temporal transport actor for bounded development evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torchvision.models import resnet18
from aic_control_interfaces.msg import TargetMode
from aic_model.policy import build_pose_from_vectors

from .RunSharedPortTCP import RunSharedPortTCP

REPO_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO_ROOT/'scripts'))
from train_temporal_transport_probe import AGES,LONG_AGES,TemporalTransport  # noqa: E402
from train_route_aware_diffusion_transport import RouteDiffusion  # noqa: E402
from train_sc_transport_goal_probe import GoalProbe  # noqa: E402
from train_sc_native_keypoints import PretrainedKeypointRefiner  # noqa: E402
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px  # noqa: E402
from shared_bc_camera_preprocessing import jpeg_roundtrip_rgb,transport_rgb_tensor  # noqa: E402
from sc_shadow_handoff_verifier import SCShadowHandoffVerifier  # noqa: E402
from handoff_region_prior import load_prior, allows as handoff_prior_allows  # noqa: E402


class RunTemporalTransport(RunSharedPortTCP):
    def __init__(self,parent_node):
        # Native RGB/JPEG preprocessing and tiny GPU models suffer when the
        # ROS process inherits the host's large default CPU thread pool.
        torch.set_num_threads(int(os.environ.get('AIC_TEMPORAL_CPU_THREADS','8')))
        cv2.setNumThreads(2)
        super().__init__(parent_node)
        # Scored Gazebo camera captures are JPEG quality 86 before the shared
        # resize path. This optional replay ablation restores that first codec
        # step for SFP while leaving the frozen policy weights unchanged.
        sfp_native_quality=os.environ.get('AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY','')
        if sfp_native_quality:
            value=int(sfp_native_quality)
            if not 1<=value<=100:raise ValueError('Invalid SFP native JPEG quality')
            self.native_jpeg_quality_by_family['sfp']=value
        path=Path(os.environ['AIC_TEMPORAL_TRANSPORT_CHECKPOINT'])
        saved=torch.load(path,map_location='cpu',weights_only=False)
        manifest=saved['source_manifest']
        stop_manifest=manifest
        diffusion_stop_path=os.environ.get('AIC_ROUTE_DIFFUSION_CHECKPOINT','')
        if diffusion_stop_path:
            stop_manifest=torch.load(diffusion_stop_path,map_location='cpu',weights_only=False)['source_manifest']
        self.zero_previous_command=bool(manifest.get('zero_previous_command',False))
        stop_text=os.environ.get('AIC_TEMPORAL_TRANSPORT_STOP_THRESHOLD','')
        self.stop_threshold=float(stop_text) if stop_text else None
        prior_path=os.environ.get('AIC_TEMPORAL_HANDOFF_PRIOR_JSON','')
        self.handoff_region_prior=load_prior(prior_path) if prior_path else None
        if self.stop_threshold is not None and self.handoff_region_prior is None:
            raise ValueError('Learned handoff requires a fitted TCP-base region veto')
        self.sfp_diagnostic_stop=os.environ.get('AIC_SFP_DIAGNOSTIC_STOP','0')=='1'
        if self.stop_threshold is not None and not (
                stop_manifest.get('handoff_aux_loss',0.) or stop_manifest.get('sfp_handoff_aux_loss',0.)):
            raise ValueError('Learned transport stop requires a trained handoff head')
        self.handoff_supported_families=stop_manifest.get('handoff_supported_families',['sc','sfp'])
        self.stop_streak_required=int(os.environ.get('AIC_TEMPORAL_TRANSPORT_STOP_STREAK','3'))
        self.stop_min_sim_s=float(os.environ.get('AIC_TEMPORAL_TRANSPORT_STOP_MIN_SIM_S','0'))
        self.stop_requires_stage=bool(stop_manifest.get('stage_aux_loss',0.))
        goal_path=os.environ.get('AIC_SC_GOAL_PROBE_CHECKPOINT','')
        self.goal_probe=None
        if goal_path:
            goal_saved=torch.load(goal_path,map_location='cpu',weights_only=False)
            if not goal_saved.get('counterfactual_port_augmentation'):
                raise ValueError('SC goal servo requires target-port contrastive supervision')
            self.goal_probe=GoalProbe().to(self.device).eval()
            self.goal_probe.load_state_dict(goal_saved['model'])
        self.goal_trigger_m=float(os.environ.get('AIC_SC_GOAL_SERVO_TRIGGER_M','.035'))
        self.goal_step_m=float(os.environ.get('AIC_SC_GOAL_SERVO_STEP_M','.008'))
        self.goal_stop_m=float(os.environ.get('AIC_SC_GOAL_SERVO_STOP_M','.006'))
        self.goal_stop_only=os.environ.get('AIC_SC_GOAL_STOP_ONLY','0')=='1'
        self.goal_stop_only_threshold_m=float(os.environ.get('AIC_SC_GOAL_STOP_ONLY_THRESHOLD_M','.030'))
        self.goal_stop_only_streak_required=int(os.environ.get('AIC_SC_GOAL_STOP_ONLY_STREAK','3'))
        if self.goal_stop_only and self.goal_probe is None:
            raise ValueError('Goal-based transport stop needs an observation-only goal probe')
        visual_path=os.environ.get('AIC_SC_VISUAL_STANDOFF_CHECKPOINT','')
        self.visual_locator=None
        if visual_path:
            if self.goal_probe is not None or self.goal_stop_only:
                raise ValueError('Select one SC visual transport diagnostic')
            saved_visual=torch.load(visual_path,map_location='cpu',weights_only=False)
            if saved_visual.get('architecture')!='dual_sc_port_spatial_heatmaps':
                raise ValueError('Expected explicit two-port SC locator')
            self.visual_locator=PretrainedKeypointRefiner(None,False,point_count=3).to(self.device).eval()
            self.visual_locator.load_state_dict(saved_visual['model'])
            calibration=json.loads(Path(os.environ['AIC_SC_VISUAL_STANDOFF_CALIBRATION']).read_text())
            if calibration['schema']!='aic_sc_visual_standoff_calibration/v1':
                raise ValueError('Unexpected SC camera/standoff calibration')
            self.visual_tcp_optical={name:np.asarray(value,dtype=float)
                                     for name,value in calibration['tcp_to_optical'].items()}
            self.visual_standoff_offset=np.asarray(
                calibration['opening_to_transport_tcp_base_offset_m'],dtype=float)
            self.visual_transport_orientation=calibration.get(
                'transport_tcp_orientation_xyzw_training_mean')
        self.visual_trigger_m=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_TRIGGER_M','.060'))
        self.visual_stop_mode=os.environ.get('AIC_SC_VISUAL_STOP_MODE','visual')
        if self.visual_stop_mode not in ('visual','transport_head'):
            raise ValueError('AIC_SC_VISUAL_STOP_MODE must be visual or transport_head')
        self.visual_step_m=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_STEP_M','.008'))
        self.visual_stop_m=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_STOP_M','.008'))
        self.visual_servo_stiffness=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_STIFFNESS','0'))
        self.visual_servo_damping=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_DAMPING','80'))
        self.visual_servo_orientation=os.environ.get('AIC_SC_VISUAL_STANDOFF_ORIENTATION','0')=='1'
        self.visual_orientation_yaw_blend=float(os.environ.get('AIC_SC_VISUAL_STANDOFF_ORIENTATION_YAW_BLEND','1'))
        if not 0<=self.visual_orientation_yaw_blend<=1:
            raise ValueError('Orientation yaw blend must be in [0,1]')
        self.sc_transport_z_floor_m=float(os.environ.get('AIC_SC_TRANSPORT_Z_FLOOR_M','0'))
        visual_orientation_path=os.environ.get('AIC_SC_VISUAL_STANDOFF_ORIENTATION_PROBE_CHECKPOINT','')
        self.visual_orientation_probe=None
        if visual_orientation_path:
            orientation_saved=torch.load(visual_orientation_path,map_location='cpu',weights_only=False)
            if not orientation_saved.get('onpolicy_adaptation'):
                raise ValueError('Visual orientation probe needs grouped actor-view adaptation')
            self.visual_orientation_probe=GoalProbe().to(self.device).eval()
            self.visual_orientation_probe.load_state_dict(orientation_saved['model'])
        if self.visual_servo_orientation and self.visual_transport_orientation is None and self.visual_orientation_probe is None:
            raise ValueError('Orientation servo needs training-only orientation calibration')
        self.visual_direct_native_rgb=os.environ.get('AIC_SC_DIRECT_NATIVE_RGB','0')=='1'
        self.shadow_direct_native_rgb=os.environ.get('AIC_SC_SHADOW_DIRECT_NATIVE_RGB','0')=='1'
        self.sc_shadow_verifier=None
        self.shadow_history_enabled=os.environ.get('AIC_SC_SHADOW_HISTORY_AUDIT','0')=='1'
        if os.environ.get('AIC_SC_SHADOW_HANDOFF','0')=='1':
            self.sc_shadow_verifier=SCShadowHandoffVerifier(
                os.environ['AIC_SC_SHADOW_LOCATOR'],
                os.environ['AIC_SC_SHADOW_IDENTITY'],
                os.environ['AIC_SC_SHADOW_ORIENTATION'],
                os.environ['AIC_SC_SHADOW_REFINER'],
                os.environ['AIC_SC_SHADOW_CALIBRATION'],
                os.environ['AIC_SC_SHADOW_BOARD_MOUNT'],
                os.environ['AIC_SC_SHADOW_TCP_PRIOR'],device=self.device,
                crop_identity_path=os.environ.get('AIC_SC_SHADOW_CROP_IDENTITY') or None,
                diagnose_identity_failure=os.environ.get(
                    'AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS','0')=='1')
        if manifest['feature_dim']!=1536 or not manifest['full_state_skip']:
            raise ValueError('Temporal checkpoint requires independent ImageNet views and full ordinary state')
        if manifest['history_ages_s'] not in (list(LONG_AGES),list(AGES)):
            raise ValueError('Temporal checkpoint history window mismatch')
        self.no_history=not saved['history_enabled']
        self.ages_s=manifest['history_ages_s']
        weights_path=Path(os.environ['AIC_TEMPORAL_TRANSPORT_IMAGENET'])
        self.backbone=resnet18(weights=None)
        self.backbone.fc=torch.nn.Identity()
        weights=torch.load(weights_path,map_location='cpu',weights_only=True)
        self.backbone.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
        self.backbone.to(self.device).eval()
        self.temporal=TemporalTransport(saved['model']['target_mean'].numpy(),
                                        saved['model']['target_std'].numpy(),
                                        raw_state_skip=True,feature_dim=1536,state_task_dim=42,
                                        route_conditioning=manifest['route_conditioning'],
                                        task_route_film=manifest.get('task_route_film',False),
                                        spatial_request_attention=manifest.get('spatial_request_attention',False),
                                        route_goal_conditioning=manifest.get('route_goal_conditioning',False))
        self.temporal.load_state_dict(saved['model'])
        self.temporal.to(self.device).eval()
        self.route_goal_ridge=None
        self.route_goal_zero_control=bool(manifest.get('route_goal_zero_control',False))
        self.route_goal_progress_gate_mm=manifest.get('route_goal_progress_gate_mm')
        if self.temporal.route_goal_conditioning:
            if not saved.get('route_goal_ridge'):
                raise ValueError('Route-goal actor checkpoint lacks its frozen probe')
            self.route_goal_ridge={k:v.cpu().numpy() for k,v in saved['route_goal_ridge'].items()}
        self.state_mean=self.actor.state_mean.detach().cpu().numpy()
        self.state_std=self.actor.state_std.detach().cpu().numpy()
        self.target_mean=self.temporal.target_mean.detach().cpu().numpy()
        self.target_std=self.temporal.target_std.detach().cpu().numpy()
        self.route_diffusion=None
        diffusion_path=os.environ.get('AIC_ROUTE_DIFFUSION_CHECKPOINT','')
        if diffusion_path:
            diffusion_saved=torch.load(diffusion_path,map_location='cpu',weights_only=False)
            diffusion_manifest=diffusion_saved['source_manifest']
            action_slot_s=float(diffusion_manifest.get('action_slot_s',.05))
            if abs(action_slot_s-1./self.control_hz)>1e-6:
                raise ValueError(f'Route actor slots are {action_slot_s}s but controller period is {1./self.control_hz}s')
            if diffusion_manifest.get('prediction_type')!='velocity':
                raise ValueError('Route diffusion runtime requires velocity prediction')
            if self.temporal.route_goal_conditioning or self.temporal.spatial_request_attention:
                raise ValueError('Route diffusion requires the ordinary 1536D temporal runtime')
            horizon=int(diffusion_manifest.get('action_horizon',4))
            self.execute_steps=int(diffusion_manifest.get('execute_steps',horizon))
            if not 1<=self.execute_steps<=horizon:raise ValueError('Invalid diffusion execution length')
            self.route_diffusion=RouteDiffusion(1536,self.target_mean,self.target_std,
                                                action_horizon=horizon,
                                                handoff_head=bool(diffusion_manifest.get('handoff_aux_loss',0.)))
            self.route_diffusion.load_state_dict(diffusion_saved['model'])
            if diffusion_manifest.get('history_ages_s'):
                ages=[float(x) for x in diffusion_manifest['history_ages_s']]
                if not ages or ages[-1]!=0. or any(a<=b for a,b in zip(ages,ages[1:])):
                    raise ValueError('Invalid diffusion history ages')
                self.ages_s=ages
            if (not np.allclose(self.route_diffusion.target_mean.cpu().numpy(),self.target_mean) or
                not np.allclose(self.route_diffusion.target_std.cpu().numpy(),self.target_std)):
                raise ValueError('Diffusion/temporal action normalization mismatch')
            self.route_diffusion.to(self.device).eval()
            self.diffusion_sample_steps=int(os.environ.get(
                'AIC_ROUTE_DIFFUSION_SAMPLE_STEPS') or diffusion_manifest['inference_steps'])
            if os.environ.get('AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS','0')=='1':
                if diffusion_manifest.get('prior_dropout')!=1.:
                    raise ValueError('Zero-prior runtime requires a checkpoint trained with prior_dropout=1')
                self.zero_previous_command=True
            if not 1<=self.diffusion_sample_steps<=100:
                raise ValueError('Invalid diffusion sample step count')
            self.get_logger().info(f'Route-aware diffusion actor: {diffusion_path}')
        self._clear_temporal()
        if self.route_diffusion is not None and os.environ.get('AIC_ROUTE_DIFFUSION_PREWARM','0')=='1':
            self._prewarm_route_models()
        self.get_logger().info(f'Causal temporal transport actor: {path}, history={not self.no_history}')

    def _prewarm_route_models(self):
        """Initialize CUDA kernels before the first scored camera decision."""
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pixels=torch.zeros((3,3,224,224),device=self.device)
            self.backbone(pixels)
            for count in (1,2,4):
                zero=lambda *shape:torch.zeros(shape,device=self.device)
                data=dict(history=torch.arange(count,device=self.device)[None],
                          mask=torch.ones((1,count),device=self.device,dtype=torch.bool),
                          features=zero(count,1536),prev=zero(count,36),
                          motion=zero(count,6),age=zero(1,count),gap=zero(1,count),
                          camera_clock=zero(count,4),state_task_full=zero(count,42))
                context=self.route_diffusion.context(data,torch.zeros(1,device=self.device,dtype=torch.long))
                self.route_diffusion.sample(context,seed=20261011,
                                            sample_steps=self.diffusion_sample_steps)
        torch.cuda.synchronize(self.device)

    def _clear_temporal(self):
        self.records=[]
        self.route_goal_start_xyz=None
        self.latest_proposal_chunk=None
        self.previous_state=None
        self.last_handoff_probability=0.
        self.last_stage5_probability=0.
        self.stop_streak=0
        self.goal_base_filtered=None
        self.goal_trigger_streak=0
        self.goal_stop_streak=0
        self.goal_servo_active=False
        self.last_goal_prediction=None
        self.goal_stop_only_streak=0
        self.visual_targets=deque(maxlen=5)
        self.visual_streak=0
        self.visual_stop_streak=0
        self.visual_active=False
        self.visual_last_target=None
        self.visual_last_good_sim=None
        self.floor_guard_active=False
        self.shadow_cooldown_until=-float('inf')
        self.shadow_inspections=0
        self.shadow_target_tcp_orientation=None
        self.shadow_history_anchor_base=None
        self.shadow_history_anchor_port_quat=None
        self.shadow_history_anchor_board_spread_deg=None
        self.shadow_history_request=None
        self.shadow_history_last_sample_tcp=None
        self.shadow_history_last_sample_sim=-float('inf')
        self.shadow_history_samples=0
        self.shadow_history_cumulative_travel_m=0.
        self.shadow_history_last_refresh_travel_m=None
        self.shadow_native_snapshot_dir=os.environ.get('AIC_SC_SHADOW_NATIVE_SNAPSHOT_DIR','')
        self.shadow_native_snapshot_count=0

    def _save_shadow_native_snapshot(self, kind, sim_time, images, state, requested, report):
        """Opt-in lossless diagnostic capture; never read by the policy."""
        limit=16 if kind=='qualified_history' else 24
        if not self.shadow_native_snapshot_dir or self.shadow_native_snapshot_count>=limit:
            return
        directory=Path(self.shadow_native_snapshot_dir)
        directory.mkdir(parents=True,exist_ok=True)
        index=self.shadow_native_snapshot_count
        stem=f'{index:02d}_{kind}_{round(sim_time*1000):08d}'
        paths={}
        for camera in ('center','left','right'):
            path=directory/f'{stem}_{camera}.png'
            if not cv2.imwrite(str(path),images[camera][...,::-1]):
                raise RuntimeError(f'Failed to write diagnostic RGB {path}')
            paths[camera]=str(path)
        payload=dict(kind=kind,sim_time_s=float(sim_time),requested_port=int(requested),
                     state=np.asarray(state,float).tolist(),images=paths,
                     anchor_opening_base_m=(self.shadow_history_anchor_base.tolist() if
                                            self.shadow_history_anchor_base is not None else None),
                     anchor_port_quat_xyzw=(self.shadow_history_anchor_port_quat.tolist() if
                                            self.shadow_history_anchor_port_quat is not None else None),
                     anchor_board_spread_deg=self.shadow_history_anchor_board_spread_deg,
                     anchor_refresh_travel_m=self.shadow_history_last_refresh_travel_m,
                     cumulative_travel_m=self.shadow_history_cumulative_travel_m,
                     report_reason=report.get('reason'),
                     report_valid=bool(report.get('valid')),
                     report_orientation_source=report.get('orientation_source'))
        (directory/f'{stem}.json').write_text(json.dumps(payload,indent=2)+'\n')
        self.shadow_native_snapshot_count+=1

    def _shadow_history_sample(self, observation, state, task_vector, sim_time):
        """Observation-only approach anchor audit; never changes control or handoff."""
        if not self.shadow_history_enabled or self.sc_shadow_verifier is None:
            return
        if task_vector[0]==1 or self.last_elapsed_sim_sec<30:
            return
        if self.previous_state is not None:
            self.shadow_history_cumulative_travel_m+=float(np.linalg.norm(
                np.asarray(state[:3],float)-np.asarray(self.previous_state[:3],float)))
        requested=int(np.argmax(task_vector[2:4]))
        if self.shadow_history_request != requested:
            self.shadow_history_request=requested
            self.shadow_history_anchor_base=None
            self.shadow_history_anchor_port_quat=None
            self.shadow_history_anchor_board_spread_deg=None
            self.shadow_history_last_refresh_travel_m=None
            self.shadow_history_last_sample_tcp=None
            self.shadow_history_last_sample_sim=-float('inf')
        position=np.asarray(state[:3],float)
        previous=self.shadow_history_last_sample_tcp
        spacing=.010 if self.shadow_history_anchor_base is None else .020
        traveled=(float(np.linalg.norm(position-previous[:3])) if previous is not None
                  else float('inf'))
        angle=(float((Rotation.from_quat(previous[3:7]).inv()*
                      Rotation.from_quat(state[3:7])).magnitude()) if previous is not None
               else float('inf'))
        if (previous is not None and traveled<spacing and angle<np.deg2rad(5)) or \
                sim_time-self.shadow_history_last_sample_sim<.2:
            return
        native={cam:self._shadow_raw_rgb(getattr(observation,cam+'_image'))
                for cam in ('center','left','right')}
        tick=time.perf_counter()
        report=self.sc_shadow_verifier.inspect(native,state,requested,
                                                diagnose_identity_failure=False)
        elapsed_ms=(time.perf_counter()-tick)*1000
        self.shadow_history_last_sample_tcp=np.asarray(state[:7],float).copy()
        self.shadow_history_last_sample_sim=sim_time
        self.shadow_history_samples+=1
        point=report.get('predicted_opening_base_m')
        eligible=(point is not None and
                  report.get('predicted_port_quat_xyzw') is not None and
                  min(report.get('spatial_identity_probability',[0]))>=.5 and
                  report.get('opening_reprojection_px',float('inf'))<=2 and
                  report.get('board_view_spread_deg',float('inf'))<=1 and
                  report.get('measured_force_norm_n',float('inf'))<=40)
        distance=None
        if eligible:
            point=np.asarray(point,float)
            if self.shadow_history_anchor_base is None:
                self.shadow_history_anchor_base=point
                self.shadow_history_anchor_port_quat=np.asarray(
                    report['predicted_port_quat_xyzw'],float)
                self.shadow_history_anchor_board_spread_deg=float(
                    report['board_view_spread_deg'])
                self.shadow_history_last_refresh_travel_m=self.shadow_history_cumulative_travel_m
                distance=0.
            else:
                distance=1000*float(np.linalg.norm(point-self.shadow_history_anchor_base))
                old=Rotation.from_quat(self.shadow_history_anchor_port_quat)
                new=Rotation.from_quat(report['predicted_port_quat_xyzw'])
                angle_deg=float(np.rad2deg((old.inv()*new).magnitude()))
                if distance<=5 and angle_deg<=2:
                    self.shadow_history_anchor_base=(.8*self.shadow_history_anchor_base+
                                                     .2*point)
                    self.shadow_history_anchor_port_quat=(old*Rotation.from_rotvec(
                        (old.inv()*new).as_rotvec()*.2)).as_quat()
                    self.shadow_history_anchor_board_spread_deg=max(
                        self.shadow_history_anchor_board_spread_deg,
                        float(report['board_view_spread_deg']))
                    self.shadow_history_last_refresh_travel_m=self.shadow_history_cumulative_travel_m
            self._save_shadow_native_snapshot('qualified_history',sim_time,native,state,
                                              requested,report)
        self._trace(event='shadow_history_sample',sim_time=sim_time,
                    requested_port=requested,spacing_m=spacing,
                    measured_travel_since_sample_mm=(traveled*1000 if np.isfinite(traveled)
                                                     else None),
                    verification_ms=elapsed_ms,reason=report['reason'],
                    identity_scores=report.get('spatial_identity_probability'),
                    opening_reprojection_px=report.get('opening_reprojection_px'),
                    anchor_distance_mm=distance,
                    anchor_base_m=(self.shadow_history_anchor_base.tolist() if
                                   self.shadow_history_anchor_base is not None else None),
                    anchor_port_quat_xyzw=(self.shadow_history_anchor_port_quat.tolist() if
                                           self.shadow_history_anchor_port_quat is not None else None),
                    geometry_input=False)

    def _bound_transport_target(self, absolute, task_vector, state):
        self.floor_guard_active=False
        if (task_vector[0]!=1 and self.sc_transport_z_floor_m>0 and
                not self.visual_active and absolute[2,3]<self.sc_transport_z_floor_m):
            original=float(absolute[2,3])
            floor=self.sc_transport_z_floor_m
            # Limit the commanded vertical correction to 8 mm per command.
            absolute=absolute.copy()
            absolute[2,3]=max(original,min(floor,float(state[2])+.008))
            if absolute[2,3]<state[2]-.008:
                absolute[2,3]=state[2]-.008
            self.floor_guard_active=True
            self._trace(event='sc_transport_z_floor_guard',sim_time=self._control_sim_time_from_state_fallback(),
                        measured_z_m=float(state[2]),raw_target_z_m=original,
                        bounded_target_z_m=float(absolute[2,3]),floor_m=floor,geometry_input=False)
        return absolute

    def _control_sim_time_from_state_fallback(self):
        return self.first_sim_time+self.last_elapsed_sim_sec if self.first_sim_time is not None else None

    def _cartesian_gains(self):
        if self.visual_active and self.visual_servo_stiffness>0:
            return ([self.visual_servo_stiffness]*3+self.cartesian_stiffness[3:],
                    [self.visual_servo_damping]*3+self.cartesian_damping[3:])
        if self.floor_guard_active:
            stiffness=list(self.cartesian_stiffness);damping=list(self.cartesian_damping)
            stiffness[2]=1200.;damping[2]=80.
            return stiffness,damping
        return super()._cartesian_gains()

    def _should_stop_transport(self,task_vector):
        if (self.sc_shadow_verifier is not None and
                (self._control_sim_time_from_state_fallback()<self.shadow_cooldown_until or
                 self.shadow_inspections>=3)):
            return False
        if (self.visual_locator is not None and self.visual_stop_mode=='visual' and
                task_vector[0]!=1 and self.visual_stop_streak>=3):
            self._trace(event='visual_standoff_complete',predicted_standoff_base_m=self.visual_last_target,
                        streak=self.visual_stop_streak,geometry_input=False)
            return True
        if self.visual_locator is not None and self.visual_stop_mode=='visual':
            return False
        if self.goal_stop_only and task_vector[0]!=1 and self.goal_stop_only_streak>=self.goal_stop_only_streak_required:
            self._trace(event='goal_stop_only_complete',predicted_relative_pose=self.last_goal_prediction,
                        streak=self.goal_stop_only_streak,geometry_input=False)
            return True
        if self.goal_probe is not None and task_vector[0]!=1 and self.goal_stop_streak>=3:
            self._trace(event='goal_servo_complete',predicted_relative_pose=self.last_goal_prediction,
                        streak=self.goal_stop_streak,geometry_input=False)
            return True
        family='sfp' if task_vector[0]==1 else 'sc'
        if family not in self.handoff_supported_families:
            return False
        if self.stop_threshold is None or (task_vector[0]==1 and not self.sfp_diagnostic_stop):
            return False
        in_region=handoff_prior_allows(self.handoff_region_prior,task_vector,
                                      self.previous_state[:3])
        eligible=(in_region and self.last_elapsed_sim_sec>=self.stop_min_sim_s and
                  self.last_handoff_probability>=self.stop_threshold and
                  (not self.stop_requires_stage or self.last_stage5_probability>=.7))
        self.stop_streak=self.stop_streak+1 if eligible else 0
        self._trace(event='transport_handoff_signal',
                    probability=self.last_handoff_probability,
                    prior_region_allows=in_region,
                    stage5_probability=self.last_stage5_probability,
                    streak=self.stop_streak,threshold=self.stop_threshold,
                    geometry_input=False)
        return self.stop_streak>=self.stop_streak_required

    @staticmethod
    def _shadow_raw_rgb(message):
        encoding=message.encoding.lower()
        channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(encoding)
        if channels is None:raise ValueError(f'Unsupported RGB encoding {message.encoding}')
        raw=np.frombuffer(message.data,np.uint8).reshape(message.height,message.step)
        rgb=raw[:,:message.width*channels].reshape(message.height,message.width,channels)[...,:3]
        return np.ascontiguousarray(rgb[...,::-1] if encoding.startswith('bgr') else rgb)

    def _inspect_transport_handoff(self, observed, state, task_vector,
                                   get_observation, move_robot, decision_sim):
        if self.sc_shadow_verifier is None or task_vector[0]==1:
            return True
        self.shadow_inspections+=1
        # Hold exactly the measured TCP pose while new camera frames arrive.
        pose=build_pose_from_vectors(state[:3],state[3:7])
        if hasattr(self._parent_node,'_target_mode'):
            self._parent_node._target_mode=TargetMode.MODE_CARTESIAN
        stiffness,damping=self._cartesian_gains()
        self.set_pose_target(move_robot=move_robot,pose=pose,frame_id='base_link',
                             stiffness=stiffness,damping=damping)
        self._trace(event='shadow_handoff_hold',sim_time=decision_sim,
                    measured_tcp_pose=state[:7].tolist(),inspection=self.shadow_inspections,
                    geometry_input=False)
        fresh=None;deadline=time.monotonic()+5.
        while time.monotonic()<deadline:
            candidate=get_observation()
            if candidate is not None and self._control_sim_time(candidate)>=decision_sim+.2:
                fresh=candidate;break
            time.sleep(.01)
        if fresh is None:
            report={'valid':False,'reason':'fresh_images_timed_out'}
            fresh_sim=decision_sim
            verify_ms=None
        else:
            fresh_sim=self._control_sim_time(fresh)
            fresh_state=self.assembler.assemble_ros(fresh)
            started=time.perf_counter()
            native={}
            for camera in ('center','left','right'):
                rgb=self._shadow_raw_rgb(getattr(fresh,camera+'_image'))
                native[camera]=(rgb if self.shadow_direct_native_rgb
                                else jpeg_roundtrip_rgb(rgb,86))
            requested=int(np.argmax(task_vector[2:4]))
            use_history_orientation=(
                os.environ.get('AIC_SC_SHADOW_USE_HISTORY_ORIENTATION','0')=='1' and
                self.shadow_history_request==requested and
                self.shadow_history_anchor_base is not None and
                self.shadow_history_anchor_port_quat is not None and
                self.shadow_history_last_refresh_travel_m is not None and
                self.shadow_history_cumulative_travel_m-
                    self.shadow_history_last_refresh_travel_m<=.150)
            report=self.sc_shadow_verifier.inspect(
                native,fresh_state,requested,
                diagnose_identity_failure=True if self.shadow_history_enabled else None,
                history_opening_base_m=(self.shadow_history_anchor_base if
                                        use_history_orientation else None),
                history_port_quat_xyzw=(self.shadow_history_anchor_port_quat if
                                        use_history_orientation else None),
                history_board_spread_deg=(self.shadow_history_anchor_board_spread_deg if
                                          use_history_orientation else None))
            self._save_shadow_native_snapshot('fresh_inspection',fresh_sim,native,
                                              fresh_state,requested,report)
            verify_ms=(time.perf_counter()-started)*1000
            if self.shadow_history_enabled:
                point=report.get('predicted_opening_base_m')
                anchor=self.shadow_history_anchor_base
                distance=(1000*float(np.linalg.norm(np.asarray(point)-anchor))
                          if point is not None and anchor is not None else None)
                travel_since_refresh_mm=(
                    1000*(self.shadow_history_cumulative_travel_m-
                          self.shadow_history_last_refresh_travel_m)
                    if self.shadow_history_last_refresh_travel_m is not None else None)
                history_match=(self.shadow_history_request==requested and
                               distance is not None and distance<=5 and
                               travel_since_refresh_mm is not None and
                               travel_since_refresh_mm<=150 and
                               report.get('opening_reprojection_px',float('inf'))<=2 and
                               report.get('board_view_spread_deg',float('inf'))<=1)
                self._trace(event='shadow_history_handoff_diagnostic',sim_time=fresh_sim,
                            requested_port=requested,anchor_distance_mm=distance,
                            travel_since_anchor_refresh_mm=travel_since_refresh_mm,
                            geometry_match=history_match,
                            would_accept_with_history=bool(history_match and
                                report.get('measured_tcp_posture_error_deg',float('inf'))<=3 and
                                report.get('measured_tcp_standoff_error_mm',float('inf'))<=15 and
                                report.get('measured_force_norm_n',float('inf'))<=40),
                            original_valid=report['valid'],original_reason=report['reason'],
                            geometry_input=False)
            if (os.environ.get('AIC_SC_SHADOW_IDENTITY_RETRY','0')=='1' and
                    report['reason']=='requested_identity_not_in_all_views'):
                scores=report.get('spatial_identity_probability',[])
                # A borderline single-camera miss can be temporal. Keep the
                # robot held and inspect newer frames; every retry still has
                # to pass the original all-view, geometry and safety gates.
                if len(scores)==3 and min(scores)>=.15 and sum(p>=.5 for p in scores)>=2:
                    for attempt in range(8):
                        retry_deadline=time.monotonic()+2.
                        newer=None
                        while time.monotonic()<retry_deadline:
                            candidate=get_observation()
                            if candidate is not None and self._control_sim_time(candidate)>=fresh_sim+.2:
                                newer=candidate;break
                            time.sleep(.01)
                        if newer is None:break
                        fresh_sim=self._control_sim_time(newer)
                        fresh_state=self.assembler.assemble_ros(newer)
                        started=time.perf_counter()
                        retry_native={c:self._shadow_raw_rgb(getattr(newer,c+'_image'))
                                      for c in ('center','left','right')}
                        retry=self.sc_shadow_verifier.inspect(retry_native,fresh_state,requested)
                        retry_ms=(time.perf_counter()-started)*1000
                        native=retry_native
                        self._trace(event='shadow_identity_retry',sim_time=fresh_sim,
                                    attempt=attempt+1,verification_ms=retry_ms,
                                    reason=retry['reason'],valid=retry['valid'],
                                    spatial_identity_probability=retry.get('spatial_identity_probability'),
                                    geometry_input=False)
                        report=retry;verify_ms=retry_ms
                        if retry['reason']!='requested_identity_not_in_all_views':break
            if os.environ.get('AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST','0')=='1':
                # Development-only frozen-model stress check. Its result never
                # changes the real requested-port control decision.
                swapped=self.sc_shadow_verifier.inspect(native,fresh_state,1-requested)
                self._trace(event='shadow_counterfactual_request',sim_time=fresh_sim,
                            actual_requested_port=requested,
                            counterfactual_requested_port=1-requested,
                            counterfactual_valid=swapped['valid'],
                            counterfactual_reason=swapped['reason'],
                            counterfactual_report=swapped,
                            geometry_input=False)
        self._trace(event='shadow_handoff_inspection',sim_time=fresh_sim,
                    proposed_at_sim_time=decision_sim,inspection=self.shadow_inspections,
                    verification_ms=verify_ms,**report)
        if report['valid']:
            return True
        # A rejected visual stop can still provide a reliable board-frame
        # orientation. Correct it while held at the transport standoff, then
        # require another fresh-image inspection before accepting handoff.
        if (report.get('reason') in ('wrong_tcp_posture','requested_identity_not_in_all_views') and
                (report.get('reason')=='wrong_tcp_posture' or
                 (os.environ.get('AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS','0')=='1' and
                  report.get('orientation_correction_candidate',False))) and
                report.get('opening_reprojection_px',float('inf'))<=2. and
                report.get('board_view_spread_deg',float('inf'))<=1. and
                report.get('measured_force_norm_n',float('inf'))<=40. and
                report.get('measured_tcp_posture_error_deg',float('inf'))<=10.):
            port=Rotation.from_quat(report['predicted_port_quat_xyzw'])
            self.shadow_target_tcp_orientation=port*self.sc_shadow_verifier.nominal_tcp_in_port
            self._trace(event='shadow_orientation_target',sim_time=fresh_sim,
                        target_tcp_quat_xyzw=self.shadow_target_tcp_orientation.as_quat().tolist(),
                        measured_error_deg=report['measured_tcp_posture_error_deg'],
                        geometry_input=False)
        self.visual_stop_streak=0
        self.shadow_cooldown_until=fresh_sim+2.
        return False

    def reset_for_task(self,task):
        super().reset_for_task(task)
        self._clear_temporal()

    def insert_cable(self,*args,**kwargs):
        self._clear_temporal()
        return super().insert_cable(*args,**kwargs)

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        started=time.perf_counter()
        family='sfp' if task_vector[0]==1 else 'sc'
        native_quality=self.native_jpeg_quality_by_family.get(family)
        native_visual=None
        center_stamp=observation.center_image.header.stamp
        center_sim=float(center_stamp.sec)+float(center_stamp.nanosec)*1e-9
        visual_due=(family=='sc' and self.visual_locator is not None and
                    self.first_sim_time is not None and center_sim-self.first_sim_time>=30.)
        if visual_due:
            actor_images=[];visual_images=[]
            for name in ('center','left','right'):
                message=getattr(observation,name+'_image')
                channels=4 if message.encoding.lower() in ('rgba8','bgra8') else 3
                raw=np.frombuffer(message.data,np.uint8).reshape(message.height,message.step)
                rgb=raw[:,:message.width*channels].reshape(message.height,message.width,channels)[...,:3]
                if message.encoding.lower().startswith('bgr'):rgb=rgb[...,::-1]
                decoded=(rgb if self.visual_direct_native_rgb else jpeg_roundtrip_rgb(rgb,86))
                actor_images.append(transport_rgb_tensor(decoded,native_already_encoded=True))
                # Matched held-out actor views show cv2 area resize preserves
                # opening p95 while avoiding three costly PIL native resizes.
                small=cv2.resize(decoded,(288,256),interpolation=cv2.INTER_AREA)
                visual_images.append(torch.from_numpy(small).permute(2,0,1))
            images=torch.stack(actor_images).to(self.device)
            native_visual=torch.stack(visual_images).to(self.device)
        else:
            images=torch.stack([self.camera_tensor(getattr(observation,name+'_image'),native_quality)
                                for name in ('center','left','right')]).to(self.device)
        state=self.assembler.assemble_ros(observation)
        if state.shape!=(32,):raise ValueError('Expected ordinary 32D measured state')
        stamps=[]
        for name in ('center','left','right'):
            stamp=getattr(observation,name+'_image').header.stamp
            stamps.append(float(stamp.sec)+float(stamp.nanosec)*1e-9)
        sim_time=stamps[0]
        if self.first_sim_time is None:self.first_sim_time=sim_time
        self.last_elapsed_sim_sec=max(0.,sim_time-self.first_sim_time)
        self._shadow_history_sample(observation,state,task_vector,sim_time)
        normalized=(state-self.state_mean)/self.state_std
        normalized[self.actor.masked_state_indices]=0
        state_task=np.r_[normalized,np.asarray(task_vector,dtype=np.float32)]
        previous=np.zeros(36,dtype=np.float32)
        if self.latest_proposal_chunk is not None and not self.zero_previous_command:
            previous=((self.latest_proposal_chunk[:4]-self.target_mean)/self.target_std).reshape(-1)
        motion=np.zeros(6,dtype=np.float32)
        if self.previous_state is not None:
            motion[:3]=state[:3]-self.previous_state[:3]
            rotation=Rotation.from_quat(self.previous_state[3:7]).inv()*Rotation.from_quat(state[3:7])
            motion[3:]=rotation.as_rotvec()
        camera_clock=np.r_[np.maximum(0.,sim_time-np.asarray(stamps))/.05,
                           0. if family=='sfp' else 1.].astype(np.float32)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            x=images.float()/255.
            x=((x-x.new_tensor((.485,.456,.406))[None,:,None,None]) /
               x.new_tensor((.229,.224,.225))[None,:,None,None])
            spatial=None
            if self.temporal.spatial_request_attention:
                x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
                x=self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(x)))
                spatial=torch.nn.functional.adaptive_avg_pool2d(x,(4,5))
                spatial=spatial.reshape(3,256,20).transpose(1,2).float()
                feature=self.backbone.avgpool(self.backbone.layer4(x)).reshape(1536).float()
            else:
                feature=self.backbone(x).reshape(1536).float()
        route_goal=None
        if self.temporal.route_goal_conditioning:
            route_goal=np.zeros(3,dtype=np.float32)
            if self.route_goal_start_xyz is None:
                self.route_goal_start_xyz=state[:3].copy()
            within_goal_region=(self.route_goal_progress_gate_mm is None or
                np.linalg.norm(state[:3]-self.route_goal_start_xyz)*1000<=self.route_goal_progress_gate_mm)
            if family=='sfp' and not self.route_goal_zero_control and within_goal_region:
                ridge=self.route_goal_ridge
                public=np.r_[feature.float().cpu().numpy(),state[:7],
                             np.asarray(task_vector,dtype=np.float32)]
                if public.shape!=(len(ridge['center']),):
                    raise ValueError('Runtime route-goal input dimension mismatch')
                base=((public-ridge['center'])/ridge['scale'])@ridge['coefficients']+ridge['target_mean']
                route_goal=(Rotation.from_quat(state[3:7]).inv().apply(base)/.05).astype(np.float32)
                if not np.isfinite(route_goal).all():
                    raise ValueError('Nonfinite route-goal prediction')
        self.records.append(dict(time=sim_time,feature=feature,previous=previous,
                                 motion=motion,state_task=state_task,camera_clock=camera_clock))
        self.previous_state=state.copy()
        chosen=[]
        times=np.asarray([record['time'] for record in self.records])
        for age in self.ages_s:
            j=np.searchsorted(times,sim_time-age,side='right')-1
            if j>=0 and (age==0 or sim_time-times[j]<=max(self.ages_s)) and j not in chosen:
                chosen.append(int(j))
        chosen.sort()
        if self.no_history:chosen=chosen[-1:]
        records=[self.records[j] for j in chosen]
        age=np.asarray([(sim_time-r['time'])/4 for r in records],dtype=np.float32)
        gap=np.asarray([0.]+[(records[j]['time']-records[j-1]['time'])/4
                               for j in range(1,len(records))],dtype=np.float32)
        # The live history uses exactly the same causal age and gap units as the
        # offline history builder, with teacher actions replaced by issued targets.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            feat=torch.stack([r['feature'] for r in records])[None]
            tensor=lambda values:torch.as_tensor(np.asarray(values),device=self.device)[None].float()
            if self.route_diffusion is None:
                action,stage,_,handoff=self.temporal(feat,tensor([r['previous'] for r in records]),
                                        tensor([r['motion'] for r in records]),tensor(age)[0:1],
                                        tensor(gap)[0:1],tensor([r['camera_clock'] for r in records]),
                                        torch.ones((1,len(records)),device=self.device,dtype=torch.bool),
                                        tensor([r['state_task'] for r in records]),
                                        spatial=(spatial[None] if spatial is not None else None),
                                        route_goal=(torch.as_tensor(route_goal,device=self.device)[None]
                                                    if route_goal is not None else None))
                target=action*self.temporal.target_std+self.temporal.target_mean
                self.last_handoff_probability=float(handoff.sigmoid()[0].item())
                self.last_stage5_probability=float(stage.softmax(-1)[0,5].item())
            else:
                diffusion_started=time.perf_counter()
                raw_clock=tensor([r['camera_clock'] for r in records])[0].clone()
                raw_clock[:,:3]*=.05
                history_data=dict(
                    history=torch.arange(len(records),device=self.device)[None],
                    mask=torch.ones((1,len(records)),device=self.device,dtype=torch.bool),
                    features=feat[0],prev=tensor([r['previous'] for r in records])[0],
                    motion=tensor([r['motion'] for r in records])[0],
                    age=tensor(age),gap=tensor(gap),
                    camera_clock=raw_clock,
                    state_task_full=tensor([r['state_task'] for r in records])[0])
                context=self.route_diffusion.context(history_data,
                    torch.zeros(1,device=self.device,dtype=torch.long))
                diffusion_context_done=time.perf_counter()
                action,predicted_route=self.route_diffusion.sample(context,
                    seed=20261011,sample_steps=self.diffusion_sample_steps)
                diffusion_sample_done=time.perf_counter()
                target=action*self.route_diffusion.target_std+self.route_diffusion.target_mean
                self.last_handoff_probability=(float(self.route_diffusion.handoff_head(context).float().sigmoid()[0,0])
                                              if self.route_diffusion.handoff_head is not None else 0.)
                self.last_stage5_probability=0.
                self._trace(event='diffusion_route_goal',sim_time=sim_time,
                    predicted_1s_tcp_body_mm=(predicted_route.float()[0].cpu().numpy()*50).tolist(),
                    before_diffusion_ms=(diffusion_started-started)*1000,
                    context_ms=(diffusion_context_done-diffusion_started)*1000,
                    sample_ms=(diffusion_sample_done-diffusion_context_done)*1000,
                    privileged_input=False)
            if family=='sc' and self.goal_probe is not None:
                goal_input=torch.cat((feature,torch.as_tensor(state_task,device=self.device)))
                goal=self.goal_probe(goal_input[None]).float()[0].cpu().numpy()
                goal=goal*np.asarray((.2,.2,.2,.3,.3,.3),dtype=np.float32)
                if not np.isfinite(goal).all():raise ValueError('Nonfinite visual standoff pose')
                self.last_goal_prediction=goal.tolist()
                tcp_rot=Rotation.from_quat(state[3:7])
                base_candidate=state[:3]+tcp_rot.apply(goal[:3])
                self.goal_base_filtered=(base_candidate if self.goal_base_filtered is None else
                                         .65*self.goal_base_filtered+.35*base_candidate)
                distance=float(np.linalg.norm(goal[:3]))
                angle_deg=float(np.linalg.norm(goal[3:])*180/np.pi)
                if self.goal_stop_only:
                    ready=(self.last_elapsed_sim_sec>=15 and
                           distance<=self.goal_stop_only_threshold_m and angle_deg<=5.)
                    self.goal_stop_only_streak=self.goal_stop_only_streak+1 if ready else 0
                eligible=(self.last_elapsed_sim_sec>=15 and
                          distance<=self.goal_trigger_m and angle_deg<=3.)
                if not self.goal_stop_only:
                    self.goal_trigger_streak=self.goal_trigger_streak+1 if eligible else 0
                    if self.goal_trigger_streak>=3:self.goal_servo_active=True
                if self.goal_servo_active:
                    relative=tcp_rot.inv().apply(self.goal_base_filtered-state[:3])
                    rel_norm=float(np.linalg.norm(relative))
                    translation=relative*min(1.,self.goal_step_m/max(rel_norm,1e-9))
                    rotation=goal[3:]*min(1.,.035/max(float(np.linalg.norm(goal[3:])),1e-9))
                    matrix=Rotation.from_rotvec(rotation).as_matrix()
                    body_pose=np.r_[translation,matrix[:,0],matrix[:,1]].astype(np.float32)
                    target=torch.as_tensor(np.tile(body_pose,(1,4,1)),device=self.device)
                    stop_eligible=(distance<=self.goal_stop_m and angle_deg<=3. and
                                   rel_norm<=.010)
                    self.goal_stop_streak=self.goal_stop_streak+1 if stop_eligible else 0
                self._trace(event='goal_servo_signal',sim_time=sim_time,
                            predicted_distance_mm=distance*1000,
                            predicted_orientation_deg=angle_deg,
                            trigger_streak=self.goal_trigger_streak,
                            active=self.goal_servo_active,stop_streak=self.goal_stop_streak,
                            predicted_relative_pose=goal.tolist(),geometry_input=False)
            if visual_due:
                visual_input=native_visual.float()/255.
                points=self.visual_locator(visual_input).float().cpu().numpy()
                points[:,0::2]*=1151;points[:,1::2]*=1023
                port=int(np.argmax(task_vector[2:4]));base_tcp=np.eye(4)
                base_tcp[:3,:3]=Rotation.from_quat(state[3:7]).as_matrix()
                base_tcp[:3,3]=state[:3]
                camera_poses={name:base_tcp@self.visual_tcp_optical[name]
                              for name in ('center','left','right')}
                pixels={name:points[j,2+2*port:4+2*port]
                        for j,name in enumerate(('center','left','right'))}
                opening=triangulate(pixels,camera_poses)
                residual=reprojection_residual_px(opening,pixels,camera_poses)
                candidate=opening+self.visual_standoff_offset
                distance=float(np.linalg.norm(candidate-state[:3]))
                steady=bool(self.visual_last_target is None or
                            np.linalg.norm(candidate-np.asarray(self.visual_last_target))<.005)
                pre_gate=(self.last_elapsed_sim_sec>=30 and distance<=.08 and
                          np.isfinite(candidate).all() and residual<=2.)
                usable=pre_gate and steady
                ready=(distance<=self.visual_trigger_m and usable)
                self.visual_streak=self.visual_streak+1 if ready else 0
                if self.visual_streak>=3:self.visual_active=True
                if usable:
                    self.visual_targets.append(candidate)
                    self.visual_last_good_sim=sim_time
                if self.visual_targets:
                    filtered=np.median(np.stack(self.visual_targets),axis=0)
                    self.visual_last_target=filtered.tolist()
                if self.visual_active and self.visual_targets:
                    relative=Rotation.from_quat(state[3:7]).inv().apply(filtered-state[:3])
                    remaining=float(np.linalg.norm(relative))
                    translation=relative*min(1.,self.visual_step_m/max(remaining,1e-9))
                    if self.visual_servo_orientation:
                        if self.shadow_target_tcp_orientation is not None:
                            desired=self.shadow_target_tcp_orientation
                            rotation=(Rotation.from_quat(state[3:7]).inv()*desired).as_rotvec()
                        elif self.visual_orientation_probe is not None:
                            goal_input=torch.cat((feature,torch.as_tensor(state_task,device=self.device)))
                            predicted=self.visual_orientation_probe(goal_input[None]).float()[0]
                            raw_relative=predicted[3:].cpu().numpy()*.3
                            if self.visual_orientation_yaw_blend<1:
                                prior=Rotation.from_quat(self.visual_transport_orientation)
                                learned=Rotation.from_quat(state[3:7])*Rotation.from_rotvec(raw_relative)
                                prior_euler=prior.as_euler('xyz',degrees=True)
                                learned_euler=learned.as_euler('xyz',degrees=True)
                                yaw_difference=(learned_euler[2]-prior_euler[2]+180.)%360.-180.
                                stable_euler=prior_euler.copy()
                                stable_euler[2]+=self.visual_orientation_yaw_blend*yaw_difference
                                desired=Rotation.from_euler('xyz',stable_euler,degrees=True)
                                rotation=(Rotation.from_quat(state[3:7]).inv()*desired).as_rotvec()
                            else:
                                rotation=raw_relative
                        else:
                            desired=Rotation.from_quat(self.visual_transport_orientation)
                            rotation=(Rotation.from_quat(state[3:7]).inv()*desired).as_rotvec()
                        angular_remaining=float(np.linalg.norm(rotation)*180/np.pi)
                        rotation*=min(1.,np.deg2rad(2.)/max(np.linalg.norm(rotation),1e-9))
                        matrix=Rotation.from_rotvec(rotation).as_matrix()
                        orientation=np.r_[matrix[:,0],matrix[:,1]]
                    else:
                        # The baseline preserves the actor's complete body-frame orientation target.
                        orientation=target[0,0,3:].float().cpu().numpy()
                    body=np.r_[translation,orientation].astype(np.float32)
                    target=torch.as_tensor(np.tile(body,(1,4,1)),device=self.device)
                    angle=angular_remaining if self.visual_servo_orientation else 0.
                    # The three-camera fit can fluctuate by a pixel at the goal.
                    # A recent accepted target is sufficient for stopping if the
                    # robot has stayed close to that target for three decisions.
                    good_age=(sim_time-self.visual_last_good_sim if self.visual_last_good_sim is not None
                              else float('inf'))
                    stale_shadow_probe=(
                        self.sc_shadow_verifier is not None and
                        os.environ.get('AIC_SC_SHADOW_ALLOW_STALE_PAUSE','0')=='1' and
                        good_age<float('inf') and remaining<=.003 and angle<=1.)
                    stop_eligible=(remaining<=self.visual_stop_m and angle<=1. and
                                   good_age<=1.0) or stale_shadow_probe
                    self.visual_stop_streak=self.visual_stop_streak+1 if stop_eligible else 0
                self._trace(event='visual_standoff_signal',sim_time=sim_time,
                            selected_port=port,reprojection_residual_px=residual,
                            predicted_standoff_base_m=candidate.tolist(),
                            predicted_distance_mm=distance*1000,steady=steady,
                            predicted_orientation_remaining_deg=angle if self.visual_active else None,
                            trigger_streak=self.visual_streak,active=self.visual_active,
                            stop_streak=self.visual_stop_streak,
                            last_good_visual_age_s=(sim_time-self.visual_last_good_sim
                                if self.visual_last_good_sim is not None else None),
                            stale_shadow_probe=stale_shadow_probe if self.visual_active else False,
                            geometry_input=False)
        self.latest_proposal_chunk=target[0].float().cpu().numpy()
        self._inference_latency_ms.append((time.perf_counter()-started)*1000)
        return self.latest_proposal_chunk.copy(),np.zeros(9,dtype=np.float32),state

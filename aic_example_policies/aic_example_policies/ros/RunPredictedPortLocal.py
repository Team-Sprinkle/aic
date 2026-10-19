"""Near-port RGB-estimated-frame insertion BC diagnostic for SFP and SC.

No simulator TF, CheatCode target, depth, or scoring data are read here.
This policy expects a near-port start; it does not implement transport or a
calibrated full-start visibility handoff.
"""
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from .RunSharedPortTCP import RunSharedPortTCP,REPO_ROOT

sys.path.insert(0,str(REPO_ROOT/'scripts'))
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px  # noqa: E402
from train_two_family_port_locator import Locator,CAMERAS,NAMES  # noqa: E402
from train_two_family_port_orientation import Model  # noqa: E402
from train_sc_native_keypoints import PretrainedKeypointRefiner  # noqa: E402
from shared_port_tcp_geometry import matrix_from_pose9  # noqa: E402
from sfp_port_cascade import SfpPortCascade  # noqa: E402


def pose9(matrix):return np.r_[matrix[:3,3],matrix[:3,:2].T.reshape(-1)].astype(np.float32)


class RunPredictedPortLocal(RunSharedPortTCP):
    def __init__(self,parent_node,checkpoint_path=None):
        # The rootless Gazebo container exposes all host CPU cores. 128-way
        # OpenMP/OpenCV pools made this small four-network pipeline take ~1 s.
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        super().__init__(parent_node,checkpoint_path)
        self.actor_device=torch.device(os.environ.get('AIC_LOCAL_ACTOR_DEVICE',str(self.device)))
        if self.actor_device!=self.device:
            torch.set_num_threads(8)
            self.actor.to(self.actor_device)
        if self.target_frame!='estimated_port':raise ValueError('Local actor must use estimated port targets')
        locator_path=Path(os.environ['AIC_LOCAL_LOCATOR_CHECKPOINT'])
        orientation_path=Path(os.environ['AIC_LOCAL_ORIENTATION_CHECKPOINT'])
        calibration_path=Path(os.environ['AIC_LOCAL_CAMERA_CALIBRATION'])
        sfp_cascade_paths=[os.environ.get(key) for key in (
            'AIC_LOCAL_SFP_COARSE_CHECKPOINT','AIC_LOCAL_SFP_BROAD_CHECKPOINT',
            'AIC_LOCAL_SFP_FINE_CHECKPOINT','AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT')]
        if any(sfp_cascade_paths) and not all(sfp_cascade_paths):
            raise ValueError('SFP cascade requires coarse, broad, fine, and orientation checkpoints')
        self.sfp_cascade=(SfpPortCascade(*sfp_cascade_paths,calibration_path,device=self.device,
                          orientation_path_secondary=os.environ.get('AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT_2'),
                          pair_orientation=os.environ.get('AIC_LOCAL_SFP_PAIR_ORIENTATION','0')=='1')
                          if all(sfp_cascade_paths) else None)
        self.locator=Locator(None).to(self.device).eval()
        self.locator.load_state_dict(torch.load(locator_path,map_location='cpu',weights_only=False)['model'])
        data=torch.load(orientation_path,map_location='cpu',weights_only=False)
        self.orientation=Model(data['model']['mean'],data['model']['std']).to(self.device).eval()
        self.orientation.load_state_dict(data['model'])
        sfp_orientation_path=os.environ.get('AIC_LOCAL_SFP_ORIENTATION_CHECKPOINT')
        self.sfp_orientation=None
        if sfp_orientation_path:
            sfp_data=torch.load(sfp_orientation_path,map_location='cpu',weights_only=False)
            self.sfp_orientation=Model(sfp_data['model']['mean'],
                                       sfp_data['model']['std']).to(self.device).eval()
            self.sfp_orientation.load_state_dict(sfp_data['model'])
        calibration=json.loads(calibration_path.read_text())
        self.tcp_to_optical={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
        refiner_path=os.environ.get('AIC_LOCAL_SC_REFINER_CHECKPOINT')
        self.sc_refiner=None
        if refiner_path:
            self.sc_refiner=PretrainedKeypointRefiner(None,False,point_count=1).to(self.device).eval()
            self.sc_refiner.load_state_dict(torch.load(refiner_path,map_location='cpu',weights_only=False)['model'])
        sfp_refiner_path=os.environ.get('AIC_LOCAL_SFP_REFINER_CHECKPOINT')
        self.sfp_refiner=None
        if sfp_refiner_path:
            self.sfp_refiner=PretrainedKeypointRefiner(None,False,point_count=1).to(self.device).eval()
            self.sfp_refiner.load_state_dict(torch.load(sfp_refiner_path,map_location='cpu',weights_only=False)['model'])
        self.last_base_port=None;self.last_good_time=None
        self.locked_base_port=None;self.lock_samples=[];self.lock_time=None
        self.sfp_lock_samples=[]
        self.sfp_required_lock_samples=int(os.environ.get('AIC_LOCAL_SFP_LOCK_SAMPLES','5'))
        self.sfp_decision_limit=int(os.environ.get('AIC_SFP_LOCAL_DECISION_LIMIT','0'))
        if not 0<=self.sfp_decision_limit<=100:
            raise ValueError('SFP diagnostic decision limit must be 0–100')
        if not 1<=self.sfp_required_lock_samples<=10:
            raise ValueError('SFP lock sample count must be 1–10')
        prior_path=os.environ.get('AIC_LOCAL_SC_ORIENTATION_PRIOR')
        self.sc_orientation_prior=(Rotation.from_quat(json.loads(Path(prior_path).read_text())[
            'nominal_tcp_in_port_orientation_xyzw']) if prior_path else None)
        self.sc_orientation_max_deg=float(os.environ.get('AIC_LOCAL_SC_ORIENTATION_MAX_DEG','5'))
        if not 0<self.sc_orientation_max_deg<=15:raise ValueError('Invalid SC orientation constraint')
        self._perception_trace=[]
        self.local_history_samples=[]

    def _constrain_sc_orientation(self,targets):
        if self.sc_orientation_prior is None:return targets,None
        corrected=targets.copy();angles=[];clamped=0
        for row in corrected:
            predicted=Rotation.from_matrix(matrix_from_pose9(row)[:3,:3])
            relative=self.sc_orientation_prior.inv()*predicted
            rotvec=relative.as_rotvec();angle=float(np.linalg.norm(rotvec))
            angles.append(float(np.rad2deg(angle)))
            limit=float(np.deg2rad(self.sc_orientation_max_deg))
            if angle>limit:
                constrained=self.sc_orientation_prior*Rotation.from_rotvec(rotvec*(limit/angle))
                row[3:9]=constrained.as_matrix()[:,:2].T.reshape(-1)
                clamped+=1
        return corrected,{'nominal_from_successful_train_targets':True,
                          'max_deviation_deg':self.sc_orientation_max_deg,
                          'raw_first_target_deviation_deg':angles[0],
                          'clamped_chunk_targets':clamped,
                          'runtime_true_geometry':False}

    @staticmethod
    def raw_rgb(message):
        enc=message.encoding.lower()
        count={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(enc)
        if count is None:raise ValueError(enc)
        raw=np.frombuffer(message.data,np.uint8).reshape(message.height,message.step)
        rgb=raw[:,:message.width*count].reshape(message.height,message.width,count)[...,:3]
        return np.ascontiguousarray(rgb[...,::-1] if enc.startswith('bgr') else rgb)

    @staticmethod
    def rgb_for_training(native,family,*,fast_locked_sc=False):
        if family=='sc' and not fast_locked_sc:
            ok,encoded=cv2.imencode('.jpg',native[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
            if not ok:raise RuntimeError('SC JPEG preprocessing failed')
            native=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        small=cv2.resize(native,(288,256),interpolation=cv2.INTER_AREA)
        if family=='sfp':
            ok,encoded=cv2.imencode('.jpg',small[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise RuntimeError('SFP JPEG preprocessing failed')
            small=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        return native,small

    def _refine_port(self,rgb,coarse,refiner,*,jpeg_crop=False):
        cx,cy=coarse;left=int(round(cx-80));top=int(round(cy-80))
        crop=cv2.warpAffine(rgb,np.array([[1,0,-left],[0,1,-top]],np.float32),
                            (160,160),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        if jpeg_crop:
            ok,encoded=cv2.imencode('.jpg',crop[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
            if not ok:raise RuntimeError('Native SFP crop JPEG preprocessing failed')
            crop=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(np.ascontiguousarray(crop)).permute(2,0,1)[None].to(self.device).float()/255.
        with torch.inference_mode(),torch.autocast(self.device.type,dtype=torch.bfloat16):
            point=refiner(tensor).float().cpu().numpy().reshape(2)*159+[left,top]
        return point

    def _estimate(self,native,small,state,task_vector,sim_time):
        family='sfp' if task_vector[0]==1 else 'sc'
        if family=='sfp' and self.sfp_cascade is not None:
            if self.locked_base_port is not None:
                return self.locked_base_port,{'static_visual_lock':True,
                    'lock_age_s':sim_time-self.lock_time,'lock_samples':self.sfp_required_lock_samples,
                    'sfp_native_cascade':True,'visibility_certified':False}
            estimate,diagnostic=self.sfp_cascade.predict(native,small,state,task_vector)
            diagnostic['sfp_native_cascade']=True
            if estimate is not None:
                self.sfp_lock_samples.append((sim_time,estimate.copy()))
                self.sfp_lock_samples=self.sfp_lock_samples[-self.sfp_required_lock_samples:]
                if len(self.sfp_lock_samples)==self.sfp_required_lock_samples:
                    matrices=[sample[1] for sample in self.sfp_lock_samples]
                    xyz=np.stack([m[:3,3] for m in matrices])
                    rotations=Rotation.from_matrix(np.stack([m[:3,:3] for m in matrices]))
                    center=rotations.mean()
                    spread_m=float(np.max(np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1)))
                    spread_deg=float(np.max((center.inv()*rotations).magnitude())*180/np.pi)
                    if spread_m<=.0005 and spread_deg<=5:
                        locked=np.eye(4);locked[:3,3]=np.median(xyz,axis=0)
                        locked[:3,:3]=center.as_matrix()
                        self.locked_base_port=locked;self.lock_time=sim_time
                        estimate=locked
                        diagnostic['static_visual_lock_established']=True
                        diagnostic['position_spread_mm']=spread_m*1000
                        diagnostic['orientation_spread_deg']=spread_deg
                self.last_base_port=estimate;self.last_good_time=sim_time
                return estimate,diagnostic
            if self.last_base_port is not None and sim_time-self.last_good_time<=1.0:
                diagnostic['carried_age_s']=sim_time-self.last_good_time
                return self.last_base_port,diagnostic
            raise RuntimeError(f'SFP selected port unavailable: {diagnostic}')
        # The selected port is rigid and stationary for one task. Once five
        # independent, mutually consistent native SC views establish its pose,
        # retain that observation-only estimate through cable/plug occlusion.
        if family=='sc' and self.locked_base_port is not None:
            return self.locked_base_port,{'static_visual_lock':True,
                                          'lock_age_s':sim_time-self.lock_time,
                                          'lock_samples':5,
                                          'visibility_certified':False}
        if family=='sfp':
            card=int(np.argmax(task_vector[4:9]));port=int(np.argmax(task_vector[2:4]))
            identity=f'sfp_card_{card}_port_{port}'
        else:identity=f'sc_port_{int(np.argmax(task_vector[2:4]))}'
        selected=NAMES.index(identity)
        batch=torch.stack([torch.from_numpy(np.ascontiguousarray(small[c])).permute(2,0,1) for c in CAMERAS]).to(self.device).float()/255.
        meta=np.zeros((3,5),np.float32);meta[:,0 if family=='sc' else 1]=1
        for i in range(3):meta[i,2+i]=1
        with torch.inference_mode(),torch.autocast(self.device.type,dtype=torch.bfloat16):
            coordinates,flag,features=self.locator(batch,torch.as_tensor(meta,device=self.device),return_features=True)
        coords=coordinates.float().cpu().numpy()*[1151,1023]
        probabilities=flag.float().sigmoid().cpu().numpy()
        base_tcp=np.eye(4);base_tcp[:3,3]=state[:3]
        base_tcp[:3,:3]=Rotation.from_quat(state[3:7]).as_matrix()
        optics={c:base_tcp@self.tcp_to_optical[c] for c in CAMERAS}
        candidate=[];mask=[];selected_pixels={};selected_base=None
        for j,name in enumerate(NAMES):
            pixels={c:coords[i,j] for i,c in enumerate(CAMERAS) if probabilities[i,j]>=.5}
            if len(pixels)>=2:
                try:point=triangulate(pixels,optics)
                except np.linalg.LinAlgError:point=np.zeros(3)
                valid=float(np.isfinite(point).all())
            else:point=np.zeros(3);valid=0.
            candidate.extend(point.tolist() if valid else [0.,0.,0.]);mask.append(valid)
            if j==selected and valid:selected_base=point;selected_pixels=pixels
        if selected_base is None:
            if self.last_base_port is not None and self.last_good_time is not None:
                return self.last_base_port,{'carried_occlusion_s':sim_time-self.last_good_time,'views':0}
            raise RuntimeError('Selected port has fewer than two RGB-supported views')
        code=np.zeros(len(NAMES),np.float32);code[selected]=1
        feature=np.r_[candidate,mask,code,state[:7],features.float().cpu().numpy().reshape(-1)].astype(np.float32)
        orientation_model=(self.sfp_orientation if family=='sfp' and self.sfp_orientation is not None
                           else self.orientation)
        with torch.inference_mode():
            rotation=orientation_model(torch.as_tensor(feature,device=self.device)[None]).float().cpu().numpy()[0]
        rough_residual=reprojection_residual_px(selected_base,selected_pixels,optics)
        used_refiner=False
        refiner=self.sc_refiner if family=='sc' else self.sfp_refiner
        if refiner is not None and np.linalg.norm(selected_base-state[:3])<.2:
            refined={}
            for cam,pixel in selected_pixels.items():
                point=self._refine_port(native[cam],pixel,refiner,jpeg_crop=family=='sfp')
                if np.linalg.norm(point-pixel)<=15:refined[cam]=point
            if len(refined)>=2:
                refined_base=triangulate(refined,optics)
                residual=reprojection_residual_px(refined_base,refined,optics)
                if residual<=4 and np.linalg.norm(refined_base-selected_base)<.008:
                    selected_base=refined_base;used_refiner=True
                    selected_pixels=refined;rough_residual=residual
        base_port=np.eye(4);base_port[:3,3]=selected_base;base_port[:3,:3]=rotation
        if rough_residual>8:
            if self.last_base_port is not None and self.last_good_time is not None:
                return self.last_base_port,{'carried_disagreement_s':sim_time-self.last_good_time,
                                            'views':len(selected_pixels),'reprojection_px':rough_residual}
            raise RuntimeError('Selected port RGB views disagree')
        if family=='sc' and used_refiner and rough_residual<=2:
            self.lock_samples.append((sim_time,base_port.copy()))
            self.lock_samples=self.lock_samples[-5:]
            if len(self.lock_samples)==5:
                matrices=[item[1] for item in self.lock_samples]
                xyz=np.stack([m[:3,3] for m in matrices])
                rotations=Rotation.from_matrix(np.stack([m[:3,:3] for m in matrices]))
                center=rotations.mean()
                spread_m=float(np.max(np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1)))
                spread_deg=float(np.max((center.inv()*rotations).magnitude())*180/np.pi)
                if spread_m<=.001 and spread_deg<=3:
                    locked=np.eye(4);locked[:3,3]=np.median(xyz,axis=0)
                    locked[:3,:3]=center.as_matrix()
                    self.locked_base_port=locked;self.lock_time=sim_time
                    base_port=locked
                    diagnostic_lock={'static_visual_lock_established':True,
                                     'position_spread_mm':spread_m*1000,
                                     'orientation_spread_deg':spread_deg}
                else:diagnostic_lock={}
            else:diagnostic_lock={}
        else:diagnostic_lock={}
        self.last_base_port=base_port;self.last_good_time=sim_time
        return base_port,{'views':len(selected_pixels),'reprojection_px':rough_residual,
                          'native_refined':used_refiner,'selected_identity':identity,
                          'visibility_certified':False,**diagnostic_lock}

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        start=time.perf_counter();family='sfp' if task_vector[0]==1 else 'sc'
        native={};small={};actor_images=[]
        fast_locked_sc=(family=='sc' and self.locked_base_port is not None and
                        os.environ.get('AIC_LOCAL_FAST_LOCKED_SC_PREPROCESS','false').lower()=='true')
        for cam in CAMERAS:
            native[cam],small[cam]=self.rgb_for_training(
                self.raw_rgb(getattr(observation,cam+'_image')),family,
                fast_locked_sc=fast_locked_sc)
            actor_rgb=small[cam]
            actor_images.append(torch.from_numpy(np.asarray(Image.fromarray(actor_rgb).resize(
                (224,224),Image.Resampling.BILINEAR)).copy()).permute(2,0,1))
        after_images=time.perf_counter()
        state=self.assembler.assemble_ros(observation)
        stamp=observation.center_image.header.stamp
        sim_time=float(stamp.sec)+float(stamp.nanosec)*1e-9
        base_tcp=np.eye(4);base_tcp[:3,3]=state[:3]
        base_tcp[:3,:3]=Rotation.from_quat(state[3:7]).as_matrix()
        try:
            base_port,diagnostic=self._estimate(native,small,state,task_vector,sim_time)
        except RuntimeError as exc:
            if family!='sfp' or self.sfp_cascade is None:raise
            base_port=self.last_base_port if self.last_base_port is not None else base_tcp
            diagnostic={'sfp_native_cascade':True,'reason':str(exc),
                        'visibility_certified':False,'static_visual_lock':False}
        diagnostic['fast_locked_sc_preprocessing']=fast_locked_sc
        after_estimate=time.perf_counter()
        estimated_pose=pose9(np.linalg.inv(base_port)@base_tcp)
        if self.actor.use_local_history:
            self.local_history_samples.append((sim_time,state.copy()))
            self.local_history_samples=self.local_history_samples[-4:]
            history=self.local_history_samples
            history=[history[max(0,len(history)-4+k)] for k in range(4)]
            local_history=[]
            for past_time,past_state in history:
                past_base_tcp=np.eye(4)
                past_base_tcp[:3,3]=past_state[:3]
                past_base_tcp[:3,:3]=Rotation.from_quat(past_state[3:7]).as_matrix()
                aligned_past_pose=pose9(np.linalg.inv(base_port)@past_base_tcp)
                local_history.append([*past_state,*aligned_past_pose,
                                      max(0.,min(2.,sim_time-past_time))])
            local_history=np.asarray(local_history,np.float32)
        else:local_history=None
        if family=='sfp' and self.sfp_cascade is not None and self.locked_base_port is None:
            # The port is static. Repeated independent RGB observations must
            # agree before a target is allowed to move the robot. This keeps a
            # single geometrically consistent wrong-port hallucination from
            # immediately becoming an insertion command.
            result=np.tile(estimated_pose,(4,1))
            diagnostic['held_for_visual_consensus']=True
        else:
            images=torch.stack(actor_images)[None].to(self.actor_device)
            with torch.inference_mode(),torch.autocast(self.actor_device.type,
                                                      dtype=torch.bfloat16,
                                                      enabled=self.actor_device.type=='cuda'):
                target,_=self.actor(images,torch.as_tensor(state,device=self.actor_device)[None],
                                    torch.as_tensor(task_vector,dtype=torch.float32,device=self.actor_device)[None],
                                    external_pose=torch.as_tensor(estimated_pose,device=self.actor_device)[None],
                                    external_mask=torch.ones(1,device=self.actor_device,dtype=torch.bool),
                                    local_history=(torch.as_tensor(local_history,device=self.actor_device)[None]
                                                   if local_history is not None else None))
            result=target[0].float().cpu().numpy()
        if family=='sc':
            result,guard=self._constrain_sc_orientation(result)
            if guard is not None:diagnostic['orientation_constraint']=guard
        after_actor=time.perf_counter()
        self._inference_latency_ms.append((time.perf_counter()-start)*1000)
        diagnostic['timing_ms']={'images':(after_images-start)*1000,
                                 'estimate':(after_estimate-after_images)*1000,
                                 'actor':(after_actor-after_estimate)*1000}
        self._perception_trace.append({'sim_time':sim_time,'diagnostic':diagnostic,
                                       'inference_ms':self._inference_latency_ms[-1]})
        self._trace(event='local_perception',sim_time=sim_time,
                    predicted_base_port_position=base_port[:3,3].tolist(),
                    predicted_tcp_pose_in_port=estimated_pose.tolist(),
                    diagnostic=diagnostic,inference_ms=self._inference_latency_ms[-1],
                    geometry_input=False)
        return result,estimated_pose,state

    def insert_cable(self,*args,**kwargs):
        self.last_base_port=None;self.last_good_time=None;self._perception_trace=[]
        self.locked_base_port=None;self.lock_samples=[];self.lock_time=None
        self.sfp_lock_samples=[]
        self.local_history_samples=[]
        return super().insert_cable(*args,**kwargs)

    def _should_stop_transport(self,task_vector):
        # A diagnostic teacher takeover sees actual actor-created states.
        # The actor itself receives no privileged geometry or teacher target.
        return bool(task_vector[0]==1 and self.sfp_decision_limit and
                    len(self._inference_latency_ms)>self.sfp_decision_limit)

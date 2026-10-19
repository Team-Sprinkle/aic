"""Observation-only SC shadow handoff check for the live transport runner.

Returns a decision but never commands insertion. Scored TF, scene files and
teacher targets are deliberately absent from this module.
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_board_lines_yaw import camera_yaw
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sc_spatial_identity_probe import SelectedHead
from train_sc_observation_crop_identity import CropIdentity, crop
from train_two_family_port_locator import CAMERAS, NAMES, Locator
from train_two_family_port_orientation import Model


class SCShadowHandoffVerifier:
    def __init__(self, locator_path, identity_path, orientation_path,
                 refiner_path, calibration_path, board_mount_path,
                 tcp_prior_path, device='cuda:0', identity_scale=1,
                 crop_identity_path=None, diagnose_identity_failure=False,
                 candidate_association_path=None, candidate_gate_mode='intersection',
                 requested_locator_path=None):
        self.device = torch.device(device)
        if identity_scale not in (1, 2):
            raise ValueError('identity_scale must be 1 or 2')
        self.identity_scale=identity_scale
        self.candidate_association=None
        if candidate_association_path is not None:
            from native_candidate_association import NativeCandidateAssociation
            self.candidate_association=NativeCandidateAssociation(candidate_association_path,device)
        from requested_candidate_gate import candidate_gate
        # Validate the opt-in contract before commanding any motion.
        candidate_gate([1.,1.,1.], [1.,1.,1.] if self.candidate_association is not None else None,
                       candidate_gate_mode)
        self.candidate_gate_mode=candidate_gate_mode
        self.requested_locator=None
        if requested_locator_path is not None:
            from sc_requested_history_observation import SCRequestedHistoryObservation
            self.requested_locator=SCRequestedHistoryObservation(requested_locator_path,device)
        self.board_yaw_search_rad=.07
        self.board_yaw_prior_line_deg=5.
        self.reject_yaw_search_boundary=False
        self.locator = Locator(None).to(self.device).eval()
        self.locator.load_state_dict(torch.load(locator_path, map_location='cpu',
                                                weights_only=False)['model'])
        identity = torch.load(identity_path, map_location='cpu', weights_only=False)
        if identity['variant'] != 'spatial':
            raise ValueError('SC shadow requires request-conditioned spatial identity')
        self.identity = SelectedHead(True, 256, 12).to(self.device).eval()
        self.identity.load_state_dict(identity['model'])
        self.identity_mask_tcp_orientation=bool(identity.get('mask_tcp_orientation',False))
        self.crop_identity=None
        self.diagnose_identity_failure=bool(diagnose_identity_failure)
        if crop_identity_path is not None:
            saved_crop=torch.load(crop_identity_path,map_location='cpu',weights_only=False)
            self.crop_identity=CropIdentity(256,12).to(self.device).eval()
            self.crop_identity.load_state_dict(saved_crop['model'])
            if saved_crop['crop_size_original_px']!=160:
                raise ValueError('Unexpected original RGB crop size')
            self.crop_state_mean=saved_crop['state_mean'].to(self.device)
            self.crop_state_std=saved_crop['state_std'].to(self.device)
        self.state_mean = identity['state_mean'].to(self.device)
        self.state_std = identity['state_std'].to(self.device)
        saved = torch.load(orientation_path, map_location='cpu', weights_only=False)
        self.orientation = Model(saved['model']['mean'], saved['model']['std']).to(self.device).eval()
        self.orientation.load_state_dict(saved['model'])
        saved_refiner=torch.load(refiner_path,map_location='cpu',weights_only=False)
        self.joint_refiner=saved_refiner.get('schema')=='joint_native_opening_refiner/v1'
        if self.joint_refiner:
            from train_joint_native_opening_refiner import JointRefiner
            self.refiner=JointRefiner(None).to(self.device).eval()
        else:
            self.refiner=PretrainedKeypointRefiner(None,False,point_count=1).to(self.device).eval()
        self.refiner.load_state_dict(saved_refiner['model'])
        calibration = json.loads(Path(calibration_path).read_text())
        self.optics = {c: np.asarray(calibration['tcp_to_optical'][c], float) for c in CAMERAS}
        self.standoff_offset = np.asarray(calibration['opening_to_transport_tcp_base_offset_m'],float)
        self.board_mount = Rotation.from_quat(json.loads(Path(board_mount_path).read_text())
                                              ['board_to_port_xyzw'])
        self.nominal_tcp_in_port = Rotation.from_quat(json.loads(Path(tcp_prior_path).read_text())
                                                      ['nominal_tcp_in_port_orientation_xyzw'])
        self.board_pool=ThreadPoolExecutor(max_workers=3,thread_name_prefix='sc_board')
        self._warmup_models()

    def reset_context(self):
        if self.candidate_association is not None:self.candidate_association.reset()
        if self.requested_locator is not None:self.requested_locator.reset()

    @torch.inference_mode()
    def retain_rgb_context(self,native_rgb,state):
        if self.candidate_association is None and self.requested_locator is None:return
        small=np.stack([cv2.resize(native_rgb[c],(288,256),interpolation=cv2.INTER_AREA) for c in CAMERAS])
        image=torch.from_numpy(small).permute(0,3,1,2).to(self.device).float()/255
        meta=torch.zeros((3,5),device=self.device);meta[:,0]=1;meta[:,2:]=torch.eye(3,device=self.device)
        _,_,_,spatial=self.locator(image,meta,return_features=True,return_spatial=True)
        if self.candidate_association is not None:self.candidate_association.observe(spatial,state)
        if self.requested_locator is not None:self.requested_locator.observe(spatial,state)

    def _refine_batch(self,images,requested_port):
        if not self.joint_refiner:return self.refiner(images)
        from train_joint_native_opening_refiner import code
        identity=f'sc_port_{requested_port}'
        task=torch.from_numpy(np.stack([code(dict(family='sc'),identity,c) for c in CAMERAS])).to(self.device)
        return self.refiner(images,task)

    @torch.inference_mode()
    def _warmup_models(self):
        """Pay CUDA first-use cost before the first real handoff decision."""
        rgb=torch.zeros((3,3,256,288),device=self.device)
        meta=torch.zeros((3,5),device=self.device)
        _,_,_,spatial=self.locator(rgb,meta,return_features=True,return_spatial=True)
        self.identity(spatial,torch.zeros((3,12),device=self.device))
        if self.crop_identity is not None:
            self.crop_identity(spatial,torch.zeros((3,3,160,160),device=self.device),
                               torch.zeros((3,12),device=self.device))
        self.orientation(torch.zeros((1,self.orientation.mean.numel()),device=self.device))
        self._refine_batch(torch.zeros((3,3,160,160),device=self.device),0)
        if self.identity_scale==2:
            larger=torch.zeros((3,3,512,576),device=self.device)
            z=(larger-self.locator.mean)/self.locator.std
            backbone=self.locator.backbone
            z=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(z))))
            z=backbone.layer3(backbone.layer2(backbone.layer1(z)))
            self.identity(z.float(),torch.zeros((3,12),device=self.device))
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        # The live first inspection also pays OpenCV's first Canny/Hough
        # setup cost unless these CPU kernels have been called at startup.
        synthetic=np.zeros((1024,1152,3),np.uint8)
        for y in (300,400,500):
            cv2.line(synthetic,(100,y),(1050,y+20),(255,255,255),2)
        def prime(_):
            return cv2.HoughLinesP(cv2.Canny(cv2.cvtColor(synthetic,cv2.COLOR_BGR2GRAY),50,150),
                                   1,math.pi/180,threshold=90,minLineLength=160,maxLineGap=25)
        list(self.board_pool.map(prime,range(3)))

    @torch.inference_mode()
    def inspect(self, native_rgb, state, requested_port, diagnose_identity_failure=None,
                history_opening_base_m=None, history_port_quat_xyzw=None,
                history_board_spread_deg=None, observation_candidate_pixels=None):
        """Inspect a fresh three-camera RGB triplet and measured 32D state."""
        diagnose = (self.diagnose_identity_failure if diagnose_identity_failure is None
                    else bool(diagnose_identity_failure))
        if requested_port not in (0, 1):
            raise ValueError('Invalid requested SC port')
        report = {'requested_port': requested_port, 'runtime_true_geometry': False,
                  'valid': False, 'reason': None}
        tick=time.perf_counter();stage={}
        if any(native_rgb[c].shape[:2] != (1024,1152) for c in CAMERAS):
            report['reason'] = 'wrong_native_image_shape';return report
        small = [cv2.resize(native_rgb[c], (288,256), interpolation=cv2.INTER_AREA)
                 for c in CAMERAS]
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2,0,1)
                         for im in small]).to(self.device).float()/255.
        stage['resize_and_upload']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        meta = np.zeros((3,5),np.float32);meta[:,0] = 1
        for j in range(3):meta[j,2+j] = 1
        points, flags, features, spatial = self.locator(
            x,torch.from_numpy(meta).to(self.device),
            return_features=True,return_spatial=True)
        points = points.float().cpu().numpy()*[1151,1023]
        flags = flags.float().sigmoid().cpu().numpy()
        if self.requested_locator is not None:
            self.requested_locator.observe(spatial,state)
            selected=self.requested_locator.predict(requested_port)
            report['legacy_locator_pixels']=points[:,requested_port].tolist()
            report['requested_locator_probability']=selected['projected_inframe_probability']
            points[:,requested_port]=np.asarray(selected['pixels'])
            flags[:,requested_port]=np.asarray(selected['projected_inframe_probability'])
            report['requested_locator_absolute_state_input']=False
        if observation_candidate_pixels is not None:
            candidate=np.asarray(observation_candidate_pixels,float)
            if candidate.shape!=(3,2) or not np.isfinite(candidate).all():
                raise ValueError('Expected three finite observation-derived crop centres')
            report['original_locator_pixels']=points[:,requested_port].tolist()
            report['crop_source']='causal_visual_position_projected_with_measured_tcp'
            # This only places crops. Identity, refinement consistency,
            # board orientation and stationary handoff gates still apply.
            points[:,requested_port]=candidate
        stage['locator']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        report['selected_locator_flags'] = flags[:,requested_port].tolist()
        report['coarse_pixels'] = points[:,requested_port].tolist()
        if not all(flags[:,requested_port] >= .5):
            report['reason'] = 'selected_locator_view_missing';return report
        if not all(0 <= points[j,requested_port,0] < 1152 and
                   0 <= points[j,requested_port,1] < 1024 for j in range(3)):
            report['reason'] = 'selected_locator_offscreen';return report
        # Spatial request check uses image features and measured TCP. It was
        # trained for projected presence, so this is conservative evidence,
        # not certified physical visibility.
        state_tensor=torch.as_tensor(np.asarray(state[:7]),device=self.device,dtype=torch.float32)
        condition=[]
        for j in range(3):
            c=np.zeros(3,np.float32);c[j]=1
            request=np.zeros(2,np.float32);request[requested_port]=1
            condition.append(np.r_[c,np.zeros(7,np.float32),request])
        condition=torch.from_numpy(np.stack(condition).astype(np.float32)).to(self.device)
        condition[:,3:10]=(state_tensor-self.state_mean)/self.state_std
        if self.identity_mask_tcp_orientation:
            condition[:,6:10]=0
        identity_spatial=spatial.float()
        if self.identity_scale==2:
            larger=[cv2.resize(native_rgb[c],(576,512),interpolation=cv2.INTER_AREA)
                    for c in CAMERAS]
            high=torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2,0,1)
                              for im in larger]).to(self.device).float()/255.
            high=(high-self.locator.mean)/self.locator.std
            backbone=self.locator.backbone
            high=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(high))))
            identity_spatial=backbone.layer3(backbone.layer2(backbone.layer1(high))).float()
        if self.crop_identity is None:
            presence=self.identity(identity_spatial,condition).sigmoid().cpu().numpy()
        else:
            crop_condition=condition.clone()
            crop_condition[:,3:10]=(state_tensor-self.crop_state_mean)/self.crop_state_std
            original_crops=np.stack([crop(native_rgb[c],points[j,requested_port])
                                     for j,c in enumerate(CAMERAS)])
            crop_tensor=torch.from_numpy(original_crops).permute(0,3,1,2).to(self.device).float()/255
            presence=self.crop_identity(spatial.float(),crop_tensor,crop_condition).sigmoid().cpu().numpy()
        stage['identity']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        report['spatial_identity_probability']=presence.tolist()
        association=None
        if self.candidate_association is not None:
            self.candidate_association.observe(spatial,state)
            association=self.candidate_association.probabilities(points[:,requested_port],requested_port)
            report['candidate_association_probability']=association.tolist()
            report['association_is_not_visibility']=True
        from requested_candidate_gate import candidate_gate
        identity_ok,effective=candidate_gate(presence,association,self.candidate_gate_mode)
        report['candidate_gate_mode']=self.candidate_gate_mode
        report['effective_identity_probability']=effective
        if not identity_ok and not diagnose:
            report['reason']='requested_identity_not_in_all_views';return report
        base_tcp=pose_matrix(state[:7])
        cameras={c:base_tcp@self.optics[c] for c in CAMERAS}
        # The full orientation head supplies a search prior, not the final
        # yaw. Its input matches the original frozen locator feature schema.
        xyz=[];mask=[]
        for k in range(len(NAMES)):
            pixels={c:points[j,k] for j,c in enumerate(CAMERAS) if flags[j,k]>=.5}
            valid=False
            if len(pixels)>=2:
                try:
                    position=triangulate(pixels,cameras)
                    valid=bool(np.isfinite(position).all())
                except np.linalg.LinAlgError:pass
            xyz.extend(position.tolist() if valid else [0.,0.,0.]);mask.append(float(valid))
        code=np.zeros(len(NAMES),np.float32);code[requested_port]=1
        feature=np.r_[xyz,mask,code,state[:7],features.float().cpu().numpy().reshape(-1)].astype(np.float32)
        rotation=self.orientation(torch.from_numpy(feature).to(self.device)[None]).float().cpu().numpy()[0]
        prior=(Rotation.from_matrix(rotation)*self.board_mount.inv()).as_euler('xyz')[2]
        stage['orientation_prior']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        # Original-resolution opening crops improve the standoff check.
        refined={};shifts={};crops=[];offsets=[]
        for j,c in enumerate(CAMERAS):
            center=points[j,requested_port]
            left,top=np.round(center-80).astype(int)
            crops.append(torch.from_numpy(np.ascontiguousarray(cv2.warpAffine(
                native_rgb[c],np.array([[1,0,-left],[0,1,-top]],np.float32),
                (160,160),borderMode=cv2.BORDER_CONSTANT))).permute(2,0,1))
            offsets.append((left,top))
        refined_batch=self._refine_batch(torch.stack(crops).to(self.device).float()/255,
                                         requested_port).float().cpu().numpy()
        for j,c in enumerate(CAMERAS):
            pixel=refined_batch[j].reshape(2)*159+offsets[j]
            center=points[j,requested_port]
            refined[c]=pixel
            shifts[c]=float(np.linalg.norm(pixel-center))
        report['refined_pixels']={c:p.tolist() for c,p in refined.items()}
        report['coarse_to_refined_px']=shifts
        if max(shifts.values())>50:
            report['reason']='refiner_large_jump';return report
        position=triangulate(refined,cameras)
        residual=reprojection_residual_px(position,refined,cameras)
        report['opening_reprojection_px']=float(residual)
        if not np.isfinite(residual) or residual>2:
            report['reason']='opening_views_disagree';return report
        stage['refiner_and_geometry']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        line_values=list(self.board_pool.map(
            lambda c:camera_yaw(native_rgb[c][...,::-1],refined[c],cameras[c],prior,
                               search_span_rad=self.board_yaw_search_rad,
                               max_prior_line_error_deg=self.board_yaw_prior_line_deg),
            CAMERAS))
        line_results=dict(zip(CAMERAS,line_values))
        stage['board_rails']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        good={c:v for c,v in line_results.items() if v is not None and v['mad_deg']<=1.
              and (not self.reject_yaw_search_boundary or v['boundary_weight_fraction']<=.5)}
        report['board_search_boundary_fractions']={c:(v['boundary_weight_fraction'] if v else None)
                                                  for c,v in line_results.items()}
        report['board_line_counts']={c:(v['lines'] if v else 0) for c,v in line_results.items()}
        if len(good)<2:
            if (history_opening_base_m is None or history_port_quat_xyzw is None or
                    history_board_spread_deg is None):
                report['reason']='insufficient_board_rails';return report
            history_point=np.asarray(history_opening_base_m,float)
            history_quat=np.asarray(history_port_quat_xyzw,float)
            if (history_point.shape!=(3,) or history_quat.shape!=(4,) or
                    not np.isfinite(history_point).all() or
                    not np.isfinite(history_quat).all() or
                    not np.isfinite(history_board_spread_deg) or
                    history_board_spread_deg>1):
                report['reason']='invalid_history_orientation';return report
            history_distance=float(np.linalg.norm(position-history_point)*1000)
            report['history_opening_agreement_mm']=history_distance
            if history_distance>5:
                report['reason']='history_opening_disagrees';return report
            port=Rotation.from_quat(history_quat)
            if good:
                history_yaw=(port*self.board_mount.inv()).as_euler('xyz')[2]
                one_yaw=next(iter(good.values()))['yaw_rad']
                difference=abs((one_yaw-history_yaw+math.pi)%(2*math.pi)-math.pi)
                report['current_single_board_disagreement_deg']=math.degrees(difference)
                if difference>math.radians(2):
                    report['reason']='history_board_disagrees';return report
            spread=float(history_board_spread_deg)
            report['board_view_spread_deg']=spread
            report['orientation_source']='causal_history'
            report['current_board_valid_views']=len(good)
        else:
            spread=math.degrees(max(v['yaw_rad'] for v in good.values())-
                                min(v['yaw_rad'] for v in good.values()))
            report['board_view_spread_deg']=spread
            if spread>1:
                report['reason']='board_views_disagree';return report
            yaw=float(np.median([v['yaw_rad'] for v in good.values()]))
            port=Rotation.from_euler('z',yaw)*self.board_mount
            report['orientation_source']='current_board'
        expected_tcp=port*self.nominal_tcp_in_port
        measured_tcp=Rotation.from_quat(state[3:7])
        posture_error=math.degrees((expected_tcp.inv()*measured_tcp).magnitude())
        standoff=np.linalg.norm(position+self.standoff_offset-np.asarray(state[:3]))*1000
        report.update(predicted_opening_base_m=position.tolist(),
                      predicted_port_quat_xyzw=port.as_quat().tolist(),
                      measured_tcp_posture_error_deg=posture_error,
                      measured_tcp_standoff_error_mm=float(standoff),
                      measured_force_norm_n=float(np.linalg.norm(state[26:29])))
        stage['final_pose']=(time.perf_counter()-tick)*1000
        report['stage_ms']=stage
        if not identity_ok:
            # Diagnostic orientation only: a missing identity may never
            # authorize insertion. Geometry and force still bound a held
            # in-place correction; no true scene pose enters inference.
            report['orientation_correction_candidate']=bool(
                residual<=2 and spread<=1 and standoff<=15 and
                report['measured_force_norm_n']<=40 and
                3<posture_error<=10)
            report['reason']='requested_identity_not_in_all_views'
            return report
        if posture_error>3:
            report['reason']='wrong_tcp_posture';return report
        if standoff>15:
            report['reason']='not_at_selected_standoff';return report
        if report['measured_force_norm_n']>40:
            report['reason']='high_measured_force';return report
        report.update(valid=True,reason=('clear_selected_port_and_safe_standoff_from_history_orientation'
                                         if report['orientation_source']=='causal_history'
                                         else 'clear_selected_port_and_safe_standoff'))
        return report

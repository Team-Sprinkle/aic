"""Frozen RGB-only selected SFP opening estimate for transport conditioning."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sfp_board_reference import BoardReferenceLocal, SIZE
from train_sfp_observed_rail_crop import crop_native
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, LOOKUP


class SelectedSfpPort:
    def __init__(self,board_checkpoint,crop_checkpoint,device='cuda',*,native_refiner_checkpoint=None,native_pose_mount_priors=None,minimum_views=3):
        if minimum_views not in (2,3):raise ValueError('Use two or three calibrated cameras')
        self.minimum_views=minimum_views
        self.device=torch.device(device)
        state=torch.load(board_checkpoint,map_location='cpu',weights_only=False)
        self.board=BoardReferenceLocal(state['init']).to(self.device).eval()
        self.board.load_state_dict(state['model'])
        self.crop=ConditionalLocator(None).to(self.device).eval()
        self.crop.load_state_dict(torch.load(crop_checkpoint,map_location='cpu',weights_only=False)['model'])
        calibration=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
        self.optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
        self.native_refiner=None
        if native_refiner_checkpoint is not None:
            from native_requested_opening_refiner import NativeRequestedOpeningRefiner
            self.native_refiner=NativeRequestedOpeningRefiner(native_refiner_checkpoint,device)
        self.fixture_mount=None
        if native_pose_mount_priors is not None:
            if self.native_refiner is None or not self.native_refiner.pose_keypoints:
                raise ValueError('Fixed-normal pose fit requires a native pose-keypoint model')
            self.fixture_mount=np.asarray(json.loads(Path(native_pose_mount_priors).read_text())['sfp'],float)

    @torch.inference_mode()
    def predict_batch(self,items):
        """items: (BGR images by camera, public state, requested card, port)."""
        batches=[]
        for images,_,_,_ in items:
            batches.append(np.stack([cv2.cvtColor(cv2.resize(images[c],SIZE,interpolation=cv2.INTER_AREA),
                                                cv2.COLOR_BGR2RGB).transpose(2,0,1)
                                     for c in CAMERAS]))
        tensor=torch.from_numpy(np.stack(batches).copy()).to(self.device).float()/255.
        ends,_,_,_=self.board(tensor)
        centers=[];patches=[];meta=[];codes=[]
        for bi,(images,_,card,port) in enumerate(items):
            one=[]
            for ci,c in enumerate(CAMERAS):
                endpoint=ends[bi,ci].float().cpu().numpy()*[1151,1023]
                center=endpoint[0]+card/4*(endpoint[1]-endpoint[0])
                one.append(center)
                patches.append(crop_native(images[c],center))
                m=np.zeros(5,np.float32);m[1]=1;m[2+ci]=1;meta.append(m)
                codes.append(LOOKUP[f'sfp_card_{card}_port_{port}'])
            centers.append(one)
        local=torch.from_numpy(np.stack(patches).copy()).to(self.device).float().permute(0,3,1,2)/255.
        xy,logit,_=self.crop(local,torch.from_numpy(np.stack(meta)).to(self.device),
                             torch.tensor(codes,dtype=torch.long,device=self.device))
        xy=xy.float().cpu().numpy().reshape(len(items),3,2)
        prob=logit.sigmoid().float().cpu().numpy().reshape(len(items),3)
        result=[]
        for bi,(images,state,card,port) in enumerate(items):
            points={c:centers[bi][ci]+(xy[bi,ci]-.5)*480
                    for ci,c in enumerate(CAMERAS) if prob[bi,ci]>=.5}
            coarse_points={c:p.tolist() for c,p in points.items()}
            refinement=None
            if self.native_refiner is not None and points:
                rgb={c:cv2.cvtColor(images[c],cv2.COLOR_BGR2RGB) for c in points}
                refinement=self.native_refiner.refine(rgb,points,family='sfp',port=port,card=card)
                points={c:np.asarray(p,float) for c,p in refinement['pixels'].items()}
            estimate=None;reprojection=None;orientation=None;pose_fit_report=None;guidance=None
            if len(points)>=self.minimum_views:
                optical={c:pose_matrix(np.asarray(state,float))@self.optics[c] for c in CAMERAS}
                try:
                    point=triangulate(points,optical)
                    residual=float(reprojection_residual_px(point,points,optical))
                    if np.isfinite(point).all() and residual<=5:
                        estimate=point;reprojection=residual
                except np.linalg.LinAlgError:pass
                if self.fixture_mount is not None:
                    from native_keypoint_pose_geometry import fit_fixed_normal_pose
                    from evaluate_sc_native_triangulation import FOCAL
                    from scipy.spatial.transform import Rotation
                    estimate=None
                    try:
                        fit=fit_fixed_normal_pose({c:np.asarray(v) for c,v in refinement['keypoints'].items()},
                            optical,np.asarray(self.native_refiner.reference_points['sfp']),self.fixture_mount,FOCAL)
                        pose_fit_report=dict(max_reprojection_px=fit['max_reprojection_px'],converged=fit['converged'],
                            cameras=list(refinement['keypoints']),minimum_views=self.minimum_views,
                            fixture_assumption=fit['fixture_assumption'],geometry_input=False)
                        if fit['all_depths_positive'] and fit['converged'] and fit['max_reprojection_px']<=5:
                            guidance=dict(position_base_m=fit['position_base_m'].tolist(),
                                quat_xyzw=Rotation.from_matrix(fit['rotation_base']).as_quat().tolist(),
                                cameras=list(refinement['keypoints']),
                                max_reprojection_px=fit['max_reprojection_px'],
                                insertion_lock_allowed=False,geometry_input=False)
                        if fit['all_depths_positive'] and fit['converged'] and fit['max_reprojection_px']<=2:
                            estimate=fit['position_base_m'];reprojection=fit['max_reprojection_px']
                            orientation=Rotation.from_matrix(fit['rotation_base']).as_quat().tolist()
                    except (ValueError,np.linalg.LinAlgError):pass
            result.append(dict(predicted_base_m=estimate,accepted=estimate is not None,
                               reprojection_px=reprojection,view_probabilities=prob[bi].tolist(),
                               coarse_pixels=coarse_points,refined_pixels=({c:p.tolist() for c,p in points.items()}
                                   if refinement is not None else None),native_refinement=refinement,
                               predicted_orientation_base_xyzw=orientation,pose_fit=pose_fit_report,
                               guidance_candidate=guidance))
        return result

    def warmup(self):
        dummy={c:np.zeros((1024,1152,3),np.uint8) for c in CAMERAS}
        self.predict_batch([(dummy,np.array([0,0,0,0,0,0,1],float),0,0)])
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

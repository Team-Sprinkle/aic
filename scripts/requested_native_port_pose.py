"""One public requested port, all current cameras and causal visual context.

Observation-only component for native five-landmark pose comparisons and
causal BC-frame preparation. This component alone is NOT a live handoff gate:
route, identity continuity, potential-region, motion and force gates remain.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from train_two_family_port_locator import Locator
from sc_requested_history_observation import RequestedPortHistoryObservation
from native_requested_opening_refiner import NativeRequestedOpeningRefiner
from native_keypoint_pose_geometry import fit_fixed_normal_pose
from evaluate_sc_native_triangulation import FOCAL
from shared_port_tcp_geometry import matrix_from_xyz_xyzw

CAMS=('center','left','right')


class RequestedNativePortPose:
    def __init__(self, backbone_checkpoint, requested_checkpoint, native_checkpoint,
                 mount_priors, *, family, device='cuda', maximum_shift_px=50.,maximum_reprojection_px=2.,
                 minimum_projection_probability=.9):
        if family not in ('sc','sfp'):raise ValueError('Unknown family')
        self.family=family;self.device=torch.device(device);self.maximum_shift_px=maximum_shift_px
        self.maximum_reprojection_px=maximum_reprojection_px
        if not 0<minimum_projection_probability<1:raise ValueError('Invalid projection probability gate')
        self.minimum_projection_probability=minimum_projection_probability
        old=Locator(None);old.load_state_dict(torch.load(backbone_checkpoint,map_location='cpu',weights_only=False)['model'])
        b=old.backbone;self.visual=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool,b.layer1,b.layer2,b.layer3).to(device).eval()
        self.mean=old.mean.to(device);self.std=old.std.to(device)
        self.coarse=RequestedPortHistoryObservation(requested_checkpoint,device)
        if self.coarse.num_requests!=(2 if family=='sc' else 10):raise ValueError('Requested-query checkpoint family mismatch')
        self.refiner=NativeRequestedOpeningRefiner(native_checkpoint,device)
        if not self.refiner.pose_keypoints:raise ValueError('Five-landmark native checkpoint required')
        cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
        self.optics={c:np.asarray(v) for c,v in cal['tcp_to_optical'].items()}
        self.mount=np.asarray(json.loads(Path(mount_priors).read_text())[family])

    def reset(self):self.coarse.reset()

    @torch.inference_mode()
    def retain_context(self, images_rgb, state):
        if any(images_rgb[c].shape!=(1024,1152,3) for c in CAMS):raise ValueError('Original native RGB required')
        small=np.stack([cv2.resize(images_rgb[c],(288,256),interpolation=cv2.INTER_AREA) for c in CAMS])
        image=torch.from_numpy(small).permute(0,3,1,2).to(self.device).float()/255
        spatial=self.visual((image-self.mean)/self.std)
        self.coarse.observe(spatial,state)
        return spatial

    @torch.inference_mode()
    def predict(self, images_rgb, state, task, *, spatial_features=None):
        if len(task)!=10:raise ValueError('Actual public ten-dimensional task code required')
        family='sfp' if task[0]>.5 else 'sc'
        if family!=self.family:raise ValueError('Family/query mismatch')
        if not np.isclose(np.sum(task[2:4]),1):raise ValueError('Exactly one selected port required')
        port=int(np.argmax(task[2:4]));card=int(np.argmax(task[4:9])) if family=='sfp' else None
        if family=='sfp' and not np.isclose(np.sum(task[4:9]),1):raise ValueError('Exactly one selected SFP card required')
        if any(images_rgb[c].shape!=(1024,1152,3) for c in CAMS):raise ValueError('Original native RGB required')
        if spatial_features is None:
            self.retain_context(images_rgb,state)
        else:self.coarse.observe(spatial_features,state)
        coarse_started=time.perf_counter()
        proposal=self.coarse.predict(port if family=='sc' else card*2+port)
        timing={'coarse_predict_and_pending_GPU_ms':(time.perf_counter()-coarse_started)*1000}
        report=dict(accepted=False,geometry_input=False,request_only=True,requested_port=port,
                    requested_card=card,projection_flags_are_not_confidence=True,
                    projected_inframe_probability=proposal['projected_inframe_probability'],
                    coarse_pixels=proposal['pixels'],stage_ms=timing)
        if min(proposal['projected_inframe_probability'])<self.minimum_projection_probability:
            return dict(report,reason='requested_projection_not_in_all_views')
        points={c:proposal['pixels'][ci] for ci,c in enumerate(CAMS)}
        if any(not (0<=p[0]<1152 and 0<=p[1]<1024) for p in points.values()):
            return dict(report,reason='requested_crop_offscreen')
        started=time.perf_counter()
        native=self.refiner.refine(images_rgb,points,family=family,port=port,card=card)
        timing['native_refiner_ms']=(time.perf_counter()-started)*1000
        report['native']=native
        if max(native['shifts_px'].values())>self.maximum_shift_px:
            return dict(report,reason='native_refinement_exceeds_crop_margin')
        cameras={c:matrix_from_xyz_xyzw(state[:7])@self.optics[c] for c in CAMS}
        started=time.perf_counter()
        fit=fit_fixed_normal_pose({c:np.asarray(v) for c,v in native['keypoints'].items()},cameras,
            np.asarray(self.refiner.reference_points[family]),self.mount,FOCAL)
        timing['pose_fit_ms']=(time.perf_counter()-started)*1000
        report['max_reprojection_px']=fit['max_reprojection_px']
        if not (fit['converged'] and fit['all_depths_positive'] and fit['max_reprojection_px']<=self.maximum_reprojection_px):
            return dict(report,reason='native_pose_geometry_rejected')
        pose=np.eye(4);pose[:3,:3]=fit['rotation_base'];pose[:3,3]=fit['position_base_m']
        return dict(report,accepted=True,reason='three_view_requested_native_pose',
                    base_port_matrix=pose.tolist(),
                    predicted_port_quat_xyzw=Rotation.from_matrix(pose[:3,:3]).as_quat().tolist(),
                    fixture_assumption=fit['fixture_assumption'])

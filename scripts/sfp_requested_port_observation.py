"""One requested SFP region from all-camera RGB history, then native pose.

No simulator geometry or scene file is an inference argument. The frozen
visual backbone supplies spatial features only; legacy all-port heads are
neither retained nor evaluated. Motion anchors use public measured TCP.
"""
import json
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


class RequestedSfpPort:
    def __init__(self,backbone_checkpoint,requested_checkpoint,refiner_checkpoint,
                 mount_priors,device='cuda',minimum_views=3,view_threshold=.9,
                 maximum_reprojection_px=2.):
        self.device=torch.device(device);self.minimum_views=minimum_views
        self.view_threshold=view_threshold
        if not 0<maximum_reprojection_px<=5:raise ValueError('Unbounded native pose residual gate')
        self.maximum_reprojection_px=maximum_reprojection_px
        # Offline callers read the original recorded JPEGs. Runtime enables
        # the collector's native JPEG86 codec once, before either model.
        self.normalize_native_codec=False
        old=Locator(None);old.load_state_dict(torch.load(backbone_checkpoint,map_location='cpu',weights_only=False)['model'])
        b=old.backbone
        self.visual=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool,b.layer1,b.layer2,b.layer3).to(device).eval()
        self.mean=old.mean.to(device);self.std=old.std.to(device)
        self.coarse=RequestedPortHistoryObservation(requested_checkpoint,device)
        if self.coarse.num_requests!=10:raise ValueError('Expected five SFP cards by two public ports')
        self.native_refiner=NativeRequestedOpeningRefiner(refiner_checkpoint,device)
        if not self.native_refiner.pose_keypoints:raise ValueError('Five-landmark native pose checkpoint required')
        cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
        self.optics={c:np.asarray(t,float) for c,t in cal['tcp_to_optical'].items()}
        self.mount=np.asarray(json.loads(Path(mount_priors).read_text())['sfp'],float)

    def reset_context(self):self.coarse.reset()

    @torch.inference_mode()
    def retain_context(self,images,state):
        if any(images[c].shape!=(1024,1152,3) for c in ('center','left','right')):
            raise ValueError('Original synchronized camera images required')
        rgb=np.stack([cv2.cvtColor(cv2.resize(images[c],(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                      for c in ('center','left','right')])
        x=torch.from_numpy(rgb).permute(0,3,1,2).to(self.device).float()/255
        spatial=self.visual((x-self.mean)/self.std)
        self.coarse.observe(spatial,state)

    @torch.inference_mode()
    def predict_batch(self,items,*,spatial_features=None):
        result=[]
        for ordinal,(images,state,card,port) in enumerate(items):
            if card not in range(5) or port not in (0,1):raise ValueError('Invalid public requested SFP port')
            if self.normalize_native_codec:
                normalized={}
                for camera,image in images.items():
                    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,86])
                    if not ok:raise RuntimeError('Native training-codec normalization failed')
                    normalized[camera]=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                images=normalized
            if spatial_features is None:self.retain_context(images,state)
            else:self.coarse.observe(spatial_features[ordinal],state)
            crop=self.coarse.predict(card*2+port)
            points={c:np.asarray(crop['pixels'][j]) for j,c in enumerate(('center','left','right'))
                    if crop['projected_inframe_probability'][j]>=self.view_threshold}
            native=(self.native_refiner.refine({c:cv2.cvtColor(images[c],cv2.COLOR_BGR2RGB) for c in points},
                points,family='sfp',card=card,port=port) if points else None)
            estimate=orientation=reprojection=fit_report=guidance=None
            if native and len(points)>=self.minimum_views:
                # Match the native training crop margin. A large hallucinated
                # jump cannot turn an empty crop into an accepted pose.
                inside=all(np.max(np.abs(np.asarray(native['pixels'][c])-points[c]))<=64 for c in points)
                if inside:
                    cameras={c:matrix_from_xyz_xyzw(state[:7])@self.optics[c] for c in points}
                    try:
                        fit=fit_fixed_normal_pose({c:np.asarray(p) for c,p in native['keypoints'].items()},
                            cameras,np.asarray(self.native_refiner.reference_points['sfp']),self.mount,FOCAL)
                        fit_report=dict(cameras=list(points),max_reprojection_px=fit['max_reprojection_px'],
                            converged=fit['converged'],fixture_assumption=fit['fixture_assumption'],geometry_input=False)
                        if fit['converged'] and fit['all_depths_positive']:
                            fit_report.update(position_base_m=fit['position_base_m'].tolist(),
                                quat_xyzw=Rotation.from_matrix(fit['rotation_base']).as_quat().tolist())
                        if fit['converged'] and fit['all_depths_positive']:
                            if fit['max_reprojection_px']<=5:
                                guidance=dict(position_base_m=fit['position_base_m'].tolist(),
                                    quat_xyzw=Rotation.from_matrix(fit['rotation_base']).as_quat().tolist(),
                                    cameras=list(points),max_reprojection_px=fit['max_reprojection_px'],
                                    geometry_input=False,insertion_lock_allowed=False)
                            if fit['max_reprojection_px']<=self.maximum_reprojection_px:
                                estimate=fit['position_base_m'];orientation=Rotation.from_matrix(fit['rotation_base']).as_quat().tolist()
                                reprojection=fit['max_reprojection_px']
                    except (ValueError,np.linalg.LinAlgError):pass
            result.append(dict(predicted_base_m=estimate,accepted=estimate is not None,
                predicted_orientation_base_xyzw=orientation,reprojection_px=reprojection,
                view_probabilities=crop['projected_inframe_probability'],
                coarse_pixels={c:p.tolist() for c,p in points.items()},
                refined_pixels=native['pixels'] if native else None,native_refinement=native,
                pose_fit=fit_report,guidance_candidate=guidance,
                request_only=True,absolute_state_input=False,geometry_input=False))
        return result

    def warmup(self):
        self.predict_batch([({c:np.zeros((1024,1152,3),np.uint8) for c in self.optics},
                             np.array([0.,0.,0.,0.,0.,0.,1.]),0,0)])
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        self.reset_context()

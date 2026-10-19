"""SC native five-reference pose plus unchanged posture/standoff/force gates.

The 64-pixel crop margin is paired with the validated multi-landmark fit, never
with the single-point margin relaxation that admitted an absent target.
"""
import cv2
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation


class NativePortHandoffVerifier:
    def __init__(self,component,standoff_offset,nominal_tcp_in_port):
        self.component=component;self.standoff_offset=np.asarray(standoff_offset)
        self.nominal_tcp_in_port=nominal_tcp_in_port;self.optics=component.optics
        self.candidate_association=None;self.requested_locator=component.coarse
        self.cached_images=None;self.cached_state=None;self.cached_spatial=None
        self.codec_pool=ThreadPoolExecutor(max_workers=3,thread_name_prefix='native_codec')
        self.codec_ms=None
        self.cached_signature=None

    def reset_context(self):
        self.component.reset();self.cached_images=self.cached_state=self.cached_spatial=None;self.cached_signature=None

    @staticmethod
    def signature(images,state):
        # A stationary TCP does not imply stationary cable/port pixels.
        # Reuse only the exact public state AND current RGB evidence.
        return (np.asarray(state).tobytes(),tuple(hashlib.sha256(memoryview(np.ascontiguousarray(images[c]))).digest()
                for c in ('center','left','right')))

    def retain_rgb_context(self,images,state):
        started=time.perf_counter()
        def normalize(item):
            c,image=item
            ok,data=cv2.imencode('.jpg',image[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
            if not ok:raise RuntimeError('Native training codec failed')
            return c,cv2.cvtColor(cv2.imdecode(data,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        normalized=dict(self.codec_pool.map(normalize,images.items()))
        self.codec_ms=(time.perf_counter()-started)*1000
        self.cached_images=normalized;self.cached_state=np.asarray(state).copy()
        self.cached_signature=self.signature(images,state)
        self.cached_spatial=self.component.retain_context(normalized,state)

    def inspect(self,images,state,requested):
        if self.cached_signature!=self.signature(images,state):
            self.retain_rgb_context(images,state)
        task=[0.,1.,float(requested==0),float(requested==1),0.,0.,0.,0.,0.,0.]
        result=self.component.predict(self.cached_images,state,task,spatial_features=self.cached_spatial)
        report=dict(requested_port=requested,runtime_true_geometry=False,valid=False,
                    reason=result['reason'],native_pose_component=result,
                    orientation_source='five_native_landmarks_fixed_fixture_normal',
                    projection_flags_are_not_confidence=True,
                    separate_candidate_classifier_used=False,native_codec='JPEG86 once')
        report['native_codec_ms']=self.codec_ms
        if not result['accepted']:return report
        pose=np.asarray(result['base_port_matrix']);point=pose[:3,3];rotation=Rotation.from_matrix(pose[:3,:3])
        error=float(np.linalg.norm(point+self.standoff_offset-np.asarray(state[:3]))*1000)
        angle=float(np.rad2deg(((rotation*self.nominal_tcp_in_port).inv()*Rotation.from_quat(state[3:7])).magnitude()))
        force=float(np.linalg.norm(state[26:29]))
        report.update(predicted_opening_base_m=point.tolist(),predicted_port_quat_xyzw=rotation.as_quat().tolist(),
                      measured_tcp_standoff_error_mm=error,measured_tcp_posture_error_deg=angle,
                      measured_force_norm_n=force)
        if abs(point[2]-.03)>.010:return dict(report,reason='outside_calibrated_fixture_height')
        if angle>10:return dict(report,reason='predicted_orientation_outside_rule_repair_envelope')
        if angle>3:return dict(report,reason='wrong_tcp_posture')
        if error>15:return dict(report,reason='not_at_selected_standoff')
        if force>40:return dict(report,reason='high_measured_force')
        return dict(report,valid=True,reason='clear_selected_port_and_safe_standoff')

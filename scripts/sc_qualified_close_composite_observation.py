"""Combine independently qualified image-only SC position and orientation."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from native_connector_pose_observation import NativeConnectorPoseObservation
from sc_spatial_grasp_observation import SCSpatialGraspObservation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SCQualifiedCloseCompositeObservation:
    rotation_only=False

    def __init__(self, checkpoint, device, *, approval, experimental_diagnostic=False):
        report=json.loads(Path(approval).read_text())
        permitted=(report.get('pose_gate_passed') is False and report.get('experimental_trial_only') is True
                   if experimental_diagnostic else report.get('pose_gate_passed') is True)
        expected_kind='experimental_composite_pose' if experimental_diagnostic else 'composite_pose'
        if (not permitted or not report.get('visual_inspection_completed') or
                report.get('prediction_kind')!=expected_kind or
                report.get('pose_gate_scope')!='SC_held_near25mm_only' or
                report.get('pose_frame_contract')!='SC_mating_reference_in_gripper_TCP' or
                report['checkpoint_sha256']!=digest(checkpoint)):
            raise ValueError('Qualified inspected exact composite pose certificate required')
        for key in ['rotation_approval','translation_approval','translation_checkpoint']:
            path=Path(report[key])
            if not path.is_absolute():path=Path(approval).parent/path
            report[key]=str(path)
        for key in ['rotation_approval','translation_approval']:
            if digest(report[key])!=report[key+'_sha256']:
                raise ValueError('Changed component approval: '+key)
        rotation=json.loads(Path(report['rotation_approval']).read_text())
        position=json.loads(Path(report['translation_approval']).read_text())
        rotation_permitted=(rotation.get('pose_gate_passed') is False and rotation.get('experimental_trial_only') is True
                            if experimental_diagnostic else rotation.get('pose_gate_passed') is True)
        if (not rotation_permitted or not rotation.get('visual_inspection_completed') or
                rotation.get('prediction_kind')!='orientation_only' or
                rotation['checkpoint_sha256']!=digest(checkpoint) or
                not position.get('translation_gate_passed') or not position.get('visual_inspection_completed') or
                position.get('prediction_kind')!='translation_only_component' or
                position.get('orientation_approved') is not False or
                position['checkpoint_sha256']!=digest(report['translation_checkpoint'])):
            raise ValueError('Both independently qualified components required; position model rotation is forbidden')
        for component in [rotation,position]:
            if (component.get('pose_gate_scope')!='SC_held_near25mm_only' or
                    component.get('pose_frame_contract')!='SC_mating_reference_in_gripper_TCP'):
                raise ValueError('Pose components use incompatible frames or deployment scopes')
        self.position=SCSpatialGraspObservation(report['translation_checkpoint'],device)
        self.rotation=NativeConnectorPoseObservation(checkpoint,device)
        if not self.rotation.rotation_only or self.rotation.plan['family']!='sc':
            raise ValueError('SC RGB orientation-only component required')
        self.approved=not experimental_diagnostic
        self.diagnostic_allowed=experimental_diagnostic

    def predict(self, images, *, already_training_jpeg=False):
        started=time.perf_counter()
        if not already_training_jpeg:
            encoded_images={}
            for camera in ['center','left','right']:
                rgb=images[camera]
                if rgb.shape!=(1024,1152,3) or rgb.dtype!=np.uint8:
                    raise ValueError('Original three-camera uint8 RGB required')
                ok,encoded=cv2.imencode('.jpg',rgb[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
                if not ok:raise RuntimeError('Training codec failed')
                encoded_images[camera]=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            images=encoded_images
        position=np.asarray(self.position.predict(images,already_training_jpeg=True)['predicted_tip_position_tcp_m'])
        orientation=self.rotation.predict(images,already_training_jpeg=True,visual_tip_position_tcp_m=position)
        return dict(predicted_tip_position_tcp_m=position.tolist(),
                    orientation_for_diagnostic_only=orientation['orientation_for_diagnostic_only'],
                    orientation_approved=self.approved,translation_reestimated=True,translation_only=False,
                    experimental_trial_only=self.diagnostic_allowed,
                    prediction_kind='composite_pose' if self.approved else 'experimental_composite_pose',runtime_true_geometry=False,
                    position_component_orientation_used=False,
                    input='Same fresh native RGB triplet, one shared JPEG86 conversion, two qualified pose components',
                    latency_ms=(time.perf_counter()-started)*1000)

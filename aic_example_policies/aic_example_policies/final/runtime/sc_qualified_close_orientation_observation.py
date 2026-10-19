"""Inference definitions extracted from sc_qualified_close_orientation_observation.py; no training entry point."""
import hashlib
import json
from pathlib import Path
from .native_connector_pose_observation import NativeConnectorPoseObservation

class SCQualifiedCloseOrientationObservation(NativeConnectorPoseObservation):

    def __init__(self, checkpoint, device, *, approval, retained_translation_checkpoint):
        report = json.loads(Path(approval).read_text())
        if not report.get('pose_gate_passed') or not report.get('visual_inspection_completed') or report.get('prediction_kind') != 'orientation_only' or (report.get('pose_gate_scope') != 'SC_held_near25mm_only') or (report['checkpoint_sha256'] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()) or (report['retained_translation_checkpoint_sha256'] != hashlib.sha256(Path(retained_translation_checkpoint).read_bytes()).hexdigest()):
            raise ValueError('Qualified inspected exact orientation model and retained RGB translation required')
        super().__init__(checkpoint, device)
        if not self.rotation_only or self.plan['family'] != 'sc':
            raise ValueError('SCrotation-only fit required')
        self.approved = True

    def predict(self, images, *, already_training_jpeg=False, visual_tip_position_tcp_m=None):
        result = super().predict(images, already_training_jpeg=already_training_jpeg, visual_tip_position_tcp_m=visual_tip_position_tcp_m)
        result.update(orientation_approved=True, translation_reestimated=False, translation_source='Retained image-estimated acquisition position, not this orientation head')
        return result

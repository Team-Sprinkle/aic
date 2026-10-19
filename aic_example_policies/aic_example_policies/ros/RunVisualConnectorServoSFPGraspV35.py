"""V34 SC unchanged; qualified wider-grasp RGB orientation for SFP only."""
import hashlib
import json
import os
from pathlib import Path
from native_connector_pose_observation import NativeConnectorPoseObservation
from sfp_repaired_pose_observation import SharedSFPRepairedPose
from .RunVisualConnectorServoRecoveryFastV34 import RunVisualConnectorServoRecoveryFastV34


class RunVisualConnectorServoSFPGraspV35(RunVisualConnectorServoRecoveryFastV34):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        checkpoint=Path(os.environ['AIC_SFP_ORIENTATION_REPAIR_CHECKPOINT'])
        approval=json.loads(Path(os.environ['AIC_SFP_ORIENTATION_REPAIR_APPROVAL']).read_text())
        if (not approval.get('orientation_approved') or not approval.get('visual_inspection_completed')
                or approval.get('checkpoint_sha256')!=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or approval.get('family')!='sfp' or approval.get('runtime_true_geometry') is not False):
            raise ValueError('Exact SFP orientation component must pass domain and image QA')
        self.replace_SFP_position=bool(approval.get('joint_position_approved'))
        self.repaired_SFP_pose=SharedSFPRepairedPose(checkpoint,str(self.local.device)) if self.replace_SFP_position else NativeConnectorPoseObservation(checkpoint,str(self.local.device))
        self.connector_models['sfp']=self.repaired_SFP_pose
        self._trace(event='SFP_wide_orientation_component_loaded',checkpoint_sha256=approval['checkpoint_sha256'],
                    position_estimator_unchanged=not self.replace_SFP_position,SC_controller_unchanged=True,
                    geometry_input=False)

    def _warm_grasp(self):
        if self.replace_SFP_position and self.grasp.plan.get('family')=='sfp':
            self.grasp=self.repaired_SFP_pose
        return super()._warm_grasp()

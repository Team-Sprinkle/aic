"""Preload both grasp encoders so shared preprocessing applies from task one."""
from .RunVisualConnectorServoFreeApproachV50 import RunVisualConnectorServoFreeApproachV50
from .RunFullConnectorInsertion import ROOT
from native_connector_pose_observation import NativeConnectorPoseObservation
import torch


class RunVisualConnectorServoPrewarmedFreeV51(RunVisualConnectorServoFreeApproachV50):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        with torch.inference_mode():
            for family,folder in (('sfp','sfp_direct_pose'),('sc','sc_orientation_jitter_v2')):
                model = NativeConnectorPoseObservation(ROOT/folder/'best.pt',str(self.local.device))
                for _ in range(3):
                    model.model(torch.zeros(1,3,3,192,192,device=self.local.device,dtype=torch.uint8))
                self.connector_models[family] = model
            if self.local.device.type=='cuda':
                torch.cuda.synchronize(self.local.device)
        self._trace(event='both_grasp_rotation_models_preloaded',
                    model_weights_unchanged=True,synthetic_kernel_warmup_sets_no_pose=True,
                    shared_JPEG_available_before_first_family_query=True,geometry_input=False)

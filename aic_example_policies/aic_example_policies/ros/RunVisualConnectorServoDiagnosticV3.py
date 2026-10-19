"""Same image-only controller, with native pose networks warmed before transport."""
import torch
import time
from .RunVisualConnectorServoDiagnosticV2 import RunVisualConnectorServoDiagnosticV2
from .RunFullConnectorInsertion import ROOT
from native_connector_pose_observation import NativeConnectorPoseObservation


class RunVisualConnectorServoDiagnosticV3(RunVisualConnectorServoDiagnosticV2):
    def _warm_grasp(self):
        started=time.perf_counter()
        super()._warm_grasp()
        with torch.inference_mode():
            for family,name in [('sc','sc_orientation_jitter_v2'),('sfp','sfp_direct_pose')]:
                if family not in self.connector_models:
                    self.connector_models[family]=NativeConnectorPoseObservation(ROOT/name/'best.pt',str(self.local.device))
                self.connector_models[family].model(torch.zeros(1,3,3,192,192,dtype=torch.uint8,device=self.local.device))
        if self.local.device.type=='cuda':torch.cuda.synchronize(self.local.device)
        self._trace(event='visual_servo_pose_networks_warmed',initialization_ms=(time.perf_counter()-started)*1000,synthetic_pixels_bound_as_pose=False,geometry_input=False)

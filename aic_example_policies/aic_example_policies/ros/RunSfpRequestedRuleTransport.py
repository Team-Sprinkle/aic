"""Rule acquisition with the full-data, single-request RGB-history locator."""
import os
from .RunSfpGuidedAnchoredRuleTransport import RunSfpGuidedAnchoredRuleTransport
from sfp_requested_port_observation import RequestedSfpPort


class RunSfpRequestedRuleTransport(RunSfpGuidedAnchoredRuleTransport):
    def _create_observer(self):
        observer=RequestedSfpPort(os.environ['AIC_SFP_REQUESTED_BACKBONE_CHECKPOINT'],
            os.environ['AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT'],
            os.environ['AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT'],
            os.environ['AIC_SFP_RULE_MOUNT_PRIORS'],str(self.device),
            view_threshold=float(os.environ.get('AIC_SFP_REQUESTED_VIEW_THRESHOLD','.9')),
            maximum_reprojection_px=float(os.environ.get('AIC_SFP_NATIVE_POSE_MAX_REPROJECTION_PX','2.')))
        observer.normalize_native_codec=True
        return observer

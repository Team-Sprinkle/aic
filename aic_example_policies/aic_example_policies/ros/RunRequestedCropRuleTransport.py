"""Controlled SC requested-crop ablation; all native geometry gates retained."""
from .RunYawMemoryTrackedRuleTransport import RunYawMemoryTrackedRuleTransport
from requested_crop_agreement import RequestedCropAgreement


class RunRequestedCropRuleTransport(RunYawMemoryTrackedRuleTransport):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        if self.verifier.candidate_gate_mode!='association_replaces_presence':
            raise ValueError('Explicit requested-crop ablation gate required')
        self.verifier.candidate_association=RequestedCropAgreement(self.verifier.requested_locator)
        original=self.raw_strict_inspect
        def inspect(*args,**kwargs):
            report=original(*args,**kwargs)
            report['candidate_check_source']='requested_locator_crop_agreement'
            report['projection_flags_are_not_confidence']=True
            report['separate_candidate_classifier_used']=False
            return report
        self.raw_strict_inspect=inspect

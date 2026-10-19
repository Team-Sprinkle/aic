"""Requested-locator crop agreement, without a second identity classifier.

These scores remain projected-in-frame flags, not visibility or confidence.
The caller MUST retain native refinement, three-view geometry and handoff
gates. This is an opt-in ablation until held-out and live checks pass.
"""
import numpy as np


class RequestedCropAgreement:
    def __init__(self,requested_locator,maximum_crop_error_px=50.):
        if requested_locator is None:raise ValueError('A task-conditioned locator is required')
        self.locator=requested_locator;self.maximum_crop_error_px=maximum_crop_error_px

    def reset(self):pass
    def observe(self,spatial,state):pass  # The verifier observes its locator itself.

    def probabilities(self,pixels,requested_port):
        expected=self.locator.predict(requested_port)
        distance=np.linalg.norm(np.asarray(pixels)-np.asarray(expected['pixels']),axis=-1)
        flags=np.asarray(expected['projected_inframe_probability'])
        return np.where(distance<=self.maximum_crop_error_px,flags,0.)

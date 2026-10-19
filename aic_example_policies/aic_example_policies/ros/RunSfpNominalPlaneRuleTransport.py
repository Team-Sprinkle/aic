"""Acquire vision at the training-prior 150 mm plane, not 200 mm above it.

Changes only the no-current-visual-lock acquisition goal. The requested-port,
precision, stability and measured final handoff gates are inherited unchanged.
"""
from .RunSfpVisualRuleTransport import RunSfpVisualRuleTransport


class RunSfpNominalPlaneRuleTransport(RunSfpVisualRuleTransport):
    nominal_plane_acquisition = True

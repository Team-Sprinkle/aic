"""Guarded opposite-row SFP view, for visual identity diagnosis only.

Uses the same force, contact, incremental motion, and measured-return guards
as CheatCodeWideViewDiagnostic. Oracle geometry is used for diagnostic motion,
not for visual identity or any deployable policy.
"""

from .CheatCodeWideViewDiagnostic import CheatCodeWideViewDiagnostic


class CheatCodeOppositeWideViewDiagnostic(CheatCodeWideViewDiagnostic):
    ROW_Y_M = -.080

"""Same bounded V32 test with baseXY500; no extra axial stiffness."""
from .RunVisualConnectorServoStiffV32 import RunVisualConnectorServoStiffV32


class RunVisualConnectorServoStiffV33(RunVisualConnectorServoStiffV32):
    lateral_stiffness=500.

"""Choose the existing CheatCode controller's initial base-frame Z offset.

This is a handoff gate, not an insertion-success detector. Success must still
come from the scorer. Coordinates follow CheatCode's base-frame XYZ controller.
"""

import math


def initial_insertion_z_offset(port_xyz, tip_xyz, port_wxyz, tip_wxyz, *, approach_offset=0.2):
    values = (*port_xyz, *tip_xyz, *port_wxyz, *tip_wxyz)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite port/tip transform")
    norms = [math.sqrt(sum(value * value for value in quat)) for quat in (port_wxyz, tip_wxyz)]
    if min(norms) < 1e-8:
        raise ValueError("Invalid port/tip quaternion")
    dot = abs(sum(a * b for a, b in zip(port_wxyz, tip_wxyz))) / (norms[0] * norms[1])
    angle = 2 * math.acos(min(1.0, dot))
    lateral = math.hypot(tip_xyz[0] - port_xyz[0], tip_xyz[1] - port_xyz[1])
    offset = tip_xyz[2] - port_xyz[2]
    # Inside the existing final-descent corridor, preserve current depth.
    # Sending the usual +0.2 m approach pose would first pull the plug out.
    if lateral <= 0.002 and angle <= 0.1 and -0.025 <= offset <= 0.005:
        return offset
    return approach_offset

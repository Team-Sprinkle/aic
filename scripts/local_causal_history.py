"""Shared observation-only history sampling for training and deployment."""
import numpy as np
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


def sample_robot_history(times, states, now, base_port, offsets, masked=False):
    times = np.asarray(times, dtype=float)
    if not len(times) or np.any(np.diff(times) < 0) or times[-1] > now + 1e-6:
        raise ValueError('History must be nonempty, ordered and causal')
    if len(offsets) != 4 or min(offsets) < 0:
        raise ValueError('Expected four nonnegative history offsets')
    indices = np.searchsorted(times, now - np.asarray(offsets), side='right') - 1
    indices = np.maximum(indices, 0)
    if masked:
        indices[:] = len(times) - 1
    port_base = np.linalg.inv(base_port)
    result = []
    for i in indices:
        state = np.asarray(states[int(i)], dtype=float)
        relative = port_base @ matrix_from_xyz_xyzw(state[:7])
        result.append([*state, *relative[:3, 3], *relative[:3, :2].T.reshape(-1),
                       max(0., min(max(offsets), now - times[i]))])
    return np.asarray(result, dtype=np.float32)

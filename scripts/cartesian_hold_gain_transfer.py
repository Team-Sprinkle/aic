"""Change impedance gains while retaining the current public holding effort."""
import numpy as np
from scipy.spatial.transform import Rotation


def transfer_hold(measured, reference, old_stiffness, new_stiffness):
    measured,reference = np.asarray(measured),np.asarray(reference)
    old,new = np.asarray(old_stiffness,float),np.asarray(new_stiffness,float)
    if old.shape!=(6,) or new.shape!=(6,) or not np.isfinite(np.r_[old,new]).all() or min(old.min(),new.min())<=0:
        raise ValueError('Finite positive Cartesian gains required')
    error = reference[:3,3]-measured[:3,3]
    angle = Rotation.from_matrix(reference[:3,:3]@measured[:3,:3].T).as_rotvec()
    adjusted = measured.copy()
    adjusted[:3,3] += old[:3]/new[:3]*error
    adjusted[:3,:3] = Rotation.from_rotvec(old[3:]/new[3:]*angle).as_matrix()@measured[:3,:3]
    if np.linalg.norm(adjusted[:3,3]-measured[:3,3])>.040 or np.linalg.norm(old[3:]/new[3:]*angle)>np.deg2rad(5.):
        raise ValueError('Unreasonable public holding-effort transfer')
    return adjusted

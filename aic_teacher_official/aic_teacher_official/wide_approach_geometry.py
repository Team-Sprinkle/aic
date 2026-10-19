"""Pure geometry for label-only insertion start perturbations, never actor input."""
import numpy as np


def tcp_target_for_tip_pose(base_port,tcp_tip,axial_m,lateral_xy_m,rotvec_rad):
    lateral=np.asarray(lateral_xy_m,float);rv=np.asarray(rotvec_rad,float)
    if lateral.shape!=(2,) or rv.shape!=(3,) or not np.isfinite(np.r_[lateral,rv,axial_m]).all():
        raise ValueError('Invalid approach reset')
    if not .03<=axial_m<=.18 or np.linalg.norm(lateral)>.02500001 or np.linalg.norm(rv)>np.deg2rad(15.00001):
        raise ValueError('Reset outside approved buffered approach envelope')
    angle=float(np.linalg.norm(rv));rotation=np.eye(3)
    if angle>1e-10:
        axis=rv/angle;x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        rotation+=np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
    port_tip=np.eye(4);port_tip[:3,:3]=rotation;port_tip[:3,3]=[*lateral,-axial_m]
    return np.asarray(base_port)@port_tip@np.linalg.inv(tcp_tip)

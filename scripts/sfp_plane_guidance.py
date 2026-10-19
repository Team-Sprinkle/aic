"""Approximate transport guidance from a visible requested-port pixel.

The nominal fixture height is a training-only prior. This estimate may place
the robot for reacquisition; it must never supply an insertion pose lock.
"""
import numpy as np

from evaluate_sc_native_triangulation import FOCAL
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


def requested_plane_guidance(prediction,state,optics,prior,*,probability=.9):
    pixels=prediction.get('refined_pixels') or {}
    coarse=prediction.get('coarse_pixels') or {}
    height=float(prior.get('guidance_nominal_height_m',prior['port_mean_base_m'][2]));points=[];names=[];bounds=[]
    for i,c in enumerate(('center','left','right')):
        if c not in pixels or c not in coarse or prediction['view_probabilities'][i]<probability:continue
        pixel=np.asarray(pixels[c],float)
        if not np.isfinite(pixel).all() or np.linalg.norm(pixel-coarse[c])>50:continue
        camera=matrix_from_xyz_xyzw(state[:7])@optics[c]
        ray=camera[:3,:3]@np.r_[(pixel-[576,512])/FOCAL,1.]
        if abs(ray[2])/np.linalg.norm(ray)<.5:continue
        depth=(height-camera[2,3])/ray[2]
        if depth<=0 or depth*np.linalg.norm(ray)>1.:continue
        point=camera[:3,3]+depth*ray
        if not (np.all(point>=prior['potential_min_base_m']) and np.all(point<=prior['potential_max_base_m'])):continue
        height_bound=prior.get('guidance_height_uncertainty_m',max(
            abs(height-prior['potential_min_base_m'][2]),
            abs(height-prior['potential_max_base_m'][2])))
        lateral_bound=float(height_bound*np.linalg.norm(ray[:2])/abs(ray[2]))
        if lateral_bound>.025:continue
        points.append(point);names.append(c);bounds.append(lateral_bound)
    if not points:return None
    point=np.median(points,axis=0)
    if max(np.linalg.norm(p[:2]-point[:2]) for p in points)>.010:return None
    return dict(position_base_m=point.tolist(),cameras=names,
        height_uncertainty_lateral_bound_mm=max(bounds)*1000,
        geometry_input=False,nominal_height_training_prior=True,
        insertion_lock_allowed=False)

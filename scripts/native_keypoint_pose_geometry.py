"""Fit one requested port from predicted landmarks and calibrated cameras.

Known reference dimensions and fixed fixture roll/pitch are calibration priors.
Target translation and yaw remain unknown and are fitted from RGB predictions.
No true target pose is an inference argument. The flat-fixture assumption must
be revalidated before use on boards with different roll/pitch.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def fit_fixed_normal_pose(predicted_pixels,cameras,reference_points,mount_rotation,focal,*,maximum_evaluations=40):
    from evaluate_sc_native_triangulation import triangulate
    names=sorted(predicted_pixels)
    if len(names)<2:raise ValueError('At least two calibrated views required')
    points=np.asarray(reference_points,float)
    if points.shape!=(5,3):raise ValueError('Expected five landmarks of ONE requested port')
    cloud=np.stack([triangulate({c:predicted_pixels[c][i] for c in names},cameras) for i in range(5)])
    direction=cloud[2]-cloud[1];yaw=np.arctan2(direction[1],direction[0])-np.arctan2(mount_rotation[1,0],mount_rotation[0,0])
    initial=np.r_[cloud[0],yaw];inverse=np.stack([np.linalg.inv(cameras[c]) for c in names]);truth=np.stack([predicted_pixels[c] for c in names])
    def project(params):
        rotation=Rotation.from_euler('z',params[3]).as_matrix()@mount_rotation
        base=points@rotation.T+params[:3]
        optical=np.einsum('cij,pj->cpi',inverse[:,:3,:3],base)+inverse[:,:3,3][:,None]
        uv=optical[...,:2]/np.maximum(optical[...,2:],1e-5)*focal+[576,512]
        return uv,optical[...,2],rotation
    def residual(params):return (project(params)[0]-truth).reshape(-1)
    result=least_squares(residual,initial,loss='soft_l1',f_scale=.75,max_nfev=maximum_evaluations,xtol=1e-8,ftol=1e-8,gtol=1e-8)
    uv,depth,rotation=project(result.x);error=np.linalg.norm(uv-truth,axis=-1)
    return dict(position_base_m=result.x[:3],rotation_base=rotation,max_reprojection_px=float(error.max()),
        rms_reprojection_px=float(np.sqrt(np.mean(error**2))),all_depths_positive=bool((depth>0).all()),
        converged=bool(result.success),evaluations=result.nfev,geometry_input=False,
        fixture_assumption='fixed calibrated roll/pitch; free translation and yaw')

"""Independent projected poses must survive triangulation and crop translations."""
from pathlib import Path
import sys,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from train_sc_spatial_grasp_to_plateau import ray_cloud,planar_pose,CAD,FOCAL


def test_multicamera_pose_and_crop_origin_roundtrip_has_finite_gradient():
    root=Path(__file__).resolve().parents[2]
    data=json.loads((root/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics=np.stack([data['tcp_to_optical'][c] for c in ['center','left','right']])
    rotations=Rotation.from_euler('xyz',[[.48,-.02,1.63],[.50,-.04,1.60],[.45,.01,1.65]]).as_matrix()
    translations=np.array([[-.001,-.012,.016],[0,-.013,.017],[.001,-.011,.015]])
    truth=np.einsum('bij,kj->bki',rotations,CAD)+translations[:,None]
    native=[]
    for optical in optics:
        inv=np.linalg.inv(optical);camera=truth@inv[:3,:3].T+inv[:3,3]
        assert camera[:,:,2].min()>0
        native.append(camera[:,:,:2]/camera[:,:,2:]*FOCAL+[576,512])
    native=np.stack(native,1)
    origins=np.array([[480,640],[455,620],[495,645]])
    shifts=np.array([[[3,-2],[-4,1],[0,4]]]*3)
    local=native-origins[None,:,None,:]+shifts[:,:,None]
    restored=local+(origins-shifts)[:,:,None]
    assert np.max(abs(restored-native))<1e-10
    pixels=torch.tensor(restored,dtype=torch.float64,requires_grad=True)
    cloud=ray_cloud(pixels,torch.tensor(optics,dtype=torch.float64))
    position,rotation=planar_pose(cloud,torch.tensor(CAD,dtype=torch.float64))
    assert np.max(abs(cloud.detach().numpy()-truth))<2e-6
    assert np.max(abs(position.detach().numpy()-translations))<2e-6
    assert np.max(abs(rotation.detach().numpy()-rotations))<1e-4
    (position.square().sum()+rotation.square().sum()+cloud.square().sum()).backward()
    assert torch.isfinite(pixels.grad).all()

"""Causal requested-SC crop proposals from RGB and relative robot motion."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F
from train_sc_requested_history_locator import RequestedHistoryLocator


class RequestedPortHistoryObservation:
    def __init__(self,checkpoint,device='cuda'):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('schema') not in ('SC_requested_history_locator/v1','SFP_requested_history_locator/v1') or saved.get('geometry_input') is not False:
            raise ValueError('Expected an observation-only requested-port locator')
        self.device=torch.device(device)
        self.num_requests=saved.get('num_requests',2)
        self.model=RequestedHistoryLocator(saved['use_history'],self.num_requests).to(device).eval()
        self.model.load_state_dict(saved['model'],strict=True);self.reset()

    def reset(self):
        self.anchors=[];self.path=0.;self.anchor_path=0.;self.previous=None;self.current=None;self.state=None

    @torch.inference_mode()
    def observe(self,spatial,state):
        state=np.asarray(state[:7],float)
        if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('Invalid public robot pose')
        current=F.adaptive_avg_pool2d(spatial.float(),(8,9))
        if current.shape!=(3,256,8,9):raise ValueError('Expected all three ordered camera feature maps')
        if self.previous is not None:self.path+=float(np.linalg.norm(state[:3]-self.previous[:3]))
        angle=0. if not self.anchors else np.degrees((Rotation.from_quat(self.anchors[-1][0][3:7]).inv()*Rotation.from_quat(state[3:7])).magnitude())
        if not self.anchors or self.path-self.anchor_path>=.04 or angle>=10:
            pooled=F.adaptive_avg_pool2d(current,(2,3)).flatten(2).permute(0,2,1)
            self.anchors.append((state.copy(),pooled))
            if len(self.anchors)>10:self.anchors=[self.anchors[0]]+self.anchors[-9:]
            self.anchor_path=self.path
        self.current=current;self.state=state.copy();self.previous=state.copy()

    @torch.inference_mode()
    def predict(self,requested_port):
        if not 0<=requested_port<self.num_requests or self.current is None:raise ValueError('Observe before requesting a valid port')
        history=self.current.new_zeros((1,10,3,6,256));motion=self.current.new_zeros((1,10,9))
        mask=torch.zeros(1,10,device=self.device,dtype=torch.bool)
        rot=Rotation.from_quat(self.state[3:7])
        for j,(old,pooled) in enumerate(self.anchors):
            history[0,j]=pooled;mask[0,j]=True
            rel=rot.inv()*Rotation.from_quat(old[3:7])
            value=np.r_[rot.inv().apply(old[:3]-self.state[:3])/.2,rel.as_matrix()[:,:2].T.reshape(-1)]
            motion[0,j]=torch.tensor(value,device=self.device,dtype=motion.dtype)
        point,flag,_=self.model(self.current[None],history,motion,mask,
                               torch.tensor([requested_port],device=self.device))
        return dict(pixels=(point[0]*point.new_tensor([1151,1023])).cpu().tolist(),
                    projected_inframe_probability=flag[0].sigmoid().cpu().tolist(),
                    geometry_input=False,absolute_state_input=False)


# Preserve the existing SC entry point and checkpoint contract.
SCRequestedHistoryObservation=RequestedPortHistoryObservation

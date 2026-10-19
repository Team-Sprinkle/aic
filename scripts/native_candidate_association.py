"""Deployment adapter for a requested SC candidate check with measured-motion history."""
import numpy as np
import torch
from torch.nn import functional as F

from train_native_candidate_history_association import CandidateHistoryAssociation


class NativeCandidateAssociation:
    def __init__(self,checkpoint,device='cuda'):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('geometry_input') is not False or saved['variant'] not in ('current','distance_history'):
            raise ValueError('Expected RGB candidate association checkpoint')
        self.device=torch.device(device)
        self.model=CandidateHistoryAssociation(saved['variant']=='distance_history',saved.get('always_current_global',False)).to(device).eval()
        self.model.load_state_dict(saved['model'],strict=True)
        self.mean=saved['state_mean'].to(device);self.std=saved['state_std'].to(device)
        self.spacing_m=saved['spacing_mm']/1000;self.reset()

    def reset(self):
        self.anchors=[];self.path_m=0.;self.anchor_path_m=0.;self.previous_state=None
        self.current=None;self.state=None

    @torch.inference_mode()
    def observe(self,spatial,state):
        """Call during transport as well as inspection. No clock-only anchors."""
        state=np.asarray(state[:7],np.float32).copy()
        if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('Invalid public measured TCP')
        state[3:]/=np.linalg.norm(state[3:])
        if state[6]<0:state[3:]*=-1
        current=F.adaptive_avg_pool2d(spatial.float(),(8,9))
        if current.shape!=(3,256,8,9):raise ValueError('Expected three ordered camera spatial maps')
        if self.previous_state is not None:self.path_m+=float(np.linalg.norm(state[:3]-self.previous_state[:3]))
        angle=0.
        if self.anchors:angle=float(np.degrees(2*np.arccos(np.clip(abs(state[3:]@self.anchors[-1][0][3:]),0,1))))
        if not self.anchors or self.path_m-self.anchor_path_m>=self.spacing_m or angle>=10:
            self.anchors.append((state.copy(),F.adaptive_avg_pool2d(current,(2,3)).flatten(2).permute(0,2,1)))
            if len(self.anchors)>10:self.anchors=[self.anchors[0]]+self.anchors[-9:]
            self.anchor_path_m=self.path_m
        self.current=current;self.state=state;self.previous_state=state

    @torch.inference_mode()
    def probabilities(self,pixels,requested_port):
        if self.current is None:raise ValueError('Observe the current camera frame before candidate check')
        if requested_port not in (0,1):raise ValueError('SC public selected port must be0or1')
        pts=np.asarray(pixels,np.float32)
        if pts.shape!=(3,2) or not np.isfinite(pts).all():raise ValueError('Expected RGB proposal for each camera')
        history=self.current.new_zeros((10,3,6,256));states=self.current.new_zeros((10,7));mask=torch.zeros(10,dtype=torch.bool,device=self.device)
        for i,(state,feature) in enumerate(self.anchors):
            history[i]=feature;states[i]=(torch.from_numpy(state).to(self.device)-self.mean)/self.std;mask[i]=True
        state=(torch.from_numpy(self.state).to(self.device)-self.mean)/self.std
        score=self.model(self.current[None].expand(3,-1,-1,-1,-1),history[None].expand(3,-1,-1,-1,-1),
            states[None].expand(3,-1,-1),mask[None].expand(3,-1),state[None].expand(3,-1),
            torch.arange(3,device=self.device),torch.from_numpy(pts).to(self.device)/self.current.new_tensor([1151,1023]),
            torch.full((3,),requested_port,device=self.device,dtype=torch.long))
        return score.sigmoid().float().cpu().numpy()

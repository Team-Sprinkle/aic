"""One qualified joint RGB pose, shared between orientation and position calls."""
import numpy as np
from torch import nn
from native_connector_pose_observation import NativeConnectorPoseObservation


class _WarmPose(nn.Module):
    def __init__(self,model):
        super().__init__();self.pose_model=model
    def forward(self,x,camera_id=None):
        return self.pose_model(x[None] if x.ndim==4 else x)


class SharedSFPRepairedPose:
    def __init__(self,checkpoint,device):
        self.estimator=NativeConnectorPoseObservation(checkpoint,device)
        if self.estimator.rotation_only:raise ValueError('A joint position component needs supervised translation')
        self.plan=self.estimator.plan;self.model=_WarmPose(self.estimator.model)
        self.last_images=None;self.last_result=None

    def predict(self,images,*,already_training_jpeg=False,visual_tip_position_tcp_m=None,estimated_TCP_axial_gap_mm=None):
        if visual_tip_position_tcp_m is None and (estimated_TCP_axial_gap_mm is None or estimated_TCP_axial_gap_mm<120):
            raise ValueError('Joint position is qualified only at free acquisition TCP gap>=120mm')
        hit=already_training_jpeg and self.last_images is not None and all(images[c] is self.last_images[c] for c in ('center','left','right'))
        if not hit:
            self.last_result=self.estimator.predict(images,already_training_jpeg=already_training_jpeg)
            self.last_images={c:images[c] for c in ('center','left','right')}
        result=dict(self.last_result,orientation_approved=True,runtime_true_geometry=False,shared_pose_cache_hit=hit)
        if hit:result['latency_ms']=0.
        if visual_tip_position_tcp_m is not None:
            position=np.asarray(visual_tip_position_tcp_m)
            result['predicted_tip_position_tcp_m']=position.tolist()
            matrix=np.asarray(result['predicted_connector_in_tcp_matrix']).copy();matrix[:3,3]=position
            result['predicted_connector_in_tcp_matrix']=matrix.tolist();result['position_source']='explicit caller placeholder; not bound as control position'
        else:result['position_source']='qualified joint RGB translation head'
        return result

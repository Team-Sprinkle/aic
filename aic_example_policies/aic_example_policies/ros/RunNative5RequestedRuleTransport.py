"""SC LLB rule with task-conditioned native five-landmark acquisition.

Controlled decoder/gate ablation; no learned transport or insertion actions.
The older single-point/checker models are initialized by the base constructor
but do not authorize this branch's decisions.
"""
import os
import cv2
from .RunRuleTransport import RunRuleTransport
from .RunSharedPortTCP import REPO_ROOT
from native_port_handoff_verifier import NativePortHandoffVerifier
from requested_native_port_pose import RequestedNativePortPose


class RunNative5RequestedRuleTransport(RunRuleTransport):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        # Tiny scipy pose fits must not create64-way BLAS work while Gazebo
        # renders on the same host. Keep independent camera codecs parallel.
        if os.environ.get('OPENBLAS_NUM_THREADS')!='1':
            raise ValueError('Native pose rollout requires OPENBLAS_NUM_THREADS=1 before Python startup')
        cv2.setNumThreads(1)
        component=RequestedNativePortPose(
            REPO_ROOT/'outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt',
            os.environ['AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT'],
            os.environ.get('AIC_SC_NATIVE_POSE_CHECKPOINT') or os.environ['AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT'],
            os.environ['AIC_SFP_RULE_MOUNT_PRIORS'],family='sc',device=str(self.device),maximum_shift_px=64.,
            maximum_reprojection_px=float(os.environ.get('AIC_SC_NATIVE_POSE_MAX_REPROJECTION_PX','2.')))
        previous=self.verifier
        self.verifier=NativePortHandoffVerifier(component,previous.standoff_offset,previous.nominal_tcp_in_port)
        # The older tracked controller allowed140mm after its two-view guide.
        # Here a current THREE-view requested native pose supplies that move.
        # The base-only80mm limit otherwise rejects valid85mm rail corrections.
        self.maximum_visual_reacquisition_m=.14
        previous.board_pool.shutdown(wait=True)

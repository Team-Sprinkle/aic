"""Bounded full-connector BC diagnostic; public RGB/state/task only.

Same rule transport and25mm approach as the controlled history comparison.
Adds a causal3-frame grasp-rotation lock; no privileged crop or actor input.
"""
import os
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .RunHistoryChunkedInsertion import RunHistoryChunkedInsertion
from native_connector_pose_observation import NativeConnectorPoseObservation


ROOT=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1')


class RunFullConnectorInsertion(RunHistoryChunkedInsertion):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        if self.local.sc_action_reference!='full_connector_pose' or self.local.tip_input_mode!='full_pose':
            raise ValueError('This diagnostic requires the exact full-connector representation')
        self.connector_models={}
        self.connector_rotation_window=[]
        self.locked_connector_rotation=None

    def _fresh_report(self,observation,task_vector):
        report,images,state,sim=super()._fresh_report(observation,task_vector)
        if not report['valid']:
            self.connector_rotation_window=[]
            return report,images,state,sim
        family='sfp' if task_vector[0]==1 else 'sc'
        if family not in self.connector_models:
            filename=ROOT/('sfp_direct_pose' if family=='sfp' else 'sc_orientation_jitter_v2')/'best.pt'
            self.connector_models[family]=NativeConnectorPoseObservation(filename,str(self.local.device))
        # The SC fit predicts rotation only. Its required position argument is
        # deliberately unused here; the approved visual position is bound by
        # the parent acquisition, never replaced with this placeholder.
        started=time.perf_counter()
        estimate=self.connector_models[family].predict(images,visual_tip_position_tcp_m=np.zeros(3))
        rotation=np.asarray(estimate['orientation_for_diagnostic_only'])
        self.connector_rotation_window=[x for x in self.connector_rotation_window if 0<=sim-x[0]<=1.]
        if self.connector_rotation_window and sim<=self.connector_rotation_window[-1][0]:
            return dict(report,valid=False,reason='repeated_orientation_timestamp'),images,state,sim
        self.connector_rotation_window=(self.connector_rotation_window+[(sim,rotation)])[-3:]
        accepted=False;span=None
        if len(self.connector_rotation_window)==3:
            rotations=[x[1] for x in self.connector_rotation_window]
            span=max(float((Rotation.from_matrix(a).inv()*Rotation.from_matrix(b)).magnitude()*180/np.pi)
                for a in rotations for b in rotations)
            if span<=1.:
                self.locked_connector_rotation=Rotation.from_matrix(np.stack(rotations)).mean().as_matrix()
                accepted=True
            else:
                self.connector_rotation_window=self.connector_rotation_window[1:]
        self._trace(event='connector_rotation_acquisition',sim_time=sim,family=family,
            rotation_tcp=rotation.tolist(),window_samples=len(self.connector_rotation_window),
            accepted=accepted,span_deg=span,orientation_model_ms=(time.perf_counter()-started)*1000,
            geometry_input=False,position_placeholder_not_used_for_control=True)
        report=dict(report,connector_rotation_consensus_accepted=bool(accepted))
        if not accepted and not getattr(self,'joint_grasp_acquisition',False):
            report=dict(report,valid=False,reason='awaiting_stable_connector_rotation')
        return report,images,state,sim

    def _lock(self,task_vector,get_observation,move_robot,*,fresh_context):
        self.connector_rotation_window=[];self.locked_connector_rotation=None
        self.local.visual_body_connector_rotation=None
        if not super()._lock(task_vector,get_observation,move_robot,fresh_context=fresh_context):return False
        if self.locked_connector_rotation is None:raise ValueError('Missing causal connector orientation')
        self.local.visual_body_connector_rotation=self.locked_connector_rotation.copy()
        self._trace(event='full_connector_grasp_bound',rotation_tcp=self.locked_connector_rotation.tolist(),
            position_tcp_m=self.local.visual_body_tip.tolist(),source='three native RGB frames before approach',
            geometry_input=False,static_within_attempt=True)
        return True

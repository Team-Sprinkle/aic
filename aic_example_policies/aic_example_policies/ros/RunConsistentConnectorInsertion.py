"""Pair full-target BC with the exact RGB connector estimators used in its rows."""
import hashlib
import os
from pathlib import Path
import cv2  # Load the image codec before torch in the ROS/Pixi environment.
import torch
import numpy as np
from .RunContinuousConnectorInsertionV2 import RunContinuousConnectorInsertionV2
from native_connector_pose_observation import NativeConnectorPoseObservation


class JointVisualPosition:
    def __init__(self,original,pose):
        self.model=original.model  # Compatibility with the earlier warmup hook.
        self.pose=pose

    def predict(self,images,**kwargs):
        if kwargs:raise ValueError('SC joint position received unexpected privileged context')
        return self.pose.predict(images)


class RunConsistentConnectorInsertion(RunContinuousConnectorInsertionV2):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        checkpoint=torch.load(os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'],map_location='cpu',weights_only=False)
        estimators=checkpoint.get('connector_estimator_fingerprints')
        if not estimators or set(estimators)!={'sc','sfp'}:raise ValueError('Actor missing exact connector-estimator fingerprints')
        self.connector_position_modes=checkpoint.get('connector_position_modes',{})
        # The retained image-only position helpers must match the source
        # grasp locks too; checking the new rotation head alone is insufficient.
        retained=checkpoint.get('retained_translation_estimator_fingerprints') or {
            'sc':{'sha256':'d10db9f6e27acb0d18f4b8a58d659e0173f62f3f483f2077eb5268040d41b14f'},
            'sfp':{'sha256':'e19cc1ff555b2c6f67b94aea84360ec7aa60e4904482e8c7d6c14c836ba3f095'}}
        for family,key in [('sc','AIC_LOCKED_GRASP_CHECKPOINT'),('sfp','AIC_SFP_GRASP_CHECKPOINT')]:
            if self.connector_position_modes.get(family)!='retained':continue
            path=Path(os.environ[key])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=retained[family]['sha256']:raise ValueError('Retained translation helper differs from actor data: '+family)
        for family,record in estimators.items():
            path=Path(record['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Actor/connector estimator mismatch: '+family)
            self.connector_models[family]=NativeConnectorPoseObservation(path,str(self.local.device))
            if self.connector_position_modes.get(family)=='joint' and self.connector_models[family].rotation_only:raise ValueError('Joint position requested from rotation-only fit')
        # Warm the image path only; synthetic pixels never bind a pose or
        # provide a handoff/seating approval.
        with torch.inference_mode():
            for model in self.connector_models.values():
                model.model(torch.zeros(1,3,3,192,192,dtype=torch.uint8,device=self.local.device))
        if self.local.device.type=='cuda':torch.cuda.synchronize(self.local.device)
        self._trace(event='connector_estimator_contract_verified',estimators=estimators,
                    position_modes=self.connector_position_modes,geometry_input=False)

    def _approach_orientation_tolerance(self):
        return np.deg2rad(.5)

    def _lock(self,task_vector,get_observation,move_robot,*,fresh_context):
        if task_vector[1]==1 and self.connector_position_modes.get('sc')=='joint' and not isinstance(self.grasp,JointVisualPosition):
            self.grasp=JointVisualPosition(self.grasp,self.connector_models['sc'])
        return super()._lock(task_vector,get_observation,move_robot,fresh_context=fresh_context)

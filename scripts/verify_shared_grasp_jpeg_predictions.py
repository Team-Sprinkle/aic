#!/usr/bin/env python3
"""Verify actual frozen SFP estimators are numerically unchanged by JPEG sharing."""
import cv2
cv2.setNumThreads(2)
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from native_connector_pose_observation import NativeConnectorPoseObservation
from sfp_native_grasp_observation import SFPNativeGraspObservation
from shared_grasp_training_jpeg import SharedGraspTrainingJPEG,SharedJPEGEstimator

torch.set_num_threads(4)
repo=Path.cwd();root=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1');old=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1');selection=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/selection.json').read_text())['auxiliary']
row=next(json.loads(line) for line in (old/'full_connector_actor_rows/full_grasp_locks.jsonl').open() if json.loads(line)['family']=='sfp');images={c:cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB) for c,path in row['source_images'].items()}
pose=NativeConnectorPoseObservation(old/'sfp_direct_pose/best.pt','cpu');position=SFPNativeGraspObservation(selection['sfp_grasp'],selection['sfp_grasp_approval'],'cpu')
first=pose.predict(images,visual_tip_position_tcp_m=np.zeros(3));second=position.predict(images,estimated_TCP_axial_gap_mm=150.)
cache=SharedGraspTrainingJPEG();a=SharedJPEGEstimator(pose,cache,begin=True);b=SharedJPEGEstimator(position,cache,begin=False)
newfirst=a.predict(images,visual_tip_position_tcp_m=np.zeros(3));newsecond=b.predict(images,estimated_TCP_axial_gap_mm=150.)
differences={k:float(np.max(np.abs(np.array(before[key])-np.array(after[key])))) for k,before,after,key in [('pose_rotation',first,newfirst,'orientation_for_diagnostic_only'),('position',second,newsecond,'predicted_tip_position_tcp_m'),('landmark_pixels',second,newsecond,'predicted_pixels')]}
assert max(differences.values())==0.,differences
record=dict(real_frozen_estimators=True,device='cpu',source_episode=row['episode'],ground_truth_runtime_input=False,maximum_absolute_output_differences=differences,sharing_encodes=cache.encodes,cache_hits=cache.cache_hits,bit_exact=True,new_model_training=False)
(root/'shared_grasp_JPEG_equivalence.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

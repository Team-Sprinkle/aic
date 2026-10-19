#!/usr/bin/env python3
"""Finish frozen8-grasp fused review, then matched32-grasp fit when data exist."""
import json
import os
from pathlib import Path
import subprocess
import time
repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python';root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');state=root/'fused_orientation_comparison_state.json'
if state.exists():raise FileExistsError('Do not duplicate fused comparison')

def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None

def save(d):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(state)

def run(command,name,env=None):
 (root/(name+'.command.json')).write_text(json.dumps(list(map(str,command)))+'\n')
 with (root/(name+'.log')).open('w') as log:subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)

save(dict(stage='waiting_for_frozen8grasp_fused_orientation_plateau'))
while not read(root/'near_fused_orientation/summary.json'):time.sleep(20)
assert read(root/'near_fused_orientation/summary.json')['plateau_reached']
run([python,repo/'scripts/review_SC_near_fused_orientation.py','--fit',root/'near_fused_orientation','--output',root/'near_fused_orientation_prediction_review'],'near_fused_orientation_prediction_review')
save(dict(stage='waiting_for_targeted24training_grasp_dataset',existing8grasp_review_complete=True,automatic_deployment=False))
while not read(root/'expanded_near_pose_dataset/summary.json'):time.sleep(20)
source=root/'expanded_near_pose_dataset';fit=root/'expanded_near_fused_orientation'
command=[python,repo/'scripts/train_SC_near_fused_orientation_to_plateau.py','--source',source,'--pixels',source/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt','--output',fit,'--family','sc','--rotation-only','--balance-new-angles','--crop-jitter','4','--updates','100000','--minutes','1440','--min-updates','5000','--patience','3000']
save(dict(stage='training_matched32grasp_fused_orientation',physical_gpu=4,compared_with='near_fused_orientation',same_architecture_initialization_augmentation_selection_budget=True,changed='24additional independent neartrainingepisodes only',automatic_deployment=False))
run(command,'expanded_near_fused_orientation',dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
assert read(fit/'summary.json')['plateau_reached']
run([python,repo/'scripts/review_SC_near_fused_orientation.py','--fit',fit,'--output',root/'expanded_near_fused_orientation_prediction_review'],'expanded_near_fused_orientation_prediction_review')
save(dict(stage='matched_fused_comparison_complete_requires_visual_and_live_tests',automatic_deployment=False))

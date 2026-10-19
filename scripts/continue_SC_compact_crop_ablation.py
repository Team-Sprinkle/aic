#!/usr/bin/env python3
"""One conditional visual ablation after both matched pose fits finish."""
import json
import os
from pathlib import Path
import subprocess
import time
repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
state=root/'compact_crop_ablation_state.json'
if state.exists():raise FileExistsError('Do not duplicate the bounded ablation')

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def save(value):
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(state)

def run(command,name,env=None):
    (root/(name+'.command.json')).write_text(json.dumps(list(map(str,command)))+'\n')
    with (root/(name+'.log')).open('w') as log:
        subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)

save(dict(stage='waiting_for_both_matched_pose_plateaus',hypothesis='A larger native object footprint may improve subpixel rigid-pose precision',maximum_new_fits=1))
while not all(read(root/n/'summary.json') for n in ['acquisition_spatial_joint','far_near_spatial_joint']):time.sleep(20)
base=read(root/'far_near_spatial_joint/summary.json')
if not base['plateau_reached']:raise RuntimeError('Baseline fit incomplete, not evidence for ablation')
if base['gate_passed']:
    save(dict(stage='not_needed_baseline_gate_passed',new_fits=0));raise SystemExit(0)
source=root/'far_and_near_pose_dataset';cache=root/'compact_crop_dataset';fit=root/'compact_far_near_spatial_joint'
save(dict(stage='building_one_train_derived_compact_native_cache',source=str(source),new_episode_collection=False))
run([python,repo/'scripts/prepare_sc_compact_native_grasp_crops.py','--source',source,'--plan',root/'far_near_spatial_joint/plan.json','--output',cache,'--cache'],'compact_crop_dataset')
assert read(cache/'crop_geometry.json')['regions']==read(root/'compact_crop_training_only_plan/crop_geometry.json')['regions']
command=[python,repo/'scripts/train_sc_compact_spatial_grasp_to_plateau.py','--source',source,'--pixels',cache/'native_crops.npy','--crop-geometry',cache/'crop_geometry.json','--checkpoint','/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt','--output',fit,'--mode','joint','--crop-jitter','0','--selection-domains','original','nominal_acquisition','near_25mm','--updates','100000','--minutes','1440','--min-updates','5000','--patience','3000']
save(dict(stage='training_one_compact_crop_ablation_to_plateau',physical_gpu=4,identical_episode_split_and_budget=True,automatic_deployment=False))
run(command,'compact_far_near_spatial_joint',dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
summary=read(fit/'summary.json')
if not summary['plateau_reached']:raise RuntimeError('Compact fit incomplete: resume before judging it')
run([python,repo/'scripts/review_sc_spatial_pose_candidate.py','--fit',fit,'--output',root/'compact_far_near_spatial_joint_prediction_review'],'compact_far_near_spatial_joint_prediction_review')
save(dict(stage='ablation_complete_requires_actual_visual_inspection_and_decision',gate_passed=summary['gate_passed'],baseline_selection_score=base['selection_score'],ablation_selection_score=summary['selection_score'],new_fits=1,automatic_deployment=False))

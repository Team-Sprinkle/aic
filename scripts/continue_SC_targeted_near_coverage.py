#!/usr/bin/env python3
"""Conditional24-training-reset repair for a measured near-pose generalization gap."""
import json
import os
from pathlib import Path
import subprocess
import time
repo=Path.cwd();python=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
state=root/'targeted_near_coverage_state.json'
if state.exists():raise FileExistsError('Inspect existing targeted continuation')

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def save(value):
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(state)

def run(command,name,env=None):
    (root/(name+'.command.json')).write_text(json.dumps(list(map(str,command)))+'\n')
    with (root/(name+'.log')).open('w') as log:
        subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)

save(dict(stage='waiting_for_compact_crop_result',conditional_budget_new_training_episodes=24,validation_episodes_unchanged=True,automatic_deployment=False))
approval=root/'compact_far_near_spatial_joint_prediction_review/approval.json'
while not read(approval):time.sleep(20)
if read(approval)['pose_gate_passed']:
    save(dict(stage='not_needed_compact_near_gate_passed',new_episodes=0));raise SystemExit(0)
# The preceding measured trainingnearR p95.252deg vsheldout2.137deg,
# with only8near traininggrasps, is the required concrete coverage hypothesis.
captures=root/'targeted_near_training_captures'
run([python,repo/'scripts/collect_sc_grasp_variations.py','--output',captures,'--viewpoint','near','--gpu','3','--domain','123','--seed','2026101829','--repeats','6','--card-counts','0','5','--episode-prefix','coverage24','--train-only','--prepare-only'],'targeted_near_prepare')
plan=read(captures/'plan.json');assert len(plan['cases'])==24 and all(c['split']=='train' for c in plan['cases'])
(captures/'plan_before_public_near_resets.json').write_text(json.dumps(plan,indent=2)+'\n')
for case in plan['cases']:
    original=Path(case['scene']);output=captures/'reset_scenes'/case['case']
    command=[python,repo/'scripts/prepare_grasp_viewpoint_reset.py','--scene',original,'--output',output,'--home-state',root/'public_home_state.json','--view-state',root/'public_near25_view_state.json']
    run(command,'targeted_near_reset_'+case['case'])
    case.update(original_scene=str(original),scene=str(output),pose_only_reset_hold=True,sampling_window_s=4.5,
        near_public_robot_viewpoint_reset=True,actual_scene_25mm_standoff_not_asserted=True,
        capture_reason='Independent random grasp/board contexts at public near robot posture; held-plug labels only')
plan.update(near_public_robot_viewpoint_reset=True,validation_rows_added=0,
    hypothesis='8neartraininggrasps memorize well(.252degp95) but heldoutgrasp error remains2.137deg. Add24independent nominal-centred near-view grasp resets without changing heldout scenes.',
    gate='First6need actual image/causal QA; stop on invalidcapture; maximum24new admitted episodes, then one fit to plateau. No insertion or BCexpert claim.')
(captures/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
save(dict(stage='collecting24training_only_near_viewpoints_requires_first6_visual_QA',physical_gpu=3,maximum_new_episodes=24,BC_expert=False,heldout_unchanged=True))
run([python,repo/'scripts/collect_sc_grasp_variations.py','--output',captures,'--viewpoint','near','--gpu','3','--domain','123','--seed','2026101829','--resume'],'targeted_near_collection')
summary=read(captures/'summary.json');assert summary['completed']==24 and all(x['valid'] for x in summary['results'])
source=root/'expanded_near_pose_dataset'
save(dict(stage='merging24additional_training_episodes_heldout_unchanged'))
run([python,repo/'scripts/merge_sc_grasp_capture_supervision.py','--source',root/'far_and_near_pose_dataset','--captures',captures,'--output',source],'expanded_near_dataset')
old=read(root/'far_and_near_pose_dataset/manifest.json');new=read(source/'manifest.json')
assert [x for x in old if x['split']=='validation']==[x for x in new if x['split']=='validation']
fit=root/'expanded_near_spatial_joint'
command=[python,repo/'scripts/train_sc_spatial_grasp_to_plateau.py','--source',source,'--pixels',source/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt','--output',fit,'--mode','joint','--crop-jitter','0','--selection-domains','near_25mm','--updates','100000','--minutes','1440','--min-updates','5000','--patience','3000']
save(dict(stage='training_targeted_near_pose_to_plateau',physical_gpu=3,selection='Nearheldoutdomain only: deployment uses only the25mmhold; other domains stillreported',selection_change_explicit=True,data_and_checkpoint_selection_changes_not_isolated=True))
run(command,'expanded_near_spatial_joint',dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
if not read(fit/'summary.json')['plateau_reached']:raise RuntimeError('Incomplete fit; resume before judging it')
run([python,repo/'scripts/review_sc_spatial_pose_candidate.py','--fit',fit,'--output',root/'expanded_near_spatial_joint_prediction_review'],'expanded_near_spatial_joint_prediction_review')
save(dict(stage='targeted_near_fit_complete_requires_visual_and_live_tests',approval=str(root/'expanded_near_spatial_joint_prediction_review/approval.json'),automatic_deployment=False))

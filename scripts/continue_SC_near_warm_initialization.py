#!/usr/bin/env python3
"""One conditional full-RGB-head warm-start test after both32-grasp fits."""
import json,os,subprocess,time
from pathlib import Path
repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python';root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');state=root/'near_warm_initialization_state.json'
if state.exists():raise FileExistsError('Do not duplicate warm-initialization diagnostic')
def read(p):
 try:return json.loads(p.read_text())
 except(FileNotFoundError,json.JSONDecodeError):return None
def save(d):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(state)
def run(cmd,name,env=None):
 (root/(name+'.command.json')).write_text(json.dumps(list(map(str,cmd)))+'\n')
 with(root/(name+'.log')).open('w') as log:subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
save(dict(stage='waiting_for_both32grasp_model_reviews',hypothesis='Previous direct fit initializes only the landmarkencoder; its orientationMLP is fresh. Test retaining the complete pretrained RGB orientation head and normalization.',maximum_new_fits=1))
paths=[root/(n+'_prediction_review/approval.json') for n in ['expanded_near_spatial_joint','expanded_near_fused_orientation']]
while not all(read(p) for p in paths):time.sleep(20)
if any(read(p)['pose_gate_passed'] for p in paths):
 save(dict(stage='not_needed_32grasp_candidate_qualified',new_fits=0));raise SystemExit(0)
fit=root/'expanded_near_fused_orientation_warm';source=root/'expanded_near_pose_dataset';warm='/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/sc_orientation_jitter_v2/best.pt'
cmd=[py,repo/'scripts/train_SC_near_fused_orientation_to_plateau.py','--source',source,'--pixels',source/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt','--native-warm-start',warm,'--output',fit,'--family','sc','--rotation-only','--balance-new-angles','--crop-jitter','4','--updates','100000','--minutes','1440','--min-updates','5000','--patience','3000']
save(dict(stage='training_one_full_head_warm_start_to_plateau',physical_gpu=4,compared_with='expanded_near_fused_orientation',only_intended_change='Full RGB head/encoder initialization, preserving pretrained physical output normalization',automatic_deployment=False))
run(cmd,'expanded_near_fused_orientation_warm',dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
assert read(fit/'summary.json')['plateau_reached']
run([py,repo/'scripts/review_SC_near_fused_orientation.py','--fit',fit,'--output',root/'expanded_near_fused_orientation_warm_prediction_review'],'expanded_near_fused_orientation_warm_prediction_review')
save(dict(stage='warm_initialization_complete_requires_visual_and_live_decision',automatic_deployment=False))

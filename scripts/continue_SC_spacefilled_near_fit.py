#!/usr/bin/env python3
"""Wait for48 bounded training captures, preserve validation, fit one warm model."""
import json,os,subprocess,time
from pathlib import Path
repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python';root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');bank=root/'spacefilled48_captures';state=root/'spacefilled_near_fit_state.json'
if state.exists():raise FileExistsError('Inspect existing continuation; do not duplicate')
def read(p):
 try:return json.loads(p.read_text())
 except(FileNotFoundError,json.JSONDecodeError):return None
def save(d):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(state)
def run(cmd,name,env=None):
 (root/(name+'.command.json')).write_text(json.dumps(list(map(str,cmd)))+'\n')
 with(root/(name+'.log')).open('w') as log:subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
save(dict(stage='waiting_for_three_16episode_workers_and_actual_QA',new_training_episodes=48,maximum_near_training_grasps=80,validation_unchanged=True))
while not all(read(bank/f'worker{i}/summary.json') for i in range(3)):time.sleep(20)
for i in range(3):
 d=read(bank/f'worker{i}/summary.json');assert d['completed']==16 and all(x['valid'] for x in d['results']);assert read(bank/f'worker{i}/first_six_visual_QA.json')['passed']
source=root/'expanded_near_pose_dataset'
for i in range(3):
 output=root/(f'spacefilled_near_merge{i}' if i<2 else 'spacefilled_near_pose_dataset')
 save(dict(stage='merging_immutable_shards',worker=i,source=str(source),output=str(output)))
 run([py,repo/'scripts/merge_sc_grasp_capture_supervision.py','--source',source,'--captures',bank/f'worker{i}','--output',output],f'spacefilled_near_merge{i}')
 source=output
old=read(root/'expanded_near_pose_dataset/manifest.json');rows=read(source/'manifest.json');assert [x for x in old if x['split']=='validation']==[x for x in rows if x['split']=='validation'];assert len({x['episode'] for x in rows if x['split']=='train' and x.get('source_role','').startswith('Near25mm')})==80
fit=root/'spacefilled_near_fused_orientation_warm';warm='/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/sc_orientation_jitter_v2/best.pt'
cmd=[py,repo/'scripts/train_SC_near_fused_orientation_to_plateau.py','--source',source,'--pixels',source/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt','--native-warm-start',warm,'--output',fit,'--family','sc','--rotation-only','--balance-new-angles','--crop-jitter','4','--updates','100000','--minutes','1440','--min-updates','5000','--patience','3000']
save(dict(stage='training_one80grasp_warm_fit_to_plateau',physical_gpu=4,compared_with='expanded_near_fused_orientation_warm',changed='Systematic training-grasp coverage and count; same representation/init/optimizer/augmentation/selection/budget',automatic_deployment=False))
run(cmd,'spacefilled_near_fused_orientation_warm',dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
assert read(fit/'summary.json')['plateau_reached']
run([py,repo/'scripts/review_SC_near_fused_orientation.py','--fit',fit,'--output',root/'spacefilled_near_fused_orientation_warm_prediction_review'],'spacefilled_near_fused_orientation_warm_prediction_review')
a=read(root/'spacefilled_near_fused_orientation_warm_prediction_review/approval.json');score=a['metrics_by_domain']['near_25mm']['orientation_p50_p95_deg'][1]
save(dict(stage='80grasp_fit_complete_requires_visual_review_and_live_decision',pose_gate_passed=a['pose_gate_passed'],near_orientation_p95_deg=score,material_improvement_threshold_deg=1.587038199302359*.85,material_improvement=score<1.587038199302359*.85,automatic_deployment=False,no_further_collection_queued=True))

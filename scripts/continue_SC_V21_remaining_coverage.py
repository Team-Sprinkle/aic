#!/usr/bin/env python3
"""Three matched SC coverage cases; stop on the first controller-gate failure."""
import json
from pathlib import Path
import shutil
import subprocess
import time

repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
out=root/'V21_remaining_SC_coverage';out.mkdir(exist_ok=False)
state=out/'state.json'
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def save(d):state.write_text(json.dumps(d,indent=2)+'\n')
def run(cmd,log):
    log.with_suffix('.command.json').write_text(json.dumps(list(map(str,cmd)))+'\n')
    with log.open('w') as stream:subprocess.run(list(map(str,cmd)),stdout=stream,stderr=subprocess.STDOUT,check=True)
save(dict(stage='waiting_for_first_sample_physical_pass_and_ablation_completion',maximum_new_episodes=3))
while not read(root/'sample_V21_experimental/analysis_summary.json'):time.sleep(20)
assert read(root/'sample_V21_experimental/analysis_summary.json')['strict_physical_insertions']==3
while not read(root/'V25_no_close_matched_ablation/runs/summary.json'):time.sleep(20)
cases=read(root/'V17_fresh_SC_development/evaluation_cases.json')['cases']
cases=[c for c in cases if c['case']!='fresh_SC_explicit_n0_p0_seed1813']
assert len(cases)==3
results=[]
for case in cases:
    case_root=out/case['case'];case_root.mkdir()
    (case_root/'evaluation_cases.json').write_text(json.dumps(dict(cases=[case]),indent=2)+'\n')
    variants=read(root/'V21_close_grasp_development/evaluation_variants.json')
    variants['variants'][0]['name']='V21_experimental_close_update_local'
    (case_root/'evaluation_variants.json').write_text(json.dumps(variants,indent=2)+'\n')
    bundle=case_root/'pose_bundle'
    run([py,repo/'scripts/prepare_SC_close_composite_bundle.py',
         '--rotation-fit',root/'spacefilled_near_fused_orientation_warm',
         '--rotation-review',root/'final80_orientation_review',
         '--translation-fit',root/'expanded_near_spatial_joint',
         '--translation-review',root/'expanded_near_translation_component_review',
         '--experimental-case',case['case'],
         '--independent-frozen-replay',root/'final80_frozen_rollout_generalization/summary.json',
         '--output',bundle],case_root/'bundle.log')
    shutil.copyfile(bundle/'selection.json',case_root/'close_pose_selection.json')
    save(dict(stage='running_one_matched_SC_case',case=case['case'],results=results,maximum_new_episodes=3))
    run([py,repo/'scripts/run_visual_servo_close_grasp_probe.py','--root',case_root,
         '--output',case_root/'runs','--gpu','3','--domain','130','--experimental-diagnostic'],case_root/'rollout.log')
    row=read(case_root/'runs/summary.json')['results'][0]
    gate=case_root/'explicit_trial_gate.json'
    run([py,repo/'scripts/qualify_SC_explicit_close_trial.py','--audit',row['audit'],'--output',gate],case_root/'gate.log')
    result=dict(case=case['case'],gate=read(gate),physical=row['raw_terminal'],audit=row['audit'])
    results.append(result)
    if not result['gate']['explicit_trial_pass']:
        save(dict(stage='first_failed_gate_requires_targeted_diagnosis',results=results,
                  no_more_rollouts_queued=True,final_scenes_sealed=True));raise SystemExit(0)
save(dict(stage='three_remaining_matched_cases_complete',results=results,
          first_n0p0_reused_from_V21_pilot=True,model_still_experimental=True,final_scenes_sealed=True))

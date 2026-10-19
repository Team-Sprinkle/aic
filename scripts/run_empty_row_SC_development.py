#!/usr/bin/env python3
"""Four frozen SC route preflights; do not launch more after a failed gate."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess

repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1');old=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
out=root/'empty_row_SC_development';out.mkdir(exist_ok=False);state=out/'state.json'
results=[]
def save(data):state.write_text(json.dumps(data,indent=2)+'\n')
def run(command,log):
    log.with_suffix('.command.json').write_text(json.dumps(list(map(str,command)),indent=2)+'\n')
    with log.open('w') as stream:subprocess.run(list(map(str,command)),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,check=True)
def one(case,gpu,domain):
    path=out/case['case'];path.mkdir()
    (path/'evaluation_cases.json').write_text(json.dumps(dict(cases=[case]),indent=2)+'\n')
    original=old/'V34_SC_matched_coverage'/case['case']
    variants=json.loads((original/'evaluation_variants.json').read_text());variants['variants'][0]['name']='SC_empty_row_route_speed_preflight'
    (path/'evaluation_variants.json').write_text(json.dumps(variants,indent=2)+'\n');shutil.copyfile(original/'close_pose_selection.json',path/'close_pose_selection.json')
    run([py,repo/'scripts/run_visual_servo_close_grasp_probe.py','--root',path,'--output',path/'runs','--gpu',gpu,'--domain',domain,'--policy','RunVisualConnectorServoSCEmptyDiagnosticV38','--empty-row-selection',root/'empty_row_bundle/selection.json','--experimental-diagnostic'],path/'rollout.log')
    row=json.loads((path/'runs/summary.json').read_text())['results'][0];gate=path/'explicit_trial_gate.json'
    run([py,repo/'scripts/qualify_SC_explicit_close_trial.py','--audit',row['audit'],'--output',gate],path/'gate.log')
    trace=[json.loads(s) for s in (Path(row['audit'])/'runtime_trace.jsonl').open()];used=any(r['event']=='empty_row_shortcut_selected' for r in trace)
    result=dict(case=case['case'],gate=json.loads(gate.read_text()),physical=row['raw_terminal'],shortcut_used=used,expected_empty=case['card_count']==0,audit=row['audit'])
    result['route_gate_passed']=result['gate']['explicit_trial_pass'] and used==(case['card_count']==0)
    return result
cases=json.loads((old/'V17_fresh_SC_development/evaluation_cases.json').read_text())['cases'];save(dict(stage='running_two_isolated_SC_route_preflights'))
failed=False
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    jobs={pool.submit(one,cases[i],str(3+i),str(141+i)):(3+i,141+i) for i in range(2)};next_case=2
    while jobs:
        done,_=concurrent.futures.wait(jobs,return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            gpu,domain=jobs.pop(future)
            try:r=future.result()
            except Exception as exc:r=dict(error=repr(exc),route_gate_passed=False)
            results.append(r);failed|=not r['route_gate_passed']
            save(dict(stage='failed_waiting_for_active_jobs' if failed else 'running_frozen_SC_preflights',results=results))
            if not failed and next_case<len(cases):jobs[pool.submit(one,cases[next_case],str(gpu),str(domain))]=(gpu,domain);next_case+=1
save(dict(stage='route_preflight_failed_requires_diagnosis' if failed else 'four_SC_route_preflights_passed',results=results,final_scenes_sealed=True))

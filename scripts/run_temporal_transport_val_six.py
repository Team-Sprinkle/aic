#!/usr/bin/env python3
"""Sequential rootless observation-only SC transport checks on held-out episodes."""
import json
import os
from pathlib import Path
import subprocess
import sys


CASES = ('llb_0005','llb_0090','llb_0112','llb_0117','llb_0074','llb_0109')
REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json'
ROOT = REPO/'outputs/experiments/2026-10-01_transport_reliability/val_six_no_previous'
MODEL = REPO/'outputs/experiments/2026-10-01_transport_reliability/history_no_previous_6000/best.pt'
BASE = REPO/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    by_case = {e['case']: e for e in json.loads(SOURCE.read_text())['episodes']}
    rows = []
    for case in CASES:
        episode = by_case[case]
        if episode['split'] != 'validation': raise ValueError(f'Not validation: {case}')
        output = ROOT/case
        env = os.environ.copy()
        env.update(AIC_TRANSPORT_EVAL_GPU='1', AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_SCENE_DIR=episode['source_episode'],
            AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(MODEL),
            AIC_TEMPORAL_TRANSPORT_IMAGENET='/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth',
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport')
        command = ['bash','scripts/run_clean_transport_eval.sh',case,str(BASE),str(output)]
        print('RUN', case, flush=True)
        if not output.exists():
            with (ROOT/f'{case}.log').open('w') as log:
                result = subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'{case}: launcher failed, see log')
        plan = Path(episode['insert_plan'])/'moveit_joint_trajectory.json'
        if not (output/'autonomous_summary.json').exists():
            subprocess.run([str(REPO/'.pixi/envs/default/bin/python'),
                'scripts/analyze_clean_transport_rollout.py',str(output),'--plan',str(plan)],
                cwd=REPO,check=True,stdout=(ROOT/f'{case}-analysis.json').open('w'))
        subprocess.run([str(REPO/'.pixi/envs/default/bin/python'),
            'scripts/measure_transport_teacher_path.py',str(output),'--plan',str(plan)],
            cwd=REPO,check=True,stdout=(ROOT/f'{case}-path.json').open('w'))
        summary = json.loads((output/'autonomous_summary.json').read_text())
        path = json.loads((output/'teacher_path_diagnostic.json').read_text())
        row = {'case':case,'card_count':episode['card_count'],'selected_port':episode['selected_port'],
               'scene_reset_group':episode['scene_reset_group'],
               'final_standoff_mm':summary['final_planned_standoff_error_mm'],
               'first_10mm_3deg_gate_time_s':path['first_10mm_3deg_gate_time_s'],
               'pre_gate_path_p95_mm':path['route_deviation_before_first_gate_mm']['p95'],
               'terminal_drift_after_gate_mm':path['terminal_drift_after_gate_mm'],
               'score':summary['official_total'],'inference_p95_ms':summary['inference_ms']['p95'],
               'artifact':str(output)}
        rows.append(row)
        (ROOT/'summary.json').write_text(json.dumps({'schema':'aic_temporal_transport_val_six/v1',
            'checkpoint':str(MODEL),'cases_requested':list(CASES),
            'cases_completed':len(rows),'results':rows,'final_scenes_opened':False},indent=2)+'\n')
        print(json.dumps(row),flush=True)
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__': main()

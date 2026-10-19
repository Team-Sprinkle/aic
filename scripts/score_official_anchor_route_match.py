#!/usr/bin/env python3
"""Extract and score frozen public-anchor paired rollouts with preset gate."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

from score_official_anchor_route_dev import score


ROOT=Path(__file__).resolve().parents[1]


def extract(audit):
    destination=audit/'scored_stream.json'
    if destination.exists():return
    env=os.environ.copy();env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
    env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
    subprocess.run([str(ROOT/'.pixi/envs/default/bin/python'),
                    str(ROOT/'scripts/extract_moveit_training_stream.py'),
                    str(audit),str(destination)],cwd=ROOT,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing score')
    frozen=json.loads((a.comparison/'plan.json').read_text())
    progress=json.loads((a.comparison/'progress.json').read_text())
    expected={arm+'_'+s['name'] for arm in ('control','candidate') for s in frozen['scenes']}
    if expected!=progress.keys():raise ValueError('Matched six-pair comparison incomplete')
    calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    rows=[]
    for name in sorted(expected):
        item=progress[name]
        if item['returncode']!=0:raise ValueError(f'Rollout error {name}')
        audit=Path(item['audit']);extract(audit)
        row=score(audit,calibration);row['arm']=item['arm'];row['scene']=item['scene']
        rows.append(row)
    by=defaultdict(list)
    for row in rows:by[(row['arm'],row['family'])].append(row)
    summary={}
    for (arm,family),group in sorted(by.items()):
        summary[arm+'_'+family]=dict(episodes=len(group),
            safe_buffers=sum(r['safe_buffer_visit'] for r in group),
            scored_contacts=sum(r['official_contact_message']!='No contact detected.' for r in group),
            latency_p95_under_300_all=all(r['inference_p50_p95_p99_ms'] and
                r['inference_p50_p95_p99_ms'][1]<300 for r in group))
    passed=all(summary['candidate_'+fam]['safe_buffers']>=2 and
               summary['candidate_'+fam]['scored_contacts']<=summary['control_'+fam]['scored_contacts'] and
               summary['candidate_'+fam]['latency_p95_under_300_all']
               for fam in ('sfp','sc'))
    result=dict(schema='aic_official_anchor_route_match_score/v1',runs=rows,
                summary=summary,preset_gate_passed=passed,
                automatic_promotion=False,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,preset_gate_passed=passed),indent=2),flush=True)


if __name__=='__main__':main()

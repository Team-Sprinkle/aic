#!/usr/bin/env python3
"""Advance sample development to frozen evaluation; never tune on evaluation."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--development',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path.cwd();py=sys.executable
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    pilot=a.pilot/'summary.json'
    while not pilot.exists():
        save(a.output/'state.json',dict(stage='waiting_for_three_public_sample_pilots',training=False));time.sleep(45)
    x=json.loads(pilot.read_text())
    if len(x['results'])!=3 or not all(r['qualified'] for r in x['results']):
        save(a.output/'decision.json',dict(status='sample_pilot_needs_targeted_repair',evidence=str(pilot),evaluation_started=False));return
    devrun=a.development/'rule_runs_v1'
    save(a.output/'state.json',dict(stage='29_sample_and_development_rollouts',source=str(a.development)))
    cmd=[py,str(repo/'scripts/run_sample_rule_development.py'),'--root',str(a.development),'--output',str(devrun),'--gpu','5']
    with (a.output/'development.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.loads((devrun/'summary.json').read_text());rows=data['results']
    sample=[r for r in rows if r['case'].startswith('sample_')]
    sc=[r for r in rows if r['family']=='sc' and r not in sample]
    sfp=[r for r in rows if r['family']=='sfp' and r not in sample]
    def correct(r):return r.get('wrong_accepted_port') is False if r.get('rule_visual_acquisition') else r.get('wrong_accepted_port') is not True
    # Prospective 90% screen plus coverage controls; no missing-acceptance
    # None flag is incorrectly interpreted as an actual wrong-port accept.
    passed=(len(rows)==29 and all(r['qualified'] for r in sample) and sum(r['qualified'] for r in sc)==6
        and sum(r['qualified'] for r in sfp)>=18 and all(correct(r) and r['no_scored_harm'] for r in rows)
        and all(sum(r['qualified'] for r in sfp if r['card_count']==n and r['port']==p)>=4 for n in [1,5] for p in range(2))
        and all(sum(r['qualified'] for r in sfp if r['requested_card']==card)>=3 for card in range(5)))
    save(a.output/'development_gate.json',dict(passed=passed,sample=sum(r['qualified'] for r in sample),
        sc=sum(r['qualified'] for r in sc),sfp=sum(r['qualified'] for r in sfp),evidence=str(devrun/'summary.json')))
    if not passed:
        save(a.output/'decision.json',dict(status='development_needs_targeted_repair',evaluation_started=False,evidence=str(devrun/'summary.json')));return
    # Freeze before any evaluation result; existing perception lineage caveat
    # remains explicit, and no model training is launched by this coordinator.
    freeze=a.output/'frozen';freeze.mkdir()
    sources=[repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',
        repo/'scripts/prepare_sample_conservative_transport.py',repo/'scripts/run_sample_rule_development.py',
        a.development/'transport_spec.json']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    for f in sources:(freeze/f.name).write_bytes(f.read_bytes())
    save(freeze/'manifest.json',dict(hashes=hashes,development_gate=str(a.output/'development_gate.json'),tuning_after_evaluation=False))
    final=a.output/'evaluation_scene_bank'
    subprocess.run([py,str(repo/'scripts/prepare_sample_conservative_transport.py'),'--phase','evaluation','--output',str(final)],check=True)
    plan=json.loads((final/'plan.json').read_text())
    if (final/'transport_spec.json').read_bytes()!=(a.development/'transport_spec.json').read_bytes():
        raise ValueError('Evaluation generator changed the frozen rule specification')
    # Six exposed official-scene evaluations are appended only after the
    # development gate and freeze. They never participate in model selection.
    exposed=Path('/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1/plan.json')
    if exposed.exists():
        e=json.loads(exposed.read_text())
        for case in e['cases']:
            if case['domain']=='exact_public' and case['arm']=='candidate' and case['replicate']<2:
                c=copy.deepcopy(case);c['case']='exposed_'+c['case'];c['arm']='frozen_sample_rule';c['split']='exposed_evaluation_only'
                plan['cases'].append(c)
    plan['planned']=len(plan['cases']);plan['rule_spec']=str(freeze/'transport_spec.json')
    plan['frozen_rule_manifest']=str(freeze/'manifest.json')
    plan['no_tuning_after_results']=True;plan['reserved_final_scenes_opened']=False
    save(final/'plan.json',plan)
    save(a.output/'state.json',dict(stage='frozen_evaluation',planned=plan['planned'],sample_derived=124,exposed_postrelease=plan['planned']-124))
    result=a.output/'evaluation_runs'
    with (a.output/'evaluation.log').open('w') as log:
        subprocess.run([py,str(repo/'scripts/run_sample_rule_development.py'),'--root',str(final),'--output',str(result),'--gpu','5'],stdout=log,stderr=subprocess.STDOUT,check=True)
    d=json.loads((result/'summary.json').read_text());groups=collections.defaultdict(list)
    for r in d['results']:
        groups[(r['domain'],r['family'],r['card_count'],r['port'],r.get('requested_card'))].append(r)
    cells=[dict(domain=k[0],family=k[1],cards=k[2],port=k[3],requested_card=k[4],episodes=len(v),
        qualified=sum(r['qualified'] for r in v),acquired=sum(bool(r['rule_visual_acquisition']) for r in v),
        wrong=sum(r['wrong_accepted_port'] is True for r in v),harm=sum(not r['no_scored_harm'] for r in v)) for k,v in groups.items()]
    save(a.output/'evaluation_cells.json',dict(cells=cells,expected_repeats_per_cell=2,complete=len(d['results'])==plan['planned']))
    save(a.output/'decision.json',dict(status='frozen_evaluation_complete_requires_review',completed=len(d['results']),
        qualified=sum(r['qualified'] for r in d['results']),cells=str(a.output/'evaluation_cells.json'),
        full_insertion_evaluated=False,training_started=False,no_postrelease_tuning=True,reserved_final_scenes_opened=False))


if __name__=='__main__':main()

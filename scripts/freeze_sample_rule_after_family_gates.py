#!/usr/bin/env python3
"""Freeze a SC-only rule revision after explicit SC and unchanged-SFP gates."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','sc-screen','sfp-screen','spec','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();repo=Path.cwd();py=sys.executable
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    while not all((f/'summary.json').exists() for f in [a.sc_screen,a.sfp_screen]):
        save(a.output/'state.json',dict(stage='waiting_for_complete_SC_revision_and_unchanged_SFP_screen',training=False,evaluation_started=False))
        time.sleep(45)
    sc=json.loads((a.sc_screen/'summary.json').read_text())['results']
    original=json.loads((a.sfp_screen/'summary.json').read_text())['results']
    sfp=[r for r in original if r['family']=='sfp'];ss=[r for r in sfp if r['case'].startswith('sample_')];sd=[r for r in sfp if r not in ss]
    spec=json.loads(a.spec.read_text());before=json.loads((a.root/'transport_spec.json').read_text())
    assert spec['families']['sfp']==before['families']['sfp'] and spec['control']==before['control'] and spec['bundle']==before['bundle']
    passed=(len(sc)==7 and all(r['qualified'] for r in sc) and len(ss)==2 and all(r['qualified'] for r in ss)
        and len(sd)==20 and sum(r['qualified'] for r in sd)>=18
        and all(sum(r['qualified'] for r in sd if r['card_count']==n and r['port']==port)>=4 for n in [1,5] for port in range(2))
        and all(sum(r['qualified'] for r in sd if r['requested_card']==card)>=3 for card in range(5))
        and all(r['no_scored_harm'] and (not r['rule_visual_acquisition'] or r['wrong_accepted_port'] is False) for r in sc+sfp))
    save(a.output/'development_gate.json',dict(passed=passed,SC_qualified=sum(r['qualified'] for r in sc),SC_episodes=len(sc),
        SFP_qualified=sum(r['qualified'] for r in sfp),SFP_episodes=len(sfp),
        SC_evidence=str(a.sc_screen/'summary.json'),SFP_reused_unchanged_family_evidence=str(a.sfp_screen/'summary.json'),
        SFP_control_and_family_spec_identical=True,new_training=False,final_scenes_opened=False))
    if not passed:
        save(a.output/'decision.json',dict(status='family_development_gate_failed',evaluation_started=False));return
    frozen=a.output/'frozen';frozen.mkdir();models=frozen/'models';models.mkdir()
    for key,value in spec['bundle'].items():
        src=Path(value);dest=models/(key+src.suffix);shutil.copyfile(src,dest)
        assert digest(src)==digest(dest);spec['bundle'][key]=str(dest)
    save(frozen/'transport_spec.json',spec)
    sources=[repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',
        repo/'aic_example_policies/aic_example_policies/ros/RunSharedPortTCP.py',
        *[repo/'scripts'/n for n in ['prepare_sample_conservative_transport.py','run_sample_rule_development.py','run_sample_rule_variant.py',
            'requested_native_port_pose.py','native_requested_opening_refiner.py','native_keypoint_pose_geometry.py','sc_requested_history_observation.py']]]
    hashes={str(f):digest(f) for f in sources}
    for f in sources:shutil.copyfile(f,frozen/f.name)
    save(frozen/'manifest.json',dict(source_hashes=hashes,model_hashes={str(f):digest(f) for f in models.iterdir()},
        rule_spec_hash=digest(frozen/'transport_spec.json'),gate=str(a.output/'development_gate.json'),
        selected_without_postrelease_feedback=True,perception_training_lineage_not_fully_audited=True))
    bank=a.output/'evaluation_scene_bank'
    subprocess.run([py,str(repo/'scripts/prepare_sample_conservative_transport.py'),'--phase','evaluation','--output',str(bank)],check=True)
    plan=json.loads((bank/'plan.json').read_text());plan['rule_spec']=str(frozen/'transport_spec.json');plan['bundle']=spec['bundle']
    # Read post-release scene definitions only after the development freeze.
    e=Path('/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1/plan.json')
    if e.exists():
        for case in json.loads(e.read_text())['cases']:
            if case['domain']=='exact_public' and case['arm']=='candidate' and case['replicate']<2:
                c=copy.deepcopy(case);c['case']='exposed_'+c['case'];c['arm']='frozen_sample_rule';c['split']='exposed_evaluation_only';plan['cases'].append(c)
    plan['planned']=len(plan['cases']);plan['no_tuning_after_results']=True;plan['frozen_manifest']=str(frozen/'manifest.json');save(bank/'plan.json',plan)
    save(a.output/'state.json',dict(stage='frozen_transport_evaluation',planned=plan['planned'],sample_derived=124,
        exposed_official=plan['planned']-124,selected_policy='RunSampleConservativeTransport',SC_side='opposite_public_row_corridor',training=False))
    output=a.output/'evaluation_runs'
    with (a.output/'evaluation.log').open('w') as log:
        subprocess.run([py,str(repo/'scripts/run_sample_rule_variant.py'),'--root',str(bank),'--output',str(output),
            '--spec',str(frozen/'transport_spec.json'),'--gpu','5','--domain','93'],stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((output/'summary.json').read_text());groups=collections.defaultdict(list)
    for r in report['results']:groups[(r['domain'],r['family'],r['card_count'],r['port'],r.get('requested_card'))].append(r)
    cells=[dict(domain=k[0],family=k[1],cards=k[2],port=k[3],requested_card=k[4],episodes=len(v),qualified=sum(r['qualified'] for r in v),
        acquired=sum(bool(r['rule_visual_acquisition']) for r in v),wrong=sum(r['wrong_accepted_port'] is True for r in v),harm=sum(not r['no_scored_harm'] for r in v)) for k,v in groups.items()]
    save(a.output/'evaluation_cells.json',dict(cells=cells,all_cells_two_episodes=all(c['episodes']==2 for c in cells)))
    save(a.output/'decision.json',dict(status='frozen_evaluation_complete_requires_review',completed=len(report['results']),
        qualified=sum(r['qualified'] for r in report['results']),new_training=False,insertion_evaluated=False,
        no_postrelease_tuning=True,reserved_final_scenes_opened=False))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Extract and admit causally joined public qualification-family transport BC."""
import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import subprocess

import yaml

from materialize_sfp_scored_transport import materialize as sfp_materialize
from materialize_sc_scored_route import materialize as sc_materialize


ROOT=Path(__file__).resolve().parents[1]
SFP_CALIBRATION=json.loads((ROOT/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())


def extract(audit):
    path=audit/'scored_stream.json'
    if path.exists():return path
    env=os.environ.copy()
    env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
    env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
    subprocess.run([str(ROOT/'.pixi/envs/default/bin/python'),
                    str(ROOT/'scripts/extract_moveit_training_stream.py'),str(audit),str(path)],
                   cwd=ROOT,env=env,check=True)
    return path


def safe_score(audit,family):
    score=yaml.safe_load((audit/'results/scoring.yaml').read_text())
    trial=next(v for k,v in score.items() if k.startswith('trial_'))
    contact=trial.get('tier_2',{}).get('categories',{}).get('contacts',{})
    tier3=float(trial['tier_3']['score'])
    return tier3>=0 and contact.get('message')=='No contact detected.'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode-index-start',type=int,required=True,
                   help='Reserved collision-free episode ID range for this complete collection')
    p.add_argument('--names-file',type=Path,
                   help='Optional newline-separated scene names for one immutable staged subset')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve immutable labels')
    a.output.mkdir(parents=True)
    plan=json.loads((a.collection/'plan.json').read_text())
    progress=json.loads((a.collection/'progress.json').read_text())
    selected=(set(a.names_file.read_text().splitlines()) if a.names_file else None)
    if selected is not None and (not selected or selected-{s['name'] for s in plan['scenes']}):
        p.error('names-file is empty or contains names outside the frozen plan')
    admitted=[];rejected=[];scenes={'train':set(),'validation':set()}
    with (a.output/'sfp_transport_chunks.jsonl').open('w') as sfp_out,\
         (a.output/'sc_transport_chunks.jsonl').open('w') as sc_out:
        for i,spec in enumerate(plan['scenes']):
            if selected is not None and spec['name'] not in selected:continue
            item=progress.get(spec['name'])
            if item is None:continue
            audit=Path(item['audit'])
            if spec.get('start_variant'):
                start_audit=audit/'start_variant_audit.json'
                if not start_audit.exists() or not json.loads(start_audit.read_text()).get('valid'):
                    rejected.append(dict(name=spec['name'],reason='unverified_actual_first_trial_start'))
                    continue
            if item['returncode']!=0 or not safe_score(audit,spec['family']):
                rejected.append(dict(name=spec['name'],reason='score_or_contact',score=item['official_tier3']))
                continue
            try:
                extract(audit)
                buffer=io.StringIO()
                if spec['family']=='sfp':
                    result=sfp_materialize(audit,SFP_CALIBRATION,a.episode_index_start+i,buffer,
                                           spec['split'],allow_transport_only=True)
                    target=sfp_out
                else:
                    result=sc_materialize(audit,a.episode_index_start+i,spec['split'],buffer)
                    target=sc_out
                reference_p95=(result['reference_target_error_mm']['p95'] if spec['family']=='sfp'
                               else result.get('reference_target_p95_mm'))
                if reference_p95 is None or reference_p95>5:
                    raise ValueError('reference target p95 >5mm')
                for line in buffer.getvalue().splitlines():
                    row=json.loads(line)
                    row['official_anchor_teacher']=True
                    row['official_anchor_name']=spec['official_anchor']
                    row['card_count']=spec['card_count']
                    target.write(json.dumps(row,separators=(',',':'))+'\n')
                scenes[spec['split']].add(result['scene_group'])
                admitted.append({**result,'name':spec['name'],
                                 'family':spec['family'],'split':spec['split'],
                                 'official_tier3':item['official_tier3'],
                                 'full_insert_teacher':item['official_tier3']==75,
                                 'transport_only_prefix':item['official_tier3']!=75})
                print(json.dumps(dict(name=spec['name'],accepted=True,chunks=result['transport_chunks'])),flush=True)
            except (ValueError,KeyError,FileNotFoundError,subprocess.CalledProcessError) as exc:
                rejected.append(dict(name=spec['name'],reason=str(exc)))
                print(json.dumps(dict(name=spec['name'],accepted=False,reason=str(exc))),flush=True)
    if scenes['train'] & scenes['validation']:
        raise ValueError('Train/development reset leak')
    manifest=dict(schema='aic_official_anchor_route_labels/v1',source=str(a.collection),
                  episode_index_start=a.episode_index_start,
                  names_file=str(a.names_file) if a.names_file else None,
                  admitted=admitted,rejected=rejected,
                  counts={str(k):v for k,v in Counter((r['family'],r['split']) for r in admitted).items()},
                  train_development_reset_overlap=0,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(admitted=len(admitted),rejected=len(rejected),counts=manifest['counts'])),flush=True)


if __name__=='__main__':main()

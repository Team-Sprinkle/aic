#!/usr/bin/env python3
"""Keep perception provenance separate from qualified successful-BC sources.

Original data are untouched. This is a source admission index, not materialized
action chunks. Causal predicted frames, ownership intervals and stall cleaning
still need to be applied by the later BC builder.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import yaml


def source(row):
    return str(Path(row.get('source_scored_stream') or row.get('source_stream') or str(Path(row['physical_reset_config']).parent/'scored_stream.json')).resolve())


def task_vector(task):
    sfp=task['plug_name'].startswith('sfp');port=int((task['port_name'] if sfp else task['target_module_name']).rsplit('_',1)[1])
    values=[float(sfp),float(not sfp),float(port==0),float(port==1)]+[0.]*6
    if sfp:values[4+int(task['target_module_name'].rsplit('_',1)[1])]=1.;values[9]=1.
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--SC-depth-audit',type=Path,required=True);p.add_argument('--SFP-depth-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    grouped=defaultdict(list)
    for l in a.labels.open():
        r=json.loads(l);grouped[source(r)].append(r)
    sc={r['episode']:r for r in json.loads(a.SC_depth_audit.read_text())['results']}
    sfp={str(Path(r['source_stream']).resolve()):r for r in map(json.loads,a.SFP_depth_audit.open())}
    admitted=[];excluded=[];counts=Counter();coverage=defaultdict(set);true_episode_counts=Counter()
    for path,rows in grouped.items():
        r=rows[0];family=r['family'];cfg=Path(r['physical_reset_config']);audit=cfg.parent
        splits={r['split'] for r in rows};assert len(splits)==1
        true_episode_counts[f'{family}/{r["split"]}']+=1
        trial=next(iter(yaml.safe_load(cfg.read_text())['trials'].values()));task=next(iter(trial['tasks'].values()))
        record=dict(family=family,split=r['split'],source_stream=path,source_audit=str(audit),
            source_episode_id=audit.name,image_stream_ids=sorted({v['episode'] for v in rows}),
            physical_reset_groups=sorted({v['scene_group'] for v in rows}),card_count=r['card_count'],
            actual_public_task=task,task_vector=task_vector(task))
        policy_log=audit/'policy.log';text=''
        if policy_log.exists():
            with policy_log.open(errors='replace') as f:text=f.read(8192)
        match=re.search(r'Using policy: ([A-Za-z0-9_]+)',text);policy=match[1] if match else None;record['source_policy']=policy
        geometry=sc.get(r['original_episode']) if family=='sc' else sfp.get(path)
        reasons=[]
        if geometry is None:reasons.append('missing independent physical-depth audit')
        elif family=='sfp' and not geometry.get('valid'):reasons.append('unavailable raw terminal geometry')
        elif not geometry.get('physical_depth_lateral_pass',geometry.get('configured_opening_depth_lateral_pass',False)):
            reasons.append('physical depth/lateral conjunction failed')
        if policy not in (('OfficialTeacherReplay',) if family=='sc' else ('CheatCode',)):
            reasons.append('not the approved route teacher or teacher ownership unverified')
        score_path=audit/'results/scoring.yaml';official=None
        if score_path.exists():
            scores=yaml.safe_load(score_path.read_text());official=next((v for k,v in scores.items() if k.startswith('trial_')),None)
        if official is None:reasons.append('missing official score/contact/force record')
        else:
            if official['tier_3']['score']!=75:reasons.append('no official successful insertion event')
            cats=official['tier_2']['categories']
            if any(cats[k]['score']<0 for k in ('contacts','insertion force')):reasons.append('official prohibited-contact or force penalty')
        record['physical_audit']=geometry;record['reasons']=reasons
        if reasons:excluded.append(record);counts['excluded/'+family]+=1;continue
        record.update(admitted_source=True,action_chunks_materialized=False,
            raw_and_original_data_unchanged=True,remaining_requirements=['teacher/supervisor interval audit','image-time state join','recorded target-frame roundtrip','stall-cleaned derivative','RGB-predicted port frame labels'])
        admitted.append(record);counts['admitted/'+family+'/'+r['split']]+=1
        identity=(f"sfp_card_{task_vector(task)[4:9].index(1.)}_port_{task_vector(task)[2:4].index(1.)}" if family=='sfp' else f"sc_port_{task_vector(task)[2:4].index(1.)}")
        coverage[f'{family}/{r["split"]}/cards{r["card_count"]}/{identity}'].update(record['physical_reset_groups'])
    train={g for r in admitted if r['split']=='train' for g in r['physical_reset_groups']};val={g for r in admitted if r['split']=='validation' for g in r['physical_reset_groups']}
    assert train.isdisjoint(val)
    for filename,rows_ in [('admitted_sources.jsonl',admitted),('excluded_sources.jsonl',excluded)]:
        with (a.output/filename).open('w') as f:
            for r in rows_:f.write(json.dumps(r)+'\n')
    summary=dict(schema='verified_insertion_source_index/v1',unique_source_episodes=len(grouped),source_episodes_by_split=dict(true_episode_counts),
        named_image_stream_ids=len({r['episode'] for rr in grouped.values() for r in rr}),counts=dict(counts),
        admitted_sources=len(admitted),admitted_physical_resets=len(train|val),train_reset_count=len(train),validation_reset_count=len(val),
        coverage_distinct_resets={k:len(v) for k,v in sorted(coverage.items())},
        image_stream_identifier_is_not_independent_episode=True,perception_can_use_excluded_success_sources_for_correct_geometry_labels=True,
        materialized_BC_dataset_ready=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Summarize frozen 4/4, 8/4, and 12/6 development comparisons."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def live(root,analysis):
    result=json.loads(analysis.read_text())
    scenes=[]
    for row in result['rows']:
        audit=Path(row['actor'])
        manifest=json.loads((audit/'eval_manifest.json').read_text())
        if not manifest['execute_complete_chunk']:
            raise ValueError(f'Complete-chunk schedule not enabled: {audit}')
        counts=Counter();members=Counter();chunks=[];chunk=[]
        for line in (audit/'runtime_trace.jsonl').open():
            event=json.loads(line)
            counts[event['event']]+=1
            if event['event']=='command':
                j=int(event['chunk_index']);members[j]+=1
                if j==0 and chunk:chunks.append(chunk);chunk=[]
                chunk.append(float(event['sim_time']))
        if chunk:chunks.append(chunk)
        gaps=[b-a for c in chunks for a,b in zip(c,c[1:])]
        scored=row['actor_score']
        scenes.append(dict(name=row['name'],family=row['family'],
            safe_buffer=bool(scored['safe_buffer_visit']),
            closest_lateral_in_axial_band_mm=scored['closest_lateral_in_axial_band_mm'],
            official_tier3=float(scored['official_tier3']),
            official_contact=scored['official_contact_message'],
            proposal_count=counts['proposal'],command_count=counts['command'],
            stale_command_drops=counts['stale_chunk_member_dropped'],
            command_members=dict(members),
            complete_chunks=sum(len(c)==len(members) for c in chunks),
            within_chunk_gap_p50_s=float(np.quantile(gaps,.5)) if gaps else None,
            within_chunk_gap_p95_s=float(np.quantile(gaps,.95)) if gaps else None,
            complete_inference_p95_ms=float(scored['inference_p50_p95_p99_ms'][1]),
            audit=str(audit),eval_manifest_sha256=sha(audit/'eval_manifest.json')))
    if len(scenes)!=6:raise ValueError(f'Expected six scenes in {root}, found {len(scenes)}')
    return dict(root=str(root),analysis=str(analysis),analysis_sha256=sha(analysis),
        scene_count=len(scenes),safe_buffers=sum(s['safe_buffer'] for s in scenes),
        full_insertions=sum(s['official_tier3']>=75 for s in scenes),
        scored_contacts=sum(s['official_contact']!='No contact detected.' for s in scenes),
        scenes=scenes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/nas/chmin/aic_archives'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing summary')
    root=a.root
    run_names={
        '4/4':'2026-10-13_route_horizon4_execute4_completeclock_v1',
        '8/4':'2026-10-13_route_horizon8_execute4_completeclock_v1',
        '12/6':'2026-10-13_route_horizon12_execute6_completeclock_v1'}
    result=dict(schema='aic_route_horizon_comparison/v1',date='2026-10-13',
                final_scenes_opened=False,actor_privileged_inputs=False,
                recorded_slot_s=.05,corrected_execution_clock=True,
                frozen_development_scene_count=6,live={})
    for name,folder in run_names.items():
        result['live'][name]=live(root/folder,root/(folder.replace('_v1','_analysis.json')))
    for name,file in (('8/4','2026-10-13_route_horizon8_execute4_matched_offline.json'),
                      ('12/6','2026-10-13_route_horizon12_execute6_v4_matched_offline.json')):
        path=root/file
        result.setdefault('offline',{})[name]=dict(path=str(path),sha256=sha(path),
            report=json.loads(path.read_text()))
    phases={
        '8/4':['2026-10-13_route_horizon8_execute4_v1','2026-10-13_route_horizon8_execute4_final_v2'],
        '12/6':['2026-10-13_route_horizon12_execute6_cadence_v2',
                 '2026-10-13_route_horizon12_execute6_final_v3',
                 '2026-10-13_route_horizon12_execute6_plateau_v4']}
    result['training']={name:[dict(path=str(root/folder),
        manifest=json.loads((root/folder/'manifest.json').read_text()),
        summary=json.loads((root/folder/'summary.json').read_text()),
        best_sha256=sha(root/folder/'best.pt')) for folder in folders]
        for name,folders in phases.items()}
    result['extended_label_manifests']={
        '200_ms':json.loads((root/'2026-10-13_route_extended_chunks_v1/manifest.json').read_text()),
        '300_ms':json.loads((root/'2026-10-13_route_extended_chunks_stride6_v1/manifest.json').read_text())}
    bounded=root/'2026-10-13_route_horizon12_execute6_sample4_analysis.json'
    if bounded.exists():
        result['bounded_four_denoising_step_check']=dict(
            analysis=str(bounded),analysis_sha256=sha(bounded),
            scenes=[dict(name=row['name'],family=row['family'],
                         safe_buffer=row['actor_score']['safe_buffer_visit'],
                         closest_lateral_in_axial_band_mm=row['actor_score']['closest_lateral_in_axial_band_mm'],
                         full_insertion=row['actor_score']['official_tier3']>=75,
                         contact=row['actor_score']['official_contact_message'])
                    for row in json.loads(bounded.read_text())['rows']])
    result['decision']='No model promoted: all three variants had 0/6 safe buffers and 0/6 full insertions; 12/6 reduced SFP lateral miss in 3/3 matched scenes but SC did not reach a usable approach.'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{key:item[key] for key in ('safe_buffers','full_insertions','scored_contacts')}
                      for name,item in result['live'].items()},indent=2))


if __name__=='__main__':main()

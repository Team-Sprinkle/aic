#!/usr/bin/env python3
"""Summarize the frozen six-scene SC timing/fit comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency(audit):
    rows=(json.loads(x) for x in (audit/'runtime_trace.jsonl').open())
    vals=[float(r['inference_ms']) for r in rows
          if r.get('event')=='proposal' and r.get('inference_ms') is not None]
    return [float(x) for x in np.quantile(vals,[.5,.95,.99])] if vals else None


def baseline(path):
    data=json.loads(path.read_text());rows={}
    for x in data['runs']:
        if x['cohort'] not in ('repeat','new'):continue
        rows[x['scene']]=dict(scene=x['scene'],best_tcp_port_3d_mm=x['best_tcp_to_port_3d_mm'],
                            axial_band_samples=x['axial_band_samples'],safe_buffer=x['safe_buffer'],
                            full_insertion=x['full_insertion'],scored_contact=x['scored_contact'],
                            departure_s=x['departure_time_s'],inference_p50_p95_p99_ms=latency(Path(x['audit'])),
                            audit=x['audit'])
    return rows


def newer(path):
    data=json.loads(path.read_text());rows={}
    for x in data['rows']:
        positions=x['matched_trace'];best=min(positions,key=lambda r:math.hypot(r['actor_axial_mm'],r['actor_lateral_mm']))
        s=x['actor_score'];d=x['first_sustained_20mm_departure']
        rows[x['name']]=dict(scene=x['name'],best_tcp_port_3d_mm=math.hypot(best['actor_axial_mm'],best['actor_lateral_mm']),
                            best_axial_mm=best['actor_axial_mm'],best_lateral_mm=best['actor_lateral_mm'],
                            axial_band_samples=s['axial_band_samples'],safe_buffer=s['safe_buffer_visit'],
                            full_insertion=s['official_tier3']>=75,
                            scored_contact=s['official_contact_message']!='No contact detected.',
                            departure_s=d['sim_s'] if d else None,
                            inference_p50_p95_p99_ms=s['inference_p50_p95_p99_ms'],
                            audit=x['actor'])
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--long',type=Path,required=True)
    p.add_argument('--overfit',type=Path,required=True)
    p.add_argument('--dense',type=Path,
                   help='Optional 70 s / 27-token 100 ms context variant')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing summary')
    arms={'old_50ms_12_6':baseline(a.baseline),'new_100ms_12_6':newer(a.long),
          'overfit_50ms_12_6':newer(a.overfit)}
    if a.dense:arms['dense70_100ms_12_6']=newer(a.dense)
    scenes=sorted(arms['old_50ms_12_6'])
    if len(scenes)!=6 or any(sorted(x)!=scenes for x in arms.values()):raise ValueError('Unmatched six-scene comparison')
    aggregate={}
    for name,rows in arms.items():
        aggregate[name]=dict(episodes=len(rows),safe_buffers=sum(x['safe_buffer'] for x in rows.values()),
                             full_insertions=sum(x['full_insertion'] for x in rows.values()),
                             scored_contacts=sum(x['scored_contact'] for x in rows.values()),
                             median_best_tcp_port_3d_mm=float(np.median([x['best_tcp_port_3d_mm'] for x in rows.values()])),
                             median_inference_p95_ms=float(np.median([x['inference_p50_p95_p99_ms'][1] for x in rows.values()])))
    result=dict(schema='aic_sc_pause_comparison/v1',scenes=scenes,models=arms,aggregate=aggregate,
                baseline_minus_100ms_best_mm={s:arms['old_50ms_12_6'][s]['best_tcp_port_3d_mm']-
                                                arms['new_100ms_12_6'][s]['best_tcp_port_3d_mm'] for s in scenes},
                source_sha256={str(x):digest(x) for x in (a.baseline,a.long,a.overfit,a.dense) if x},
                metric_note='Posthoc TCP-to-opening 3D distance; not plug-tip insertion error',
                geometry_actor_input=False,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()

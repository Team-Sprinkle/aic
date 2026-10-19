#!/usr/bin/env python3
"""Posthoc three-arm same-scene transport comparison, including deterministic transformer."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from score_official_anchor_route_dev import score
from score_official_anchor_route_match import extract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transformer',type=Path,required=True)
    p.add_argument('--diffusion-score',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous comparison')
    plan=json.loads((a.transformer/'plan.json').read_text())
    progress=json.loads((a.transformer/'progress.json').read_text())
    expected={s['name'] for s in plan['scenes']}
    if progress.keys()!=expected:raise ValueError('Incomplete transformer denominator')
    calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    transformer=[]
    for scene in plan['scenes']:
        item=progress[scene['name']]
        if item['returncode']:raise ValueError(f'Failed rollout {scene["name"]}')
        audit=Path(item['audit']);extract(audit)
        row=score(audit,calibration)
        row.update(arm='transformer',scene=scene['name'])
        transformer.append(row)
    old=json.loads(a.diffusion_score.read_text())
    runs=old['runs']+transformer
    by=defaultdict(list)
    for row in runs:by[(row['arm'],row['family'])].append(row)
    summary={}
    for (arm,family),group in sorted(by.items()):
        summary[f'{arm}_{family}']=dict(episodes=len(group),
            safe_buffers=sum(r['safe_buffer_visit'] for r in group),
            axial_band_visits=sum(r['axial_band_samples']>0 for r in group),
            scored_contacts=sum(r['official_contact_message']!='No contact detected.' for r in group),
            closest_lateral_in_band_mm=[r['closest_lateral_in_axial_band_mm'] for r in group],
            complete_inference_p95_ms=[r['inference_p50_p95_p99_ms'][1]
                                        if r['inference_p50_p95_p99_ms'] else None for r in group])
    out=dict(schema='aic_official_anchor_three_arm_route_match/v1',runs=runs,summary=summary,
             same_six_frozen_development_scenes=True,simulation_budget_s=40,
             actor_privileged_inputs=False,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

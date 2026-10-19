#!/usr/bin/env python3
"""Expose only public measured TCP state to an otherwise sealed RGB review."""
import argparse
import json
from pathlib import Path


def rows(path):
    for line in path.open(): yield json.loads(line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    output=a.review/'public_observation_by_alias.json'
    if output.exists():p.error('Preserve existing public-state packet')
    blind={x['alias']:x for x in json.loads((a.review/'blind_manifest.json').read_text())}
    sealed=json.loads((a.review/'sealed_mapping.json').read_text())
    indexed={(x['episode'],round(float(x['time_s']),6)):x for x in rows(a.source)}
    packet=[]
    for item in sealed:
        alias=item['alias'];key=(item['source_episode'],round(float(item['time_s']),6))
        row=indexed[key]
        if row['selected_identity']!=blind[alias]['requested']:
            raise ValueError(f'Request join failed for {alias}')
        history_key=(item['source_episode'],round(float(item['history_time_s']),6)) if item['history_time_s'] is not None else None
        history=indexed[history_key] if history_key else None
        packet.append(dict(alias=alias,requested=blind[alias]['requested'],
                           current_tcp_base_pose_xyzw=row['state'][:7],
                           earlier_tcp_base_pose_xyzw=history['state'][:7] if history else None,
                           camera_names=['center','left','right']))
    packet.sort(key=lambda x:x['alias'])
    output.write_text(json.dumps(dict(schema='sfp_public_observation_review/v1',
        observations=packet,base_link_pose_from_official_controller=True,
        known_cad_rule='physical SFP slots are 40 mm apart and number 0 to 4 along board +y',
        simulator_target_geometry_included=False,final_scenes_opened=False),indent=2)+'\n')
    form=a.review/'public_state_review_form.md'
    lines=['# SFP RGB + public-state review form','','These are original RGB triplets and the measured TCP pose available to the deployed pipeline. Fixed CAD says slots 0–4 increase along board +y with 40 mm spacing. No per-scene board or target pose is supplied. Mark the requested card, both port openings (camera and pixel), and visible/occluded/uncertain; abstain if the fixed slot origin cannot be established. Save answers before opening `sealed_mapping.json`.','','| Alias | Requested | Panel | Current TCP base XYZ (m) | Review notes |','|---|---|---|---|---|']
    for x in packet:
        xyz=', '.join(f'{v:.3f}' for v in x['current_tcp_base_pose_xyzw'][:3])
        lines.append(f"| {x['alias']} | {x['requested']} | [view](panels/{x['alias']}.jpg) | {xyz} |  |")
    form.write_text('\n'.join(lines)+'\n')
    print(output)


if __name__=='__main__':main()

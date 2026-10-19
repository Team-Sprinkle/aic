#!/usr/bin/env python3
"""Audit complete-reset route training coverage against released trial anchors."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import yaml


def first_rows(path):
    seen = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            seen.setdefault(int(row['episode_index']), row)
    return seen


def sfp_record(row):
    scene = (Path(row['source_scene_yaml']) if row.get('source_scene_yaml')
             else Path(row['source_scored_stream']).parent/'eval_config.yaml')
    config = yaml.safe_load(scene.read_text())
    trial = next(iter(config['trials'].values()))
    board = trial['scene']['task_board']
    task = next(iter(trial['tasks'].values()))
    cards = [i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
    # Paired-request examples reuse one physical scene with a different public
    # task code. The label's request, rather than the source teacher YAML, is
    # authoritative for training coverage.
    code=np.asarray(row['task_vector'])
    target_card=int(code[4:9].argmax())
    target_port=int(code[2:4].argmax())
    source_target=(int(task['target_module_name'].split('_')[-1]),
                   int(task['port_name'].split('_')[-1]))
    xyz = row.get('observed_tcp_pose_port_frame', [None]*3)[:3]
    return dict(family='sfp', split=row['split'], episode=int(row['episode_index']),
                scene=row['scene_sha256'], source_scene_yaml=str(scene),
                card_count=len(cards), cards=cards, target_card=target_card,
                target_port=target_port, source_teacher_request=list(source_target),
                initial_port_frame_xyz_mm=[round(float(x)*1000,2)
                    for x in xyz] if all(x is not None for x in xyz) else None,
                board_xy_yaw=[board['pose'][k] for k in ('x','y','yaw')])


def sc_record(row):
    scene=Path(row['source_episode'])/'eval_config.yaml'
    config=yaml.safe_load(scene.read_text())
    board=next(iter(config['trials'].values()))['scene']['task_board']
    cards=[i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
    return dict(family='sc', split=row['split'], episode=row['case'],
                scene=row['scene_reset_group'], card_count=int(row['card_count']),
                target_port=int(row['selected_port']), case=row['case'],
                source_episode=row['source_episode'], cards=cards,
                board_xy_yaw=[board['pose'][k] for k in ('x','y','yaw')])


def summarize(records):
    by = defaultdict(list)
    for r in records:
        by[(r['family'], r['split'])].append(r)
    result = {}
    for (family, split), group in sorted(by.items()):
        key = family+'_'+split
        hist = Counter((r['card_count'], r.get('target_card', -1), r['target_port']) for r in group)
        values = [r['initial_port_frame_xyz_mm'] for r in group
                  if r.get('initial_port_frame_xyz_mm') is not None]
        result[key] = dict(episodes=len(group), reset_groups=len({r['scene'] for r in group}),
                           by_cards_target={str(k):v for k,v in sorted(hist.items())},
                           initial_port_xyz_mm_quantiles={str(q):np.quantile(values,q,axis=0).round(1).tolist()
                               for q in (0,.1,.5,.9,1)} if values else None)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sfp-labels',type=Path,required=True)
    p.add_argument('--sc-manifest',type=Path,required=True)
    p.add_argument('--official-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sfp=[sfp_record(row) for row in first_rows(a.sfp_labels).values()]
    sc=[sc_record(row) for row in json.loads(a.sc_manifest.read_text())['episodes']]
    official=yaml.safe_load(a.official_config.read_text())
    anchors=[]
    for name,trial in official['trials'].items():
        board=trial['scene']['task_board'];task=next(iter(trial['tasks'].values()))
        cards=[i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
        anchors.append(dict(name=name,family='sfp' if task['plug_type']=='sfp' else 'sc',
                            cards=cards,card_count=len(cards),
                            target_card=(int(task['target_module_name'].split('_')[-1])
                                if task['plug_type']=='sfp' else None),
                            target_port=(int(task['port_name'].split('_')[-1])
                                if task['plug_type']=='sfp' else
                                int(task['target_module_name'].split('_')[-1])),
                            board_xy_yaw=[board['pose'][k] for k in ('x','y','yaw')]))
    result=dict(schema='aic_route_diffusion_coverage_audit/v1',
                sources=dict(sfp=str(a.sfp_labels),sc=str(a.sc_manifest),
                             official=str(a.official_config)),
                official_anchors=anchors,summary=summarize(sfp+sc),episodes=sfp+sc,
                final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(anchors=anchors,summary=result['summary']),indent=2))


if __name__=='__main__':main()

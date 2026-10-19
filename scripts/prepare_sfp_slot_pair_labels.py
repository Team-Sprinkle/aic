#!/usr/bin/env python3
"""Group existing original-RGB SFP labels for a slot-first identity probe."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml

from audit_sc_port_targets import matrix,pose_matrix
from build_sc_native_pose_labels import project
from build_sfp_port_tcp_waypoints import transform
from train_two_family_port_locator import CAMERAS


def all_slot_centers(row,cal):
    audit=Path(row['source_scored_stream']).parent
    cfg=yaml.safe_load((audit/'eval_config.yaml').read_text())
    trial=next(iter(cfg['trials'].values()))
    board=trial['scene']['task_board'];bp=board['pose']
    world_board=transform([bp[k] for k in ('x','y','z')],[bp[k] for k in ('roll','pitch','yaw')])
    world_base=matrix(cal['sfp']['world_to_base'])
    mount=cal['sfp']['board_to_nic_mount']
    optics={c:np.asarray(cal['camera']['tcp_to_optical'][c],float) for c in CAMERAS}
    tcp=pose_matrix(row['state'])
    output=[]
    for card in range(5):
        rail=board[f'nic_rail_{card}'];present=bool(rail['entity_present'])
        rp=rail['entity_pose'] if present else dict(translation=0.,roll=0.,pitch=0.,yaw=0.)
        board_nic=transform([mount['x_nominal_m']+rp['translation'],
                             mount['y_rail0_m']+mount['y_step_per_rail_m']*card,
                             mount['z_m']],[rp[k] for k in ('roll','pitch','yaw')])
        ports=[]
        for port in (0,1):
            ports.append(np.linalg.inv(world_base)@world_board@board_nic@
                         matrix(cal['sfp']['nic_mount_to_port_opening'][str(port)]))
        midpoint=np.eye(4);midpoint[:3,3]=(ports[0][:3,3]+ports[1][:3,3])/2
        output.append(dict(card=card,present=present,position_base_m=midpoint[:3,3].tolist(),
                           cameras={c:project(midpoint,tcp@optics[c]) for c in CAMERAS}))
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old',type=Path,required=True)
    ap.add_argument('--fresh',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve prior label set')
    old=[json.loads(s) for s in (args.old/'rows.jsonl').open()]
    fresh=[json.loads(s) for s in (args.fresh/'rows.jsonl').open()]
    if len({r['scene_group'] for r in old if r['split']=='train'})!=50 or \
       len({r['scene_group'] for r in old if r['split']=='validation'})!=10 or \
       len({r['scene_group'] for r in fresh})!=10:
        raise ValueError('Unexpected source episode groups')
    source={}
    for r in old:
        source.setdefault(r['scene_group'],r['source_scored_stream'])
    # The synthetic card4/port1 reset perturbs an old validation reset; keep
    # it out of fitting to avoid a near-duplicate validation configuration.
    rows=[]
    for r in old:
        rows.append({**r,'origin':'old_balanced_teacher','fit_split':r['split']})
    for r in fresh:
        rows.append({**r,'origin':'fresh_failure_diagnostic',
                     'fit_split':'excluded_near_duplicate' if r['episode']=='card4_port1' else 'train'})
    train={r['scene_group'] for r in rows if r['fit_split']=='train'}
    val={r['scene_group'] for r in rows if r['fit_split']=='validation'}
    if train&val:raise ValueError('Complete reset leak')
    cal={'sfp':json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text()),
         'camera':json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())}
    # Identical static slot positions can be cached per scene, but their image
    # projections change with the measured wrist TCP at every observation.
    for row in rows:
        row['slot_targets']=all_slot_centers(row,cal)
        if len(row['slot_targets'])!=5:raise ValueError('Wrong slot count')
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    summary=dict(schema='sfp_slot_pair_grouped_labels/v1',rows=len(rows),
                 group_count={s:len({r['scene_group'] for r in rows if r['fit_split']==s})
                              for s in ('train','validation','excluded_near_duplicate')},
                 row_count=dict(Counter(r['fit_split'] for r in rows)),
                 training_geometry_label_only=True,
                 absent_slot_is_nominal_hypothetical_coordinate=True,
                 projected_inframe_is_not_occlusion_visibility=True,
                 source_agent_actions_used=False,final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()

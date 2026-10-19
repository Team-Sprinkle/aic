#!/usr/bin/env python3
"""Audit causal actor rows and count independent150/180mm approach coverage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9
from audit_wide_estimated_frames import audit as audit_wide_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','commands','source-index','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior audit')
    metadata={x['episode']:x for x in json.load((a.commands/'summary.json').open())['episodes']}
    originals={str(Path(x['source_stream']).resolve()):x for x in map(json.loads,a.source_index.open())}
    groups=defaultdict(list);splits={};coverage=defaultdict(set);bins=defaultdict(set)
    for row in map(json.loads,(a.dataset/'rows.jsonl').open()):
        if (not row['actor_training_ready'] or row['true_geometry_actor_input']
                or max(row['grasp_lock_time_s'],row['perception_lock_time_s'])>row['time_s']
                or not row['recorded_complete_targets'] or row['action_owner']!='teacher'):
            raise ValueError('Unverified/acausal actor input')
        scene=row['scene']
        if scene in splits and splits[scene]!=row['split']:raise ValueError('Physical reset leakage')
        splits[scene]=row['split'];groups[row['episode']].append(row)
    episodes=[];wide_items=[]
    for key,rows in groups.items():
        m=metadata[key];rows.sort(key=lambda r:r['time_s'])
        source=str(Path(m['source']).resolve())
        original=originals.get(source)
        match=re.search(r'_n([0-5])_p([01])',m['source_episode_name'])
        cards=original['card_count'] if original else int(match[1]) if match else None
        if cards is None:raise ValueError('Unknown scene count provenance')
        family=m['family'];split=m['split'];port=int(np.argmax(rows[0]['task'][2:4]))
        xyz=np.array([r['predicted_tip_port_pose9'][:3] for r in rows])*1000
        offsets=np.array([r['action_target_offsets_s'] for r in rows])
        if not np.allclose(offsets,[.05,.1,.15,.2]):raise ValueError('Wrong command horizon')
        targets=np.array([r['recorded_tcp_targets_predicted_port_pose9'] for r in rows])
        # Complete target decode must recover the original recorded TCP command
        # with the same predicted body offset, including rotated targets.
        for r in rows:
            body=np.array(r['predicted_body_tip_position_m'])
            for point,tcp in zip(r['teacher_tcp_targets_predicted_port_pose9'],r['recorded_tcp_targets_predicted_port_pose9']):
                decoded=matrix_from_pose9(point);decoded[:3,3]-=decoded[:3,:3]@body
                if not np.allclose(decoded,matrix_from_pose9(tcp),atol=2e-7,rtol=0):
                    raise ValueError('Plug-control-point target does not roundtrip')
        wide=original is None
        if wide:wide_items.append((m,rows[0]))
        e=dict(episode=key,family=family,split=split,cards=cards,port=port,scene=rows[0]['scene'],
            rows=len(rows),wide_reset_source=wide,maximum_gap_mm=float(max(-xyz[:,2])),
            maximum_lateral_mm=float(max(np.linalg.norm(xyz[:,:2],axis=1))))
        episodes.append(e);coverage[f'{family}/{split}/cards{cards}/port{port}'].add(key)
        for lower,upper in [(0,40),(40,100),(100,150),(150,185)]:
            if np.any((-xyz[:,2]>=lower)&(-xyz[:,2]<upper)):
                bins[f'{family}/{split}/gap{lower}_{upper}mm'].add(key)
    gates={}
    for family in ('sc','sfp'):
        train=[x for x in episodes if x['family']==family and x['split']=='train']
        val=[x for x in episodes if x['family']==family and x['split']=='validation']
        wider_train=[x for x in train if x['wide_reset_source'] and x['maximum_gap_mm']>=145]
        wider_val=[x for x in val if x['wide_reset_source'] and x['maximum_gap_mm']>=145]
        gates[family]=dict(train_sources=len(train),validation_sources=len(val),
            wide_far_train_sources=len(wider_train),wide_far_validation_sources=len(wider_val),
            coverage_cells={k:len(v) for k,v in coverage.items() if k.startswith(family+'/')},
            passed=len(train)>=(50 if family=='sc' else 100) and len(wider_train)>=6 and len(wider_val)>=6)
    wide_geometry=audit_wide_frames(wide_items)
    a.output.with_name(a.output.stem+'_wide_geometry.json').write_text(json.dumps(wide_geometry,indent=2)+'\n')
    report=dict(schema='shared_insertion_envelope_audit/v1',passed=all(x['passed'] for x in gates.values()) and wide_geometry['passed'],
        wider_geometry_gate_passed=wide_geometry['passed'],
        gates=gates,episodes=episodes,independent_episode_bins={k:len(v) for k,v in bins.items()},
        task_code_is_public=True,geometry_runtime_input=False,original_data_modified=False,
        caveat='Counts and decoder roundtrips are data gates, not autonomous insertion success.',final_scenes_opened=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__=='__main__':main()

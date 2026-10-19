#!/usr/bin/env python3
"""Read-only command-change and normalization audit on complete held-out episodes."""
import argparse,json
from pathlib import Path
import numpy as np
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups={}
for line in (a.source/'rows.jsonl').open():
 r=json.loads(line)
 if r['split']!='validation':continue
 near=-.04<=r['predicted_tip_port_pose9'][2]<=0
 targets=np.asarray(r['recorded_tcp_targets_predicted_port_pose9'])[:,:3];tcp=matrix_from_xyz_xyzw(r['state'][:7]);relative=matrix_from_pose9(r['predicted_port_tcp_pose9']);rotation=(relative@np.linalg.inv(tcp))[:3,:3];prior=relative[:3,3]+rotation@np.asarray(r['state'][13:16]);change=np.linalg.norm(targets-prior,axis=1)*1000
 groups.setdefault(r['family']+('/near' if near else '/other'),[]).append(dict(episode=r['episode'],first_change_mm=float(change[0]),last_change_mm=float(change[-1]),chunk_span_mm=float(np.linalg.norm(targets[-1]-targets[0])*1000),axial_span_mm=float((targets[-1,2]-targets[0,2])*1000)))
report={}
for group,rows in groups.items():
 report[group]=dict(rows=len(rows),episodes=len({r['episode'] for r in rows}),percentiles={k:np.percentile([r[k] for r in rows],[50,90,95,99]).tolist() for k in ['first_change_mm','last_change_mm','chunk_span_mm','axial_span_mm']},fraction_first_change_below_0p1mm=float(np.mean([r['first_change_mm']<.1 for r in rows])),fraction_chunk_span_below_0p1mm=float(np.mean([r['chunk_span_mm']<.1 for r in rows])))
a.output.write_text(json.dumps(dict(reports=report,heldout_episodes_only=True,original_data_modified=False,interpretation='Small command changes can be valid while impedance tracking continues. This audit diagnoses objective signal; do not blindly delete stationary references or substitute measured future poses for recorded commands.'),indent=2)+'\n');print(json.dumps(report),flush=True)

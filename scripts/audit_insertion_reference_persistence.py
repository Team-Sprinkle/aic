#!/usr/bin/env python3
"""Held-out causal persistence baselines; no privileged state or deployment model."""
import argparse,json
from pathlib import Path
import numpy as np
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();errors={}
 for line in (a.source/'rows.jsonl').open():
  r=json.loads(line)
  if r['split']!='validation' or not -.04<=r['predicted_tip_port_pose9'][2]<=0:continue
  tcp=matrix_from_xyz_xyzw(r['state'][:7]);relative=matrix_from_pose9(r['predicted_port_tcp_pose9']);base_to_port=relative@np.linalg.inv(tcp)
  target=matrix_from_pose9(r['recorded_tcp_targets_predicted_port_pose9'][0]);error=np.asarray(r['state'][13:16])
  for name,offset in [('hold_measured',np.zeros(3)),('controller_error_as_base',error),('controller_error_as_TCP',tcp[:3,:3]@error)]:
   predicted=relative[:3,3]+base_to_port[:3,:3]@offset
   key=r['family']+'/'+name;errors.setdefault(key,[]).append((r['episode'],(predicted-target[:3,3])*1000))
 report={}
 for key,items in errors.items():
  e=np.stack([i[1] for i in items]);report[key]=dict(samples=len(e),episodes=len({i[0] for i in items}),lateral_target_error_mm_p50_p95=np.percentile(np.linalg.norm(e[:,:2],axis=1),[50,95]).tolist(),position_target_error_mm_p50_p95=np.percentile(np.linalg.norm(e,axis=1),[50,95]).tolist())
 a.output.write_text(json.dumps(dict(source=str(a.source),reports=report,causal_public_state=True,diagnostic_only=True,warning='Low command prediction error may largely copy the previous reference. This cannot demonstrate learned alignment, forward progress, or physical insertion.'),indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Public-wrench tip-moment diagnosis; ground truth labels score direction only."""
import bisect,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from prepare_verified_insertion_command_rows import message_pose
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');out=root/'public_tip_torque_audit';out.mkdir(exist_ok=False)
labels=json.loads((root/'close_reestimation_audit_fresh4_v4/records.json').read_text())+json.loads((root/'close_reestimation_audit_V19_v2/records.json').read_text());sources={}
for row in labels:
 if row['phase']=='hold_25mm':sources.setdefault(str(Path(row['source_stream']).parent),[]).append(np.asarray(row['truth_pose']))
reports=[]
for path,truth in sources.items():
 run=Path(path);data=json.loads((run/'scored_stream.json').read_text());trace=[json.loads(x) for x in (run/'runtime_trace.jsonl').open()];metric=json.loads((run/'shared150mm_metrics.json').read_text());Ptrue=np.asarray(metric['raw_terminal']['measured_base_opening_matrix']);Gtrue=Rotation.from_matrix(np.stack([x[:3,:3] for x in truth])).mean().as_matrix();starts=[x['sim_time'] for x in trace if x['event']=='visual_servo_started'];steps=[x for x in trace if x['event']=='visual_servo_step'];w=data['streams']['/fts_broadcaster/wrench'];wt=[x[0] for x in w];c=data['streams']['/aic_controller/controller_state'];ct=[x[0] for x in c];assert {x[1]['header']['frame_id'] for x in w}=={'ati/tool_link'}
 for attempt,start in enumerate(starts):
  end=starts[attempt+1] if attempt+1<len(starts) else float('inf');rows=[]
  lock=max([x for x in trace if x['event']=='shared_visual_lock' and x['sim_time']<=start],key=lambda x:x['sim_time']);P=matrix_from_xyz_xyzw([*lock['report']['predicted_opening_base_m'],*lock['report']['predicted_port_quat_xyzw']]);Ghat=np.asarray(lock['body_tip_position_tcp_m'])
  for e in steps:
   if not start<=e['sim_time']<end:continue
   stamp=round(e['sim_time']*1e9);j=bisect.bisect_right(wt,stamp)-1;k=bisect.bisect_right(ct,stamp)-1
   if min(j,k)<0 or max(stamp-wt[j],stamp-ct[k])>5_000_000:continue
   v=w[j][1]['wrench'];vector=np.array([v[q][axis] for q in ['force','torque'] for axis in 'xyz']);T=message_pose(c[k][1]['tcp_pose']);rows.append((e,vector,T))
  free=[x for x in rows if -.008<=x[0]['estimated_tip_port_m'][2]<=-.005]
  if len(free)<3:continue
  baseline=np.median(np.stack([x[1] for x in free[-5:]]),0);Rref=free[-1][2][:3,:3];moments=[];desired=[];force=[];points=[]
  for e,v,T in rows:
   if e['estimated_tip_port_m'][2]<=0 or Rotation.from_matrix(Rref.T@T[:3,:3]).magnitude()>np.deg2rad(.2):continue
   delta=v-baseline
   if np.linalg.norm(delta[:3])<.5:continue
   # Tool-sensor orientation equals TCP; sensor origin172mm behind TCP.
   moment_tip=delta[3:]+np.cross(np.array([0,0,-.172])-Ghat,delta[:3]);Rport=P[:3,:3].T@T[:3,:3];tau=Rport@moment_tip;F=Rport@delta[:3];actual=Ptrue[:3,:3].T@T[:3,:3]@Gtrue;correction=-Rotation.from_matrix(actual).as_rotvec();moments.append(tau);desired.append(correction);force.append(F);points.append(dict(time=e['sim_time'],estimated_depth_mm=e['estimated_tip_port_m'][2]*1000,tip_torque_port_Nm=tau.tolist(),force_port_N=F.tolist(),desired_correction_xy_deg=np.degrees(correction[:2]).tolist()))
  if not moments:continue
  moments=np.stack(moments);desired=np.stack(desired);dot=np.sum(moments[:,:2]*desired[:,:2],1);reports.append(dict(audit=path,attempt=attempt,samples=len(points),baseline_free_samples=len(free),median_tip_moment_xy_Nm=np.median(moments[:,:2],0).tolist(),median_desired_correction_xy_deg=np.degrees(np.median(desired[:,:2],0)).tolist(),fraction_torque_aligned_with_corrective_rotation=float(np.mean(dot>0)),median_axial_force_N=float(np.median(np.stack(force)[:,2])),rows=points,GT_scoring_only=True,public_force_only_for_candidate_control=True))
(out/'summary.json').write_text(json.dumps(dict(results=reports,inference_policy_changed=False,causal_grasp_label_before_contact=True,port_truth_static_same_episode_pre_reset=True,important='Includes contact/freeze intervals; do not treat directioncorrelation as a proven compliantcontroller.'),indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in reports],indent=2),flush=True)

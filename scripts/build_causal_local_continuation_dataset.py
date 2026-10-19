#!/usr/bin/env python3
"""Build label-only expert continuations; exclude autonomous actor commands.

Teacher ownership is joined by its explicitly logged command matrices and
causal timestamps. All actor inputs come from the recorded RGB handoff,
measured state and public task. True grasp geometry only changes labels.
"""
import argparse,bisect,json,hashlib
from pathlib import Path
from collections import Counter
import cv2,numpy as np,yaml
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw,matrix_from_pose9
from train_sc_grasp_pose_probe import interpolate
from analyze_local_actor_scored_tip import record_at


def at(rows,t):
 j=bisect.bisect_right(rows,t,key=lambda r:r[0])-1
 if j<0:raise ValueError('No preceding sample')
 return rows[j]


def pose9(T):return [*T[:3,3],*T[:3,:2].T.reshape(-1)]


def pose_message(d):
 return matrix_from_xyz_xyzw([*[d['position'][k] for k in 'xyz'],
                              *[d['orientation'][k] for k in ['x','y','z','w']]])


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--audits',nargs=3,type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error('Preserve earlier derivatives')
 a.output.mkdir(parents=True);rows=[];reports=[];groups={};failures=[]
 for index,audit in enumerate(a.audits):
  stream=json.loads((audit/'scored_stream.json').read_text());events=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
  teacher=[r for r in events if r['event']=='oracle_local_command'];terminal=events[-1]
  last=teacher[-1];tip=np.asarray(last['measured_tip_point_port_m']);score=yaml.safe_load((audit/'results/scoring.yaml').read_text())[stream['trial_id']]
  safe=all('No ' in score['tier_2']['categories'][k]['message'] for k in ['contacts','insertion force'])
  passed=(tip[2]>=.014 and np.linalg.norm(tip[:2])<=.0005 and last['measured_tip_orientation_error_deg']<=1 and stream['official_tier3']>=75 and safe)
  if not passed:failures.append(str(audit));continue
  lock=next(r for r in events if r['event']=='locked_local_switch')
  assert lock['geometry_input'] is False and lock['privileged_actor_guidance'] is False
  report=lock['report'];port=np.eye(4);port[:3,3]=report['predicted_opening_base_m'];port[:3,:3]=Rotation.from_quat(report['predicted_port_quat_xyzw']).as_matrix();inv=np.linalg.inv(port)
  NNbody=np.asarray(lock['predicted_body_tip_position_m']);assert lock['tip_orientation_input'] is False
  takeover=next(r for r in events if r['event']=='teacher_takeover');oracle=next(r for r in events if r['event']=='oracle_local_switch')
  assert takeover['privileged_geometry'] and oracle['privileged_geometry']
  commands=[c for c in stream['commands'] if c['kind']=='pose'];matched=[];used=set();previous=-1
  for event in teacher:
   expected=np.asarray(event['command_base']);t=round(event['sim_time']*1e9);candidate=[]
   # Match exact logged teacher pose to serialized commands. Timestamp alone
   # cannot disambiguate the last actor command at the handoff boundary.
   lo=bisect.bisect_left(commands,t-100000000,key=lambda c:c['sim_ns']);hi=bisect.bisect_right(commands,t+1000000000,key=lambda c:c['sim_ns'])
   for j in range(lo,hi):
    if j<=previous or j in used:continue
    c=commands[j]
    if c['message']['header']['frame_id']!='base_link':continue
    T=pose_message(c['message']['pose'])
    if np.max(np.abs(T-expected))<1e-7:candidate.append((j,c,T))
   if not candidate:raise ValueError(f'No causal exact-command ownership match: {audit} {t}')
   j,c,T=candidate[0];previous=j;used.add(j);matched.append((t,T,j))
  assert len(matched)==len(teacher) and len(used)==len(teacher)
  episode=audit.name;split='validation' if index==2 else 'train'
  group=hashlib.sha256(json.dumps({k:stream['reset_identity'][k] for k in ['scene','robot']},sort_keys=True).encode()).hexdigest();groups[episode]=(split,group)
  e=stream['edges'];wb=record_at(e['world|tabletop'],0)@record_at(e['tabletop|base_link'],0);calibration=None;counts=Counter();times=[m[0] for m in matched]
  # Stop expert anchors when physical success first appears. Continued commands
  # after engine gripper freezing cannot supervise insertion dynamics.
  physical_cutoff=min(r['sim_time'] for r in teacher if r['measured_tip_point_port_m'][2]>=.014 and np.linalg.norm(r['measured_tip_point_port_m'][:2])<=.0005)
  for frame,(t,T,command_index) in enumerate(matched):
   if t>=round(physical_cutoff*1e9):counts['post_physical_success_excluded']+=1;continue
   try:
    if abs(commands[command_index]['sim_ns']-t)>100000000:raise ValueError('Delayed command anchor excluded')
    # The ROS node's command header can lag the observation's controller
    # timestamp. Recover the actual input triplet by its preceding controller
    # identity, never by that command header. The complete logged state is
    # verified below, including joints and wrench.
    input_candidates=[]
    for image_t,_ in stream['cameras']['center']:
     if not t<=image_t<=t+100000000:continue
     ct,controller=at(stream['streams']['/aic_controller/controller_state'],image_t)
     if ct==t:input_candidates.append(image_t)
    if len(input_candidates)!=1:raise ValueError('Observation triplet not uniquely recovered')
    image_t=input_candidates[0];images={};stamps={}
    for cam in ['center','left','right']:
     it,file=at(stream['cameras'][cam],image_t)
     if abs(image_t-it)>1000000:raise ValueError('Unsynchronized images')
     if not Path(file).exists():raise ValueError('Missing image')
     images[cam]=file;stamps[cam]=it
    ct,controller=at(stream['streams']['/aic_controller/controller_state'],image_t)
    jt,joints=at(stream['streams']['/joint_states'],image_t);wt,wrench=at(stream['streams']['/fts_broadcaster/wrench'],image_t)
    if max(image_t-ct,image_t-jt,image_t-wt)>100000000:raise ValueError('Stale proprioception')
    current=pose_message(controller['tcp_pose']);v=controller['tcp_velocity'];w=wrench['wrench'];q=dict(zip(joints['name'],joints['position']))
    xyzq=controller['tcp_pose'];state=[*[xyzq['position'][k] for k in 'xyz'],*[xyzq['orientation'][k] for k in ['x','y','z','w']],*[v[k][axis] for k in ['linear','angular'] for axis in 'xyz'],*controller['tcp_error'],*[q[k] for k in stream['joint_order']],q['gripper/left_finger_joint']/2.,*[w[k][axis] for k in ['force','torque'] for axis in 'xyz']]
    qstate=np.asarray(state[3:7]);qstate/=np.linalg.norm(qstate)
    if qstate[3]<0:qstate=-qstate
    state[3:7]=qstate.tolist()
    assert len(state)==32 and lock['sim_time']<=image_t/1e9
    if not np.allclose(state,teacher[frame]['state'],atol=1e-4,rtol=0):raise ValueError('Logged complete observation state does not match recovered triplet')
    if calibration is None:
     true_tip=interpolate(e['aic_world|cable_1'],ct)@interpolate(e['cable_1|cable_1/sc_tip_link'],ct)
     calibration=(np.linalg.inv(wb@current)@true_tip)[:3,3]
    ids=[bisect.bisect_right(times,t+h*50000000)-1 for h in range(4)]
    if any(j<frame or times[j]>=round(physical_cutoff*1e9) or t+h*50000000-times[j]>100000000 or abs(commands[matched[j][2]]['sim_ns']-times[j])>100000000 for h,j in enumerate(ids)):raise ValueError('Invalid pre-terminal teacher horizon')
    original=[pose9(inv@matched[j][1]) for j in ids];targets=[]
    for value in original:
     target=matrix_from_pose9(value);point=target[:3,3]+target[:3,:3]@calibration
     out=list(value);out[:3]=point.tolist();targets.append(out)
     assert np.linalg.norm(point-target[:3,:3]@calibration-target[:3,3])<1e-12
    relative=inv@current;NNpoint=relative[:3,3]+relative[:3,:3]@NNbody;requested=int(stream['task']['target_module_name'].rsplit('_',1)[-1])
    rows.append({'schema':'aic_causal_local_continuation_bc/v1','family':'sc','episode':episode,'scene':group,'split':split,'frame':frame,'time_s':t/1e9,'images':images,'cache_index':None,'state':state,'task':[0.,1.,float(requested==0),float(requested==1),*([0.]*6)],'predicted_port_tcp_pose9':pose9(relative),'teacher_tcp_targets_predicted_port_pose9':targets,'recorded_tcp_targets_predicted_port_pose9':original,'predicted_tip_port_pose9':[*NNpoint,1.,0.,0.,0.,1.,0.],'predicted_tip_valid':True,'tip_orientation_input':False,'onpolicy_teacher_match':True,'action_owner':'teacher','actor_prefix_in_bc':False,'source_command_indices':[matched[j][2] for j in ids],'source_decision_controller_stamps_ns':[times[j] for j in ids],'source_published_command_stamps_ns':[commands[matched[j][2]]['sim_ns'] for j in ids],'controller_stamp_ns':ct,'image_stamps_ns':stamps,'geometry_label_only':True,'true_geometry_actor_input':False,'action_reference':'SCplug_tip_translation_and_TCP_orientation'})
    counts['admitted']+=1
   except ValueError as err:counts[str(err)]+=1
  reports.append({'audit':str(audit),'episode':episode,'split':split,'group':group,'teacher_commands':len(teacher),'matched_unique_teacher_commands':len(matched),'non_teacher_commands_excluded':len(commands)-len(matched),'command_header_minus_observed_controller_ms_p5_p50_p95':np.percentile([(commands[j]['sim_ns']-t)/1e6 for t,_,j in matched],[5,50,95]).tolist(),'counts':dict(counts),'physical_cutoff_s':physical_cutoff,'label_body_translation_only':calibration.tolist() if calibration is not None else None,'input_body_translation_camera_only':NNbody.tolist(),'official_tier3':stream['official_tier3']})
 if len(reports)<2:raise ValueError('Continuation feasibility gate failed; no training data promoted')
 train_groups={g for s,g in groups.values() if s=='train'};val_groups={g for s,g in groups.values() if s=='validation'}
 if train_groups&val_groups:raise ValueError('Reset configuration leaks across split')
 with (a.output/'rows.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
 summary={'schema':'aic_causal_local_continuation_dataset/v1','rows':len(rows),'episodes':reports,'failed_pilots_excluded':failures,'input_contract':'Camera-only static port/body-tip lock, measured robot state and public task; true transforms only re-express recorded teacher commands.','actor_commands_in_expert_bc':False,'reset_group_split_checked':True,'post_physical_success_anchors_excluded':True,'rows_sha256':hashlib.sha256((a.output/'rows.jsonl').read_bytes()).hexdigest(),'final_scenes_opened':False}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='episodes'}))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Command-derived TCP-body transport and port-frame local chunks from scored successes.
Privileged geometry is stored under training labels, never actor input features.
"""
import argparse,bisect,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from aic_teacher_official.expert_generator.moveit_py_backend import MoveItPyBackendConfig,MoveItPyPlanningBackend
ROOT=Path(__file__).resolve().parents[1]
def path(s):
 return str(ROOT/Path(s).relative_to('/repo')) if s.startswith('/repo/') else (str(Path('/var/tmp/chmin_aic_20260918_act')/Path(s).relative_to('/history')) if s.startswith('/history/') else s)
def matrix(d):
 m=np.eye(4);m[:3,3]=[d['translation'][k] for k in ['x','y','z']];m[:3,:3]=Rotation.from_quat([d['rotation'][k] for k in ['x','y','z','w']]).as_matrix();return m
def posematrix(d):return matrix(dict(translation=d['position'],rotation=d['orientation']))
def pose(m):return list(m[:3,3])+list(Rotation.from_matrix(m[:3,:3]).as_quat(canonical=True))
def at(rows,t):
 i=bisect.bisect_right(rows,t,key=lambda r:r[0])-1
 if i<0:raise ValueError('no causal sample')
 return rows[i]
def main():
 p=argparse.ArgumentParser();p.add_argument('--streams',type=Path,nargs='*',default=[]);p.add_argument('--verified-transport-prefix-streams',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);p.add_argument('--validation-transport-streams',type=Path,nargs='*',default=[]);p.add_argument('--diagnostic-validation-transport-streams',type=Path,nargs='*',default=[]);p.add_argument('--validation-full-success-streams',type=Path,nargs='*',default=[]);a=p.parse_args();assert a.streams or a.verified_transport_prefix_streams;assert not a.output.exists() or not any(a.output.iterdir()), 'Preserve existing/partial dataset; choose a new output';a.output.mkdir(parents=True,exist_ok=True)
 cfg=MoveItPyBackendConfig(workspace_dir=str(ROOT),use_sim_time=False)
 backend=MoveItPyPlanningBackend(cfg);mi=MoveItPy(node_name='aic_command_label_fk',config_dict=backend._moveit_config_dict());fk=RobotState(mi.get_robot_model());counts=Counter();episodes=[];rejected=[]
 handles={k:(a.output/(k+'_chunks.jsonl')).open('w') for k in ['transport','local']}
 for episode_index,source in enumerate(a.streams+a.verified_transport_prefix_streams+a.validation_transport_streams+a.diagnostic_validation_transport_streams+a.validation_full_success_streams,start=1400000):
  split='validation' if source in a.validation_transport_streams+a.diagnostic_validation_transport_streams+a.validation_full_success_streams else 'train'
  r=json.loads(source.read_text());r['task_reset_group_id']=r['scene_group_id'];r['scene_group_id']=hashlib.sha256(json.dumps({k:r['reset_identity'][k] for k in ['scene','robot']},sort_keys=True).encode()).hexdigest();audit=Path(path(r['audit']));score=r['official_tier3'];report=dict(episode_index=episode_index,source_stream=str(source.resolve()),audit=str(audit),scene_group_id=r['scene_group_id'],split=split,official_tier3=score,task_family='sc_to_sc')
  if source in a.validation_full_success_streams:
   assert score==75, 'Held-out full-success stream lacks official full insertion'
  elif split=='validation':
   route_audit=json.loads((audit/'route_audit.json').read_text())
   if r.get('teacher_interface',{}).get('transport_action_mode')=='absolute_cartesian_pose_base_link':
    assert route_audit['terminal_before_reset'], 'Missing terminal observation'
    if source in a.diagnostic_validation_transport_streams:
     assert route_audit['whole_run_offlimit_contact_free'] and route_audit['measured_vs_plan_final_tcp_mm']<=20. and route_audit['orientation_error_deg']<=3., 'Invalid diagnostic target-regression validation'
     report['validation_scope']='diagnostic recorded-target regression only; teacher endpoint gate did not pass'
    else:
     assert route_audit['passed'], 'Unverified Cartesian route-only validation'
   else:
    assert route_audit['pose_command_count']==0 and route_audit['measured_vs_plan_final_tcp_mm']<=6.0, 'Unverified route-only validation'
  prefix_only=source in a.verified_transport_prefix_streams
  prefix_cutoff=math.inf
  if prefix_only:
   admission=json.loads((audit/'transport_prefix_admission.json').read_text())
   assert admission['passed'] and not admission['video_review_required'], 'Unreviewed or invalid transport prefix'
   prefix_cutoff=admission['cutoff_sim_ns'];report['admission_purpose']='verified_transport_prefix_only'
  if score!=75 and split!='validation' and not prefix_only:rejected.append({**report,'reason':'not full insertion; failed insertion excluded; transport-prefix fallback disabled while successes exist'});continue
  commands=r['commands'];streams=r['streams'];edges=r['edges'];port=r['task']['target_module_name'];port_index=int(port.rsplit('_',1)[-1]);times=[c['sim_ns'] for c in commands];base_port=None;tip=None;rows=[];bad=Counter();joint_fk_cache={}
  # Ownership is reconstructed from explicit teacher events, never guessed from action shape.
  trace=audit/'runtime_trace.jsonl'
  if not trace.exists() and split!='validation':raise FileNotFoundError(trace)
  # Route-only held-out episodes have no local/recovery events and cannot enter BC training.
  events=[json.loads(l) for l in trace.open()] if trace.exists() else []
  cartesian_transport=r.get('teacher_interface',{}).get('transport_action_mode')=='absolute_cartesian_pose_base_link'
  local_events=[e['time_sec'] for e in events if e['event']=='local_preinsert_align_started']
  if cartesian_transport and split=='train' and not local_events:raise ValueError('Missing explicit Cartesian transport/local boundary')
  local_start_ns=round(min(local_events)*1e9) if local_events else math.inf
  recovery_starts=[int(e['time_sec']*1e9) for e in events if e['event']=='recovery_backoff_started']
  # Successful normal episodes are currently required; recovering episodes need explicit full ownership labels first.
  if recovery_starts and not prefix_only:rejected.append({**report,'reason':'recovery ownership needs explicit supervisor/actor joins before BC'});continue
  assert not prefix_only or all(t>=prefix_cutoff for t in recovery_starts), 'Recovery inside transport prefix'
  for i,c in enumerate(commands):
   t=c['sim_ns']
   if t>=prefix_cutoff:bad['excluded_local_and_recovery_after_verified_prefix']+=1;continue
   try:
    ct,controller=at(streams['/aic_controller/controller_state'],t);jt,joints=at(streams['/joint_states'],t);wt,wrench=at(streams['/fts_broadcaster/wrench'],t)
    if max(t-ct,t-jt,t-wt)>100000000:bad['stale_state']+=1;continue
    if c['kind']=='joint' and c['timestamp_age_wall_ns']>30000000:bad['uncertain_unstamped_command']+=1;continue
    images={};image_stamps={}
    for cam,camera in r['cameras'].items():
     it,f=at(camera,t)
     if t-it>100000000:raise ValueError('stale image')
     f=path(f)
     if not Path(f).exists():raise ValueError('missing image')
     images[cam]=f;image_stamps[cam]=it
    # AicAdapter synchronizes all three images, then chooses preceding state
    # samples at that image stamp; /joint_states is NOT the actor observation.
    observation_t=min(image_stamps.values())
    if max(image_stamps.values())-observation_t>1000000:raise ValueError('unsynchronized RGB triplet')
    ct,controller=at(streams['/aic_controller/controller_state'],observation_t)
    jt,joints=at(streams['/joint_states'],observation_t)
    wt,wrench=at(streams['/fts_broadcaster/wrench'],observation_t)
    if max(observation_t-ct,observation_t-jt,observation_t-wt)>100000000:raise ValueError('stale adapted state')
    tcp=posematrix(controller['tcp_pose'])
    def edge(key):return matrix(at(edges[key],t)[1])
    if base_port is None:
     world_base=edge('world|tabletop')@edge('tabletop|base_link')
     wp=edge('aic_world|task_board')@edge('task_board|task_board/'+port)@edge('task_board/'+port+'|task_board/'+port+'/sc_port_base_link_entrance')
     base_port=np.linalg.inv(world_base)@wp
     world_tip=edge('aic_world|cable_1')@edge('cable_1|cable_1/sc_tip_link')
     tip=np.linalg.inv(tcp)@np.linalg.inv(world_base)@world_tip
    if c['kind']=='joint':
     q=tuple(c['message']['target_state']['positions'])
     if len(q)!=6:raise ValueError('joint order')
     if q not in joint_fk_cache:
      fk.set_joint_group_active_positions(cfg.planning_group,np.asarray(q));fk.update();joint_fk_cache[q]=posematrix({'position':{k:getattr(fk.get_pose(cfg.end_effector_link).position,k) for k in ['x','y','z']},'orientation':{k:getattr(fk.get_pose(cfg.end_effector_link).orientation,k) for k in ['x','y','z','w']}})
     target=joint_fk_cache[q];phase='transport'
    else:
     if c['message']['header']['frame_id']!='base_link':raise ValueError('non-base absolute command')
     target=posematrix(c['message']['pose']);phase='transport' if cartesian_transport and t<local_start_ns else 'local'
    v=controller['tcp_velocity'];w=wrench['wrench']
    # Match AicAdapter::ReorderJointState BEFORE the runtime state assembler.
    # The adapter orders named arm joints and divides the left-finger value by2.
    named_joints=dict(zip(joints['name'],joints['position']))
    observation_joints=[named_joints[k] for k in r['joint_order']]+[named_joints['gripper/left_finger_joint']/2.]
    state=pose(tcp)+[v[key][axis] for key in ['linear','angular'] for axis in ['x','y','z']]+list(controller['tcp_error'])+observation_joints+[w[key][axis] for key in ['force','torque'] for axis in ['x','y','z']]
    assert len(state)==32
    # Label observed plug geometry from the causal simulator tip transform;
    # do not assume a perfectly rigid grasp throughout forceful motion.
    world_tip_now=matrix(at(edges['aic_world|cable_1'],observation_t)[1])@matrix(at(edges['cable_1|cable_1/sc_tip_link'],observation_t)[1])
    measured_tip=np.linalg.inv(tcp)@np.linalg.inv(world_base)@world_tip_now
    rows.append(dict(index=i,t=t,phase=phase,tcp=tcp,target=target,images=images,image_stamps=image_stamps,state=state,tip=measured_tip))
   except (ValueError,KeyError):bad['missing_or_invalid_causal_join']+=1
  row_by_index={x['index']:x for x in rows};chunk_counts=Counter();roundtrip_m=0.;roundtrip_deg=0.
  for anchor in rows:
   indices=[bisect.bisect_right(times,anchor['t']+h*50000000)-1 for h in range(4)]
   if any(j not in row_by_index for j in indices):bad['missing_chunk_member']+=1;continue
   chunk=[row_by_index[j] for j in indices]
   if any(x['phase']!=anchor['phase'] for x in chunk):bad['ownership_or_frame_boundary']+=1;continue
   if anchor['t']+200000000>times[-1]:bad['no_terminal_coverage']+=1;continue
   if any(anchor['t']+h*50000000-x['t']>150000000 for h,x in enumerate(chunk)):bad['command_hold_too_long']+=1;continue
   body=[np.linalg.inv(anchor['tcp'])@x['target'] for x in chunk];port_targets=[np.linalg.inv(base_port)@x['target'] for x in chunk]
   for x,b in zip(chunk,body):
    recovered=anchor['tcp']@b;roundtrip_m=max(roundtrip_m,float(np.linalg.norm(recovered[:3,3]-x['target'][:3,3])));roundtrip_deg=max(roundtrip_deg,float((Rotation.from_matrix(recovered[:3,:3]).inv()*Rotation.from_matrix(x['target'][:3,:3])).magnitude()*180/np.pi))
   data=dict(schema='aic_verified_transport_local_chunk/v1',episode_index=episode_index,episode_id=audit.name,scene_sha256=r['scene_group_id'],split=split,frame=anchor['index'],sim_time=anchor['t']/1e9,episode_start_sim_time=times[0]/1e9,images=anchor['images'],image_sim_ns=anchor['image_stamps'],state=anchor['state'],task_vector=[0.,1.,*([1.,0.] if port_index==0 else [0.,1.]),*([0.]*6)],action_owner='teacher',phase=anchor['phase'],supervisor_retreat_in_bc=False,command_indices=indices,command_time_offsets_s=[0,.05,.1,.15],source_command_time_offsets_s=[(x['t']-anchor['t'])/1e9 for x in chunk],teacher_tcp_target_poses_base_frame=[pose(x['target']) for x in chunk],teacher_tcp_target_poses_body_frame=[pose(x) for x in body],teacher_tcp_target_poses_opening_frame=[pose(x) for x in port_targets],observed_tcp_pose_opening_frame=pose(np.linalg.inv(base_port)@anchor['tcp']),observed_sc_tip_pose_opening_frame=pose(np.linalg.inv(base_port)@anchor['tcp']@anchor['tip']),geometry_label_only=True,label_source='recorded_absolute_pose_command_or_FK_of_recorded_joint_target',official_tier3=score,admission_purpose='heldout_transport_only' if split=='validation' else ('verified_transport_prefix_only' if prefix_only else 'full_success_teacher_bc'))
   handles[anchor['phase']].write(json.dumps(data)+'\n');chunk_counts[anchor['phase']]+=1;counts[anchor['phase']]+=1
  episodes.append({**report,'target_port':port_index,'commands':len(commands),'admitted_chunks':dict(chunk_counts),'rejected_rows_or_chunks':dict(bad),'target_roundtrip_max_m':roundtrip_m,'target_roundtrip_max_deg':roundtrip_deg})
 for h in handles.values():h.close()
 train_groups={e['scene_group_id'] for e in episodes if e['split']=='train'}
 val_groups={e['scene_group_id'] for e in episodes if e['split']=='validation'}
 assert not train_groups & val_groups, 'Reset configuration leaks across train/validation'
 out=dict(schema='aic_transport_local_admission/v1',episodes=episodes,rejected_episodes=rejected,counts=dict(counts),split_policy='all repeated reset identities grouped; full-success episodes or explicitly audited transport-only prefixes for training; explicit measured route-only episodes for held-out transport validation; no reset overlap',label_frames={'transport':'TCP-body full target poses','local':'selected opening-frame full TCP target poses'},final_scenes_opened=False,actor_geometry_input=False)
 (a.output/'manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'episodes':len(episodes),'counts':dict(counts),'reset_groups':len(set(e['scene_group_id'] for e in episodes)),'rejected':len(rejected)}))
if __name__=='__main__':main()

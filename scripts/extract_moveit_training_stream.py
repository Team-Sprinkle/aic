#!/usr/bin/env python3
"""Extract immutable scored streams for command-target BC; requires official ROS runtime.
No image from outside the scored bag's wall-time window is admitted.
"""
import argparse,json,csv,bisect,hashlib
from pathlib import Path
from collections import defaultdict
import yaml
try:
 import rosbag2_py
except ImportError:
 rosbag2_py=None
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
JOINTS=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
def needed_transform(parent,child,task):
 selected={task['cable_name'],f"{task['cable_name']}/{task['plug_name']}_link"}
 return (child in ({'aic_world','tabletop','base_link','task_board','cable_1','cable_1/sc_tip_link'}|selected)
         or any(k in parent+' '+child for k in ['sc_port','sfp_port','nic_card_mount'])
         or any(k in parent+' '+child for k in ['camera/','cam_mount','ati/','gripper/','tool0']))
def stamp(msg):return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
def main():
 p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 bags=list((a.audit/'results').glob('bag_trial*'));assert len(bags)==1,bags;bag=bags[0]
 cfg=yaml.safe_load((a.audit/'eval_config.yaml').read_text());assert len(cfg['trials'])==1
 trial_id,trial=next(iter(cfg['trials'].items()));score=yaml.safe_load((a.audit/'results/scoring.yaml').read_text());assert trial_id in score
 task=next(iter(trial['tasks'].values()))
 selected_geometry_frames={task['cable_name'],f"{task['cable_name']}/{task['plug_name']}_link"}
 meta=yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information'];wall0=meta['starting_time']['nanoseconds_since_epoch'];wall1=wall0+meta['duration']['nanoseconds']
 streams=defaultdict(list);edges=defaultdict(list);commands=[]
 wanted={'/joint_states','/aic_controller/controller_state','/fts_broadcaster/wrench','/aic_controller/joint_commands','/aic_controller/pose_commands','/tf','/tf_static','/scoring/tf'}
 if rosbag2_py is not None:
  reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
  types={r.name:get_message(r.type) for r in reader.get_all_topics_and_types() if r.name in wanted}
  def messages():
   while reader.has_next():yield reader.read_next()
 else:
  # Rootless Docker can reach the per-UID kernel key limit after a successful
  # rollout. Read the same MCAP/C DR bytes with the host Python runtime, using
  # the installed ROS interface packages for deserialization.
  from mcap.reader import make_reader
  raw_file=next(bag.glob('*.mcap')).open('rb')
  reader=make_reader(raw_file)
  summary=reader.get_summary()
  types={c.topic:get_message(summary.schemas[c.schema_id].name)
         for c in summary.channels.values() if c.topic in wanted}
  def messages():
   for schema,channel,message in reader.iter_messages(topics=list(wanted)):
    yield channel.topic,message.data,message.log_time
 latest_controller_time=None;latest_controller_wall=None
 for topic,raw,wall in messages():
  if topic not in types:continue
  msg=deserialize_message(raw,types[topic])
  if topic in ['/tf','/tf_static','/scoring/tf']:
   for t in msg.transforms:
    parent=t.header.frame_id;child=t.child_frame_id
    needed=needed_transform(parent,child,task)
    if not needed:continue
    key=parent+'|'+child
    ts=0 if topic=='/tf_static' else t.header.stamp.sec*10**9+t.header.stamp.nanosec
    previous=edges[key][-1] if edges[key] else None
    if previous and ts-previous[0]<50000000:continue
    d=message_to_ordereddict(t.transform)
    # Keep selected plug/body timestamp heartbeats even while motion freezes.
    # Otherwise a valid terminal observation can look stale after a hold.
    if previous and d==previous[1] and child not in selected_geometry_frames:continue
    edges[key].append([ts,d])
   continue
  d=message_to_ordereddict(msg)
  if topic=='/aic_controller/joint_commands':
   if latest_controller_time is None:continue
   ts=latest_controller_time
  else:ts=stamp(msg)
  if topic=='/aic_controller/controller_state':latest_controller_time=ts;latest_controller_wall=wall
  if topic.endswith('joint_commands'):commands.append(dict(sim_ns=ts,wall_ns=wall,kind='joint',message=d,timestamp_source='latest_preceding_controller',timestamp_age_wall_ns=wall-latest_controller_wall))
  elif topic.endswith('pose_commands'):commands.append(dict(sim_ns=ts,wall_ns=wall,kind='pose',message=d))
  else:streams[topic].append([ts,d])
 cameras={k:[] for k in ['center','left','right']}
 for r in csv.DictReader((a.audit/'frame_manifest.csv').open()):
  if r['camera'] in cameras and wall0<=int(r['wall_ns'])<=wall1:
   cameras[r['camera']].append([int(r['sim_ns']),str(a.audit/'frames'/r['file'])])
 for v in [*streams.values(),*edges.values(),*cameras.values()]:v.sort(key=lambda r:r[0])
 task=next(iter(trial['tasks'].values()));reset_identity={'scene':trial['scene'],'robot':cfg.get('robot'),'task':{k:task[k] for k in ['plug_type','port_type','target_module_name','cable_name']}}
 group=hashlib.sha256(json.dumps(reset_identity,sort_keys=True).encode()).hexdigest()
 out=dict(schema='aic_scored_command_stream/v1',audit=str(a.audit),trial_id=trial_id,scene_group_id=group,reset_identity=reset_identity,official_tier3=score[trial_id]['tier_3']['score'],commands=sorted(commands,key=lambda r:r['sim_ns']),streams=dict(streams),edges=dict(edges),cameras=cameras,scored_wall_bounds_ns=[wall0,wall1],joint_order=JOINTS,task=task,bag=str(bag))
 interface_manifest=a.audit/'teacher_interface_manifest.json'
 if interface_manifest.exists():out['teacher_interface']=json.loads(interface_manifest.read_text())
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out)+'\n');print(json.dumps({'output':str(a.output),'commands':len(commands),'camera_frames':{k:len(v) for k,v in cameras.items()},'tier3':out['official_tier3'],'scene_group_id':group}))
if __name__=='__main__':main()

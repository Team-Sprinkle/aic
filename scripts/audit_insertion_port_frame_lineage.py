#!/usr/bin/env python3
"""Bounded training-frame audit; simulator geometry is read only for offline metrics."""
import argparse,bisect,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from audit_sfp_raw_bag_terminal_geometry import matrix,stored_matrix
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw
from audit_sc_port_targets import matrix as configured_matrix


def past(rows,stamp):
 i=bisect.bisect_right(rows,stamp,key=lambda r:r[0])-1
 if i<0:raise ValueError('No causal transform')
 return stored_matrix(rows[i][1])


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();selected={f:[] for f in ['sc','sfp']};seen=set()
 for line in (a.source/'rows.jsonl').open():
  r=json.loads(line)
  if r['split']!='validation' or r['episode'] in seen or len(selected[r['family']])>=10:continue
  selected[r['family']].append(r);seen.add(r['episode'])
  if all(len(x)==10 for x in selected.values()):break
 typ=get_message('tf2_msgs/msg/TFMessage');results=[]
 for row in sum(selected.values(),[]):
  d=json.loads(Path(row['source_stream']).read_text());bag=Path(d['bag']);
  if not bag.exists():bag=Path(row['source_stream']).parent/'results'/bag.name
  with next(bag.glob('*.mcap')).open('rb') as f:
   wa=None
   for _,_,m in make_reader(f).iter_messages(topics=['/tf_static']):
    for t in deserialize_message(m.data,typ).transforms:
     if (t.header.frame_id,t.child_frame_id)==('world','aic_world'):wa=matrix(t.transform)
  if wa is None:raise ValueError('Missing world calibration')
  e=d['edges'];stamp=round(row['time_s']*1e9);prefix='task_board/'+d['task']['target_module_name']
  if row['family']=='sc':
   base_world=np.linalg.inv(past(e['world|tabletop'],stamp)@past(e['tabletop|base_link'],stamp))
   truth=base_world@wa@past(e['aic_world|task_board'],stamp)@past(e['task_board|'+prefix],stamp)@past(e[prefix+'|'+prefix+'/sc_port_base_link'],stamp)@configured_matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
   truth_source='Same-episode past port TF and previously calibrated opening offset'
  else:
   # Legacy extracted SFP streams omit NIC transforms. Use the already
   # validated static YAML/CAD calibration, explicitly not a raw-TF claim.
   from build_two_family_port_projection_labels import sfp_ports
   calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
   card=int(d['task']['target_module_name'].rsplit('_',1)[1]);port=int(d['task']['port_name'].rsplit('_',1)[1])
   truth=next(m for c,p,m in sfp_ports(dict(trial_yaml=str(Path(d['audit'])/'eval_config.yaml'),trial_id=d['trial_id']),calibration) if (c,p)==(card,port))
   truth_source='Static scene YAML plus previously raw-validated SFP CAD/base calibration; legacy NIC TF absent'
  estimate=matrix_from_xyz_xyzw(row['state'][:7])@np.linalg.inv(matrix_from_pose9(row['predicted_port_tcp_pose9']));delta=np.linalg.inv(truth)@estimate
  results.append(dict(episode=row['episode'],family=row['family'],position_mm=float(np.linalg.norm(delta[:3,3])*1000),lateral_mm=float(np.linalg.norm(delta[:2,3])*1000),orientation_deg=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude())),source_stream=row['source_stream'],truth_source=truth_source))
 training=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/shared_full_causal_port_locks_v1/plan.json').read_text());runtime=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/transport_spec.json').read_text());lineage={k:dict(runtime_path=v,sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest(),identical_to_training=hashlib.sha256(Path(v).read_bytes()).hexdigest() in training['models'].values()) for k,v in runtime['bundle'].items()}
 report=dict(results=results,lineage=lineage,summary={f:{k:np.percentile([r[k] for r in results if r['family']==f],[50,95,100]).tolist() for k in ['position_mm','lateral_mm','orientation_deg']} for f in ['sc','sfp']},scope='First10complete held-out episodes per family; bounded frame/lineage diagnosis, not autonomous evaluation',geometry_metric_only=True,actors_not_modified=True)
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()

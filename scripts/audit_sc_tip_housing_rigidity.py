#!/usr/bin/env python3
"""Check label-frame attachment to visible SC housing at causal image times."""
import argparse,bisect,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from audit_sfp_raw_bag_terminal_geometry import matrix


def main():
 p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=json.loads((a.captures/'summary.json').read_text());typ=get_message('tf2_msgs/msg/TFMessage');v=list(map(float,ET.parse('aic_assets/models/SC Plug/model.sdf').getroot().find("model/link[@name='sc_tip_link']/pose").text.split()));expected=np.eye(4);expected[:3,3]=v[:3];expected[:3,:3]=Rotation.from_euler('xyz',v[3:]).as_matrix();results=[]
 for record in source['results']:
  run=Path(record['audit']);d=json.loads((run/'scored_stream.json').read_text());row=json.loads((run/'grasp_labels.json').read_text())[0];stamp=min(row['image_stamps_ns'].values());wall=int(Path(row['images']['center']).stem.split('_')[0]);cable=d['task']['cable_name'];keys=[(cable,cable+'/sc_plug_link'),(cable,cable+'/sc_tip_link')];latest={}
  with next((run/'results').glob('bag_*/*.mcap')).open('rb') as f:
   for _,_,msg in make_reader(f).iter_messages(topics=['/scoring/tf'],start_time=wall-250_000_000,end_time=wall+1):
    for tf in deserialize_message(msg.data,typ).transforms:
     key=(tf.header.frame_id,tf.child_frame_id);ts=tf.header.stamp.sec*1_000_000_000+tf.header.stamp.nanosec
     if key in keys and ts<=stamp and (key not in latest or ts>latest[key][0]):latest[key]=(ts,matrix(tf.transform))
  if len(latest)!=2 or any(stamp-latest[k][0]>4_000_000 for k in keys):raise ValueError('Missing synchronous past housing/tip labels')
  observed=np.linalg.inv(latest[keys[0]][1])@latest[keys[1]][1];delta=np.linalg.inv(expected)@observed;results.append(dict(case=record['case']['case'],translation_discrepancy_mm=float(np.linalg.norm(delta[:3,3])*1000),orientation_discrepancy_deg=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude())),TF_age_ms=[(stamp-latest[k][0])/1e6 for k in keys],observed_housing_to_tip=observed.tolist()))
 report=dict(expected_housing_to_tip=expected.tolist(),results=results,episodes=len(results),summary={k:np.percentile([r[k] for r in results],[50,95,100]).tolist() for k in ['translation_discrepancy_mm','orientation_discrepancy_deg']},scope='Offline oracle label-frame audit only; no policy geometry input')
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()

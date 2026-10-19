#!/usr/bin/env python3
"""Image-time SFP grasp labels from qualified raw episodes, without GT crops.

Select early quiet observations from the immutable command derivative. Actual
plug transforms provide supervision only. Native crop centres will be fixed
from training data in a later materialization stage, never from held-out poses.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from audit_sfp_raw_bag_terminal_geometry import matrix, stored_matrix
from prepare_verified_insertion_command_rows import message_pose

CAMS=('center','left','right')
# Virtual, frame-attached references; these are not claimed physical corners.
REFERENCE=np.array([[0,0,0],[-.0064,-.0035,-.0025],[.0064,-.0035,-.0025],
                    [-.0064,.0035,-.0025],[.0064,.0035,-.0025]])


def label(reader,typ,d,row,world_aic,world_base,optics):
    stamp=min(row['image_stamps_ns'].values())
    wall=int(Path(row['images']['center']).stem.split('_')[0])
    if wall<1_000_000_000_000_000_000:raise ValueError('No capture wall timestamp')
    cable=d['task']['cable_name'];tip=f"{cable}/{d['task']['plug_name']}_link"
    needed={('aic_world',cable),(cable,tip)};observed={k:[] for k in needed}
    # JPEG capture can lag the corresponding TF by >250ms of wall time
    # under NAS/CPU contention. Search more past log time, while retaining
    # the exact same strict simulation-time freshness checks below. The
    # upper wall cutoff still forbids later/post-reset messages.
    for _,_,message in reader.iter_messages(topics=['/tf','/scoring/tf'],start_time=wall-2_000_000_000,end_time=wall+1):
        for tf in deserialize_message(message.data,typ).transforms:
            key=tf.header.frame_id,tf.child_frame_id;ts=tf.header.stamp.sec*1_000_000_000+tf.header.stamp.nanosec
            if key in needed and ts<=stamp:observed[key].append((ts,matrix(tf.transform)))
    if any(not x for x in observed.values()):raise ValueError('Missing past image-time plug TF')
    common=min(max(t for t,_ in x) for x in observed.values());transforms={};ages=[]
    for k,rows in observed.items():
        ts,t=max((x for x in rows if x[0]<=common),key=lambda x:x[0]);transforms[k]=t;ages.append(common-ts)
    if max(ages)>2_000_000 or stamp-common>10_000_000:raise ValueError('Stale/asynchronous image-time plug TF')
    controllers=d['streams']['/aic_controller/controller_state'];j=bisect_right(controllers,common,key=lambda x:x[0])-1
    if j<0 or common-controllers[j][0]>4_000_000:raise ValueError('Stale same-time TCP')
    base_tip=np.linalg.inv(world_base)@world_aic@transforms[('aic_world',cable)]@transforms[(cable,tip)]
    body=np.linalg.inv(message_pose(controllers[j][1]['tcp_pose']))@base_tip
    tcp_points=REFERENCE@body[:3,:3].T+body[:3,3];pixels={}
    from evaluate_sc_native_triangulation import FOCAL
    for c in CAMS:
        inv=np.linalg.inv(optics[c]);q=tcp_points@inv[:3,:3].T+inv[:3,3]
        if min(q[:,2])<=0:raise ValueError('Grasp references behind a camera')
        pixels[c]=(q[:,:2]/q[:,2:]*FOCAL+[576,512]).tolist()
    return dict(family='sfp',episode=row['episode'],source_episode_name=row['source_episode_name'],
        scene=row['scene'],split=row['split'],time_s=row['time_s'],images=row['images'],
        image_stamps_ns=row['image_stamps_ns'],body_tip_matrix=body.tolist(),
        body_translation_m=body[:3,3].tolist(),body_quat_xyzw=Rotation.from_matrix(body[:3,:3]).as_quat().tolist(),
        projected_virtual_reference_pixels=pixels,geometry_label_only=True,
        label_stamp_ns=common,image_to_label_age_ms=(stamp-common)/1e6,
        TCP_to_label_age_ms=(common-controllers[j][0])/1e6,
        source_stream=row['source_stream'],post_completion_excluded=True,
        sample_selection='early quiet public-state observations; no target pose or model-output selection',
        GT_crop_selection=False)


def qa(row,output):
    columns=[]
    for c in CAMS:
        image=cv2.imread(row['images'][c]);assert image.shape[:2]==(1024,1152)
        for i,p in enumerate(row['projected_virtual_reference_pixels'][c]):
            cv2.circle(image,tuple(np.rint(p).astype(int)),5,(0,255,0),2)
            cv2.putText(image,str(i),tuple(np.rint(np.asarray(p)+[6,-6]).astype(int)),0,.5,(0,255,0),1)
        cv2.putText(image,c+' GT LABEL QA ONLY',(15,35),0,.8,(0,255,0),2)
        columns.append(cv2.resize(image,(576,512)))
    cv2.imwrite(str(output),np.concatenate(columns,axis=1))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commands',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    cv2.setNumThreads(2);typ=get_message('tf2_msgs/msg/TFMessage')
    sources={x['source_stream']:x for x in map(json.loads,a.sources.open()) if x['family']=='sfp'}
    plan=json.loads((a.commands/'summary.json').read_text());episodes=[e for e in plan['episodes'] if e['family']=='sfp']
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text());optics={c:np.array(t) for c,t in cal['tcp_to_optical'].items()}
    progress=a.output/'progress.jsonl';done=[json.loads(l) for l in progress.open()] if a.resume and progress.exists() else [];seen={x['source'] for x in done}
    for index,e in enumerate(episodes):
        if e['source'] in seen:continue
        rows=[json.loads(l) for l in (a.commands/e['rows_file']).open()]
        quiet=[x for x in rows if np.linalg.norm(x['state'][7:10])<=.010 and np.linalg.norm(x['state'][10:13])<=.10][:30]
        chosen=[quiet[i] for i in np.linspace(0,len(quiet)-1,min(10,len(quiet)),dtype=int)] if quiet else []
        s=sources[e['source']];d=json.loads(Path(e['source']).read_text());wb=stored_matrix(d['edges']['world|tabletop'][0][1])@stored_matrix(d['edges']['tabletop|base_link'][0][1]);results=[];errors=Counter();bag=next((Path(s['source_audit'])/'results').glob('bag_*/*.mcap'))
        with bag.open('rb') as f:
            reader=make_reader(f);wa=None
            for _,_,message in reader.iter_messages(topics=['/tf_static']):
                for tf in deserialize_message(message.data,typ).transforms:
                    if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):wa=matrix(tf.transform)
            if wa is None:raise ValueError('Missing explicit world/aic_world calibration')
            for row in chosen:
                try:results.append(label(reader,typ,d,row,wa,wb,optics))
                except (ValueError,KeyError) as exc:errors[str(exc)]+=1
        filename=f'{index:03d}_{e["episode"]}.jsonl'
        (a.output/filename).write_text(''.join(json.dumps(x)+'\n' for x in results))
        report=dict(source=e['source'],episode=e['episode'],split=e['split'],rows=len(results),rows_file=filename,rejected=dict(errors));done.append(report)
        with progress.open('a') as f:f.write(json.dumps(report)+'\n')
        if index<2 and results:qa(results[len(results)//2],a.output/f'first{index+1}_label_QA.jpg')
        print(json.dumps(dict(completed=len(done),planned=len(episodes),last=report)),flush=True)
        if index==1:
            wait=time.monotonic()
            while not (a.output/'first2_QA_passed.json').exists():
                if time.monotonic()-wait>3600:raise TimeoutError('Native label visual QA must precede continuation')
                time.sleep(5)
    summary=dict(source_episodes=len(done),sources_with_at_least3_rows=sum(x['rows']>=3 for x in done),rows=sum(x['rows'] for x in done),episodes=done,virtual_reference_points_m=REFERENCE.tolist(),tcp_to_optical={c:t.tolist() for c,t in optics.items()},GT_crop_selection=False,geometry_training_label_only=True,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()

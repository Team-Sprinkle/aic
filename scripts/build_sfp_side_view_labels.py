#!/usr/bin/env python3
"""Join three measured side-view events to original RGB; geometry is scoring-only."""
import argparse
from bisect import bisect_left,bisect_right
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import CAMERAS,from_controller,sfp_ports


STAGES=('baseline_view','positive_view','negative_view')
EVENT_FILE='side_view_events.json'
SCHEMA='sfp_side_view_event_join/v1'
CASES=(('card1_port0_v2',1,0,'card1_port0'),
       ('card2_port0_v3',2,0,'card2_port0'),
       ('card0_port1_v3',0,1,'card0_port1'),
       ('card4_port1_v3',4,1,'card4_port1'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source-plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous labels')
    source={x['name']:x for x in json.loads(args.source_plan.read_text())['scenes']}
    cal=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    optics_json=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(optics_json['tcp_to_optical'][c],float) for c in CAMERAS}
    rows=[];summary=[]
    for folder,card,port,source_name in CASES:
        audit=args.root/folder
        event_data=json.loads((audit/EVENT_FILE).read_text())
        events={x['stage']:x for x in event_data['events'] if x['stage'] in STAGES}
        if set(events)!=set(STAGES):
            summary.append(dict(episode=source_name,complete=False,stages=sorted(events)));continue
        stream=json.loads((audit/'scored_stream.json').read_text())
        ports=sfp_ports({'trial_yaml':str(audit/'eval_config.yaml'),
                         'trial_id':stream['trial_id']},cal)
        controllers=stream['streams']['/aic_controller/controller_state']
        episode_rows=[]
        for stage in STAGES:
            event=events[stage];stamp=round(event['sim_time_s']*1e9)
            images={};camera_gaps={}
            for cam in CAMERAS:
                seq=stream['cameras'][cam]
                j=bisect_left(seq,stamp,key=lambda x:x[0]);j=min(max(j,0),len(seq)-1)
                if j>0 and abs(seq[j-1][0]-stamp)<abs(seq[j][0]-stamp):j-=1
                camera_gaps[cam]=abs(seq[j][0]-stamp)/1e6
                if camera_gaps[cam]>75:raise ValueError(f'{folder} {stage} {cam}: stale camera')
                virtual=seq[j][1]
                image=audit/virtual[len('/audit/'):] if virtual.startswith('/audit/') else Path(virtual)
                if not image.is_file():raise FileNotFoundError(image)
                images[cam]=str(image)
            j=bisect_right(controllers,stamp,key=lambda x:x[0])-1
            if j<0 or stamp-controllers[j][0]>100_000_000:
                raise ValueError(f'{folder} {stage}: stale controller')
            state=from_controller(controllers[j][1]);tcp=pose_matrix(state)
            targets=[]
            for c,p,m in ports:
                targets.append(dict(identity=f'sfp_card_{c}_port_{p}',
                                    position_base_m=m[:3,3].tolist(),
                                    orientation_base_xyzw=Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                    cameras={view:project(m,tcp@optics[view]) for view in CAMERAS}))
            row=dict(family='sfp',episode=source_name,stage=stage,
                     scene_group=source[source_name]['reset_sha256'],split='validation',
                     time_s=event['sim_time_s'],state=state.tolist(),images=images,
                     image_kind='native_1152x1024_rgb_jpeg',targets=targets,
                     selected_identity=f'sfp_card_{card}_port_{port}',
                     card_count=source[source_name]['card_count'],
                     camera_gap_ms=camera_gaps,force_n=event['force_n'],
                     oracle_contact_pairs=event['contact_pairs'],
                     measured_tcp_base_m=event['tcp_base_m'],
                     geometry_training_label_only=True,
                     source_scored_stream=str(audit/'scored_stream.json'))
            rows.append(row);episode_rows.append(row)
        summary.append(dict(episode=source_name,complete=True,stages=list(STAGES),
                            camera_gap_ms_max=max(max(r['camera_gap_ms'].values()) for r in episode_rows)))
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    report=dict(schema=SCHEMA,rows=len(rows),episodes=summary,
                simulator_geometry_label_only=True,final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Project actual board-visual screw centers into saved native SFP RGB.

All geometry here is label generation / read-only audit. It cannot select a
runtime crop or be passed to an autonomous policy.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import project
from build_sfp_port_tcp_waypoints import transform


CAMS=('center','left','right')
# Centers of the two lower screw heads in base_visual.glb, board local XYZ.
SCREWS=((-0.145,-0.2075,0.010),(0.145,-0.2075,0.010))


def key(row):
    return row['scene_group'],row['episode'],round(float(row['time_s']),6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail-rows',type=Path,required=True)
    p.add_argument('--source-rows',type=Path,required=True)
    p.add_argument('--review-manifest',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--sfp-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    a.output.mkdir(parents=True)
    rail=[json.loads(line) for line in a.rail_rows.open()]
    sources={key(r):r for r in (json.loads(line) for line in a.source_rows.open())}
    selections=json.loads(a.review_manifest.read_text())
    camera_cal=json.loads(a.camera_calibration.read_text())
    optics={cam:np.asarray(camera_cal['tcp_to_optical'][cam],float) for cam in CAMS}
    sfp=json.loads(a.sfp_calibration.read_text())
    base_world=np.linalg.inv(matrix(sfp['world_to_base']))
    scenes={};records=[]
    for row in rail:
        if row['split']!='validation' or row['autonomous_actor_diagnostic']:continue
        source=sources[key(row)]
        if row['images']!=source['images']:raise ValueError('Camera provenance mismatch')
        scene=row['scene_group']
        if scene not in scenes:
            trial=next(iter(yaml.safe_load(Path(row['trial_yaml_label_only']).read_text())['trials'].values()))
            bp=trial['scene']['task_board']['pose']
            board_world=transform([bp[k] for k in ('x','y','z')],
                                  [bp[k] for k in ('roll','pitch','yaw')])
            scenes[scene]=base_world@board_world
        camera={cam:pose_matrix(source['state'])@optics[cam] for cam in CAMS}
        points=[]
        for xyz in SCREWS:
            q=np.eye(4);q[:3,3]=xyz
            points.append(scenes[scene]@q)
        projections={cam:[project(q,camera[cam]) for q in points] for cam in CAMS}
        records.append({'scene_group':scene,'episode':row['episode'],
                        'time_s':row['time_s'],'images':row['images'],
                        'screw_projections_training_label_only':projections,
                        'corner_projections_training_label_only':row['corner_projections_training_label_only']})
    indexed={key(r):r for r in records}
    if len({r['scene_group'] for r in records})!=10:raise ValueError('Expected ten development groups')
    panels=[]
    for selection in selections:
        r=indexed[key(selection)]
        cam=selection['camera'];end=selection['endpoint']
        screw=r['screw_projections_training_label_only'][cam][end]
        corner=r['corner_projections_training_label_only'][cam][end]
        panels.append({**selection,'screw_projection':screw,
                       'old_collision_corner_projection':corner,
                       'offset_px':float(np.linalg.norm(np.asarray(screw[:2])-corner[:2]))})
    for sheet in range(2):
        canvas=np.zeros((3*400,5*580,3),np.uint8)
        for local,item in enumerate(panels[sheet*15:(sheet+1)*15]):
            source=cv2.imread(item['image']);s=np.asarray(item['screw_projection'][:2])
            c=np.asarray(item['old_collision_corner_projection'][:2])
            full=cv2.resize(source,(384,341),interpolation=cv2.INTER_AREA)
            crop=cv2.getRectSubPix(source,(160,160),tuple(float(x) for x in s))
            for xy,color in ((s,(0,255,0)),(c,(0,0,255))):
                cv2.circle(full,tuple(np.rint(xy/3).astype(int)),5,color,2)
                v=np.rint(xy-s+79.5).astype(int)
                if np.all(v>=0) and np.all(v<160):cv2.circle(crop,tuple(v),5,color,2)
            tile=np.zeros((400,580,3),np.uint8)
            tile[35:376,5:389]=full;tile[35:195,400:560]=crop
            cv2.putText(tile,f'#{item["panel"]:02d} {item["camera"]} screw{item["endpoint"]}',
                        (7,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
            cv2.putText(tile,f'offset {item["offset_px"]:.1f}px',(400,220),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            canvas[(local//5)*400:(local//5+1)*400,(local%5)*580:(local%5+1)*580]=tile
        cv2.imwrite(str(a.output/f'screw_sheet_{sheet+1}.jpg'),canvas,
                    [cv2.IMWRITE_JPEG_QUALITY,94])
    (a.output/'review_manifest.json').write_text(json.dumps(panels,indent=2)+'\n')
    with (a.output/'development_screw_projections.jsonl').open('w') as f:
        for row in records:f.write(json.dumps(row)+'\n')
    by_scene=defaultdict(list)
    for row in records:by_scene[row['scene_group']].append(row)
    coverage={}
    for scene,rr in by_scene.items():
        rr.sort(key=lambda x:x['time_s']);early=[r for r in rr if r['time_s']-rr[0]['time_s']<=3]
        coverage[scene]={f'{cam}_{end}':any(r['screw_projections_training_label_only'][cam][end][3]
                                for r in early) for cam in CAMS for end in (0,1)}
    summary={'schema':'sfp_visual_mesh_screw_projection/v1','development_groups':10,
             'review_panels':len(panels),'screw_board_xyz_m':SCREWS,
             'cad_window_only_for_audit':True,'runtime_geometry_input':False,
             'early_projected_inframe_by_scene':coverage,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'panels':len(panels),'median_old_to_screw_px':
          float(np.median([r['offset_px'] for r in panels]))}))


if __name__=='__main__':main()

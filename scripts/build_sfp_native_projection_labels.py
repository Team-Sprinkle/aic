#!/usr/bin/env python3
"""Project every present SFP opening into scored native RGB replays.

Projection and in-frame flags are training labels, not true visibility. The
selected candidate is identified by the public task code. No action from the
source agent collection enters this dataset.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import CAMERAS,from_controller,sfp_ports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--sfp-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    manifest=json.loads((a.selection/'collection_manifest.json').read_text())['rows']
    cam_cal=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(cam_cal['tcp_to_optical'][c],float) for c in CAMERAS}
    sfp_cal=json.loads(a.sfp_calibration.read_text())
    rows=[];rejected=Counter();episodes=[];scenes={'train':set(),'validation':set()}
    def host_image_path(value, case):
        # The scored-bag extractor runs in Docker and records /repo/... paths.
        # Training and evaluation scripts run on the host from this checkout.
        path=Path(value)
        if path.parts[:2]==('/', 'repo'):
            path=Path.cwd().joinpath(*path.parts[2:])
        elif path.parts[:3]==('/', 'audit', 'frames'):
            # Rootless scored replays mount each saved audit directory at
            # /audit. Resolve against that episode, never by basename search.
            path=case/'frames'/path.name
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    for item in manifest:
        case=a.selection/item['case'];path=case/'scored_stream.json'
        if not path.exists():
            if a.allow_incomplete:continue
            raise FileNotFoundError(path)
        stream=json.loads(path.read_text());bounds=stream['scored_wall_bounds_ns']
        scene=stream['scene_group_id'];split=item['split'];scenes[split].add(scene)
        task=stream['task'];card=int(task['target_module_name'].rsplit('_',1)[-1])
        port=int(task['port_name'].rsplit('_',1)[-1])
        identity=f'sfp_card_{card}_port_{port}'
        trial={'trial_yaml':str(case/'eval_config.yaml'),'trial_id':stream['trial_id']}
        ports=sfp_ports(trial,sfp_cal)
        controllers=stream['streams']['/aic_controller/controller_state']
        cameras=stream['cameras'];count=0
        for t,image in cameras['center']:
            try:wall=int(Path(image).stem.split('_')[0])
            except (ValueError,IndexError):rejected['bad_filename']+=1;continue
            if not bounds[0]<=wall<=bounds[1]:rejected['outside_scored_window']+=1;continue
            j=bisect_right(controllers,t,key=lambda r:r[0])-1
            if j<0 or t-controllers[j][0]>100_000_000:
                rejected['stale_controller']+=1;continue
            images={};valid=True
            for c in CAMERAS:
                entries=cameras[c];k=bisect_right(entries,t,key=lambda r:r[0])-1
                if k<0 or t-entries[k][0]>100_000_000:valid=False;break
                images[c]=host_image_path(entries[k][1],case)
            if not valid:rejected['unsynchronized_camera']+=1;continue
            state=from_controller(controllers[j][1]);tcp=pose_matrix(state)
            targets=[]
            for cand_card,cand_port,m in ports:
                targets.append({'identity':f'sfp_card_{cand_card}_port_{cand_port}',
                                'position_base_m':m[:3,3].tolist(),
                                'orientation_base_xyzw':Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                'cameras':{c:project(m,tcp@optics[c]) for c in CAMERAS}})
            rows.append({'family':'sfp','episode':item['case'],'scene_group':scene,
                         'split':split,'time_s':t/1e9,'state':state.tolist(),
                         'images':images,'image_kind':'native_1152x1024_rgb_jpeg',
                         'source_stream':str(path),'targets':targets,
                         'selected_identity':identity,'card_count':item['card_count'],
                         'expert_action_source':False,'tier3':stream['official_tier3'],
                         'geometry_training_label_only':True})
            count+=1
        episodes.append({'case':item['case'],'split':split,'rows':count,
                         'scene_group':scene,'tier3':stream['official_tier3']})
    if scenes['train']&scenes['validation']:raise ValueError('Scene/reset leakage')
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as file:
        for row in rows:file.write(json.dumps(row,separators=(',',':'))+'\n')
    summary={'schema':'aic_sfp_native_projection_labels/v1','rows':len(rows),
             'episodes':episodes,'scene_groups':{k:len(v) for k,v in scenes.items()},
             'rejected':dict(rejected),'true_visibility_labels':False,
             'native_crop_available':True,'source_actions_used':False,
             'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'episodes':len(episodes),
                      'scene_groups':summary['scene_groups'],'rejected':dict(rejected)}))


if __name__=='__main__':main()

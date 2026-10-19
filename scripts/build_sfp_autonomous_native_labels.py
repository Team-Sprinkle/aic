#!/usr/bin/env python3
"""Sample saved autonomous SFP RGB/state for posthoc perception diagnosis.

The task request chooses a label identity. Simulator port geometry labels the
saved images only; it does not choose sampling times or candidate crops.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import CAMERAS, sfp_ports


def image_at(run, camera, second):
    paths = list((run/'frames').glob(f'*_{camera}.jpg'))
    if not paths:
        raise FileNotFoundError(f'No {camera} frames: {run}')
    selected = min(paths, key=lambda x: abs(int(x.stem.split('_')[1])/1e9-second))
    stamp = int(selected.stem.split('_')[1])/1e9
    return selected, abs(stamp-second)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--camera-calibration', type=Path, required=True)
    p.add_argument('--sfp-calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-second', type=int, default=4)
    p.add_argument('--last-second', type=int, default=40)
    p.add_argument('--stride-second', type=int, default=1)
    a = p.parse_args()
    if a.output.exists():
        p.error(f'Output exists: {a.output}')
    cal = json.loads(a.camera_calibration.read_text())
    optics = {c: np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    sfp_cal = json.loads(a.sfp_calibration.read_text())
    rows = []
    omitted = Counter()
    scenes = set()
    for run in a.run:
        config = yaml.safe_load((run/'eval_config.yaml').read_text())
        trial = next(iter(config['trials']))
        task = next(iter(config['trials'][trial]['tasks'].values()))
        card = int(task['target_module_name'].rsplit('_',1)[-1])
        port = int(task['port_name'].rsplit('_',1)[-1])
        selected = f'sfp_card_{card}_port_{port}'
        board = config['trials'][trial]['scene']['task_board']
        count = sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
        scene = hashlib.sha256((run/'eval_config.yaml').read_bytes()).hexdigest()
        if scene in scenes:
            raise ValueError(f'Duplicate scene config: {run}')
        scenes.add(scene)
        ports = sfp_ports({'trial_yaml':str(run/'eval_config.yaml'),
                           'trial_id':trial},sfp_cal)
        trace = [json.loads(line) for line in (run/'runtime_trace.jsonl').open()]
        proposals = [r for r in trace if r.get('event')=='proposal'
                     and len(r.get('state',[]))==32]
        if not proposals:
            raise ValueError(f'No causal actor states: {run}')
        for second in range(a.first_second,a.last_second+1,a.stride_second):
            state_row = min(proposals,key=lambda r:abs(r['sim_time']-second))
            if abs(state_row['sim_time']-second)>.15:
                omitted['state_age']+=1
                continue
            images={};ages={}
            for c in CAMERAS:
                path,age=image_at(run,c,second)
                images[c]=str(path.resolve())
                ages[c]=age
            if max(ages.values())>.075:
                omitted['camera_age']+=1
                continue
            state=state_row['state'][:7]
            tcp=pose_matrix(state)
            targets=[]
            for cand_card,cand_port,m in ports:
                targets.append({'identity':f'sfp_card_{cand_card}_port_{cand_port}',
                                'position_base_m':m[:3,3].tolist(),
                                'orientation_base_xyzw':Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                'cameras':{c:project(m,tcp@optics[c]) for c in CAMERAS}})
            rows.append({'family':'sfp','episode':run.name,'scene_group':scene,
                         'split':'validation','time_s':float(second),
                         'sampled_actor_sim_time_s':state_row['sim_time'],
                         'state':state,'images':images,'image_age_s':ages,
                         'image_kind':'native_1152x1024_rgb_jpeg',
                         'targets':targets,'selected_identity':selected,
                         'card_count':count,'geometry_training_label_only':True,
                         'expert_action_source':False,'autonomous_actor':True,
                         'source_run':str(run)})
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,separators=(',',':'))+'\n')
    summary={'schema':'sfp_autonomous_native_labels/v1','rows':len(rows),
             'runs':[str(x) for x in a.run],'scene_groups':len(scenes),
             'sampling_seconds':[a.first_second,a.last_second,a.stride_second],
             'omitted':dict(omitted),'geometry_actor_input':False,
             'true_visibility_labels':False,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main()

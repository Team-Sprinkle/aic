#!/usr/bin/env python3
"""Diagnose observability at the SC tilt-hold to return stage boundary."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image


def pixels(path):
    with Image.open(path) as f:return np.asarray(f.convert('RGB').resize((64,64)),dtype=np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=json.loads(a.spec.read_text())
    episodes={e['case']:e for e in spec['episodes'] if e['split']=='validation'}
    grouped=defaultdict(list)
    with a.rows.open() as f:
        for line in f:
            r=json.loads(line)
            if r['episode_id'] in episodes:grouped[r['episode_id']].append(r)
    report=[]
    for case,e in sorted(episodes.items()):
        plan=json.loads((Path(e['insert_plan'])/'moveit_joint_trajectory.json').read_text())['waypoints']
        start_return=next(float(w['timestamp']) for w in plan
                          if w['diagnostics']['segment']=='return_toward_port')
        rows=sorted(grouped[case],key=lambda r:r['sim_time'])
        before=[r for r in rows if r['sim_time']-r['episode_start_sim_time']<start_return]
        after=[r for r in rows if r['sim_time']-r['episode_start_sim_time']>=start_return]
        assert before and after
        last=before[-1];first=after[0]
        image_mad={cam:float(np.abs(pixels(last['images'][cam])-pixels(first['images'][cam])).mean())
                   for cam in ('center','left','right')}
        measured_mm=float(np.linalg.norm(np.asarray(last['state'][:3])-
                                         np.asarray(first['state'][:3]))*1e3)
        command_mm=float(np.linalg.norm(np.asarray(last['teacher_tcp_target_poses_base_frame'][0][:3])-
                                        np.asarray(first['teacher_tcp_target_poses_base_frame'][0][:3]))*1e3)
        prior=[r for r in before if last['sim_time']-1<=r['sim_time']<=last['sim_time']]
        prior_motion_mm=float(np.linalg.norm(np.asarray(prior[0]['state'][:3])-
                                             np.asarray(prior[-1]['state'][:3]))*1e3)
        report.append(dict(case=case,card_count=e['card_count'],port=e['selected_port'],
                           before_sim_s=last['sim_time'],after_sim_s=first['sim_time'],
                           transition_gap_ms=(first['sim_time']-last['sim_time'])*1e3,
                           camera_pixel_mad_64=image_mad,measured_tcp_step_mm=measured_mm,
                           command_step_mm=command_mm,prior_one_second_tcp_motion_mm=prior_motion_mm,
                           before_images=last['images'],after_images=first['images']))
    def quantile(key):
        x=np.asarray([r[key] for r in report])
        return dict(median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(max(x)))
    result=dict(schema='aic_temporal_tilt_boundary_diagnosis/v1',episodes=len(report),
                measured_tcp_step_mm=quantile('measured_tcp_step_mm'),
                command_step_mm=quantile('command_step_mm'),
                prior_one_second_tcp_motion_mm=quantile('prior_one_second_tcp_motion_mm'),
                center_pixel_mad_64=dict(median=float(np.median([r['camera_pixel_mad_64']['center'] for r in report])),
                                         p95=float(np.quantile([r['camera_pixel_mad_64']['center'] for r in report],.95))),
                cases=report,final_scenes_opened=False,rl_started=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()

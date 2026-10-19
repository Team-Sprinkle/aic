#!/usr/bin/env python3
"""Relabel verified local teacher targets into a causal RGB-estimated port frame.

Teacher geometry is used only to convert recorded complete TCP targets between
frames. The actor input pose comes solely from the frozen RGB estimator and
measured robot state. No true port geometry enters actor observations.
"""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix


def pose9(matrix):
    return [*matrix[:3,3].tolist(),*matrix[:3,:2].T.reshape(-1).tolist()]


def pose7_matrix(value):return pose_matrix(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sfp-source',type=Path,required=True)
    p.add_argument('--sc-source',type=Path,required=True)
    p.add_argument('--perception',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sfp-local-start-z-m',type=float,default=-.13)
    p.add_argument('--max-prediction-age-s',type=float,default=.6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    poses=defaultdict(list)
    for item in map(json.loads,a.perception.open()):
        poses[item['episode']].append(item)
    times={key:[item['time_s'] for item in sorted(value,key=lambda p:p['time_s'])]
           for key,value in poses.items()}
    poses={key:sorted(value,key=lambda p:p['time_s']) for key,value in poses.items()}
    counters=Counter();errors=defaultdict(list);episodes=defaultdict(set)
    with (a.output/'rows.jsonl').open('w') as out:
        for family,path in [('sfp',a.sfp_source),('sc',a.sc_source)]:
            for source in map(json.loads,path.open()):
                true_pose=(source['observed_tcp_pose_port_frame'] if family=='sfp'
                           else source['observed_tcp_pose_opening_frame'])
                if family=='sfp' and true_pose[2]<a.sfp_local_start_z_m:
                    counters['sfp_before_buffered_local_start']+=1;continue
                episode=(source.get('episode_id',f"sfp_{source['episode_index']}")
                         if family=='sfp' else source['episode_id'])
                t=float(source['timestamp_s'] if family=='sfp' else source['sim_time'])
                if episode not in poses:
                    counters['missing_episode_perception']+=1;continue
                j=bisect_right(times[episode],t)-1
                if j<0 or t-poses[episode][j]['time_s']>a.max_prediction_age_s:
                    counters['stale_or_missing_perception']+=1;continue
                estimate=poses[episode][j]
                if estimate['position_base_m'] is None or estimate['orientation_base_xyzw'] is None:
                    counters['insufficient_visible_views']+=1;continue
                observed_base_tcp=pose7_matrix(source['state'][:7])
                true_port_tcp=pose7_matrix(true_pose)
                true_base_port=observed_base_tcp@np.linalg.inv(true_port_tcp)
                pred_base_port=pose7_matrix([*estimate['position_base_m'],*estimate['orientation_base_xyzw']])
                pred_port_tcp=np.linalg.inv(pred_base_port)@observed_base_tcp
                teacher_targets=(source['teacher_tcp_target_poses_port_frame'] if family=='sfp'
                                 else source['teacher_tcp_target_poses_opening_frame'])
                pred_targets=[pose9(np.linalg.inv(pred_base_port)@true_base_port@pose7_matrix(target))
                              for target in teacher_targets]
                if len(pred_targets)!=4:raise ValueError('Four complete recorded target poses required')
                error=(np.linalg.inv(true_base_port)@pred_base_port)[:3,3]
                errors[(family,source['split'])].append(float(np.linalg.norm(error)*1000))
                record={'schema':'aic_predicted_port_local_bc/v1','family':family,'episode':episode,
                        'episode_index':source['episode_index'],'scene':source['scene_sha256'],
                        'split':source['split'],'frame':source['frame_index'] if family=='sfp' else source['frame'],
                        'time_s':t,'images':source.get('images'),'image_kind':source.get('image_kind'),
                        'cache_index':source.get('cache_frame_index'),
                        'state':source['state'],'task':source['task_vector'],
                        'predicted_port_tcp_pose9':pose9(pred_port_tcp),
                        'teacher_tcp_targets_predicted_port_pose9':pred_targets,
                        'supervision_only_true_tcp_axial_m':true_pose[2],
                        'perception_label_row':estimate['label_row'],
                        'perception_age_s':t-estimate['time_s'],
                        'perception_accepted_views':len(estimate['accepted_views']),
                        'perception_reprojection_residual_px':estimate['reprojection_residual_px'],
                        'true_geometry_actor_input':False,'recorded_complete_targets':True}
                out.write(json.dumps(record,separators=(',',':'))+'\n')
                counters[(family,source['split'])]+=1;episodes[(family,source['split'])].add(episode)
    summary={'schema':'aic_predicted_port_local_bc_dataset/v1','source':str(a.perception),
             'rows':{str(k):v for k,v in counters.items()},
             'episodes':{str(k):len(v) for k,v in episodes.items()},
             'training_only_port_estimate_error_mm':{str(k):{'median':float(np.median(v)),
                    'p95':float(np.quantile(v,.95))} for k,v in errors.items()},
             'sfp_local_start_z_m':a.sfp_local_start_z_m,
             'true_geometry_actor_input':False,'teacher_targets_reexpressed_in_predicted_frame':True,
             'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()

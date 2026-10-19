#!/usr/bin/env python3
"""Posthoc scored SFP handoff comparison; scene geometry is never actor input."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_two_family_port_projection_labels import sfp_ports, matrix
from shared_port_tcp_geometry import matrix_from_pose9


def stat(values):
    return {'n':len(values),'p50':float(np.median(values)),
            'p95':float(np.quantile(values,.95)),
            'p99':float(np.quantile(values,.99))} if values else {'n':0}


def summarize(name,audit,calibration):
    manifest=json.loads((audit/'eval_manifest.json').read_text())
    cfg=yaml.safe_load((audit/'eval_config.yaml').read_text())
    trial_id=next(iter(cfg['trials']))
    trial=cfg['trials'][trial_id]
    task=next(iter(trial['tasks'].values()))
    card=int(task['target_module_name'].rsplit('_',1)[-1])
    port=int(task['port_name'].rsplit('_',1)[-1])
    calibration_data=json.loads(calibration.read_text())
    ports=sfp_ports({'trial_yaml':str(audit/'eval_config.yaml'),'trial_id':trial_id},
                    calibration_data)
    base_port=next(m for c,p,m in ports if c==card and p==port)
    port_base=np.linalg.inv(base_port)
    entries=[json.loads(line) for line in (audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in entries if r['event']=='proposal']
    perceptions=[r for r in entries if r['event']=='local_perception' and
                 'predicted_base_port_position' in r]
    if not proposals:raise ValueError(f'No local actor proposals: {audit}')
    position=[];orientation=[]
    for row in proposals:
        local=port_base@pose_matrix(row['state'][:7])
        position.append(local[:3,3]*1000)
        orientation.append(float(np.rad2deg(Rotation.from_matrix(local[:3,:3]).magnitude())))
    positions=np.asarray(position)
    closest=int(np.argmax(positions[:,2]))
    score=yaml.safe_load((audit/'results/scoring.yaml').read_text())
    official=float(score[next(k for k in score if k.startswith('trial_'))]['tier_3']['score'])
    inference=[float(r['inference_ms']) for r in proposals if 'inference_ms' in r]
    result={'name':name,'audit':str(audit),'checkpoint':manifest['checkpoint'],
            'scene_case':manifest['case'],'selected_card':card,'selected_port':port,
            'policy':manifest['policy'],'privileged_transport_prefix':manifest['policy'].endswith('CheatCodeThenLocal'),
            'actor_true_geometry_input':False,'tier3':official,'proposals':len(proposals),
            'first_tcp_in_true_port_mm':positions[0].tolist(),
            'closest_tcp_in_true_port_mm':positions[closest].tolist(),
            'terminal_tcp_in_true_port_mm':positions[-1].tolist(),
            'max_tcp_axial_progress_mm':float(positions[closest,2]-positions[0,2]),
            'closest_lateral_offset_mm':float(np.linalg.norm(positions[closest,:2])),
            'closest_orientation_from_port_deg':orientation[closest],
            'complete_inference_ms':stat(inference),
            'note':'TCP coordinates are not plug-tip insertion depth'}
    if perceptions:
        perception_errors=[]
        orientation_errors=[]
        reconstructed_port_position_disagreement=[]
        for row in perceptions:
            predicted=np.r_[row['predicted_base_port_position'],1.]
            perception_errors.append((port_base@predicted)[:3]*1000)
        for perception,proposal in zip(perceptions,proposals):
            relative=matrix_from_pose9(perception['predicted_tcp_pose_in_port'])
            predicted_port=pose_matrix(proposal['state'][:7])@np.linalg.inv(relative)
            angle=Rotation.from_matrix(base_port[:3,:3]).inv()*Rotation.from_matrix(predicted_port[:3,:3])
            orientation_errors.append(float(np.rad2deg(angle.magnitude())))
            reconstructed_port_position_disagreement.append(float(np.linalg.norm(
                predicted_port[:3,3]-perception['predicted_base_port_position'])*1000))
        perception_errors=np.asarray(perception_errors)
        result['rgb_port_position_error_true_port_mm']={
            'first':perception_errors[0].tolist(),
            'first_ten_lateral':stat(np.linalg.norm(perception_errors[:10,:2],axis=1).tolist()),
            'all_lateral':stat(np.linalg.norm(perception_errors[:,:2],axis=1).tolist()),
            'all_axial_abs':stat(np.abs(perception_errors[:,2]).tolist()),
            'geometry_posthoc_scoring_only':True}
        result['rgb_port_orientation_error_true_port_deg']={
            'first':orientation_errors[0],
            'first_ten':stat(orientation_errors[:10]),
            'all':stat(orientation_errors),
            'reconstructed_position_disagreement_mm':stat(reconstructed_port_position_disagreement),
            'geometry_posthoc_scoring_only':True}
    tip_track=audit/'scored_tip_track.jsonl'
    if tip_track.is_file():
        port_world=np.linalg.inv(matrix(calibration_data['world_to_base'])@base_port)
        def pose(data):
            transform=np.eye(4)
            transform[:3,3]=data['xyz_m']
            transform[:3,:3]=Rotation.from_quat(data['quat_xyzw']).as_matrix()
            return transform
        tip_times=[];tip_points=[];all_tip_times=[];all_tip_points=[]
        first_time=proposals[0]['sim_time'];last_time=proposals[-1]['sim_time']
        for line in tip_track.open():
            row=json.loads(line);t=row['sim_time_s']
            point=(port_world@pose(row['world_cable'])@pose(row['cable_tip']))[:3,3]*1000
            all_tip_times.append(t);all_tip_points.append(point)
            if first_time<=t<=last_time:
                tip_times.append(t);tip_points.append(point)
        if tip_points:
            points=np.asarray(tip_points);nearest=int(np.argmax(points[:,2]))
            result['plug_tip_scored_tf']={'samples':len(points),
                'first_in_true_port_mm':points[0].tolist(),
                'max_axial_in_true_port_mm':points[nearest].tolist(),
                'max_axial_time_s':tip_times[nearest],
                'terminal_in_true_port_mm':points[-1].tolist(),
                'max_axial_progress_mm':float(points[nearest,2]-points[0,2]),
                'geometry_posthoc_scoring_only':True}
            handoff_file=audit/'physical_tip_handoff.json'
            if handoff_file.is_file():
                handoff=json.loads(handoff_file.read_text())
                if 'sim_time_s' in handoff:
                    handoff_time=float(handoff['sim_time_s'])
                    handoff_index=int(np.argmin(np.abs(np.asarray(all_tip_times)-handoff_time)))
                    handoff_point=np.asarray(all_tip_points[handoff_index])
                    first_actor_point=points[0]
                    result['handoff_attribution']={
                        'handoff_sim_time_s':handoff_time,
                        'first_actor_sim_time_s':first_time,
                        'plug_tip_at_handoff_true_port_mm':handoff_point.tolist(),
                        'plug_tip_at_first_actor_true_port_mm':first_actor_point.tolist(),
                        'axial_motion_during_actor_startup_mm':float(first_actor_point[2]-handoff_point[2]),
                        'axial_motion_during_actor_mm':float(points[nearest,2]-first_actor_point[2]),
                        'teacher_already_past_opening_at_first_actor':bool(first_actor_point[2]>0),
                        'geometry_posthoc_scoring_only':True}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',nargs=2,action='append',metavar=('NAME','AUDIT'),required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    runs=[summarize(name,Path(path),a.calibration) for name,path in a.run]
    if len({r['scene_case'] for r in runs})!=1:
        raise ValueError('Matched comparison requires identical scene cases')
    if len({(r['selected_card'],r['selected_port']) for r in runs})!=1:
        raise ValueError('Matched comparison requires identical target identity')
    out={'schema':'aic_sfp_matched_local_actor_live/v1','runs':runs,
         'geometry_posthoc_scoring_only':True,'reserved_final_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'runs':[{key:r[key] for key in ('name','tier3','max_tcp_axial_progress_mm',
                  'closest_lateral_offset_mm','complete_inference_ms')} for r in runs]}))


if __name__=='__main__':main()

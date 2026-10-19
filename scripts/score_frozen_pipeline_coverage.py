#!/usr/bin/env python3
"""Score a frozen rollout at pre-return observations; geometry is posthoc only."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from analyze_local_actor_scored_tip import record_at,transform
from audit_sc_port_targets import matrix
from build_two_family_port_projection_labels import sfp_ports


def score(run,case,arm):
    data=json.loads((run/'scored_stream.json').read_text())
    trace=[json.loads(l) for l in (run/'runtime_trace.jsonl').open()]
    terminal=[r for r in trace if r['event'] in ['terminal_before_return','visual_approach_failed']]
    if not terminal:
        raise ValueError('No pre-return observation: do not score post-reset geometry')
    terminal=terminal[-1]
    terminal_time_source='recorded_terminal_sim_time'
    if 'sim_time' not in terminal:
        # Frozen force-stop logging omitted its timestamp. Recover only an
        # exact float32 7D TCP match in the recorded observation stream.
        # Do not choose a nearest pose or a terminal/post-reset bag sample.
        target=np.asarray(terminal['state'][:7],dtype=np.float32)
        matches=[]
        prior=[float(r['sim_time']) for r in trace[:trace.index(terminal)] if 'sim_time' in r]
        lower=max(prior) if prior else 0.
        for stamp,message in data['streams']['/aic_controller/controller_state']:
            pose=message['tcp_pose']
            value=np.asarray([pose['position'][k] for k in 'xyz']+
                             [pose['orientation'][k] for k in 'xyzw'],dtype=np.float32)
            if lower<=stamp/1e9<=lower+1. and np.array_equal(value,target):
                matches.append(stamp)
        if len(matches)!=1:
            raise ValueError(f'Terminal timestamp needs exactly one causal TCP match: {matches}')
        terminal=dict(terminal,sim_time=matches[0]/1e9)
        terminal_time_source='unique_exact_float32_terminal_TCP_match_within_1s_after_last_trace'
        (run/'terminal_timestamp_recovery.json').write_text(json.dumps({
            'source':terminal_time_source,'sim_time':terminal['sim_time'],
            'lower_bound_sim_time':lower,'matches':matches,'raw_trace_unchanged':True},indent=2)+'\n')
    t=float(terminal['sim_time']);edges=data['edges']
    family=case['family'];port=case['port'];worldbase=None
    for base in ['world|tabletop','aic_world|tabletop']:
        if base in edges:
            worldbase=record_at(edges[base],0)@record_at(edges['tabletop|base_link'],0)
            break
    if worldbase is None:
        raise ValueError('Missing robot-base transform')
    all_ports={}
    if family=='sc':
        cal=json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())
        config=yaml.safe_load((run/'eval_config.yaml').read_text())
        board=config['trials'][data['trial_id']]['scene']['task_board']
        for p in range(2):
            if not board[f'sc_rail_{p}']['entity_present']:continue
            prefix=f'task_board/sc_port_{p}'
            all_ports[p]=(record_at(edges['aic_world|task_board'],0)@
                record_at(edges[f'task_board|{prefix}'],0)@
                record_at(edges[f'{prefix}|{prefix}/sc_port_base_link'],0)@matrix(cal['base_to_opening']))
        worldport=all_ports[port]
    else:
        cal=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
        all_ports={ (c,p):worldbase@m for c,p,m in sfp_ports(
            {'trial_yaml':str(run/'eval_config.yaml'),'trial_id':data['trial_id']},cal)}
        worldport=all_ports[case['requested_card'],port]
    inv=np.linalg.inv(worldport);cable=data['task']['cable_name']
    parent=f'aic_world|{cable}'
    if parent not in edges:
        parent=f'world|{cable}'
    tipkey=f'{cable}|{cable}/{family}_tip_link'
    native=[]
    if family=='sfp' and (parent not in edges or tipkey not in edges):
        file=run/'physical_pose_proto.txt'
        if not file.exists():
            raise ValueError('SFP plug TF not recorded; native read-only pose recording required')
        for chunk in re.split(r'(?m)^header \{',file.read_text())[1:]:
            header=chunk.split('\npose {',1)[0]
            sec=re.search(r'\n    sec: (\d+)',header);ns=re.search(r'\n    nsec: (\d+)',header)
            stamp=(int(sec[1]) if sec else 0)*1000000000+(int(ns[1]) if ns else 0)
            if stamp>t*1e9:continue
            found={}
            for block in re.findall(r'(?ms)^pose \{(.*?)^\}',chunk):
                name=re.search(r'name: "([^"]+)"',block)
                if not name:continue
                key=name[1].split('::')[-1].split('/')[-1]
                if key not in [cable,'sfp_tip_link']:continue
                pose={}
                for field,keys in [('position','xyz'),('orientation','xyzw')]:
                    section=re.search(field+r' \{(.*?)\n  \}',block,re.S)
                    values={k:float(v) for k,v in re.findall(r'([xyzw]): ([-+\deE.]+)',section[1])} if section else {}
                    pose[field]={k:values.get(k,0.) for k in keys}
                if np.linalg.norm(list(pose['orientation'].values()))>0:
                    if key in found:raise ValueError('Ambiguous native model/link identity')
                    found[key]=pose
            if cable in found and 'sfp_tip_link' in found:
                matrices=[]
                for key in [cable,'sfp_tip_link']:
                    m=np.eye(4);m[:3,3]=[found[key]['position'][k] for k in 'xyz'];m[:3,:3]=Rotation.from_quat([found[key]['orientation'][k] for k in 'xyzw']).as_matrix();matrices.append(m)
                # SceneBroadcaster: model pose is world-relative; each link
                # pose is relative to its model. Never treat local link xyz
                # as world xyz (the robot model itself is nearly identity).
                native.append((stamp,matrices[0]@matrices[1]))
        if not native:raise ValueError('No native physical SFP-tip poses')
        native.sort(key=lambda x:x[0]);native_times=[r[0] for r in native]
    cache=({k:([r[0] for r in edges[k]],edges[k]) for k in [parent,tipkey]} if not native else {})
    def tip_at(seconds):
        mats=[];ages=[]
        if native:
            i=bisect_right(native_times,seconds*1e9)-1
            if i<0:raise ValueError('Only post-terminal native tip poses exist')
            ages=[(seconds*1e9-native_times[i])/1e6]
            relative=inv@native[i][1]
        else:
            for k,(times,rows) in cache.items():
                i=bisect_right(times,seconds*1e9)-1
                if i<0:
                    raise ValueError('Only post-terminal tip TF exists')
                ages.append((seconds*1e9-times[i])/1e6)
                mats.append(transform(rows[i][1]))
            relative=inv@mats[0]@mats[1]
        result={'time_s':seconds,'axial_mm':float(relative[2,3]*1000),
            'lateral_mm':float(np.linalg.norm(relative[:2,3])*1000),
            'distance_mm':float(np.linalg.norm(relative[:3,3])*1000),'TF_age_max_ms':float(max(ages)),
            'relative_quat_xyzw':Rotation.from_matrix(relative[:3,:3]).as_quat().tolist(),
            'axis_error_unsigned_deg':float(np.rad2deg(np.arccos(np.clip(abs(relative[2,2]),0,1))))}
        if family=='sc':
            prior=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/oracle_mating_prior_v1.json').read_text())
            result['orientation_error_deg']=float((Rotation.from_quat(prior['tip_orientation_port_xyzw']).inv()*Rotation.from_matrix(relative[:3,:3])).magnitude()*180/np.pi)
        else:
            result['orientation_error_deg']=float(np.degrees(Rotation.from_matrix(relative[:3,:3]).magnitude()))
            result['orientation_reference']='calibrated SFP tip relative to opening frame'
        return result
    final=tip_at(t)
    if final['TF_age_max_ms']>200:
        raise ValueError(f'Terminal tip transform stale by {final["TF_age_max_ms"]}ms')
    official=yaml.safe_load((run/'results/scoring.yaml').read_text())[data['trial_id']]
    categories=official['tier_2']['categories']
    no_harm=all(categories[k]['score']>=0 for k in ['contacts','insertion force'])
    events={r['event'] for r in trace}
    sfp_rule=family=='sfp' and 'SFP_rule_visual_inspection' in events
    acquired=any(r['event']=='rule_terminal' and r.get('acquired') for r in trace)
    rule_terminal=next((r for r in reversed(trace) if r['event']=='rule_terminal'),{})
    inspections=[r for r in trace if r['event'] in ('rule_visual_inspection','SFP_rule_visual_inspection')]
    last_visual=inspections[-1].get('report',{}) if inspections else {}
    acquisition_geometry=None
    if family=='sc':
        observation=next((r for r in reversed(trace) if isinstance(r.get('sim_time'),(int,float)) and r['sim_time']<=rule_terminal.get('sim_time',-1) and len(r.get('state',[]))==32),None)
        if observation:
            state=np.asarray(observation['state'])
            opening_base=np.linalg.inv(worldbase)@worldport
            config=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
            offset=np.asarray(config['opening_to_transport_tcp_base_offset_m'])
            prior=json.loads(Path('outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json').read_text())
            desired=Rotation.from_matrix(opening_base[:3,:3])*Rotation.from_quat(prior['nominal_tcp_in_port_orientation_xyzw'])
            acquisition_geometry={'time_s':observation['sim_time'],
                'TCP_position_error_to_training_standoff_mm':float(np.linalg.norm(state[:3]-opening_base[:3,3]-offset)*1000),
                'TCP_posture_error_to_training_prior_deg':float((desired.inv()*Rotation.from_quat(state[3:7])).magnitude()*180/np.pi),
                'note':'Posthoc geometric reference only; does not establish port visibility or authorize handoff'}
    approach=next((r for r in trace if r['event']=='observation_approach_actor_start'),None)
    approach_truth=tip_at(approach['sim_time']) if approach else None
    local_started='locked_local_perception' in events
    wrong=None;opening_error=None
    if (family=='sc' or sfp_rule) and acquired:
        inspections=[r for r in trace if r['event'] in ('rule_visual_inspection','SFP_rule_visual_inspection') and r.get('report',{}).get('valid')]
        if inspections:
            report=inspections[-1]['report'];pred=np.r_[report['predicted_opening_base_m'],1.]
            predworld=(worldbase@pred)[:3];dist={p:float(np.linalg.norm(predworld-m[:3,3])) for p,m in all_ports.items()}
            requested=port if family=='sc' else (case['requested_card'],port)
            wrong=min(dist,key=dist.get)!=requested;opening_error=dist[requested]*1000
    sample_times=sorted({float(r['sim_time']) for r in trace if r['event'] in ['proposal','observation_only_approach','visual_approach_failed','observation_approach_actor_start','rule_terminal'] and r.get('sim_time',0)<=t})
    curve=[]
    for ts in sample_times:
        try:
            x=tip_at(ts)
            if x['TF_age_max_ms']<=200:curve.append(x)
        except ValueError:
            pass
    visits=[x for x in curve if -100<=x['axial_mm']<=-15 and x['lateral_mm']<=10]
    latency=[r['inference_ms'] for r in trace if r['event']=='proposal' and 'inference_ms' in r]
    reasons=[{k:r.get(k) for k in ['event','sim_time','reason']} for r in trace
             if any(tag in r['event'] for tag in ['failed','rejected','terminal'])]
    starts=data['streams'].get('/joint_states',[])
    start=None
    if starts:
        obs=starts[0][1];start=dict(zip(obs['name'],obs['position']))
    result={'schema':'aic_frozen_pipeline_episode/v1','case':case['case'],'arm':arm,'family':family,
        'domain':case['domain'],'card_count':case['card_count'],'port':port,'replicate':case['replicate'],
        'requested_card':case['requested_card'],'audit':str(run),'scene_group_id':data['scene_group_id'],
        'official_tier3':official['tier_3']['score'],'no_scored_harm':no_harm,
        'contact_message':categories['contacts']['message'],'force_message':categories['insertion force']['message'],
        'rule_visual_acquisition':acquired if family=='sc' or sfp_rule else None,
        'rule_failure':rule_terminal.get('failure'),
        'last_visual_reason':last_visual.get('reason'),
        'acquisition_geometry':acquisition_geometry,
        'wrong_accepted_port':wrong,'accepted_opening_error_mm':opening_error,
        'near_approach_completed':bool(approach) if arm=='near_approach' else None,
        'near_approach_true_geometry':approach_truth,'insertion_actor_ran':local_started,
        'terminal':final,'SC_full_physical_insertion':bool(local_started and final['axial_mm']>=14 and final['lateral_mm']<=.5 and no_harm) if family=='sc' else None,
        'SFP_full_pipeline_evaluation':('rule_transport_only_no_insertion_actor' if sfp_rule else 'not_implemented_in_selected_hybrid') if family=='sfp' else None,
        'posthoc_broad_near_port_visit':bool(visits),'best_broad_visit':min(visits,key=lambda x:x['distance_mm']) if visits else None,
        'closest_geometric_tip_visit':min(curve,key=lambda x:x['distance_mm']) if curve else None,
        'inference_p50_p95_p99_ms':np.percentile(latency,[50,95,99]).tolist() if latency else None,
        'first_measured_joint_positions':start,'configured_start_joints':case['configured_start_joints'],
        'terminal_reason':terminal.get('reason'),'terminal_trace_event':terminal['event'],
        'terminal_time_source':terminal_time_source,'diagnostic_reasons':reasons[-5:],
        'runtime_geometry_input':False,'geometry_posthoc_only':True,'post_reset_geometry_used':False,'final_scenes_opened':False}
    result['physical_tip_measurement_source']='native_world_model_pose_times_model_relative_tip_link_pose' if native else 'recorded_scoring_TF'
    (run/'coverage_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/'tip_curve.json').write_text(json.dumps(curve)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--arm',required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());case=next(c for c in plan['cases'] if c['case']==a.case)
    r=score(a.run,case,a.arm)
    print(json.dumps({k:r[k] for k in ['case','arm','rule_visual_acquisition','near_approach_completed','insertion_actor_ran','SC_full_physical_insertion','terminal']}))


if __name__=='__main__':main()

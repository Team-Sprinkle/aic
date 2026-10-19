#!/usr/bin/env python3
"""Posthoc SC physical score at a logged terminal observation, never after reset."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from analyze_local_actor_scored_tip import record_at
from audit_sc_port_targets import matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads((args.rollout/'scored_stream.json').read_text())
    trace=[json.loads(line) for line in (args.rollout/'runtime_trace.jsonl').open()]
    candidates=[r for r in trace if r['event'] in ('terminal_before_return','visual_approach_failed')]
    if not candidates:
        raise ValueError('No retained terminal observation')
    terminal=candidates[-1]
    ts=terminal['sim_time']*1e9
    edges=data['edges']
    port=int(data['task']['target_module_name'].rsplit('_',1)[-1])
    prefix=f'task_board/sc_port_{port}'
    calibration=json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())
    opening=(record_at(edges['aic_world|task_board'],0)@
        record_at(edges[f'task_board|{prefix}'],0)@
        record_at(edges[f'{prefix}|{prefix}/sc_port_base_link'],0)@matrix(calibration['base_to_opening']))
    ages={}
    for key in ['aic_world|cable_1','cable_1|cable_1/sc_tip_link']:
        rows=edges[key]
        index=bisect_right([r[0] for r in rows],ts)-1
        if index<0:
            raise ValueError('Only post-terminal geometry available')
        ages[key]=(ts-rows[index][0])/1e6
        if ages[key]>200:
            raise ValueError(f'Terminal geometry too stale: {key} {ages[key]}ms')
    tip=np.linalg.inv(opening)@record_at(edges['aic_world|cable_1'],ts)@record_at(edges['cable_1|cable_1/sc_tip_link'],ts)
    prior=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/oracle_mating_prior_v1.json').read_text())
    angle=float((Rotation.from_quat(prior['tip_orientation_port_xyzw']).inv()*Rotation.from_matrix(tip[:3,:3])).magnitude()*180/np.pi)
    score=yaml.safe_load((args.rollout/'results/scoring.yaml').read_text())[data['trial_id']]
    categories=score['tier_2']['categories']
    depth=float(tip[2,3]*1000)
    lateral=float(np.linalg.norm(tip[:2,3])*1000)
    # Scoring penalties cover prohibited contacts and sustained excessive
    # insertion force. Raw oracle contacts can include harmless cable touch.
    no_harm=(categories['contacts']['score']>=0 and
             categories['insertion force']['score']>=0)
    report={'schema':'aic_locked_local_terminal_physical/v1','trial':data['trial_id'],
        'terminal_event':terminal['event'],'terminal_sim_time_s':terminal['sim_time'],
        'terminal_reason':terminal.get('reason'),'selected_port':port,
        'depth_mm':depth,'lateral_mm':lateral,'orientation_error_deg':angle,
        'official_tier3':score['tier_3']['score'],
        'contact_message':categories['contacts']['message'],
        'force_message':categories['insertion force']['message'],
        'contact_penalty':categories['contacts']['score'],
        'force_penalty':categories['insertion force']['score'],
        'physical_gate_passed':depth>=14 and lateral<=.5 and no_harm,
        'orientation_gate':'Reported explicitly; no new retrospective orientation threshold',
        'terminal_tf_age_ms':ages,'post_reset_geometry_used':False,
        'true_geometry_posthoc_only':True,'final_scenes_opened':False}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

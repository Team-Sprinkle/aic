#!/usr/bin/env python3
"""Posthoc requested-port transport scoring for matched SFP/SC development runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from build_two_family_port_projection_labels import sfp_ports
from materialize_sc_scored_route import port_opening


def score(audit,sfp_calibration):
    stream=json.loads((audit/'scored_stream.json').read_text())
    task=stream['task']
    if task['plug_type']=='sfp':
        family='sfp';config=yaml.safe_load((audit/'eval_config.yaml').read_text())
        trial_id=next(iter(config['trials']))
        card=int(task['target_module_name'].split('_')[-1]);port=int(task['port_name'].split('_')[-1])
        opening={(c,p):matrix for c,p,matrix in sfp_ports(
            {'trial_yaml':str(audit/'eval_config.yaml'),'trial_id':trial_id},sfp_calibration)}[card,port]
    else:
        family='sc';card=None;port=int(task['target_module_name'].split('_')[-1])
        opening=port_opening(stream,port)
    trace=[json.loads(line) for line in (audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in trace if r.get('event')=='proposal' and len(r.get('state',[]))==32]
    if not proposals:raise ValueError(f'No autonomous actor proposals: {audit}')
    measured=[]
    for row in proposals:
        if row.get('task_vector') is not None:
            code=np.asarray(row['task_vector'])
            if int(code[2:4].argmax())!=port or (family=='sfp' and int(code[4:9].argmax())!=card):
                raise ValueError(f'Task request changed in {audit}')
        relative=opening[:3,:3].T@(np.asarray(row['state'][:3])-opening[:3,3])*1000
        measured.append(dict(sim_time_s=row['sim_time'],axial_mm=float(relative[2]),
                             lateral_mm=float(np.linalg.norm(relative[:2])),
                             inference_ms=float(row.get('inference_ms',np.nan))))
    band=[r for r in measured if -100<=r['axial_mm']<=-70]
    visits=[r for r in band if r['lateral_mm']<=10]
    latency=np.asarray([r['inference_ms'] for r in measured]);latency=latency[np.isfinite(latency)]
    score_yaml=yaml.safe_load((audit/'results/scoring.yaml').read_text())
    trial=next(v for k,v in score_yaml.items() if k.startswith('trial_'))
    return dict(audit=str(audit),family=family,card=card,port=port,
                reset_group=stream['scene_group_id'],actor_proposals=len(proposals),
                safe_buffer_visit=bool(visits),safe_buffer_samples=len(visits),
                closest_lateral_in_axial_band_mm=(min(r['lateral_mm'] for r in band) if band else None),
                axial_band_samples=len(band),
                official_tier3=float(trial.get('tier_3',{}).get('score',0)),
                official_contact_message=trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('message'),
                inference_p50_p95_p99_ms=(np.percentile(latency,[50,95,99]).tolist() if len(latency) else None),
                privileged_geometry_actor_input=False,geometry_posthoc_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing comparison')
    sfp_calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    rows=[score(run,sfp_calibration) for run in a.run]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='aic_official_anchor_matched_route_score/v1',
        runs=rows,final_scenes_opened=False),indent=2)+'\n')
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent physical SC expert check at the last recorded pose command, before cleanup."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_local_actor_scored_tip import record_at
from audit_sc_port_targets import matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();episodes={}
    for line in a.labels.open():
        r=json.loads(line)
        if r['family']=='sc':episodes.setdefault(r['episode'],r)
    cal=json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())
    prior=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/oracle_mating_prior_v1.json').read_text())
    results=[]
    for name,r in episodes.items():
        d=json.loads(Path(r['source_stream']).read_text());e=d['edges'];port=int(r['selected_identity'].split('_')[-1]);prefix=f'task_board/sc_port_{port}'
        wp=record_at(e['aic_world|task_board'],0)@record_at(e[f'task_board|{prefix}'],0)@record_at(e[f'{prefix}|{prefix}/sc_port_base_link'],0)@matrix(cal['base_to_opening'])
        stamp=max(x['sim_ns'] for x in d['commands'] if x['kind']=='pose')
        parts=[];ages=[]
        for key in ['aic_world|cable_1','cable_1|cable_1/sc_tip_link']:
            rows=e[key];j=bisect_right([x[0] for x in rows],stamp)-1
            if j<0:raise ValueError('Only future tip geometry exists')
            ages.append((stamp-rows[j][0])/1e6);parts.append(record_at(rows,stamp))
        if max(ages)>200:raise ValueError(f'Stale physical tip at last command: {name} {ages}')
        relative=np.linalg.inv(wp)@parts[0]@parts[1]
        depth=float(relative[2,3]*1000);lateral=float(np.linalg.norm(relative[:2,3])*1000)
        angle=float((Rotation.from_quat(prior['tip_orientation_port_xyzw']).inv()*Rotation.from_matrix(relative[:3,:3])).magnitude()*180/np.pi)
        results.append({'episode':name,'split':r['split'],'cards':r['card_count'],'port':port,
            'last_pose_command_time_s':stamp/1e9,'TF_age_ms':ages,'depth_mm':depth,'lateral_mm':lateral,
            'orientation_error_deg':angle,'physical_depth_lateral_pass':bool(depth>=14 and lateral<=.5),
            'official_tier3':d['official_tier3'],'geometry_training_label_only':True})
    summary={'schema':'aic_SC_expert_physical_depth_audit/v1','episodes':len(results),
             'depth_lateral_pass':sum(x['physical_depth_lateral_pass'] for x in results),
             'note':'Depth/lateral check is separate from force/contact and visibility checks. Nonpassing episodes may supply transport prefixes, not successful insertion labels.',
             'results':results,'final_scenes_opened':False}
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['episodes','depth_lateral_pass']}),flush=True)


if __name__=='__main__':main()

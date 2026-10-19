#!/usr/bin/env python3
"""Training-only fixed priors for an observation-only SFP acquisition route."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_sc_port_targets import pose_matrix
from prepare_verified_insertion_source_index import source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous priors')
    sources=[json.loads(l) for l in a.sources.open() if json.loads(l)['family']=='sfp' and json.loads(l)['split']=='train'];wanted={s['source_stream']:s for s in sources};labels={}
    for l in a.labels.open():
        r=json.loads(l);path=source(r)
        if path in wanted and path not in labels:labels[path]=r
    positions=defaultdict(list);orientations=defaultdict(list);body=[];port_tcp=[];coverage=[]
    for s in sources:
        d=json.loads(Path(s['source_stream']).read_text());task=d['task'];card=int(task['target_module_name'].rsplit('_',1)[1]);port=int(task['port_name'].rsplit('_',1)[1]);key=f'card{card}_port{port}'
        identity=f'sfp_card_{card}_port_{port}';target=next(t for t in labels[s['source_stream']]['targets'] if t['identity']==identity)
        audit=s['physical_audit'];stamp=round(audit['last_command_sim_s']*1e9);states=d['streams']['/aic_controller/controller_state'];i=bisect_right(states,stamp,key=lambda r:r[0])-1
        if i<0 or stamp-states[i][0]>20_000_000:raise ValueError('No causal terminal TCP')
        pp=states[i][1]['tcp_pose'];tcp=pose_matrix([*[pp['position'][k] for k in 'xyz'],*[pp['orientation'][k] for k in 'xyzw']]);rtcp=Rotation.from_matrix(tcp[:3,:3]);rport=Rotation.from_quat(target['orientation_base_xyzw'])
        body.append(rtcp.inv().apply(np.asarray(audit['raw_tip_position_base_m'])-tcp[:3,3]));port_tcp.append((rport.inv()*rtcp).as_quat())
        positions[key].append(target['position_base_m']);orientations[key].append(target['orientation_base_xyzw']);coverage.append(dict(source=s['source_stream'],task=key))
    body=np.asarray(body);mean_body=np.median(body,axis=0);mean_rotation=Rotation.from_quat(port_tcp).mean()
    items={}
    for key,values in positions.items():
        x=np.asarray(values);r=Rotation.from_quat(orientations[key]).mean();tcp_r=r*mean_rotation
        tcp_position=x.mean(0)+r.apply([0,0,-.150])-tcp_r.apply(mean_body)
        items[key]=dict(training_episodes=len(values),port_mean_base_m=x.mean(0).tolist(),port_mean_quat_xyzw=r.as_quat().tolist(),
            potential_min_base_m=(x.min(0)-.02).tolist(),potential_max_base_m=(x.max(0)+.02).tolist(),nominal_TCP_standoff_base_m=tcp_position.tolist(),nominal_TCP_quat_xyzw=tcp_r.as_quat().tolist())
    if len(items)!=10:raise ValueError('All5cards x2ports require training support')
    report=dict(schema='SFP_rule_training_priors/v1',tasks=items,mean_body_tip_position_m=mean_body.tolist(),
        body_tip_prior_error_p50_p95_mm=np.percentile(np.linalg.norm(body-mean_body,axis=1)*1000,[50,95]).tolist(),
        nominal_TCP_in_port_xyzw=mean_rotation.as_quat().tolist(),approach_standoff_m=.150,
        geometry_runtime_input=False,priors_use_training_geometry=True,validation_geometry_used=False,
        source_sha256=hashlib.sha256(a.sources.read_bytes()).hexdigest(),sources=coverage,final_scenes_opened=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(tasks=len(items),body_prior=mean_body.tolist(),body_prior_errors=report['body_tip_prior_error_p50_p95_mm'],example=items['card0_port0'])),flush=True)


if __name__=='__main__':main()

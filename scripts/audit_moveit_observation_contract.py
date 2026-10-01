#!/usr/bin/env python3
"""Audit the full raw-topic -> official adapter -> actor observation contract."""
import argparse,bisect,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'aic_utils/lerobot_robot_aic'))
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation
ORDER=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint','gripper/left_finger_joint']


def obj(d):
    return SimpleNamespace(**{k:obj(v) if isinstance(v,dict) else v for k,v in d.items()})


def previous(rows,t):
    i=bisect.bisect_right(rows,t,key=lambda r:r[0])-1
    if i<0:raise ValueError('No preceding sample')
    return rows[i][1]


def adapted(j):
    names=dict(zip(j['name'],j['position']))
    position=[names[k] for k in ORDER];position[-1]/=2.
    return dict(name=ORDER[:-1]+['gripper'],position=position)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
    p.add_argument('--live-stream',type=Path);p.add_argument('--live-trace',type=Path)
    a=p.parse_args();manifest=json.loads((a.dataset/'manifest.json').read_text())
    streams={e['episode_index']:json.loads(Path(e['source_stream']).read_text()) for e in manifest['episodes']}
    n=0;err=0.;badw=0
    for line in (a.dataset/'transport_chunks.jsonl').open():
        row=json.loads(line);r=streams[row['episode_index']];t=min(row['image_sim_ns'].values())
        c=previous(r['streams']['/aic_controller/controller_state'],t)
        j=adapted(previous(r['streams']['/joint_states'],t));w=previous(r['streams']['/fts_broadcaster/wrench'],t)
        state=base_state_from_ros_observation(obj(dict(controller_state=c,joint_states=j,wrist_wrench=w)))
        err=max(err,float(np.max(np.abs(state-np.asarray(row['state'],np.float32)))));badw+=row['state'][6]<0;n+=1
    assert err<1e-6 and badw==0
    result=dict(schema='aic_raw_adapter_actor_state_audit/v1',rows=n,max_abs_float32_error=err,negative_w=badw,
                adapter_order=ORDER[:-1]+['gripper'],adapter_finger_scale=.5,
                preceding_state_join='synchronized image timestamp, not command timestamp',passed=True,
                source_reference='aic_adapter/src/aic_adapter.cpp:80-86, 172-226, 262-287')
    if a.live_stream and a.live_trace:
        r=json.loads(a.live_stream.read_text());trace=[json.loads(l) for l in a.live_trace.open()]
        distances=[];raw_distances=[]
        for q in [x for x in trace if x['event']=='proposal']:
            t=round(q['sim_time']*1e9);rows=r['streams']['/joint_states'];i=bisect.bisect_right(rows,t,key=lambda x:x[0]);pool=rows[max(0,i-20):i+1]
            truth=np.asarray(q['state'][19:26],np.float32)
            distances.append(min(float(np.max(np.abs(truth-np.asarray(adapted(x[1])['position'],np.float32)))) for x in pool))
            raw_distances.append(min(float(np.max(np.abs(truth-np.asarray(x[1]['position'][:7],np.float32)))) for x in pool))
        result['live_joint_contract']=dict(proposals=len(distances),adapted_max_abs_error_rad_or_m=max(distances),
            adapted_p95=float(np.quantile(distances,.95)),raw_prefix_p95=float(np.quantile(raw_distances,.95)),
            note='Nearest preceding raw joint sample among last20; isolates adapter ordering/scale, not complete timing fidelity')
        assert max(distances)<1e-5,'Actual actor observations do not match adapter conversion'
    (a.dataset/'adapter_state_contract_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

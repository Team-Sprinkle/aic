#!/usr/bin/env python3
"""Audit recorded approach supervision before rebuilding insertion BC labels.

Reads no reserved scenes and generates no simulator rollouts. The terminal-TCP
distance below is a coverage proxy, not plug depth or a physical success claim.
Teacher commands are decoded in their recorded frame at command time, not in
the later image frame. Outputs are an audit, never an admitted BC dataset.
"""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def transform(p):
    m=np.eye(4);m[:3,3]=[p['position'][k] for k in 'xyz'];m[:3,:3]=Rotation.from_quat([p['orientation'][k] for k in 'xyzw']).as_matrix();return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);grouped=defaultdict(list)
    for line in a.labels.open():
        r=json.loads(line);source=r.get('source_stream') or r.get('source_scored_stream') or str(Path(r['physical_reset_config']).parent/'scored_stream.json')
        grouped[source].append(r)
    reports=[];coverage=defaultdict(Counter);max_roundtrip_m=0.;rejections=Counter()
    for n,(source,rows) in enumerate(grouped.items()):
        stream=json.loads(Path(source).read_text());ctrl=stream['streams']['/aic_controller/controller_state'];times=[v[0] for v in ctrl]
        commands=[v for v in stream['commands'] if v['kind']=='pose'];commands.sort(key=lambda v:v['sim_ns'])
        if not commands:rejections['no_pose_commands']+=len(rows);continue
        stamp=commands[-1]['sim_ns'];j=bisect_right(times,stamp)-1
        if j<0 or stamp-times[j]>100_000_000:rejections['no_causal_terminal_TCP']+=len(rows);continue
        final=transform(ctrl[j][1]['tcp_pose']);frame_counts=Counter();join_bad=Counter();decoded=[]
        for c in commands:
            frame=c['message']['header']['frame_id'];frame_counts[frame]+=1
            j=bisect_right(times,c['sim_ns'])-1
            if j<0 or c['sim_ns']-times[j]>100_000_000:join_bad['command_TCP_stale_or_future']+=1;continue
            tcp=transform(ctrl[j][1]['tcp_pose']);raw=transform(c['message']['pose'])
            if frame=='gripper/tcp':target=tcp@raw
            elif frame=='base_link':target=raw
            else:join_bad['unsupported_command_frame']+=1;continue
            body=np.linalg.inv(tcp)@target;roundtrip=tcp@body
            max_roundtrip_m=max(max_roundtrip_m,float(np.linalg.norm(roundtrip[:3,3]-target[:3,3])))
            decoded.append((c['sim_ns'],target))
        per=Counter();tip_keys=[k for k in stream['edges'] if k.endswith('/'+stream['task']['plug_name']+'_link')]
        for r in rows:
            stamps=list(r['image_sim_ns'].values())
            if max(stamps)-min(stamps)>1_000_000:
                per['async_triplet_not_for_actor_or_common_TCP_triangulation']+=1;continue
            t=r['image_sim_ns']['center'];j=bisect_right(times,t)-1
            if j<0 or t-times[j]>10_000_000:per['image_TCP_stale_or_future']+=1;continue
            now=transform(ctrl[j][1]['tcp_pose'])
            target=next(v for v in r['targets'] if v['identity']==r['selected_identity']);port_rot=Rotation.from_quat(target['orientation_base_xyzw']).as_matrix()
            remaining=port_rot.T@(final[:3,3]-now[:3,3])*1000
            remaining_axial=float(remaining[2]);lateral=float(np.linalg.norm(remaining[:2]))
            orientation=float((Rotation.from_matrix(final[:3,:3]).inv()*Rotation.from_matrix(now[:3,:3])).magnitude()*180/np.pi)
            band=('0to50mm' if 0<=remaining_axial<=50 else '50to150mm' if 50<remaining_axial<=150 else '150to200mm' if 150<remaining_axial<=200 else 'other')
            coverage[r['family']+'/'+r['split']][band]+=1
            if 0<=remaining_axial<=200 and lateral<=25 and orientation<=15:
                per['aligned_approach_candidate_frames']+=1
                per['aligned_'+band]+=1
                if lateral>5:per['approach_lateral_over5mm']+=1
                if lateral>15:per['approach_lateral_over15mm']+=1
                if orientation>5:per['approach_orientation_over5deg']+=1
                if orientation>10:per['approach_orientation_over10deg']+=1
        audit=Path(r['physical_reset_config']).parent
        # A corrective/student source needs timestamped ownership before any
        # recorded command is allowed into supervised actor labels.
        takeover=audit/'corrective_teacher_takeover.json'
        reports.append(dict(source_stream=source,family=r['family'],split=r['split'],
            recorded_episodes=sorted({v['episode'] for v in rows}),physical_resets=sorted({v['scene_group'] for v in rows}),
            official_tier3=stream.get('official_tier3'),commands=len(commands),decoded_commands=len(decoded),
            command_frames=dict(frame_counts),command_join_rejections=dict(join_bad),coverage=dict(per),
            raw_plug_TF_present=bool(tip_keys),corrective_takeover_file=str(takeover) if takeover.exists() else None,
            teaching_ownership_still_requires_explicit_audit=True))
        if (n+1)%25==0:print(json.dumps(dict(sources_completed=n+1,total_sources=len(grouped))),flush=True)
    summary=dict(schema='insertion_150mm_teacher_source_audit/v1',unique_source_streams=len(reports),
        coverage_terminal_TCP_distance_proxy={k:dict(v) for k,v in coverage.items()},
        body_action_roundtrip_max_m=max_roundtrip_m,rejections=dict(rejections),sources=reports,
        limitations=['Terminal TCP distance is not distance of plug to opening; includes remaining insertion travel.',
            'Official score75 alone is not an independent physical depth/force gate.',
            'Need explicit teacher/supervisor ownership, predicted port frames, and held-out live insertion before BC promotion.',
            'Unrecorded SFP plug TF must not be silently invented from a later/reset observation.'],
        admitted_for_BC=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()

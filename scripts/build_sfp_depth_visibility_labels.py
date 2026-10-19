#!/usr/bin/env python3
"""Build label-only SFP aperture depth visibility from a scored native replay.

Geometric exposed fraction is a proxy, not human-certified identifiability.
Only the original synchronized RGB is retained as a future actor input.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_two_family_port_projection_labels import sfp_ports


def latest(rows,t):
    i=bisect_right(rows,t,key=lambda r:r[0])-1
    if i<0:raise ValueError('No causal source')
    return rows[i]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stream',type=Path)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();stream=json.loads(a.stream.read_text());audit=Path(stream['audit'])
    a.output.mkdir(parents=True,exist_ok=True)
    calibration=json.loads(a.calibration.read_text())
    episode={'trial_yaml':str(audit/'eval_config.yaml'),'trial_id':stream['trial_id']}
    ports={(card,port):matrix for card,port,matrix in sfp_ports(episode,calibration)}
    task=stream['task'];card=int(task['target_module_name'].rsplit('_',1)[-1])
    port=int(task['port_name'].rsplit('_',1)[-1]);opening=ports[(card,port)]
    camera_calibration=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(camera_calibration['tcp_to_optical'][c],float) for c in ('center','left','right')}
    # Conservative inner-face patch; opening dimensions should be checked
    # against the exact SFP mesh before using this as a certified aperture mask.
    grid=np.array([[x,y,0.,1.] for x in np.linspace(-.005,.005,11)
                                  for y in np.linspace(-.002,.002,7)]).T
    bounds=stream['scored_wall_bounds_ns']
    pairs=[json.loads(l) for l in (audit/'visibility_pairs.jsonl').open()]
    pairs=[r for r in pairs if bounds[0]<=r['wall_ns']<=bounds[1]]
    results=[];counts=Counter();reject=Counter()
    for pair in pairs:
        try:
            t=pair['sim_ns'];cam=pair['camera']
            state_t,state=latest(stream['streams']['/aic_controller/controller_state'],t)
            if t-state_t>100_000_000:raise ValueError('Stale controller state')
            tcp=pose_matrix([*[state['tcp_pose']['position'][k] for k in 'xyz'],
                             *[state['tcp_pose']['orientation'][k] for k in 'xyzw']])
            optical=tcp@optics[cam]
            points=(np.linalg.inv(optical)@opening@grid)[:3]
            intr=pair['intrinsics'];h,w=intr['height'],intr['width']
            k=np.asarray(intr['k']).reshape(3,3)
            proj=k@points;uv=(proj[:2]/proj[2:]).T
            inside=(points[2]>.07)&(points[2]<20)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            depth=np.load(audit/'visibility_frames'/pair['depth'])['depth_m']
            if depth.shape!=(h,w):raise ValueError('Depth shape mismatch')
            pixel=np.rint(uv).astype(int);pixel[:,0]=np.clip(pixel[:,0],0,w-1);pixel[:,1]=np.clip(pixel[:,1],0,h-1)
            observed=depth[pixel[:,1],pixel[:,0]]
            valid=(np.isfinite(observed)&(observed>0))|np.isposinf(observed)
            facing=float(np.dot(-opening[:3,2],optical[:3,3]-opening[:3,3])/
                         (np.linalg.norm(optical[:3,3]-opening[:3,3])+1e-12))
            exposed=inside&valid&(observed>=points[2]-.001)&(facing>.1)
            fraction=float(np.mean(exposed));inframe=float(np.mean(inside))
            original_t,original=latest(stream['cameras'][cam],t)
            if t-original_t>50_000_000:raise ValueError('No synchronized original RGB')
            aux=cv2.imread(str(audit/'visibility_frames'/pair['rgb']))
            rgb=cv2.imread(original)
            if aux is None or rgb is None or aux.shape!=rgb.shape:raise ValueError('RGB shape mismatch')
            diff=float(np.mean(np.abs(aux.astype(float)-rgb.astype(float))))
            if inside.any():
                lo=np.maximum(np.floor(uv[inside].min(0)).astype(int)-5,0)
                hi=np.minimum(np.ceil(uv[inside].max(0)).astype(int)+6,[w,h])
                region=float(np.mean(np.abs(aux[lo[1]:hi[1],lo[0]:hi[0]].astype(float)-
                                            rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float))))
                if region>10:raise ValueError('Selected RGB region differs')
                if np.mean(~valid[inside])>.1:raise ValueError('Selected depth invalid')
            else:
                region=None
                if diff>5:raise ValueError('Offscreen auxiliary RGB differs from original')
            kind='visible' if fraction>=.7 else 'partial' if fraction>0 else 'hidden_or_offscreen'
            counts[kind]+=1
            results.append({'schema':'aic_sfp_depth_visibility/v1','episode_id':audit.name,
                            'scene_group_id':stream['scene_group_id'],'split':a.split,
                            'sim_ns':t,'camera':cam,'image':str(Path(original).resolve()),
                            'selected_card':card,'selected_port':port,
                            'geometric_visible_fraction':fraction,'in_frame_fraction':inframe,
                            'front_facing_cos':facing,'projected_opening_px':uv.tolist(),
                            'sample_visible':exposed.tolist(),
                            'relative_opening_pose_tcp':list((np.linalg.inv(tcp)@opening)[:3,3])+
                                list(Rotation.from_matrix((np.linalg.inv(tcp)@opening)[:3,:3]).as_quat()),
                            'rgb_mae':diff,'selected_region_rgb_mae':region,
                            'identifiable_visibility':None,'geometry_training_label_only':True,
                            'depth_deployable_input':False})
        except (ValueError,KeyError,TypeError) as exc:reject[str(exc)]+=1
    with (a.output/'labels.jsonl').open('w') as file:
        for row in results:file.write(json.dumps(row)+'\n')
    report={'schema':'aic_sfp_depth_visibility_summary/v1','rows':len(results),
            'counts':dict(counts),'rejected':dict(reject),'selected_card':card,
            'selected_port':port,'identifiable_labels_certified':False,
            'aperture_patch_m':'x +/-5mm, y +/-2mm; geometric proxy pending mesh verification',
            'actor_has_depth_or_geometry':False,'final_scene':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

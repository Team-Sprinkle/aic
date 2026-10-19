#!/usr/bin/env python3
"""Build selected-SC-aperture occlusion supervision from label-only RGBD.

This is geometric visible-area supervision, not certified visual identifiability.
Do not use its projected polygons or depth maps to select autonomous crops.
"""
import argparse
import bisect
from collections import Counter
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def host(s,audit=None):
    if s.startswith('/repo/'):
        return str(ROOT/Path(s).relative_to('/repo'))
    if audit is not None and s.startswith('/audit/'):
        return str(audit/Path(s).relative_to('/audit'))
    return s


def transform(d):
    m=np.eye(4)
    m[:3,3]=[d['translation'][k] for k in 'xyz']
    m[:3,:3]=Rotation.from_quat([d['rotation'][k] for k in 'xyzw']).as_matrix()
    return m


def select(rows,t):
    i=bisect.bisect_right(rows,t,key=lambda r:r[0])-1
    if i<0:raise ValueError('No causal state')
    return rows[i]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stream',type=Path);p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.stream.read_text())
    audit=a.stream.parent if r['audit']=='/audit' else Path(host(r['audit']))
    pairs=[json.loads(l) for l in (audit/'visibility_pairs.jsonl').open()]
    pairs=[v for v in pairs if r['scored_wall_bounds_ns'][0]<=v['wall_ns']<=r['scored_wall_bounds_ns'][1]]
    a.output.mkdir(parents=True,exist_ok=True)
    edges=r['edges'];counts=Counter();rows=[];alignment=[];rejected=Counter()
    # Aperture interiors from SC Port/model.sdf collision bounds: side walls at
    # +/-0.012047 with width .001687; centre divider width .002992;
    # top/bottom interior near +/- .0038. Leave a .1 mm boundary guard.
    x=np.r_[np.linspace(-.0111,-.0016,12),np.linspace(.0016,.0111,12)]
    y=np.linspace(-.0038,.0038,9)
    grid=np.array([[xx,yy,0.,1.] for xx in x for yy in y]).T
    selected=r['task']['target_module_name']
    for pair in pairs:
        t=pair['sim_ns'];cam=pair['camera']
        try:
            st,state=select(r['streams']['/aic_controller/controller_state'],t)
            if t-st>100000000:raise ValueError('Stale controller')
            def edge(parent,child):return transform(select(edges[parent+'|'+child],t)[1])
            world_base=edge('world','tabletop')@edge('tabletop','base_link')
            world_opening=edge('aic_world','task_board')@edge('task_board','task_board/'+selected)@edge('task_board/'+selected,'task_board/'+selected+'/sc_port_base_link_entrance')
            base_opening=np.linalg.inv(world_base)@world_opening
            mount_tcp=edge('cam_mount/cam_mount_link','ati/base_link')@edge('ati/base_link','ati/tool_link')@edge('ati/tool_link','gripper/hande_base_link')@edge('gripper/hande_base_link','gripper/tcp')
            mount_optical=edge('cam_mount/cam_mount_link',cam+'_camera/camera_link')@edge(cam+'_camera/camera_link',cam+'_camera/sensor_link')@edge(cam+'_camera/sensor_link',cam+'_camera/optical')
            tcp=transform(dict(translation=state['tcp_pose']['position'],rotation=state['tcp_pose']['orientation']))
            base_optical=tcp@np.linalg.inv(mount_tcp)@mount_optical
            optical_opening=np.linalg.inv(base_optical)@base_opening
            xyz=(optical_opening@grid)[:3]
            intr=pair['intrinsics'];assert intr is not None
            k=np.array(intr['k']).reshape(3,3)
            proj=k@xyz
            uv=(proj[:2]/proj[2:]).T
            h,w=intr['height'],intr['width']
            inside=(xyz[2]>.07)&(xyz[2]<20)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            depth=np.load(audit/'visibility_frames'/pair['depth'])['depth_m']
            assert depth.shape==(h,w)
            pixels=np.rint(uv).astype(int);pixels[:,0]=np.clip(pixels[:,0],0,w-1);pixels[:,1]=np.clip(pixels[:,1],0,h-1)
            sampled=depth[pixels[:,1],pixels[:,0]]
            outward=-base_opening[:3,2]
            ray=base_optical[:3,3]-base_opening[:3,3]
            facing=float(np.dot(outward,ray)/(np.linalg.norm(ray)+1e-12))
            # Reject NaN / -inf / zero-depth, allow +inf only as no rendered
            # obstruction. Hole backgrounds should be behind the entrance plane.
            valid_depth=(np.isfinite(sampled)&(sampled>0))|np.isposinf(sampled)
            exposed=inside & valid_depth & (sampled>=xyz[2]-.001) & (facing>.1)
            fraction=float(np.mean(exposed));inframe=float(np.mean(inside))
            rgb_path=audit/'visibility_frames'/pair['rgb']
            # Check colocated RGB against the unchanged ordinary camera stream.
            ct,original=select(r['cameras'][cam],t)
            if t-ct<=50000000:
                rgb=cv2.imread(str(rgb_path));main=cv2.imread(host(original,audit))
                assert rgb.shape==main.shape==(h,w,3)
                mae=float(np.mean(np.abs(rgb.astype(float)-main.astype(float))))
                region_mae=None
                if inside.any():
                    valid_uv=uv[inside]
                    lo=np.maximum(np.floor(valid_uv.min(0)).astype(int)-5,0)
                    hi=np.minimum(np.ceil(valid_uv.max(0)).astype(int)+6,[w,h])
                    aa=rgb[lo[1]:hi[1],lo[0]:hi[0]]
                    bb=main[lo[1]:hi[1],lo[0]:hi[0]]
                    region_mae=float(np.mean(np.abs(aa.astype(float)-bb.astype(float))))
                alignment.append(dict(camera=cam,sim_ns=t,original_age_ms=(t-ct)/1e6,
                                      rgb_mae=mae,selected_region_mae=region_mae))
                if mae>5.:
                    rejected['auxiliary_rgb_disagrees_with_original_camera']+=1
                    continue
                if region_mae is not None and region_mae>10.:
                    rejected['auxiliary_selected_region_disagrees']+=1
                    continue
                if inside.any() and np.mean(~valid_depth[inside])>.1:
                    rejected['selected_region_invalid_depth_not_invisible_label']+=1
                    continue
                # The actor is trained on the ordinary RGB sensor, not the auxiliary rendering.
                rgb_path=Path(host(original,audit))
            else:
                rejected['no_synchronized_original_rgb']+=1
                continue
            counts['positive_geometric_visibility' if fraction>=.7 else 'partial_geometric_visibility' if fraction>0 else 'hidden_or_offscreen']+=1
            record=dict(schema='aic_sc_depth_visibility/v1',episode_id=audit.name,scene_group_id=r['scene_group_id'],split=a.split,
                        sim_ns=t,camera=cam,image=str(rgb_path.resolve()),target_port=int(selected.rsplit('_',1)[-1]),
                        task_vector=[0.,1.,*([1.,0.] if selected.endswith('_0') else [0.,1.]),*([0.]*6)],
                        geometric_visible_fraction=fraction,in_frame_fraction=inframe,front_facing_cos=facing,
                        identifiable_visibility=None,identifiability_review_required=True,
                        projected_opening_px=uv.tolist(),sample_visible=exposed.tolist(),
                        relative_opening_pose_tcp=list((np.linalg.inv(tcp)@base_opening)[:3,3])+list(Rotation.from_matrix((np.linalg.inv(tcp)@base_opening)[:3,:3]).as_quat()),
                        precise_pose_loss_enabled=False,geometry_label_only=True,depth_deployable_input=False)
            rows.append(record)
        except (ValueError,KeyError) as exc:rejected[str(exc)]+=1
    with (a.output/'labels.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    # Diagnostic overlays show label rays, explicitly not model predictions.
    for camera in ('center','left','right'):
        pool=[v for v in rows if v['camera']==camera]
        panels=[]
        for i in np.linspace(0,max(0,len(pool)-1),8).astype(int):
            if not pool:continue
            row=pool[i];im=cv2.imread(row['image'])
            for uv,visible in zip(row['projected_opening_px'],row['sample_visible']):
                if all(np.isfinite(uv)) and 0<=uv[0]<im.shape[1] and 0<=uv[1]<im.shape[0]:cv2.circle(im,tuple(np.rint(uv).astype(int)),2,(0,255,0) if visible else (0,0,255),-1)
            cv2.putText(im,f"LABEL ONLY {row['sim_ns']/1e9:.1f}s exposed {row['geometric_visible_fraction']:.2f}",(12,35),0,.8,(255,255,255),2)
            panels.append(cv2.resize(im,(576,512)))
        if len(panels)==8:cv2.imwrite(str(a.output/(camera+'_label_review.jpg')),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,8,2)]))
    summary=dict(schema='aic_sc_depth_visibility_summary/v1',stream=str(a.stream.resolve()),rows=len(rows),counts=dict(counts),rejected=dict(rejected),
                 rgb_alignment=alignment,occlusion_tolerance_m=.001,identifiable_labels_certified=False,
                 selected_target=selected,aperture_geometry_source='aic_assets/models/SC Port/model.sdf',
                 actor_has_depth_or_geometry=False,final_scene=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rgb_alignment'}))


if __name__=='__main__':main()

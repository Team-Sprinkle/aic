#!/usr/bin/env python3
"""Create native SFP perception labels from recorded transport observations, not executed policies."""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import sfp_ports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-scored-streams',action='store_true',help='Use original complete recordings, not only transport prefixes')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing projection labels')
    a.output.mkdir(parents=True)
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    sfp=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    optics={k:np.asarray(v) for k,v in cal['tcp_to_optical'].items()}
    cache={};last={};count=Counter();episodes={}
    def observations():
        if not a.full_scored_streams:
            for line in a.source.open():yield json.loads(line)
            return
        representatives={}
        for line in a.source.open():
            r=json.loads(line)
            if r.get('images') and r.get('source_scored_stream'):
                representatives.setdefault(r['source_scored_stream'],r)
        for index,(path,representative) in enumerate(representatives.items()):
            stream=json.loads(Path(path).read_text());camera_rows=stream['cameras']
            stamps={c:[x[0] for x in camera_rows[c]] for c in optics}
            commands=[x['sim_ns'] for x in stream['commands'] if x['kind']=='pose']
            if not commands:count['source_has_no_pose_commands']+=1;continue
            stop=max(commands);last_stamp=-1e20
            for stamp,image in camera_rows['center']:
                if stamp>stop or stamp-last_stamp<300_000_000:continue
                images={}
                for c in optics:
                    j=bisect_right(stamps[c],stamp)-1
                    if j<0 or stamp-stamps[c][j]>50_000_000:break
                    image=Path(camera_rows[c][j][1])
                    if not image.is_file():image=Path(path).parent/'frames'/image.name
                    if not image.is_file():break
                    images[c]=str(image)
                if len(images)!=3:count['full_stream_unsynchronized_or_missing_images']+=1;continue
                last_stamp=stamp
                yield dict(representative,images=images,timestamp_s=stamp/1e9,_full_stream=stream)
            print(json.dumps({'stage':'full_stream_projection','sources_completed':index+1,'sources_total':len(representatives),'rows':count['rows']}),flush=True)
    with (a.output/'rows.jsonl').open('w') as out:
        for r in observations():
            if not r.get('images') or any(not isinstance(r['images'].get(c),str) for c in optics):
                count['no_original_paths']+=1;continue
            scene=r['scene_sha256'];t=float(r['timestamp_s'])
            stream_path=r.get('source_scored_stream')
            sampling_key=stream_path or (scene,str(r['episode_index']))
            if t-last.get(sampling_key,-1e9)<.299999:continue
            cfg=r.get('source_scene_yaml') or (str(Path(stream_path).parent/'eval_config.yaml') if stream_path else None)
            if not cfg or not Path(cfg).exists():count['missing_scene_config']+=1;continue
            if cfg not in cache:
                config=yaml.safe_load(Path(cfg).read_text());tid=next(iter(config['trials']))
                board=config['trials'][tid]['scene']['task_board']
                if not stream_path or not Path(stream_path).exists():
                    count['missing_causal_controller_stream']+=1;continue
                stream=r.get('_full_stream') or json.loads(Path(stream_path).read_text())
                controllers=stream['streams']['/aic_controller/controller_state']
                pose_rows=[]
                for stamp,message in controllers:
                    pose=message['tcp_pose']
                    pose_rows.append([stamp,[pose['position'][k] for k in 'xyz']+[pose['orientation'][k] for k in 'xyzw']])
                cache[cfg]=(sfp_ports({'trial_yaml':cfg,'trial_id':tid},sfp),sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5)),
                            [x[0] for x in pose_rows],pose_rows)
            ports,n,controller_times,controllers=cache[cfg]
            camera_tcp={};camera_stamps={};ages={}
            for c,path in r['images'].items():
                if c not in optics:continue
                stamp=int(Path(path).stem.split('_')[-2]);j=bisect_right(controller_times,stamp)-1
                if j<0 or stamp-controller_times[j]>10_000_000:break
                camera_tcp[c]=pose_matrix(controllers[j][1]);camera_stamps[c]=stamp;ages[c]=(stamp-controller_times[j])/1e6
            if len(camera_tcp)!=3:count['missing_or_stale_image_time_pose']+=1;continue
            if max(camera_stamps.values())-min(camera_stamps.values())>50_000_000:
                count['asynchronous_triplet']+=1;continue
            j=bisect_right(controller_times,camera_stamps['center'])-1
            image_state=controllers[j][1];targets=[]
            for card,port,m in ports:
                targets.append(dict(identity=f'sfp_card_{card}_port_{port}',position_base_m=m[:3,3].tolist(),
                                    orientation_base_xyzw=Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                    cameras={c:project(m,camera_tcp[c]@optics[c]) for c in optics}))
            task=np.asarray(r['task_vector']);identity=f'sfp_card_{int(task[4:9].argmax())}_port_{int(task[2:4].argmax())}'
            episode=str(r.get('episode_id',r['episode_index']))
            row=dict(family='sfp',episode=episode,scene_group=scene,split=r['split'],time_s=t,
                     state=image_state,images=r['images'],image_kind='native_path_pending_header_audit',
                     image_pose_age_ms=ages,image_sim_ns=camera_stamps,label_pose_join='latest_past_controller_at_each_image_stamp',
                     targets=targets,selected_identity=identity,card_count=n,geometry_training_label_only=True,
                     source_scene_yaml=cfg,source_transport_labels=str(a.source),projection_is_not_visibility=True)
            row['source_scored_stream']=stream_path;row['complete_recording_sampled']=a.full_scored_streams
            out.write(json.dumps(row,separators=(',',':'))+'\n');last[sampling_key]=t
            count['rows']+=1;episodes[sampling_key]={'episode':episode,'scene_group':scene,'source_stream':stream_path,'split':r['split'],'cards':n,'selected_identity':identity}
    report={'rows_and_rejections':dict(count),'episodes':list(episodes.values()),'geometry_runtime_input':False,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':dict(count),'episodes':len(episodes)}),flush=True)


if __name__=='__main__':main()

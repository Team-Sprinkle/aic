#!/usr/bin/env python3
"""Build perception-only SFP labels from existing Gazebo agent replays.

Actions and scores are never treated as expert demonstrations. Trial geometry
is used only to label projected openings. Complete reset groups are excluded
from held-out native validation and the existing verified cache.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import CAMERAS,sfp_ports


def reset_hash(config):
    trial=next(iter(config['trials'].values()))
    task=next(iter(trial['tasks'].values()))
    payload={'scene':trial['scene'],'robot':config.get('robot'),'task':task}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def candidates(root,native_manifest,old_cache,difficulty):
    excluded={r['scene_reset_sha256'] for r in json.loads(native_manifest.read_text())['rows']}
    old=json.loads((old_cache/'cache.json').read_text())
    old_trials={str(Path(e['trial_yaml']).resolve()) for e in old['episodes'] if e['task']['task_family']=='sfp_to_nic'}
    old_resets={reset_hash(yaml.safe_load(Path(path).read_text())) for path in old_trials}
    buckets=defaultdict(list);audit=Counter()
    for parent in sorted(root.glob('nic_cards_mixed/n400__ec2_*_sfp_*')):
        if difficulty!='both' and not parent.name.endswith('_'+difficulty):continue
        attempts=(parent/'agent_generation/replay_attempts')
        if not attempts.exists():attempts=parent/'replay_attempts'
        for attempt in sorted(attempts.glob('attempt_*')):
            dataset=attempt/'dataset'
            if not (dataset/'meta/info.json').exists():continue
            if int(json.loads((dataset/'meta/info.json').read_text()).get('total_episodes',0))!=1:
                audit['incomplete_dataset_episode_count']+=1;continue
            if not next((dataset/'data').rglob('*.parquet'),None):
                audit['incomplete_dataset_no_parquet']+=1;continue
            if len(list((dataset/'videos').rglob('*.mp4')))<3:
                audit['incomplete_dataset_no_video']+=1;continue
            if not next((dataset/'meta/episodes').rglob('*.parquet'),None):
                audit['incomplete_dataset_no_episode_metadata']+=1;continue
            bits=attempt.name.split('_');trial_id=f'trial_{bits[1]}'
            trial_path=(parent/'trials'/f'{trial_id}.yaml').resolve()
            if not trial_path.exists():audit['no_trial']+=1;continue
            if str(trial_path) in old_trials:audit['verified_duplicate']+=1;continue
            config=yaml.safe_load(trial_path.read_text());trial=next(iter(config['trials'].values()))
            task=next(iter(trial['tasks'].values()));board=trial['scene']['task_board']
            card=int(task['target_module_name'].rsplit('_',1)[-1]);port=int(task['port_name'].rsplit('_',1)[-1])
            count=sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
            digest=reset_hash(config)
            if digest in excluded:audit['native_reset_duplicate']+=1;continue
            if digest in old_resets:audit['verified_reset_duplicate']+=1;continue
            if not board[f'nic_rail_{card}']['entity_present']:audit['missing_selected_card']+=1;continue
            buckets[(card,port,count)].append((digest,trial_path,dataset,trial_id))
    return buckets,audit


def video_frames(dataset,info,meta,indices,camera):
    key='observation.images.'+camera+'_camera'
    prefix='videos/'+key
    path=dataset/info['video_path'].format(video_key=key,
        chunk_index=int(meta[prefix+'/chunk_index']),file_index=int(meta[prefix+'/file_index']))
    fps=float(info['fps']);offset=float(meta[prefix+'/from_timestamp'])
    wanted={int(i):offset+int(i)/fps for i in indices}
    result={};index=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        for frame in container.decode(stream):
            if index in wanted:
                actual=float(frame.pts*stream.time_base);expected=wanted[index]
                if abs(actual-expected)>.027:
                    raise ValueError(f'Video/state mismatch {path} frame {index}: {actual} vs {expected}')
                result[index]=frame.to_ndarray(format='rgb24')
                if len(result)==len(wanted):break
            index+=1
    if len(result)!=len(wanted):raise ValueError(f'Incomplete video {path}: {len(result)}/{len(wanted)}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/agent'))
    p.add_argument('--native-manifest',type=Path,required=True)
    p.add_argument('--old-cache',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--sfp-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes-per-setting',type=int,default=4)
    p.add_argument('--max-episodes',type=int,default=200)
    p.add_argument('--stride-frames',type=int,default=20)
    p.add_argument('--difficulty',choices=('easy','hard','both'),default='both')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    if a.stride_frames<1:raise ValueError('Invalid stride')
    buckets,audit=candidates(a.source,a.native_manifest,a.old_cache,a.difficulty)
    selection=[];used=set()
    for setting in sorted(buckets):
        for digest,trial,dataset,trial_id in buckets[setting]:
            if digest in used:continue
            selection.append((setting,digest,trial,dataset,trial_id));used.add(digest)
            if sum(x[0]==setting for x in selection)>=a.episodes_per_setting:break
    selection=selection[:a.max_episodes]
    if not selection:raise ValueError('No suitable existing agent replays')
    camera=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(camera['tcp_to_optical'][c],float) for c in CAMERAS}
    sfp_cal=json.loads(a.sfp_calibration.read_text())
    a.output.mkdir(parents=True);(a.output/'images').mkdir()
    episodes=[];total=0
    with (a.output/'rows.jsonl').open('w') as out:
        for number,(setting,digest,trial_path,dataset,trial_id) in enumerate(selection):
            info=json.loads((dataset/'meta/info.json').read_text())
            frame=pd.read_parquet(next((dataset/'data').rglob('*.parquet'))).sort_values('frame_index')
            meta=pd.read_parquet(next((dataset/'meta/episodes').rglob('*.parquet'))).iloc[0]
            if int(info['total_episodes'])!=1:raise ValueError(f'Expected one episode: {dataset}')
            indices=[i for i in range(0,len(frame),a.stride_frames)]
            frames={c:video_frames(dataset,info,meta,indices,c) for c in CAMERAS}
            ports=sfp_ports({'trial_yaml':str(trial_path),'trial_id':trial_id},sfp_cal)
            card,port,_=setting;episode=f'agent_perception_{number:04d}'
            directory=a.output/'images'/episode;directory.mkdir()
            for index in indices:
                state=np.asarray(frame.iloc[index]['observation.state'],float)
                if not np.isfinite(state[:7]).all():raise ValueError(f'Nonfinite state {dataset} {index}')
                tcp=pose_matrix(state);targets=[]
                for cand_card,cand_port,m in ports:
                    targets.append({'identity':f'sfp_card_{cand_card}_port_{cand_port}',
                                    'position_base_m':m[:3,3].tolist(),
                                    'orientation_base_xyzw':Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                    'cameras':{c:project(m,tcp@optics[c]) for c in CAMERAS}})
                images={}
                for c in CAMERAS:
                    path=directory/f'{index:06d}_{c}.jpg'
                    rgb=frames[c][index]
                    cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                    images[c]=str(path.resolve())
                row={'family':'sfp','episode':episode,'scene_group':digest,'split':'train',
                     'time_s':float(frame.iloc[index]['timestamp']),'state':state[:7].tolist(),
                     'images':images,'image_kind':'agent_288x256_rgb_jpeg',
                     'source_dataset':str(dataset),'targets':targets,
                     'selected_identity':f'sfp_card_{card}_port_{port}',
                     'card_count':setting[2], 'expert_action_source':False,
                     'geometry_training_label_only':True,'literal_visibility_label':False}
                out.write(json.dumps(row,separators=(',',':'))+'\n');total+=1
            episodes.append({'episode':episode,'source_dataset':str(dataset),'trial_yaml':str(trial_path),
                             'scene_group':digest,'setting':setting,'rows':len(indices),
                             'actions_used':False})
            if (number+1)%20==0:print(json.dumps({'episodes':number+1,'rows':total}),flush=True)
    summary={'schema':'aic_sfp_agent_perception_only/v1','episodes':len(episodes),'rows':total,
             'difficulty':a.difficulty,
             'settings':{str(k):sum(x['setting']==k for x in episodes) for k in sorted(buckets)},
             'episode_manifest':episodes,'excluded':dict(audit),
             'native_validation_reset_overlap':False,'verified_cache_episode_overlap':False,
             'actions_used':False,'scores_used_as_expert':False,'geometry_runtime':False,
             'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'episodes':len(episodes),'rows':total,'excluded':dict(audit)}))


if __name__=='__main__':main()

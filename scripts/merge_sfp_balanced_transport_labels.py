#!/usr/bin/env python3
"""Freeze a new SFP BC train/development manifest without old skewed validation."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new-train',type=Path,required=True)
    p.add_argument('--new-development',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous/partial merge')
    a.output.mkdir(parents=True)
    counts=Counter();episodes=defaultdict(set);groups=defaultdict(set);source_groups=defaultdict(set)
    old_cutoff_seen=set();old_last_frame={};old_pre_cutoff_rows=0
    with (a.output/'sfp_labels.jsonl').open('w') as out:
        for source,path in [('old',a.old),('new_train',a.new_train),
                            ('new_development',a.new_development)]:
            for line in path.open():
                row=json.loads(line)
                if source=='old':
                    if row['split']!='train':continue
                    episode=int(row['episode_index']);frame=int(row['frame_index'])
                    if frame<=old_last_frame.get(episode,-1):
                        raise ValueError(f'Old SFP frame order regressed in episode {episode}')
                    old_last_frame[episode]=frame
                    if episode in old_cutoff_seen:continue
                    rel=row['observed_tcp_pose_port_frame']
                    old_pre_cutoff_rows+=1
                    if rel[2]>=-.070 and math.hypot(rel[0],rel[1])<=.010:
                        old_cutoff_seen.add(episode)
                elif row['split']!=('train' if source=='new_train' else 'validation'):
                    raise ValueError(f'Split mismatch: {source} {row["split"]}')
                task=np.asarray(row['task_vector'],float)
                if task.shape!=(10,) or task[0]!=1:raise ValueError('Invalid SFP task code')
                card=int(np.argmax(task[4:9]));port=int(np.argmax(task[2:4]))
                if task[4+card]!=1 or task[2+port]!=1:raise ValueError('Invalid selected target code')
                split=row['split'];episode=int(row['episode_index']);scene=row['scene_sha256']
                counts[(source,split,card,port)]+=1
                episodes[(source,split,card,port)].add(episode)
                groups[split].add(scene)
                source_groups[source].add(scene)
                out.write(line)
    if groups['train'] & groups['validation']:
        raise ValueError(f'Complete reset leak: {len(groups["train"] & groups["validation"])}')
    if old_cutoff_seen!={e for key,values in episodes.items() if key[0]=='old' for e in values}:
        raise ValueError('Some old scored-successful training episodes never reached a transport cutoff')
    if len({(key[2],key[3]) for key in episodes if key[0]=='new_train'})!=10:
        raise ValueError('New train controls do not cover all ten card/port requests')
    if len({(key[2],key[3]) for key in episodes if key[0]=='new_development'})!=10:
        raise ValueError('New development controls do not cover all ten card/port requests')
    manifest=dict(schema='aic_sfp_balanced_transport_merge/v1',
        source_sha256={str(path):sha(path) for path in (a.old,a.new_train,a.new_development)},
        rows_by_source_split_target={str(k):v for k,v in sorted(counts.items())},
        episodes_by_source_split_target={str(k):len(v) for k,v in sorted(episodes.items())},
        reset_groups={k:len(v) for k,v in groups.items()},
        old_new_train_reset_overlap=len(source_groups['old'] & source_groups['new_train']),
        old_validation_excluded=True,new_validation_complete_episode_grouped=True,
        old_train_truncated_at_first_buffered_arrival=True,
        old_train_episodes_with_cutoff=len(old_cutoff_seen),old_pre_cutoff_rows=old_pre_cutoff_rows,
        invalid_agent_actions_used=False,reserved_final_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(rows=sum(counts.values()),new_train_episodes=sum(len(v) for k,v in episodes.items() if k[0]=='new_train'),
        new_development_episodes=sum(len(v) for k,v in episodes.items() if k[0]=='new_development'),
        reset_groups=manifest['reset_groups'])))


if __name__=='__main__':main()

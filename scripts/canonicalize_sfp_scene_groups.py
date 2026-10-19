#!/usr/bin/env python3
"""Canonicalize historical SFP reset IDs to scene+robot JSON hashes.

The historical curated labels used another hash convention. Paired new teacher
controls already use the scene+robot convention; this metadata-only revision
allows their same-board original controls to be recognized by the sampler.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import yaml


def canonical(source):
    config=yaml.safe_load(Path(source).read_text())
    trial=next(iter(config['trials'].values()))
    value={'scene':trial['scene'],'robot':config.get('robot')}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output-labels',type=Path,required=True)
    p.add_argument('--output-cache',type=Path,required=True)
    a=p.parse_args()
    if a.output_labels.exists() or a.output_cache.exists():p.error('Preserve previous outputs')
    a.output_labels.mkdir(parents=True);a.output_cache.mkdir(parents=True)
    source_hash={};episode_scene={};split_scene=defaultdict(set);changed=defaultdict(set)
    with (a.output_labels/'sfp_labels.jsonl').open('w') as out:
        for line in a.labels.open():
            row=json.loads(line)
            source=row.get('source_scene_yaml')
            if source:
                if source not in source_hash:source_hash[source]=canonical(source)
                old=row['scene_sha256'];row['scene_sha256']=source_hash[source]
                if old!=row['scene_sha256']:
                    row['source_scene_sha256_precanonical']=old
                    changed[row['split']].add(row['episode_index'])
            key=int(row['episode_index'])
            if key in episode_scene and episode_scene[key]!=row['scene_sha256']:
                raise ValueError(f'Changing scene within episode {key}')
            episode_scene[key]=row['scene_sha256']
            split_scene[row['split']].add(row['scene_sha256'])
            out.write(json.dumps(row,separators=(',',':'))+'\n')
    if split_scene['train'] & split_scene['validation']:
        raise ValueError('Canonicalization exposed train/validation leakage')
    scenes=defaultdict(set)
    for line in (a.output_labels/'sfp_labels.jsonl').open():
        row=json.loads(line)
        if row['split']=='train':
            task=row['task_vector']
            scenes[row['scene_sha256']].add((task[4:9].index(1),task[2:4].index(1)))
    paired={scene for scene,requests in scenes.items() if len(requests)>1}
    if len(paired)<20:raise ValueError(f'Unexpectedly few same-board paired scenes: {len(paired)}')
    for name in ('features.npy','camera_offsets.npy'):
        os.link(a.cache/name,a.output_cache/name)
    cache_rows=0
    with (a.output_cache/'rows.jsonl').open('w') as out:
        for line in (a.cache/'rows.jsonl').open():
            row=json.loads(line)
            if row['family']=='sfp':
                row['scene']=episode_scene[int(row['episode'])]
            out.write(json.dumps(row,separators=(',',':'))+'\n')
            cache_rows+=1
    source_manifest=json.loads((a.cache/'manifest.json').read_text())
    label_sha=hashlib.sha256()
    with (a.output_labels/'sfp_labels.jsonl').open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):label_sha.update(block)
    source_manifest.update(schema='aic_temporal_transport_frozen_features_scene_rekey/v1',
        source_cache=str(a.cache),source_labels=str(a.labels),
        canonical_labels=str(a.output_labels/'sfp_labels.jsonl'),
        canonical_labels_sha256=label_sha.hexdigest(),
        scene_metadata_only_rekey=True,features_and_camera_offsets_hardlinked=True,
        historical_episodes_rekeyed={k:len(v) for k,v in changed.items()},
        paired_train_scenes=len(paired),train_validation_scene_overlap=0,
        final_scenes_opened=False)
    (a.output_cache/'manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    summary=dict(labels=str(a.output_labels/'sfp_labels.jsonl'),cache=str(a.output_cache),
        labels_sha256=label_sha.hexdigest(),
        historical_episodes_rekeyed={k:len(v) for k,v in changed.items()},
        paired_train_scenes=len(paired),cache_rows=cache_rows,
        features_changed=False,actions_changed=False,final_scenes_opened=False)
    (a.output_labels/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

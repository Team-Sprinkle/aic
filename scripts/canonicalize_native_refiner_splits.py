#!/usr/bin/env python3
"""Enforce physical scene/robot grouping across perception sources without opening final data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve split audit')
    a.output.mkdir(parents=True);rows=[];configs={};groups=defaultdict(set);rejected=Counter();cutoffs={}
    for line in a.source.open():
        r=json.loads(line)
        path=Path(r.get('source_scene_yaml') or str(Path(r['images']['center']).parent.parent/'eval_config.yaml'))
        if not path.is_file():
            rejected['no_physical_reset_config']+=1;continue
        if str(path) not in configs:
            config=yaml.safe_load(path.read_text());trial=next(iter(config['trials'].values()))
            # Public target request and episode name do not make a new physical reset.
            physical={'scene':trial['scene'],'robot':config.get('robot')}
            key=hashlib.sha256(json.dumps(physical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            configs[str(path)]=key
        key=configs[str(path)];groups[key].add(r['split'])
        stream=r.get('source_stream') or r.get('source_scored_stream') or str(path.parent/'scored_stream.json')
        if stream not in cutoffs:
            if not Path(stream).exists():
                rejected['missing_terminal_command_stream']+=1;continue
            data=json.loads(Path(stream).read_text())
            commands=[x['sim_ns'] for x in data['commands'] if x['kind']=='pose']
            if not commands:
                rejected['no_recorded_pose_commands']+=1;continue
            cutoffs[stream]=max(commands)
        stamps={c:int(Path(image).stem.split('_')[-2]) for c,image in r['images'].items()}
        if max(stamps.values())>cutoffs[stream]:
            rejected['images_after_last_pose_command']+=1;continue
        r['original_row_time_s']=r['time_s'];r['time_s']=stamps['center']/1e9
        r['image_sim_ns']=stamps;r['terminal_clipped_to_last_pose_command']=True
        r['original_episode']=r['episode']
        recording=str(Path(r['images']['center']).parent)
        r['episode']=r['episode']+'__recording'+hashlib.sha256(recording.encode()).hexdigest()[:10]
        r['original_scene_group']=r['scene_group'];r['scene_group']=key;r['physical_reset_config']=str(path)
        rows.append(r)
    mixed={key for key,value in groups.items() if len(value)>1}
    excluded=[r for r in rows if r['scene_group'] in mixed];kept=[r for r in rows if r['scene_group'] not in mixed]
    rejected['rows_quarantined_cross_source_reset_split_overlap']=len(excluded)
    with (a.output/'rows.jsonl').open('w') as out:
        for r in kept:out.write(json.dumps(r,separators=(',',':'))+'\n')
    episodes=defaultdict(set);coverage=defaultdict(set)
    for r in kept:
        episodes[f"{r['family']}/{r['split']}"].add(r['episode'])
        coverage[f"{r['family']}/{r['split']}/cards{r['card_count']}/{r['selected_identity']}"].add(r['scene_group'])
    summary={'schema':'aic_native_physical_reset_split_audit/v1','rows':len(kept),
             'episodes':{k:len(v) for k,v in episodes.items()},'coverage_distinct_reset_configs':{k:len(v) for k,v in sorted(coverage.items())},
             'physical_reset_groups':len({r['scene_group'] for r in kept}),
             'quarantined_groups':sorted(mixed),'rejected':dict(rejected),'target_request_excluded_from_reset_identity':True,
             'source':str(a.source),'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['rows','episodes','physical_reset_groups','rejected']}),flush=True)


if __name__=='__main__':main()

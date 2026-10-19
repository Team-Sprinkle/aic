#!/usr/bin/env python3
"""Link only verified full teacher takeovers for causal SFP DAgger relabeling."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-episodes',type=int,default=6)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous selection')
    report=json.loads((a.collection/'summary.json').read_text())
    selected=report['full_successes']
    if len(selected)<a.min_episodes:
        raise ValueError(f'Only {len(selected)} full takeovers; need {a.min_episodes}')
    scene_groups=set();items=[]
    a.output.mkdir(parents=True)
    for row in selected:
        if not row['eligible'] or row['tier3']!=75 or row['actor_commands']<1:
            raise ValueError('Nonexpert takeover in selection')
        audit=Path(row['audit']).resolve()
        stream=json.loads((audit/'scored_stream.json').read_text())
        if stream['official_tier3']!=75 or not (audit/'teacher_takeover.json').is_file():
            raise ValueError(f'Score/takeover mismatch: {audit}')
        group=stream['scene_group_id']
        if group in scene_groups:raise ValueError('Duplicate reset configuration')
        scene_groups.add(group)
        case=f"{row['case']}_dagger_r{row['repeat']}"
        (a.output/case).symlink_to(audit)
        parts=row['case'].split('_')
        card=int(parts[0].removeprefix('card'))
        port=int(parts[1].removeprefix('port'))
        items.append({'case':case,'split':'train','card_count':5,
                      'selected_card':card,'selected_port':port,
                      'source_audit':str(audit),'scored_scene_group_id':group,
                      'teacher_takeover_after_actor':True,'reserved_final':False})
    (a.output/'collection_manifest.json').write_text(json.dumps({
        'schema':'aic_sfp_stage_preserving_dagger_selection/v1',
        'rows':items,'full_insertion_only':True,'post_takeover_commands_only':True,
        'all_training_scenes':True,'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps({'episodes':len(items),'scene_groups':len(scene_groups),
                      'output':str(a.output)}))

if __name__=='__main__':main()

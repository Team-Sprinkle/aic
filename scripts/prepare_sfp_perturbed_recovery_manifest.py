#!/usr/bin/env python3
"""Build a fixed ten-identity, training-only label manifest for recovery runs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-selection',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--negative-pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior manifest')
    rows=json.loads((a.original_selection/'collection_manifest.json').read_text())['rows']
    chosen=sorted((r for r in rows if r['split']=='train' and 'extra0' in r['case']),
                  key=lambda r:(r['selected_card'],r['selected_port']))
    if len(chosen)!=10 or {(r['selected_card'],r['selected_port']) for r in chosen}!={(c,p) for c in range(5) for p in range(2)}:
        raise ValueError('Exactly one new training reset per card/port required')
    items=[]
    for r in chosen:
        card=r['selected_card'];port=r['selected_port']
        audit=(a.negative_pilot if (card,port)==(2,1) else
               a.collection/(r['case']+'_perturb')).resolve()
        if not (audit/'scored_stream.json').is_file() or not (audit/'near_port_perturbation.json').is_file():
            raise FileNotFoundError(audit)
        stream=json.loads((audit/'scored_stream.json').read_text())
        nominal=json.loads((a.original_selection/r['case']/'scored_stream.json').read_text())
        if stream['official_tier3']!=75 or stream['scene_group_id']!=nominal['scene_group_id']:
            raise ValueError(f'Expert score or reset mismatch: {audit}')
        case=r['case']+'_perturb'
        items.append({'case':case,'split':'train','card_count':5,
                      'selected_card':card,'selected_port':port,
                      'scene_reset_sha256':r['scene_reset_sha256'],
                      'scored_scene_group_id':stream['scene_group_id'],
                      'source_audit':str(audit),'perturb_sign':1 if (card+port)%2==0 else -1,
                      'reserved_final':False})
    a.output.mkdir(parents=True)
    for item in items:
        (a.output/item['case']).symlink_to(item['source_audit'])
    (a.output/'collection_manifest.json').write_text(json.dumps({
        'schema':'aic_sfp_perturbed_recovery_selection/v1','rows':items,
        'all_training_scenes':True,'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps({'cases':len(items),'output':str(a.output)}))


if __name__=='__main__':main()

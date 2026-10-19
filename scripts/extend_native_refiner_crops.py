#!/usr/bin/env python3
"""Add existing acquisition-state RGB crops without copying the old46GB bank.

Crop centres are the requested coarse model's predictions. Geometry supplies
supervision and training eligibility only. No new simulator collection occurs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from sc_requested_history_observation import RequestedPortHistoryObservation
from train_joint_native_opening_refiner import code,CAMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','context','coarse','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--family',choices=('sc','sfp'),default='sfp')
    p.add_argument('--source-domain',default='SFP_acquisition_gap')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    coarse=RequestedPortHistoryObservation(a.coarse,'cpu')
    context=[r for r in map(json.loads,(a.context/'frame_context.jsonl').open())
             if r.get('training_source')==a.source_domain and r['family']==a.family]
    if not context:raise ValueError('No matching source domain/family')
    spatial=np.load(a.context/'coarse_spatial_8x9.npy',mmap_mode='r')
    extra_path=(a.output/'acquisition_crops_192.npy').resolve()
    array=np.lib.format.open_memmap(extra_path,mode='w+',dtype=np.uint8,
        shape=(len(context)*6,192,192,3))
    samples=[];rejected=Counter()
    for ordinal,r in enumerate(context):
        coarse.reset();coarse.observe(torch.from_numpy(np.array(spatial[r['feature_row']])).float(),r['state'])
        wanted={r['selected_identity'],r['selected_identity'].rsplit('_',1)[0]+'_'+str(1-int(r['selected_identity'].rsplit('_',1)[1]))}
        images={c:cv2.cvtColor(cv2.imread(r['images'][c]),cv2.COLOR_BGR2RGB) for c in CAMS}
        for t in r['targets']:
            if t['identity'] not in wanted:continue
            parts=t['identity'].split('_');q=(int(parts[2])*2+int(parts[-1]) if a.family=='sfp' else int(parts[-1]));pred=coarse.predict(q)
            for ci,c in enumerate(CAMS):
                gt=t['cameras'][c]
                if not gt[3]:rejected['projected_offscreen']+=1;continue
                centre=np.asarray(pred['pixels'][ci]);truth=np.array(gt[:2]);inside=np.max(np.abs(truth-centre))<=64
                if r['split']=='train' and not inside:rejected['training_truth_outside_predicted_crop_margin']+=1;continue
                origin=np.round(centre-96).astype(int);index=len(samples)
                array[index]=cv2.warpAffine(images[c],np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(192,192))
                samples.append(dict(index=index,crop_store=str(extra_path),family=a.family,episode=r['episode'],
                    scene_group=r['scene_group'],split=r['split'],camera=c,identity=t['identity'],
                    selected=t['identity']==r['selected_identity'],cards=r['card_count'],time_s=r['time_s'],
                    state=r['state'][:7],target_position_base_m=t['position_base_m'],
                    target_orientation_base_xyzw=t['orientation_base_xyzw'],task=code(r,t['identity'],c).tolist(),
                    truth_local_192=(truth-origin).tolist(),inside_margin=bool(inside),source_image=r['images'][c],
                    coarse_xy=centre.tolist(),truth_xy=truth.tolist(),source_domain=a.source_domain))
        if ordinal%100==0:print(json.dumps(dict(frames=ordinal,total=len(context),added_crops=len(samples))),flush=True)
    array.flush()
    original=a.original/'samples.jsonl'
    with (a.output/'samples.jsonl').open('w') as writer:
        with original.open() as reader:
            for line in reader:writer.write(line)
        for sample in samples:writer.write(json.dumps(sample)+'\n')
    (a.output/'native_crops_192.npy').symlink_to((a.original/'native_crops_192.npy').resolve())
    report=dict(schema='acquisition_native_refiner_extension/v1',added_frames=len(context),
        family=a.family,source_domain=a.source_domain,
        added_views=dict(Counter(r['split'] for r in samples)),
        added_episodes={s:len({r['episode'] for r in samples if r['split']==s}) for s in ('train','validation')},
        source_checkpoint_sha256=hashlib.sha256(a.coarse.read_bytes()).hexdigest(),
        original_manifest_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),rejected=dict(rejected),
        original_data_modified=False,new_collection=False,crops_chosen_from_RGB_prediction=True,
        labels_are_not_certified_visibility=True,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

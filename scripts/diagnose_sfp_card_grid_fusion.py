#!/usr/bin/env python3
"""Check whether an RGB-predicted card/port grid repairs wrong target identity.

The fit uses only per-camera RGB locator outputs and public card/port indices.
Label projections are read only after fitting for grouped diagnostic scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def estimate(pred,selected):
    data=[]
    for key,item in pred.items():
        if not key.startswith('sfp_card_') or item['projected_inframe_probability']<.7:continue
        words=key.split('_');card=int(words[2]);port=int(words[4])
        data.append((np.asarray([1.,card,port]),np.asarray(item['xy']),
                     float(item['projected_inframe_probability'])))
    if len(data)<6:return None
    design=np.asarray([r[0] for r in data]);xy=np.asarray([r[1] for r in data]);weight=np.asarray([r[2] for r in data])
    rng=np.random.default_rng(20261002)
    best=None
    for _ in range(150):
        chosen=rng.choice(len(data),3,replace=False)
        if np.linalg.matrix_rank(design[chosen])<3:continue
        coef=np.linalg.lstsq(design[chosen],xy[chosen],rcond=None)[0]
        residual=np.linalg.norm(design@coef-xy,axis=1)
        inliers=residual<18.
        score=float(np.sum(weight[inliers]))
        if best is None or score>best[0]:best=(score,inliers)
    if best is None or best[1].sum()<6:return None
    inliers=best[1];coef=np.linalg.lstsq(design[inliers],xy[inliers],rcond=None)[0]
    residual=np.linalg.norm(design[inliers]@coef-xy[inliers],axis=1)
    words=selected.split('_');card=int(words[2]);port=int(words[4])
    return {'xy':(np.asarray([1.,card,port])@coef).tolist(),
            'inliers':int(inliers.sum()),'fit_p95_px':float(np.quantile(residual,.95))}


def stats(values):
    return {'n':len(values),'p50_px':float(np.median(values)) if values else None,
            'p95_px':float(np.quantile(values,.95)) if values else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    pred={(r['row'],r['camera']):r for r in map(json.loads,a.predictions.open())}
    errors={split:{name:[] for name in ('direct','grid','grid_supported')} for split in ('selection','test')}
    grouped=[]
    for i,r in enumerate(rows):
        if r['split']!='validation':continue
        card=int(r['selected_identity'].split('_')[2]);port=int(r['selected_identity'].split('_')[4])
        split='selection' if (card+port)%2==0 else 'test'
        truth=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        for cam in ('center','left','right'):
            if not truth['cameras'][cam][3]:continue
            item=pred[i,cam];direct=item['predicted'][r['selected_identity']]['xy']
            label=np.asarray(truth['cameras'][cam][:2])
            fit=estimate(item['predicted'],r['selected_identity'])
            de=float(np.linalg.norm(np.asarray(direct)-label));errors[split]['direct'].append(de)
            if fit:
                ge=float(np.linalg.norm(np.asarray(fit['xy'])-label))
                errors[split]['grid_supported'].append(ge)
                errors[split]['grid'].append(ge)
            else:errors[split]['grid'].append(de)
            grouped.append({'episode':r['episode'],'split':split,'camera':cam,
                            'direct_error_px':de,'grid_error_px':ge if fit else None,
                            'grid':fit,'truth_scoring_only':label.tolist()})
    summary={k:{'direct':stats(v['direct']),'grid_fallback_direct':stats(v['grid']),
                'grid_supported_only':stats(v['grid_supported']),
                'grid_support_fraction':len(v['grid_supported'])/len(v['direct'])}
             for k,v in errors.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_sfp_card_grid_diagnostic/v1','metrics':summary,
        'rows':grouped,'runtime_true_geometry':False,'truth_scoring_only':True,
        'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()

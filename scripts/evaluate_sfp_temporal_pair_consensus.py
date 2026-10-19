#!/usr/bin/env python3
"""Observation-only static-port lock from repeated two-camera triangulations.

Ground-truth target geometry is used only after the lock for scoring. This is
a bounded diagnostic for selected-port confusion in one camera.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate

CAMERAS=('center','left','right')
PAIRS=tuple(itertools.combinations(CAMERAS,2))


def stats(values):
    return {'n':len(values),'median_mm':float(np.median(values)) if values else None,
            'p95_mm':float(np.quantile(values,.95)) if values else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--fine-predictions',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=int,default=15)
    p.add_argument('--radius-mm',type=float,default=.5)
    p.add_argument('--min-support',type=int,default=5)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing result')
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    pred={(q['row'],q['camera']):q for q in map(json.loads,a.fine_predictions.open())}
    cal=json.loads(a.calibration.read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='validation':groups[r['episode']].append((i,r))
    results=[]
    for episode,group in groups.items():
        group.sort(key=lambda q:q[1]['time_s'])
        candidates=[];lock=None
        for i,r in group:
            base_tcp=pose_matrix(r['state'][:7]);poses={c:base_tcp@optics[c] for c in CAMERAS}
            points={c:pred[i,c]['predicted'][r['selected_identity']]['xy'] for c in CAMERAS}
            frame=[]
            for pair in PAIRS:
                try:xyz=triangulate({c:points[c] for c in pair},poses)
                except np.linalg.LinAlgError:continue
                if np.isfinite(xyz).all() and np.linalg.norm(xyz)<2:
                    frame.append((pair,xyz))
            candidates.append((r['time_s'],frame))
            candidates=candidates[-a.window:]
            best=None
            for anchor_time,anchor_frame in candidates:
                for anchor_pair,anchor in anchor_frame:
                    support=[]
                    for time,frame_values in candidates:
                        near=[(pair,point) for pair,point in frame_values
                              if np.linalg.norm(point-anchor)*1000<=a.radius_mm]
                        if near:
                            chosen=min(near,key=lambda x:np.linalg.norm(x[1]-anchor))
                            support.append((time,chosen[0],chosen[1]))
                    if best is None or len(support)>len(best):best=support
            if best is not None and len(best)>=a.min_support:
                center=np.median([x[2] for x in best],axis=0)
                spread=max(np.linalg.norm(x[2]-center)*1000 for x in best)
                if spread<=a.radius_mm:
                    target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
                    truth=np.asarray(target['position_base_m'])
                    rotation=Rotation.from_quat(target['orientation_base_xyzw']).as_matrix()
                    delta=rotation.T@(center-truth)*1000
                    lock={'episode':episode,'identity':r['selected_identity'],
                          'time_s':r['time_s'],'support_frames':len(best),
                          'support_pairs':{str(pair):sum(x[1]==pair for x in best) for pair in PAIRS},
                          'spread_mm':float(spread),'lateral_mm':float(np.linalg.norm(delta[:2])),
                          'axial_mm':float(abs(delta[2])),'error_3d_mm':float(np.linalg.norm(delta)),
                          'predicted_base_position_m':center.tolist(),
                          'truth_scoring_only':truth.tolist()}
                    break
        if lock is None:lock={'episode':episode,'identity':group[0][1]['selected_identity'],'no_lock':True}
        results.append(lock)
    selection=[r for r in results if int(r['identity'].split('_')[2])%2==0]
    test=[r for r in results if int(r['identity'].split('_')[2])%2==1]
    def report(items):
        accepted=[r for r in items if not r.get('no_lock')]
        return {'episodes':len(items),'locked':len(accepted),'lateral':stats([r['lateral_mm'] for r in accepted]),
                'axial':stats([r['axial_mm'] for r in accepted]),
                'within_0p5mm_fraction':sum(r['lateral_mm']<=.5 for r in accepted)/len(accepted) if accepted else None}
    output={'schema':'sfp_temporal_pair_consensus/v1','method':{'window':a.window,
            'radius_mm':a.radius_mm,'min_support':a.min_support},
            'selection':report(selection),'development_test':report(test),'episodes':results,
            'runtime_geometry_input':False,'reserved_final_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('selection','development_test')}))


if __name__=='__main__':main()

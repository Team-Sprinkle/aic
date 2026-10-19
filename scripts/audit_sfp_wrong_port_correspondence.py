#!/usr/bin/env python3
"""Posthoc pixel audit of accepted SFP poses against requested and neighbor ports."""

import argparse
import json
from pathlib import Path

import numpy as np


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--cascade',type=Path,required=True)
    p.add_argument('--episode',required=True)
    p.add_argument('--times-s',nargs='+',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    labels=[json.loads(x) for x in a.labels.open()]
    labels={float(r['time_s']):r for r in labels if r['episode']==a.episode}
    cascade=json.loads(a.cascade.read_text())
    cascade={float(r['time_s']):r for r in cascade['episode_rows'] if r['episode']==a.episode}
    records=[]
    for time_s in a.times_s:
        row=labels[time_s];inference=cascade[time_s]
        requested=row['selected_identity']
        if not inference['accepted']:raise ValueError(f'Not an accepted prediction at {time_s}')
        candidates=[]
        for target in row['targets']:
            distances={}
            for camera,point in zip(CAMERAS,inference['diagnostic']['fine_pixels_px']):
                projection=target['cameras'][camera]
                distances[camera]={'error_px':float(np.linalg.norm(np.asarray(point)-projection[:2])),
                                   'projected_inframe':bool(projection[3])}
            candidates.append({'identity':target['identity'],'by_camera':distances,
                'max_error_px':max(x['error_px'] for x in distances.values())})
        requested_record=next(x for x in candidates if x['identity']==requested)
        nearest=min(candidates,key=lambda x:x['max_error_px'])
        records.append({'time_s':time_s,'requested_identity':requested,
            'predicted_nearest_identity_audit_only':nearest['identity'],
            'requested_by_camera':requested_record['by_camera'],
            'nearest_by_camera':nearest['by_camera'],
            'all_requested_projected_inframe':all(x['projected_inframe']
                for x in requested_record['by_camera'].values()),
            'fine_pixels_px':inference['diagnostic']['fine_pixels_px'],
            'images':row['images'],'reprojection_residual_px':inference['diagnostic']['reprojection_px']})
    result={'schema':'sfp_wrong_port_correspondence_audit/v1',
        'geometry_used_only_to_score_saved_predictions':True,
        'runtime_true_geometry':False,'episode':a.episode,'records':records,
        'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'episode':a.episode,'records':[{k:v for k,v in r.items()
        if k in ('time_s','requested_identity','predicted_nearest_identity_audit_only',
                 'all_requested_projected_inframe','reprojection_residual_px')}
        for r in records]},indent=2))


if __name__=='__main__':main()

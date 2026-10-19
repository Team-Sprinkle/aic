#!/usr/bin/env python3
"""Causal RGB-rail 3D reference gate for frozen SFP port candidates."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import FOCAL, WIDTH, HEIGHT
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px

CAMS=('center','left','right')


def fit_horizontal_pair(rail,camera,used,seed):
    """Fit a 160 mm horizontal CAD rail to observation-predicted image points."""
    observed={(c,end):np.asarray(rail['endpoint_xy_normalized'][CAMS.index(c)][end])
              *[1151,1023] for c in used for end in (0,1)}
    inverses={c:np.linalg.inv(camera[c]) for c in used}

    def points(v):
        x,y,z,yaw=v
        return (np.array([x,y,z]),np.array([x+.16*np.cos(yaw),y+.16*np.sin(yaw),z]))

    def residual(v):
        xyz=points(v);out=[]
        for c in used:
            for end in (0,1):
                q=inverses[c][:3,:3]@xyz[end]+inverses[c][:3,3]
                if q[2]<=.02:
                    out.extend((1e4,1e4))
                else:
                    pixel=np.array([WIDTH/2+FOCAL*q[0]/q[2],
                                    HEIGHT/2+FOCAL*q[1]/q[2]])
                    out.extend(pixel-observed[(c,end)])
        return np.asarray(out)

    delta=seed[1]-seed[0]
    yaw=float(np.clip(np.arctan2(delta[1],delta[0]),1.201,1.799))
    lower=np.array([-.6,0.,-.05,1.2]);upper=np.array([-.25,.4,.2,1.8])
    initial=np.clip(np.r_[seed[0],yaw],lower+1e-5,upper-1e-5)
    fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',
                      f_scale=5.,max_nfev=60)
    vectors=residual(fit.x).reshape(2,2,2)
    worst=float(np.linalg.norm(vectors,axis=-1).max())
    return worst,np.stack(points(fit.x))


def read(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return row['scene_group'], row['episode'], round(float(row['time_s']),6)


def groups_report(rows,field):
    scenes=defaultdict(list)
    for r in rows:scenes[r['scene_group']].append(r)
    result={'groups':len(scenes),'accepted_rows':0,'wrong_accepted_rows':0,
            'early_correct_groups':0,'early_wrong_groups':0,
            'ever_correct_groups':0,'ever_wrong_groups':0}
    for rr in scenes.values():
        rr.sort(key=lambda r:r['time_s']);first=rr[0]['time_s']
        accepted=[r for r in rr if r[field]]
        result['accepted_rows']+=len(accepted)
        result['wrong_accepted_rows']+=sum(r['wrong_identity_label_only'] for r in accepted)
        early=[r for r in accepted if r['time_s']-first<=3]
        result['early_correct_groups']+=any(not r['wrong_identity_label_only'] for r in early)
        result['early_wrong_groups']+=any(r['wrong_identity_label_only'] for r in early)
        result['ever_correct_groups']+=any(not r['wrong_identity_label_only'] for r in accepted)
        result['ever_wrong_groups']+=any(r['wrong_identity_label_only'] for r in accepted)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rail-predictions','candidates','labels','calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-mode', choices=('all_visible','best_two',
                                               'horizontal_pair_fit'),
                   default='all_visible')
    args=p.parse_args()
    if args.output.exists():p.error('Preserve existing output')
    rails={key(r):r for r in read(args.rail_predictions)}
    labels={key(r):r for r in read(args.labels)}
    candidates=read(args.candidates)
    if set(rails)!=set(labels) or set(rails)!={key(r) for r in candidates}:
        raise ValueError('Mismatched same-frame keys')
    cal=json.loads(args.calibration.read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMS}
    by_scene=defaultdict(list)
    for item in candidates:by_scene[item['scene_group']].append(item)
    records=[]
    for scene,rr in sorted(by_scene.items()):
        rr.sort(key=lambda r:r['time_s'])
        references=[]
        occupancy=[]
        for old in rr:
            k=key(old);rail=rails[k];label=labels[k]
            camera={c:pose_matrix(np.asarray(label['state'][:7],float))@optics[c]
                    for c in CAMS}
            eligible={c for ci,c in enumerate(CAMS)
                      if all(rail['endpoint_inframe_prob'][ci][end]>=.5 for end in (0,1))}
            camera_sets=(list(combinations(sorted(eligible),2))
                         if args.reference_mode in ('best_two','horizontal_pair_fit')
                         else [tuple(sorted(eligible))])
            proposals=[]
            for used in camera_sets:
                if len(used)<2:continue
                points=[];residuals=[]
                for end in (0,1):
                    pixels={c:np.asarray(rail['endpoint_xy_normalized'][CAMS.index(c)][end])
                            *[1151,1023] for c in used}
                    try:
                        xyz=triangulate(pixels,{c:camera[c] for c in used})
                        residual=reprojection_residual_px(
                            xyz,pixels,{c:camera[c] for c in used})
                    except np.linalg.LinAlgError:
                        break
                    points.append(xyz);residuals.append(residual)
                if len(points)==2:
                    positions=np.stack(points)
                    if args.reference_mode=='horizontal_pair_fit':
                        fitted_error,positions=fit_horizontal_pair(rail,camera,used,positions)
                    else:
                        fitted_error=max(residuals)
                    proposals.append((fitted_error,float(np.linalg.norm(
                        positions[1]-positions[0])),tuple(used),positions))
            new_reference=False
            reject_reason='insufficient_two_view_endpoints'
            selected_cameras=[]
            if proposals:
                valid_span=[q for q in proposals if .12<=q[1]<=.20]
                if not valid_span:
                    reject_reason='rail_span_outside_120_200_mm'
                else:
                    residual,_,used,positions=min(valid_span,key=lambda q:q[0])
                    if residual>5:
                        reject_reason='rail_reprojection_above_5_px'
                    else:
                        references.append(positions)
                        new_reference=True
                        selected_cameras=list(used)
                        reject_reason='accepted'
            occupancy.append((old['time_s'],np.asarray(rail['occupied_prob'])))
            recent=[v for t,v in occupancy if 0<=old['time_s']-t<=3]
            card=int(old['requested'].split('_')[2])
            occupied=float(np.median(recent,axis=0)[card])
            candidate=old['candidates'][old['requested']]
            candidate_xyz=candidate.get('predicted_base_m')
            baseline=bool(candidate['three_view_consistent'] and candidate_xyz is not None)
            reference=np.median(references[-3:],axis=0) if references else None
            slot_index=None
            if reference is not None and candidate_xyz is not None:
                vector=reference[1]-reference[0]
                slot_index=4*float(np.dot(np.asarray(candidate_xyz)-
                                          reference[0],vector)/np.dot(vector,vector))
            accept=bool(baseline and occupied>=.5 and slot_index is not None
                        and abs(slot_index-card)<=.5)
            nearest=(min(label['targets'],key=lambda t:np.linalg.norm(
                np.asarray(candidate_xyz)-t['position_base_m']))
                     if candidate_xyz is not None else None)
            wrong=nearest is not None and nearest['identity']!=old['requested']
            records.append({'scene_group':scene,'episode':old['episode'],
                            'time_s':old['time_s'],'requested':old['requested'],
                            'new_board_reference':new_reference,
                            'reference_cameras':selected_cameras,
                            'board_reference_reason':reject_reason,
                            'board_reference_available':reference is not None,
                            'requested_occupancy_prob':occupied,
                            'requested_slot_index':slot_index,
                            'baseline_accept':baseline,'rail_3d_accept':accept,
                            'wrong_identity_label_only':wrong,
                            'nearest_identity_label_only':nearest['identity'] if nearest else None})
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    result={'schema':'sfp_rail_candidate_causal_3d/v1',
            'reference_mode':args.reference_mode,
            'fixed_rule':'triangulated two-view rail endpoints, spacing 120-200 mm, '
                         'reprojection <=5 px, last three valid references, '
                         'causal three-second occupancy >=0.5, requested 3D rail '
                         'index within 0.5, existing three-view port consistency',
            'reference_groups':len({r['scene_group'] for r in records
                                    if r['board_reference_available']}),
            'baseline':groups_report(records,'baseline_accept'),
            'rail_3d':groups_report(records,'rail_3d_accept'),
            'geometry_runtime_input':False,'final_scenes_opened':False,
            'offline_teacher_route_only':True}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

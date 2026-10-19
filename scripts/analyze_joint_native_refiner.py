#!/usr/bin/env python3
"""Score opening refinements and save predicted-centred native QA panels.

Input predictions were made without target poses. Truth is accessed here only
for scoring/green comparison marks. Missing/offscreen GT views were omitted
by the crop dataset: three-view results are conditional accuracy, NOT an
autonomous acquisition/visibility rate. This report cannot promote a pipeline.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px


def stats(v):
    v=np.asarray(v,float);return dict(n=len(v),p50=float(np.median(v)) if len(v) else None,
        p90=float(np.percentile(v,90)) if len(v) else None,p95=float(np.percentile(v,95)) if len(v) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    rows=[json.loads(l) for l in (a.run/'validation_predictions.jsonl').open()]
    optics={c:np.asarray(m,float) for c,m in json.loads(a.calibration.read_text())['tcp_to_optical'].items()}
    groups=defaultdict(dict);pixel=defaultdict(list)
    for r in rows:
        err=float(np.linalg.norm(np.asarray(r['predicted_xy'])-r['truth_xy']))
        for k in (r['family'],f"{r['family']}/camera_{r['camera']}",f"{r['family']}/cards{r['cards']}",f"{r['family']}/{r['identity']}"):
            pixel[k].append(err)
        if r['inside_margin']:pixel[r['family']+'/inside_crop'].append(err)
        if r['selected']:groups[(r['family'],r['episode'],r['time_s'],r['identity'])][r['camera']]=r
    predictions=[];errors=defaultdict(lambda:defaultdict(list));reject=Counter()
    for key,views in groups.items():
        if len(views)!=3:reject[f'{key[0]}/fewer_than_three_labelled_views']+=1;continue
        stamps=[int(Path(v['source_image']).stem.split('_')[-2]) for v in views.values()]
        if max(stamps)-min(stamps)>1_000_000:
            reject[f'{key[0]}/asynchronous_images_excluded_from_common_TCP_triangulation']+=1;continue
        r=next(iter(views.values()));family=r['family'];base_tcp=pose_matrix(r['state'])
        cameras={c:base_tcp@optics[c] for c in views}
        refined={c:v['predicted_xy'] for c,v in views.items()};coarse={c:v['coarse_xy'] for c,v in views.items()}
        target=np.asarray(r['target_position_base_m']);rotation=Rotation.from_quat(r['target_orientation_base_xyzw']).as_matrix()
        for variant,points in [('coarse',coarse),('refined',refined)]:
            try:point=triangulate(points,cameras);residual=float(reprojection_residual_px(point,points,cameras))
            except np.linalg.LinAlgError:reject[f'{family}/{variant}/singular']+=1;continue
            delta=rotation.T@(point-target)*1000
            tcp_axial=float((rotation.T@(base_tcp[:3,3]-target))[2]*1000)
            record=dict(family=family,episode=r['episode'],time_s=r['time_s'],identity=r['identity'],
                cards=r['cards'],variant=variant,predicted_position_base_m=point.tolist(),
                xyz_error_mm=float(np.linalg.norm(delta)),lateral_error_mm=float(np.linalg.norm(delta[:2])),
                axial_error_mm=float(abs(delta[2])),reprojection_px=residual,observed_TCP_axial_mm=tcp_axial,
                all_crops_inside_margin=all(v['inside_margin'] for v in views.values()),
                geometry_access='scoring only; not used in prediction or acceptance')
            predictions.append(record)
            buckets=[f'{family}/{variant}/all_three_labelled_views',f'{family}/{variant}/cards{r["cards"]}',f'{family}/{variant}/{r["identity"]}']
            if all(v['inside_margin'] for v in views.values()):buckets.append(f'{family}/{variant}/inside_all_crops')
            for gate in (2,5):
                if residual<=gate:buckets.append(f'{family}/{variant}/RGB_reprojection_le{gate}px')
            phase=('TCP_axial_le50mm' if abs(tcp_axial)<=50 else 'TCP_axial_50to150mm' if abs(tcp_axial)<=150 else 'TCP_axial_gt150mm')
            buckets.append(f'{family}/{variant}/{phase}')
            for b in buckets:
                for name in ('xyz_error_mm','lateral_error_mm','axial_error_mm','reprojection_px'):errors[b][name].append(record[name])
    with (a.output/'triangulation_predictions.jsonl').open('w') as f:
        for r in predictions:f.write(json.dumps(r)+'\n')
    # Deterministic coverage sampling by family/card count/port; original crop
    # centres are model predictions, never true coordinates.
    panels=defaultdict(list)
    for key,views in groups.items():
        if len(views)==3:
            r=next(iter(views.values()));panels[(r['family'],r['cards'],r['identity'].rsplit('_',1)[1])].append((key,views))
    qa=[]
    for ci,(cell,options) in enumerate(sorted(panels.items())):
        if ci>=20:break
        _,views=options[len(options)//2];parts=[]
        for c in ('center','left','right'):
            r=views[c];image=cv2.imread(r['source_image']);pred=np.asarray(r['predicted_xy']);truth=np.asarray(r['truth_xy'])
            cv2.drawMarker(image,tuple(np.rint(pred).astype(int)),(0,0,255),cv2.MARKER_CROSS,22,2)
            cv2.drawMarker(image,tuple(np.rint(truth).astype(int)),(0,255,0),cv2.MARKER_TILTED_CROSS,18,1)
            # A prediction-centred detail crop is valid even if the estimate is wrong.
            origin=np.rint(pred-80).astype(int)
            detail=cv2.warpAffine(image,np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(160,160))
            full=cv2.resize(image,(576,512));detail=cv2.resize(detail,(512,512),interpolation=cv2.INTER_NEAREST)
            pad=np.zeros((512,64,3),np.uint8);column=np.concatenate((full,np.concatenate((detail,pad),axis=1)),axis=0)
            cv2.putText(column,f'{c}: red prediction / green GT scoring only',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1)
            parts.append(column)
        name=f"{cell[0]}_cards{cell[1]}_port{cell[2]}.jpg";cv2.imwrite(str(a.output/name),np.concatenate(parts,axis=1))
        qa.append(dict(file=name,episode=r['episode'],time_s=r['time_s'],source_images={c:v['source_image'] for c,v in views.items()},crop_selection='predicted point, NOT ground truth'))
    report=dict(schema='joint_native_refiner_analysis/v1',pixel_errors_px={k:stats(v) for k,v in pixel.items()},
        three_view_conditional_errors={k:{n:stats(v) for n,v in fields.items()} for k,fields in errors.items()},
        rejections=dict(reject),validation_episodes={f:len({r['episode'] for r in rows if r['family']==f}) for f in ('sc','sfp')},
        caveat='GT projected-inframe determines which crop samples exist. This is conditional accuracy, NOT an acquisition/visibility/safe handoff rate. Orientation is not predicted by this opening-point refiner.',
        qa=qa,final_scenes_opened=False,pipeline_promoted=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='conditional_scoring_complete_not_promoted',panels=len(qa),triplet_predictions=len(predictions))))


if __name__=='__main__':main()

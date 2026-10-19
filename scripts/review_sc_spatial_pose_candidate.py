#!/usr/bin/env python3
"""Review held-out RGB-only grasp predictions; labels score, never choose crops."""
import cv2
cv2.setNumThreads(2)
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sc_spatial_grasp_observation import SCSpatialGraspObservation
from evaluate_sc_native_triangulation import FOCAL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    summary=json.loads((a.fit/'summary.json').read_text())
    source=Path(summary['plan']['source'])
    rows=json.loads((source/'manifest.json').read_text())
    indices=np.load(a.fit/'heldout_indices.npy')
    pixels=np.load(a.fit/'heldout_pixel_predictions.npy')
    poses=np.load(a.fit/'heldout_pose_predictions.npy')
    model=SCSpatialGraspObservation(a.fit/'best.pt','cpu')
    domains={}
    for j,index in enumerate(indices):
        role=rows[int(index)].get('source_role','')
        domain='near_25mm' if role.startswith('Near25mm') else 'nominal_acquisition' if role.startswith('Predeadline') else 'nominal_home' if role else 'original'
        domains.setdefault(domain,[]).append(j)
    checks=[]
    for domain,positions in domains.items():
        # Predeclared evenly spaced held-out episodes, not cherry-picked frames.
        episodes=list(dict.fromkeys(rows[int(indices[j])]['episode'] for j in positions))
        selected=[episodes[i] for i in np.linspace(0,len(episodes)-1,min(3,len(episodes)),dtype=int)]
        for n,episode in enumerate(selected):
            choices=[j for j in positions if rows[int(indices[j])]['episode']==episode]
            j=choices[len(choices)//2]; row=rows[int(indices[j])]
            images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB) for c in model.cameras}
            prediction=model.predict(images,already_training_jpeg=True)
            rotation=np.asarray(prediction['orientation_for_diagnostic_only'])
            position=np.asarray(prediction['predicted_tip_position_tcp_m'])
            delta_px=float(np.max(np.abs(np.asarray(prediction['predicted_pixels'])-pixels[j])))
            delta_mm=float(np.linalg.norm(position-poses[j,:3,3])*1000)
            delta_deg=float(np.degrees(Rotation.from_matrix(rotation.T@poses[j,:3,:3]).magnitude()))
            panels=[]
            for ci,camera in enumerate(model.cameras):
                image=cv2.cvtColor(images[camera],cv2.COLOR_RGB2BGR)
                left,top=np.rint(np.asarray(model.centres[camera])-96).astype(int)
                patch=cv2.warpAffine(image,np.array([[1,0,-left],[0,1,-top]],np.float32),(192,192))
                for point in np.asarray(prediction['predicted_pixels'])[ci]-[left,top]:
                    cv2.circle(patch,tuple(np.rint(point).astype(int)),2,(255,255,0),1,cv2.LINE_AA)
                cloud=np.vstack([position,position[None]+.008*rotation.T])
                inv=np.linalg.inv(model.optics[camera]);q=cloud@inv[:3,:3].T+inv[:3,3]
                projected=q[:,:2]/q[:,2:]*FOCAL+[576-left,512-top]
                if np.all(q[:,2]>0):
                    start=tuple(np.rint(projected[0]).astype(int))
                    for k,color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
                        cv2.arrowedLine(patch,start,tuple(np.rint(projected[k+1]).astype(int)),color,1,cv2.LINE_AA,tipLength=.2)
                patch=cv2.copyMakeBorder(patch,22,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(patch,camera+' predicted axes',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
                panels.append(patch)
            filename=f'{domain}_{n}_prediction_only.png'
            cv2.imwrite(str(a.output/filename),np.hstack(panels))
            checks.append(dict(domain=domain,episode=episode,index=int(indices[j]),
                pixel_parity_max_px=delta_px,translation_parity_mm=delta_mm,
                rotation_parity_deg=delta_deg,image=filename,
                GT_markers_or_crops=False,geometry_used_for_prediction=False))
    # CPU/GPU float32 convolution can differ by about0.011nativepixel with
    # bit-exact cached/live input. Preserve strict physical-pose parity and
    # use0.05px only as the additional intermediate-pixel diagnostic bound.
    parity=all(x['pixel_parity_max_px']<=.05 and x['translation_parity_mm']<=.01 and x['rotation_parity_deg']<=.01 for x in checks)
    near=summary['metrics_by_domain'].get('near_25mm')
    # V21 uses this model ONLY during the held25mm observation. The old far
    # model remains deployed at acquisition. Report the original all-domain
    # training gate separately; do not claim a far/home model replacement.
    near_gate=bool(near and near['episodes']>=4 and
        near['translation_p50_p95_mm'][1]<=.4 and
        near['episode_balanced_translation_p50_p95_mm'][1]<=.4 and
        near['orientation_p50_p95_deg'][1]<=1 and
        near['episode_balanced_orientation_p50_p95_deg'][1]<=1)
    qualified=bool(summary['plateau_reached'] and near_gate and parity)
    report=dict(checkpoint_sha256=hashlib.sha256((a.fit/'best.pt').read_bytes()).hexdigest(),
        pose_gate_passed=qualified,training_plateau_reached=summary['plateau_reached'],
        pose_gate_scope='SC_held_near25mm_only',near_domain_gate_passed=near_gate,
        original_all_domain_training_gate_passed=summary['gate_passed'],
        metrics_by_domain=summary['metrics_by_domain'],runtime_parity_passed=parity,
        parity_tolerances=dict(pixel_max_px=.05,translation_mm=.01,orientation_deg=.01),
        near_view_heldout_present='near_25mm' in domains,checks=checks,
        visual_inspection_completed=False,live_SC_success_not_asserted=True,
        inference_inputs='Three native RGB fixed192crops and known camera calibration only',
        caution='Stable predictions do not establish absolute accuracy. Gate does not imply insertion success.')
    (a.output/'approval.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pose_gate_passed=qualified,runtime_parity_passed=parity,images=len(checks))),flush=True)


if __name__=='__main__':
    main()

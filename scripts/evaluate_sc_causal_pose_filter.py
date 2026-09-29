#!/usr/bin/env python3
"""Test causal stationary-port filtering of RGB-predicted SC relative pose.

The filter uses only a trained image/state pose prediction, measured TCP
state, and fixed TCP-to-tip calibration. True port TF is read solely to score
the held-out error, never to update or select the filter state.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from audit_sc_port_targets import matrix, pose_list, pose_matrix
from train_sc_full_crop_fusion import FullCropFusion, PairedFrames


WINDOWS = (0, 3, 5, 10, 20, 40)


def evaluate_rows(rows, estimates):
    out={}
    for split in ("train","validation"):
        selected=[]
        for row, estimate in zip(rows,estimates):
            if row["split"]!=split:continue
            estimate=np.asarray(estimate)
            truth=np.asarray(row["observed_sc_tip_pose_opening_frame"])
            if abs(truth[2])>=.03:continue
            delta=(estimate[:3]-truth[:3])*1000
            selected.append({"trial":row["trial"],"tier3":row.get("official_tier3"),
                             "lateral_mm":float(np.linalg.norm(delta[:2])),
                             "axial_mm":float(abs(delta[2]))})
        if not selected:
            out[split]={"near_count":0,"episodes":0}
            continue
        out[split]={"near_count":len(selected),"episodes":len({r['trial'] for r in selected}),
                    "lateral_median_mm":float(np.median([r['lateral_mm'] for r in selected])),
                    "lateral_p95_mm":float(np.percentile([r['lateral_mm'] for r in selected],95)),
                    "axial_p95_mm":float(np.percentile([r['axial_mm'] for r in selected],95)),
                    "per_episode_lateral_p95_mm":{
                        trial:float(np.percentile([r['lateral_mm'] for r in selected if r['trial']==trial],95))
                        for trial in sorted({r['trial'] for r in selected})},
                    "per_episode_near_count":{
                        trial:sum(r['trial']==trial for r in selected)
                        for trial in sorted({r['trial'] for r in selected})},
                    "per_episode_official_tier3":{
                        trial:next(r['tier3'] for r in selected if r['trial']==trial)
                        for trial in sorted({r['trial'] for r in selected})}}
    return out


def causal_filter(rows, predictions, tcp_tip, window):
    if window==0:return predictions
    histories={}
    last_update_time={}
    outputs=[]
    for row,pred in zip(rows,predictions):
        base_tip=pose_matrix(row["state"][:7])@tcp_tip
        relative=pose_matrix(pred)
        base_open=base_tip@np.linalg.inv(relative)
        history=histories.setdefault(row["episode_id"],deque(maxlen=window))
        # Only use observations while the image model itself places the tip
        # within 100 mm of the opening. No true phase/port geometry is read.
        # Native training capture was sampled every roughly 0.55–0.9 s.
        # Keep the same minimum physical time spacing at a 20 Hz runtime.
        elapsed=float(row['sim_time'])-last_update_time.get(row['episode_id'],-float('inf'))
        if pred[2]>-.1 and elapsed>=.5:
            history.append(base_open)
            last_update_time[row['episode_id']]=float(row['sim_time'])
        if len(history)<2:
            outputs.append(pred)
            continue
        smooth=np.eye(4)
        smooth[:3,3]=np.median(np.stack([pose[:3,3] for pose in history]),axis=0)
        smooth[:3,:3]=Rotation.from_matrix(np.stack([pose[:3,:3] for pose in history])).mean().as_matrix()
        outputs.append(pose_list(np.linalg.inv(smooth)@base_tip))
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-labels",type=Path,required=True)
    p.add_argument("--crop-labels",type=Path,required=True)
    p.add_argument("--base-checkpoint",type=Path,required=True)
    p.add_argument("--fusion-checkpoint",type=Path,required=True)
    p.add_argument("--pretrained-resnet",type=Path,required=True)
    p.add_argument("--tcp-tip-calibration",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frozen-window",type=int,choices=WINDOWS,
                   help="Use a preselected history length; do not select from these episodes")
    args=p.parse_args()
    torch.set_num_threads(8)
    full=[json.loads(line) for line in args.full_labels.open()]
    crops=[json.loads(line) for line in args.crop_labels.open()]
    if [(r['trial'],r['frame']) for r in full]!=[(r['trial'],r['frame']) for r in crops]:
        raise ValueError("Unmatched full/crop observations")
    base=torch.load(args.base_checkpoint,map_location="cpu",weights_only=False)
    dataset=PairedFrames(full,crops,base['state_mean'],base['state_std'],base['image_size'])
    loader=DataLoader(dataset,batch_size=16,num_workers=4,shuffle=False,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=FullCropFusion(base,args.pretrained_resnet).to(device).eval()
    model.load_state_dict(torch.load(args.fusion_checkpoint,map_location="cpu",weights_only=False)['model'],strict=True)
    predictions=[]
    with torch.inference_mode():
        for full_images,crop_images,state,_,_ in loader:
            xyz,q=model(full_images.to(device),crop_images.to(device),state.to(device))
            predictions.extend(np.concatenate((xyz.cpu().numpy(),q.cpu().numpy()),axis=1).tolist())
    tcp_tip=matrix(json.loads(args.tcp_tip_calibration.read_text())['tcp_to_sc_tip'])
    results={str(window):evaluate_rows(full,causal_filter(full,predictions,tcp_tip,window)) for window in WINDOWS}
    # The scene is static. Prefer the longest observation-only history that
    # still meets the training corridor, rather than optimizing the fit of a
    # near-memorized training model. This rule is frozen before new-scene
    # confirmation; the first four validation scenes motivated the rule.
    if args.frozen_window is not None:
        selected=args.frozen_window
    else:
        admissible=[window for window in WINDOWS if results[str(window)]['train']['lateral_p95_mm']<=.5]
        selected=max(admissible) if admissible else 0
    selected_predictions=causal_filter(full,predictions,tcp_tip,selected)
    error_rows=[]
    for row,pred in zip(full,selected_predictions):
        if row['split']!='validation':continue
        truth=np.asarray(row['observed_sc_tip_pose_opening_frame'])
        if abs(truth[2])>=.03:continue
        pred=np.asarray(pred)
        error_rows.append({"trial":row['trial'],"frame":row['frame'],"sim_time":row['sim_time'],
                           "official_tier3":row.get('official_tier3'),
                           "lateral_error_mm":float(np.linalg.norm((pred[:2]-truth[:2])*1000)),
                           "axial_error_mm":float(abs(pred[2]-truth[2])*1000),
                           "true_translation_mm":(truth[:3]*1000).tolist(),
                           "predicted_translation_mm":(pred[:3]*1000).tolist(),
                           "images":row['images']})
    worst=sorted(error_rows,key=lambda r:r['lateral_error_mm'],reverse=True)[:20]
    result={"schema":"sc_causal_stationary_port_filter/v1",
            "runtime_inputs":["RGB-only learned pose","measured 32D state","fixed TCP-to-tip calibration"],
            "minimum_history_update_interval_s":0.5,
            "true_port_geometry_runtime":False,"selected_window_on_train":selected,
            "selection_rule":"frozen external window" if args.frozen_window is not None else
                             "longest window with train near-port lateral p95 <= 0.5 mm; exploratory rule set after first four development validation scenes, frozen before three new confirmations",
            "candidate_windows":results,
            "worst_near_port_validation_rows_training_label_diagnostics":worst,
            "gate":{"lateral_p95_mm":0.5,
                    "selected_validation_passed":results[str(selected)]['validation']['lateral_p95_mm']<=.5},
            "checkpoints":{"base":str(args.base_checkpoint),"fusion":str(args.fusion_checkpoint)}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"selected":selected,"validation_p95_mm":results[str(selected)]['validation']['lateral_p95_mm'],
                      "all_validation_p95_mm":{str(k):results[str(k)]['validation']['lateral_p95_mm'] for k in WINDOWS}}))


if __name__=='__main__':main()

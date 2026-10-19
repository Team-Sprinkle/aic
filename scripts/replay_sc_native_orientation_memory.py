#!/usr/bin/env python3
"""Deterministic no-memory/memory comparison on lossless SC runtime snapshots.

The captured state, requested port, native RGB and earlier request-bound
visual anchor are observation-only. Gazebo geometry is never read here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import sc_shadow_handoff_verifier as verifier_module

SCShadowHandoffVerifier=verifier_module.SCShadowHandoffVerifier


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--simulate-missing-rails',action='store_true',
                        help='Code-path check only: force the board-line detector to return no lines')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite a saved replay')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    root=Path(__file__).resolve().parents[1]
    model_paths={
        'locator':root/'outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt',
        'identity':Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt'),
        'orientation':Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_orientation_oneport.pt'),
        'refiner':root/'outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_v1/best.pt',
        'calibration':root/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json',
        'board_mount':root/'configs/hierarchical_recovery/sc_board_to_port_training_fixed.json',
        'tcp_prior':root/'outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json'}
    verifier=SCShadowHandoffVerifier(*(model_paths[k] for k in (
        'locator','identity','orientation','refiner','calibration','board_mount','tcp_prior')),
        device=args.device)
    if args.simulate_missing_rails:
        verifier_module.camera_yaw=lambda *unused:None
    metadata=[json.loads(path.read_text()) for path in sorted(args.snapshots.glob('*.json'))]
    rows=[]
    for snapshot in metadata:
        if snapshot['kind']!='fresh_inspection':
            continue
        image_paths={camera:args.snapshots/Path(snapshot['images'][camera]).name
                     for camera in ('center','left','right')}
        images={}
        for camera,path in image_paths.items():
            bgr=cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f'Missing native PNG {path}')
            images[camera]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        requested=int(snapshot['requested_port'])
        state=np.asarray(snapshot['state'],np.float32)
        anchor_available=(snapshot['anchor_opening_base_m'] is not None and
                          snapshot['anchor_port_quat_xyzw'] is not None and
                          snapshot['anchor_board_spread_deg'] is not None and
                          snapshot['anchor_refresh_travel_m'] is not None and
                          snapshot['cumulative_travel_m']-
                          snapshot['anchor_refresh_travel_m']<=.150)
        no_memory=verifier.inspect(images,state,requested)
        memory=verifier.inspect(
            images,state,requested,
            history_opening_base_m=(snapshot['anchor_opening_base_m'] if anchor_available else None),
            history_port_quat_xyzw=(snapshot['anchor_port_quat_xyzw'] if anchor_available else None),
            history_board_spread_deg=(snapshot['anchor_board_spread_deg'] if anchor_available else None))
        # Deliberately pass the current request's anchor to the absent sibling:
        # this is stricter than runtime's request-bound anchor guard.
        wrong=verifier.inspect(
            images,state,1-requested,
            history_opening_base_m=(snapshot['anchor_opening_base_m'] if anchor_available else None),
            history_port_quat_xyzw=(snapshot['anchor_port_quat_xyzw'] if anchor_available else None),
            history_board_spread_deg=(snapshot['anchor_board_spread_deg'] if anchor_available else None))
        rows.append(dict(sim_time_s=snapshot['sim_time_s'],requested_port=requested,
                         source_reason=snapshot['report_reason'],
                         source_valid=snapshot['report_valid'],
                         native_images={c:str(p) for c,p in image_paths.items()},
                         image_sha256={c:sha256(image_paths[c]) for c in images},
                         anchor_available=anchor_available,
                         anchor_travel_since_refresh_mm=(1000*(snapshot['cumulative_travel_m']-
                             snapshot['anchor_refresh_travel_m']) if anchor_available else None),
                         no_memory=no_memory,memory=memory,wrong_request_with_anchor=wrong))
    result=dict(schema='aic_sc_native_orientation_memory_replay/v1',
                snapshots=str(args.snapshots),models_sha256={k:sha256(v) for k,v in model_paths.items()},
                simulated_board_rail_dropout=args.simulate_missing_rails,
                geometry_runtime_input=False,final_scenes_opened=False,rows=rows,
                counts=dict(inspections=len(rows),
                            no_memory_accepts=sum(r['no_memory']['valid'] for r in rows),
                            memory_accepts=sum(r['memory']['valid'] for r in rows),
                            memory_only_accepts=sum(r['memory']['valid'] and not r['no_memory']['valid'] for r in rows),
                            wrong_request_accepts=sum(r['wrong_request_with_anchor']['valid'] for r in rows)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':
    main()

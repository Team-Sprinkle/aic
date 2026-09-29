#!/usr/bin/env python3
"""Compare a shared BC checkpoint with scored teacher frames on one SC scene."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain
from shared_port_tcp_geometry import matrix_from_pose9, inferred_base_port, body_delta_from_target
from train_shared_port_tcp_bc import SharedPortTCPActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--trial', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--every', type=int, default=50)
    a = p.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    weights = saved['model']; keys = ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')
    model = SharedPortTCPActor({k: weights[k].numpy() for k in keys}, pretrained=False,
                               use_time=bool(saved.get('include_elapsed', False))).cuda().eval()
    model.load_state_dict(weights)
    scored = next(row for row in json.loads(a.edges.read_text())['rows'] if row['trial'] == a.trial)
    index = scored['target_module_name'].rsplit('_', 1)[-1]
    parent = f'task_board/sc_port_{index}'
    world_base = chain(scored['edges'], ['world', 'tabletop', 'base_link'])
    world_port = chain(scored['edges'], ['aic_world', 'task_board', parent, parent+'/sc_port_base_link'])
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    base_port = np.linalg.inv(world_base) @ world_port @ opening_offset
    rows = [json.loads(line) for line in (a.episode/'frames.jsonl').open()]
    first_time = float(rows[0]['sim_time'])
    report = []
    for row in rows[::a.every]:
        images = []
        for cam in ('center','left','right'):
            with Image.open(a.episode/row['images'][cam]) as f:
                image = f.convert('RGB').resize((224,224), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2,0,1))
        rgb = torch.stack(images)[None].cuda()
        state = torch.tensor(row['state'], dtype=torch.float32, device='cuda')[None]
        task = torch.tensor([0.,1.,float(index=='0'),float(index=='1'),0.,0.,0.,0.,0.,0.],device='cuda')[None]
        with torch.inference_mode():
            clock = (float(row['nominal_expert_time'])
                     if saved.get('time_coordinate', 'sim_elapsed') == 'nominal_decision'
                     else max(0., float(row['sim_time']) - first_time))
            elapsed = torch.tensor([clock],
                                   device='cuda', dtype=torch.float32)
            target, pose = model(rgb, state, task, elapsed)
        target, pose = target[0].cpu().numpy(),pose[0].cpu().numpy()
        observed_tcp = pose_matrix(row['state'][:7])
        true_port_tcp = np.linalg.inv(base_port) @ observed_tcp
        predicted_port_tcp = matrix_from_pose9(pose)
        teacher_target = pose_matrix(row['teacher_target_pose'])
        teacher_body = np.linalg.inv(observed_tcp) @ teacher_target
        inferred = inferred_base_port(row['state'][:7], pose)
        predicted_body = body_delta_from_target(row['state'][:7], inferred, target[0])
        # Diagnostic upper bound: keep the frozen action head, replace only its
        # inferred port frame with scored TF after the rollout has completed.
        # This never enters the actor or the command stream.
        oracle_body = body_delta_from_target(row['state'][:7], base_port, target[0])
        report.append({'frame': row['frame'], 'sim_time': row['sim_time'],
                       'decision_time_s': clock,
                       'pose_position_error_mm': float(np.linalg.norm(predicted_port_tcp[:3,3]-true_port_tcp[:3,3])*1000),
                       'pose_orientation_error_deg': float(np.degrees((Rotation.from_matrix(predicted_port_tcp[:3,:3]).inv()*Rotation.from_matrix(true_port_tcp[:3,:3])).magnitude())),
                       'predicted_body_translation_mm': (predicted_body[:3]*1000).tolist(),
                       'teacher_body_translation_mm': (teacher_body[:3,3]*1000).tolist(),
                       'first_command_translation_error_mm': float(np.linalg.norm(predicted_body[:3]-teacher_body[:3,3])*1000),
                       'oracle_pose_first_command_translation_error_mm':
                           float(np.linalg.norm(oracle_body[:3]-teacher_body[:3,3])*1000),
                       'teacher_target_gap_mm': float(np.linalg.norm(teacher_body[:3,3])*1000)})
    out = {'schema':'shared_port_tcp_teacher_scene_diagnostic/v1',
           'trial':a.trial,'checkpoint':str(a.checkpoint),'rows':report,
           'time_coordinate':saved.get('time_coordinate','sim_elapsed'),
           'geometry_use':'scored port TF is posthoc diagnostic only; actor receives RGB/state/task'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'trial':a.trial,'samples':len(report),'first':report[0]}))


if __name__=='__main__':main()

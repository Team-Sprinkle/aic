#!/usr/bin/env python3
"""Construct real checkpoint/runtime and check decode on held-out RGB inputs."""
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'aic_example_policies'),str(REPO/'aic_model')]
import cv2
import json
import numpy as np
import torch
from types import SimpleNamespace
from project_gpu_guard import selected_gpu_reservation
from shared_port_tcp_geometry import matrix_from_xyz_xyzw,matrix_from_pose9


def main():
    import rclpy
    from rclpy.node import Node
    from aic_example_policies.ros.RunFullConnectorInsertion import RunFullConnectorInsertion
    rclpy.init();node=Node('full_connector_preflight');policy=RunFullConnectorInsertion(node)
    root=Path('/home/nas/chmin/aic_archives/2026-10-17_full_connector_live_probe_v1')
    source=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/full_connector_actor_rows/rows.jsonl')
    examples={}
    for line in source.open():
        row=json.loads(line)
        if row['split']=='validation' and row['family'] not in examples:examples[row['family']]=row
        if len(examples)==2:break
    errors=[];local=policy.local
    for family,row in examples.items():
        state=np.asarray(row['state'],np.float32);sim=row['time_s'];base=matrix_from_xyz_xyzw(state[:7])
        relative=matrix_from_pose9(row['predicted_port_tcp_pose9'])
        local.visual_port=base@np.linalg.inv(relative);local.visual_lock_sim_time=sim
        local.visual_body_tip=np.asarray(row['predicted_body_tip_position_m'])
        local.visual_body_connector_rotation=np.asarray(row['estimated_connector_in_tcp'])[:3,:3]
        local.assembler.assemble_ros=lambda value:state.copy();local._control_sim_time=lambda value:sim
        local.retain_history_observation=lambda value:None
        local.robot_history=[(sim-x,state.copy()) for x in [1.2,.6,.2,0]]
        images={}
        for c,path in row['images'].items():
            rgb=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
            images[c+'_image']=SimpleNamespace(encoding='rgb8',data=rgb.tobytes(),height=1024,width=1152,step=3456)
        captured=[];local._trace=lambda **kwargs:captured.append(kwargs)
        result,_,_=local.predict_chunk(SimpleNamespace(**images),row['task'])
        trace=next(x for x in captured if x['event']=='locked_local_perception')
        original=np.asarray(trace['actor_control_point_targets_port'])
        grasp=np.asarray(row['estimated_connector_in_tcp'])
        recovered=np.stack([matrix_from_pose9(x)@np.linalg.inv(grasp) for x in original])
        actual=np.stack([matrix_from_pose9(x) for x in result])
        error=float(np.max(abs(recovered-actual)));errors.append(error)
        assert trace['tip_orientation_input'] and error<1e-5
    report=dict(passed=True,families=list(examples),full_checkpoint_constructor=True,real_RGB_prediction=True,
        decoder_max_abs_error=max(errors),orientation_input_unmasked=not local.actor.masked_tip_indices,
        physical_rollout=False,history_is_synthetic_for_interface_check_only=True)
    (root/'runtime_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    with selected_gpu_reservation():main()

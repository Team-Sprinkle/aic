#!/usr/bin/env python3
"""Training-only public measured TCP/tip speed during successful SC entry."""
import json
from pathlib import Path
import numpy as np
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw
source=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/full_connector_actor_rows');values=[];episodes=set()
for line in (source/'rows.jsonl').open():
 row=json.loads(line)
 if row['family']!='sc' or row['split']!='train' or not .003<=row['predicted_tip_port_pose9'][2]<=.014:continue
 state=np.array(row['state']);tcp=matrix_from_xyz_xyzw(state[:7]);relative=matrix_from_pose9(row['predicted_port_tcp_pose9']);port_base=(relative@np.linalg.inv(tcp))[:3,:3]
 grasp=np.array(row['estimated_connector_in_tcp']);velocity=state[7:10]+np.cross(state[10:13],tcp[:3,:3]@grasp[:3,3]);axial=float((port_base@velocity)[2]*1000)
 if axial>5:values.append(axial);episodes.add(row['episode'])
d=dict(source=str(source),training_only=True,positive_axial_motion_above_5mmps_rows=len(values),episodes=len(episodes),measured_tip_speed_mmps_p10_p50_p90_p95=np.percentile(values,[10,50,90,95]).tolist() if values else None,scope='Public measured TCP linear/angular velocities mapped through the source RGB grasp and port frame, at estimated3-14mmdepth; not exact raw-scoring-tip differentiation',physics='OfficialSCcontactlatchstartsafter1secondexclusivecontact. Slow6mmpsentry may latch before15.6mmdepth even after orientation is repaired.',new_training=False)
output=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1/SC_expert_entry_speed.json');output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)

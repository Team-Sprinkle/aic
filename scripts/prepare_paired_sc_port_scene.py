#!/usr/bin/env python3
"""One bounded matched-world SC port-code contrast from an existing training scene."""
from pathlib import Path
import hashlib,json,shutil,yaml
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[1]
base=root/'outputs/experiments/2026-09-30_transport_continuation'
src=base/'scenes/port1_five_shiftminus';dest=base/'scenes/port0_five_paired'
if dest.exists():raise FileExistsError(dest)
dest.mkdir(parents=True)
for name in ('world_audit.sdf','bridge_audit.yaml','measured_start_joint_state.json'):
 shutil.copy2(src/name,dest/name)
cfg=yaml.safe_load((src/'eval_config.yaml').read_text());task=next(iter(next(iter(cfg['trials'].values()))['tasks'].values()))
assert task['target_module_name']=='sc_port_1';task['target_module_name']='sc_port_0'
(dest/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
x=json.loads((base/'streams/cartesian_port1_five.json').read_text());edges=x['edges']
def matrix(key):
 data=edges[key][0][1];p=data['translation'];q=data['rotation'];a=np.eye(4);a[:3,:3]=Rotation.from_quat([q[k] for k in 'xyzw']).as_matrix();a[:3,3]=[p[k] for k in 'xyz'];return a
world_base=matrix('world|tabletop')@matrix('tabletop|base_link')
def opening(port):
 name=f'task_board/sc_port_{port}'
 return (np.linalg.inv(world_base)@matrix('aic_world|task_board')@matrix(f'task_board|{name}')@matrix(f'{name}|{name}/sc_port_base_link_entrance'))[:3,3]
delta=opening(0)-opening(1)
assert .04<np.linalg.norm(delta)<.09 and abs(delta[2])<.0001
route=json.loads((src/'route_plan.json').read_text());assert route['waypoints'][4]['name']=='sweep_after_row' and route['waypoints'][5]['name']=='return_toward_port'
route['port_link_base_m']=(np.asarray(route['port_link_base_m'])+delta).tolist()
for i in (4,5,6):
 p=np.asarray(route['waypoints'][i]['tcp_base_m']);p[0]+=delta[0]
 if i in (5,6):p[1]+=delta[1]
 route['waypoints'][i]['tcp_base_m']=p.tolist()
(dest/'route_plan.json').write_text(json.dumps(route,indent=2)+'\n')
manifest=dict(schema='aic_paired_sc_port_scene/v1',source=str(src),scene=str(dest),parent_world_sha256=hashlib.sha256((src/'world_audit.sdf').read_bytes()).hexdigest(),paired_world_sha256=hashlib.sha256((dest/'world_audit.sdf').read_bytes()).hexdigest(),same_world=True,changed='selected task port and port-referenced route ending; all robot/card/board/cable/reset starts copied',target_port_delta_base_m=delta.tolist(),source_target='sc_port_1',paired_target='sc_port_0',split='train',final_scene=False,admission='transport prefix only after scored route and video review; failed insertion excluded',stopping='one paired teacher planning/replay, one BC retry, two held-out starts; no broader collection unless gate passes')
(dest/'paired_scene_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

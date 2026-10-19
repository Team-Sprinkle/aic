"""Independent causal geometry audit for supervised wide-reset sources.

Only this posthoc audit reads object TF. Neither labels nor simulator poses
are returned as actor features, crop choices or action targets.
"""
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw


def tf_matrix(value):
    return matrix_from_xyz_xyzw([*[value['translation'][k] for k in 'xyz'],
                                 *[value['rotation'][k] for k in 'xyzw']])


def past(rows,stamp):
    i=bisect_right(rows,stamp,key=lambda x:x[0])-1
    if i<0:raise ValueError('No same-episode past geometry; refusing future/reset join')
    return rows[i]


def one(item):
    metadata,row=item;d=json.loads(Path(metadata['source']).read_text());edges=d['edges'];task=d['task']
    def edge(key,t):
        try:return tf_matrix(past(edges[key],t)[1])
        except ValueError as exc:
            raise ValueError(f'{key}: requested{t}, first{edges[key][0][0]}: {exc}') from exc
    def base_aic(t):return np.linalg.inv(edge('world|tabletop',t)@edge('tabletop|base_link',t))@edge('world|aic_world',t)
    stamp=round(row['perception_lock_time_s']*1e9)
    family=row['family'];parent='task_board/'+task['target_module_name']
    if family=='sc':
        base_board=base_aic(stamp)@edge('aic_world|task_board',stamp)
        true=base_board@edge('task_board|'+parent,stamp)@edge(parent+'|'+parent+'/sc_port_base_link_entrance',stamp)
        label_source='same_episode_past_fixed_fixture_TF'
    else:
        cal=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
        from build_two_family_port_projection_labels import sfp_ports
        card=int(task['target_module_name'].rsplit('_',1)[1]);port=int(task['port_name'].rsplit('_',1)[1])
        candidates={(c,p):m for c,p,m in sfp_ports(dict(
            trial_yaml=str(Path(metadata['source']).parent/'eval_config.yaml'),trial_id=d['trial_id']),cal)}
        true=candidates[card,port]
        # Configuration is known before the episode. Never substitute a later
        # board-TF observation when the first recorded broadcast arrives late.
        label_source='configured_fixed_mount_and_preexisting_calibrated_opening'
    estimated=matrix_from_xyz_xyzw(row['state'][:7])@np.linalg.inv(matrix_from_pose9(row['predicted_port_tcp_pose9']))
    delta=true[:3,:3].T@(estimated[:3,3]-true[:3,3])*1000
    angle=np.degrees((Rotation.from_matrix(true[:3,:3]).inv()*Rotation.from_matrix(estimated[:3,:3])).magnitude())
    # A common past cable/root/tip timestamp and its own past controller pose
    # avoid apparent grasp motion caused by asynchronous TF during a tilt.
    grasp_stamp=round(row['grasp_lock_time_s']*1e9);cable=task['cable_name'];tip=f"{cable}/{task['plug_name']}_link"
    tip_rows=edges[cable+'|'+tip];root_rows=edges['aic_world|'+cable]
    controller=d['streams']['/aic_controller/controller_state'];candidate=bisect_right(tip_rows,grasp_stamp,key=lambda x:x[0])-1
    actual=None;age=None
    while candidate>=0:
        t,tip_tf=tip_rows[candidate]
        if grasp_stamp-t>50_000_000:break
        root_time,root_tf=past(root_rows,t);ct,control=past(controller,t)
        if t-root_time<=2_000_000 and t-ct<=4_000_000:
            p=control['tcp_pose'];tcp=matrix_from_xyz_xyzw([*[p['position'][k] for k in 'xyz'],*[p['orientation'][k] for k in 'xyzw']])
            actual=(np.linalg.inv(tcp)@base_aic(t)@tf_matrix(root_tf)@tf_matrix(tip_tf))[:3,3]
            age=(grasp_stamp-t)/1e6;break
        candidate-=1
    if actual is None:raise ValueError('No synchronized causal grasp label at wide-reset lock')
    return dict(episode=row['episode'],family=family,split=row['split'],scene=row['scene'],
        port_position_mm=float(np.linalg.norm(delta)),port_lateral_mm=float(np.linalg.norm(delta[:2])),
        port_orientation_deg=float(angle),grasp_translation_mm=float(np.linalg.norm(actual-row['predicted_body_tip_position_m'])*1000),
        grasp_label_age_ms=age,port_label_source=label_source,
        object_geometry_posthoc_only=True,post_reset_geometry_used=False)


def audit(items):
    with ProcessPoolExecutor(max_workers=4) as pool:all_rows=list(pool.map(one_safe,items))
    errors=[x for x in all_rows if 'audit_error' in x];rows=[x for x in all_rows if 'audit_error' not in x]
    groups={}
    for family in ('sc','sfp'):
        for split in ('train','validation'):
            subset=[x for x in rows if x['family']==family and x['split']==split]
            groups[family+'/'+split]=dict(episodes=len(subset),**{k:dict(p50=float(np.median([x[k] for x in subset])) if subset else None,p95=float(np.percentile([x[k] for x in subset],95)) if subset else None,max=max(x[k] for x in subset) if subset else None) for k in ['port_position_mm','port_lateral_mm','port_orientation_deg','grasp_translation_mm']})
    passed=all(g['episodes']>=6 and g['port_position_mm']['p95']<=1 and
        g['port_lateral_mm']['p95']<=.5 and g['port_orientation_deg']['p95']<=3 and
        g['grasp_translation_mm']['p95']<=.5 and g['port_position_mm']['max']<=5 and
        g['grasp_translation_mm']['max']<=2 for g in groups.values())
    return dict(passed=passed and not errors,groups=groups,rows=rows,errors=errors,geometry_posthoc_only=True,
        original_actor_rows_modified=False,final_scenes_opened=False)


def one_safe(item):
    try:return one(item)
    except (ValueError,KeyError) as exc:
        return dict(episode=item[1]['episode'],family=item[1]['family'],split=item[1]['split'],audit_error=str(exc))

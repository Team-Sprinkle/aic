#!/usr/bin/env python3
"""Matched explicit SC visual servo with no/local/hierarchical measured recovery."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from shared_recovery_eval_environment import environment, effective_environment


def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def run(command, log, repo, env=None, ros=False):
    if ros:
        prefix=('source .pixi/envs/default/setup.bash\n'
                'export PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n')
        command=['bash','-c',prefix+shlex.join(list(map(str,command)))]
    with log.open('w') as f:subprocess.run(list(map(str,command)),cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def discontinuity(trace):
    local=[t for t in trace if t['event']=='locked_local_perception']
    if not local:return dict(actor_ran=False)
    start=local[0]['sim_time']
    commands=[t for t in trace if t['event']=='command' and t['sim_time']>=start]
    values={k:[] for k in ['within_mm','boundary_mm','within_deg','boundary_deg']}
    for a,b in zip(commands,commands[1:]):
        kind='boundary' if b['chunk_index']==0 else 'within'
        aa=np.asarray(a['absolute_tcp_target']);bb=np.asarray(b['absolute_tcp_target'])
        values[kind+'_mm'].append(np.linalg.norm(aa[:3,3]-bb[:3,3])*1000)
        values[kind+'_deg'].append((Rotation.from_matrix(aa[:3,:3]).inv()*Rotation.from_matrix(bb[:3,:3])).magnitude()*180/np.pi)
    return dict(actor_ran=True,statistics={k:np.percentile(v,[50,95,100]).tolist() if v else None for k,v in values.items()},
        predicted_lengths=dict(Counter(len(t['actor_control_point_targets_port']) for t in local)),
        executed_indices=dict(Counter(t['chunk_index'] for t in commands)),
        insertion_stale_drops=sum(t['event']=='stale_chunk_member_dropped' and t.get('sim_time',-1)>=start for t in trace))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='5');p.add_argument('--domain',default='115')
    p.add_argument('--experimental-diagnostic', action='store_true')
    p.add_argument('--policy',default='RunVisualConnectorServoRecoveryV21')
    p.add_argument('--empty-row-selection',type=Path)
    a=p.parse_args();repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
    if a.output.exists():raise FileExistsError('Preserve previous live comparison')
    a.output.mkdir(parents=True)
    prior=Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1')
    plan=json.loads((prior/'scenario_bank/plan.json').read_text())
    cases=[next(c for c in plan['cases'] if c['family']==f and c['replicate']==2 and c['card_count']==3 and c['port']==1
        and (f=='sc' or c['requested_card']==2)) for f in ['sc','sfp']]
    if (a.root/'evaluation_cases.json').exists():
        cases=json.loads((a.root/'evaluation_cases.json').read_text())['cases']
    if not cases or any(c['family']!='sc' or 'configured_start_joints' not in c for c in cases):raise ValueError('Explicit SC development manifest and configured starts required')
    variants=json.loads((a.root/'evaluation_variants.json').read_text())['variants']
    modeldir=a.output/'models';modeldir.mkdir()
    for v in variants:
        source=Path(v['checkpoint']);dest=modeldir/(v['name']+'.pt');shutil.copyfile(source,dest);v['checkpoint']=str(dest)
    files=[repo/'aic_example_policies/aic_example_policies/ros'/n for n in [
        'RunVisualConnectorServoRecoveryV21.py','RunVisualConnectorServoRecoveryV18.py','RunVisualConnectorServoRecoveryV17.py','RunVisualConnectorServoRecoveryV14.py','RunVisualConnectorServoRecoveryV13.py','RunVisualConnectorServoRecoveryV12.py','RunVisualConnectorServoRecoveryV10.py','RunVisualConnectorServoRecoveryV8.py','RunVisualConnectorServoRecoveryV7.py','RunVisualConnectorServoRecoveryV6.py','RunVisualConnectorServoRecoveryV5.py','hierarchical_recovery.py','RunVisualConnectorServoDiagnosticV4.py','RunVisualConnectorServoDiagnosticV3.py','RunVisualConnectorServoDiagnosticV2.py','RunContinuousConnectorInsertionV2.py','RunContinuousConnectorInsertion.py','RunFullConnectorInsertion.py','RunHistoryChunkedInsertion.py','RunSample25mmThenInsertion.py','RunSampleThenSharedInsertion.py',
        'RunRuleThenSharedRecovery.py','RunLockedPortLocal.py','RunSharedPortTCP.py']]
    files += [repo/'scripts'/n for n in ['close_grasp_consensus.py','sc_spatial_grasp_observation.py','sc_qualified_close_orientation_observation.py','sc_qualified_close_composite_observation.py','train_sc_native_keypoints.py','sc_native_grasp_observation.py','chunked_port_actor.py','run_close_grasp_insertion_eval.sh','run_visual_servo_close_grasp_probe.py','correlated_lateral_retry.py','causal_force_reference.py','insertion_reference_governor_v2.py','insertion_reference_governor.py','fine_connector_orientation.py','native_connector_pose_observation.py','train_native_connector_pose.py','full_connector_geometry.py']]
    if a.policy!='RunVisualConnectorServoRecoveryV21':
        files.append(repo/'aic_example_policies/aic_example_policies/ros'/(a.policy+'.py'))
    if a.policy in ('RunVisualConnectorServoContextV28','RunVisualConnectorServoAlignedV29','RunVisualConnectorServoTrackedV30','RunVisualConnectorServoBiasSeedV31','RunVisualConnectorServoStiffV32','RunVisualConnectorServoStiffV33','RunVisualConnectorServoRecoveryFastV34'):
        files.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoRecoveryV26.py')
    if a.policy in ('RunVisualConnectorServoAlignedV29','RunVisualConnectorServoTrackedV30','RunVisualConnectorServoBiasSeedV31','RunVisualConnectorServoStiffV32','RunVisualConnectorServoStiffV33','RunVisualConnectorServoRecoveryFastV34'):
        files.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoContextV28.py')
    if a.policy in ('RunVisualConnectorServoTrackedV30','RunVisualConnectorServoBiasSeedV31','RunVisualConnectorServoStiffV32','RunVisualConnectorServoStiffV33','RunVisualConnectorServoRecoveryFastV34'):
        files.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoAlignedV29.py')
    if a.policy in ('RunVisualConnectorServoStiffV32','RunVisualConnectorServoStiffV33','RunVisualConnectorServoRecoveryFastV34'):
        files.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoBiasSeedV31.py')
    if a.policy in ('RunVisualConnectorServoStiffV33','RunVisualConnectorServoRecoveryFastV34'):
        files.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoStiffV32.py')
    if a.policy=='RunVisualConnectorServoRecoveryFastV34':
        files.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoStiffV33.py')
        files.append(repo/'scripts/shared_grasp_training_jpeg.py')
    import ast
    ros_dir=repo/'aic_example_policies/aic_example_policies/ros'
    pending=[ros_dir/(a.policy+'.py')];visited=set()
    while pending:
        path=pending.pop()
        if path in visited:continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                dependency=ros_dir/(node.module.rsplit('.',1)[-1]+'.py')
                if dependency.exists():pending.append(dependency)
    files=list(dict.fromkeys([*files,*sorted(visited),repo/'scripts/empty_row_observation.py',repo/'scripts/shared_grasp_training_jpeg.py']))
    extra_environment={}
    if a.empty_row_selection:
        empty=json.loads(a.empty_row_selection.read_text())
        for key in ('checkpoint','approval'):
            dest=modeldir/('empty_row_'+Path(empty[key]).name)
            shutil.copyfile(empty[key],dest);empty[key]=str(dest)
        extra_environment.update(AIC_EMPTY_ROW_CHECKPOINT=empty['checkpoint'],AIC_EMPTY_ROW_APPROVAL=empty['approval'])
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    close=json.loads((a.root/'close_pose_selection.json').read_text())
    approval=json.loads(Path(close['approval']).read_text())
    if a.experimental_diagnostic:
        if (len(cases)!=1 or len(variants)!=1 or approval.get('pose_gate_passed') is not False or
                not approval.get('experimental_trial_only') or
                approval.get('allowed_development_case')!=cases[0]['case']):
            raise ValueError('Experimental permission is restricted to one exact development case')
    elif not approval['pose_gate_passed']:
        raise ValueError('Qualified model required outside the bounded diagnostic')
    if not approval['visual_inspection_completed']:
        raise ValueError('Actual visual inspection is required')
    extra_close_paths=[]
    if approval.get('prediction_kind') in ('composite_pose','experimental_composite_pose'):
        bundle=a.output/'close_pose_bundle'
        shutil.copytree(Path(close['approval']).parent,bundle)
        close=dict(checkpoint=str(bundle/Path(close['checkpoint']).name),approval=str(bundle/Path(close['approval']).name))
        extra_close_paths=[p for p in bundle.iterdir() if p.is_file()]
    for path in extra_close_paths:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ['checkpoint','approval']:
        path=Path(close[key]);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in modeldir.iterdir()})
    orientation_root=Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1')
    for name in ['sc_orientation_jitter_v2','sfp_direct_pose']:
        p=orientation_root/name/'best.pt';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in files:shutil.copyfile(p,a.output/p.name)
    save(a.output/'plan.json',dict(cases=cases,variants=variants,source_model_hashes=hashes,
        scenes='Explicit SC development manifest; sample-derived starts, no sealed final scenes.',
        gate='Strict causal terminal insertion, correct requested port, no harm and live latency; no automatic promotion.',
        recovery='Matched none/local/hierarchical explicit-controller contact-stall recovery, no actor or RL',
        policy=a.policy,experimental_trial_only=a.experimental_diagnostic,new_RL=False))
    selected=json.loads((prior/'frozen/selection.json').read_text());spec=json.loads((prior/'frozen/transport_spec.json').read_text())
    results=[]
    for c in cases:
        for v in variants:
            assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in hashes.items())
            name=c['case']+'__'+v['name'];audit=a.output/name;casefile=a.output/(name+'.case.json');save(casefile,c)
            env=environment(repo,Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1'),spec['bundle'],c,
                policy='aic_example_policies.ros.'+a.policy,domain=a.domain,actor=Path(v['checkpoint']),recovery=v['recovery'])
            env.update(extra_environment)
            aux=selected['auxiliary'];env.update(AIC_TRANSPORT_EVAL_GPU=a.gpu,AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
                AIC_RULE_TRANSPORT_TEMPLATE=str(prior/'frozen/transport_spec.json'),AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'],
                AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'],
                AIC_INSERTION_EXECUTE_STEPS=str(v['execute']),
                AIC_SC_CLOSE_GRASP_CHECKPOINT=close['checkpoint'],AIC_SC_CLOSE_GRASP_APPROVAL=close['approval'],
                AIC_SC_CLOSE_DIAGNOSTIC_TRIAL='1' if a.experimental_diagnostic else '0')
            command=['bash',repo/'scripts/run_close_grasp_insertion_eval.sh',name,
                repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt',audit]
            save(a.output/'current_job.json',dict(case=c,variant=v,audit=str(audit)))
            save(a.output/(name+'.command.json'),dict(command=list(map(str,command)),environment={k:x for k,x in env.items() if k.startswith('AIC_')}))
            run(command,a.output/(name+'.log'),repo,env)
            effective_environment(audit,env)
            actual=dict(x.split('=',1) for x in json.loads((audit/'container_environment.json').read_text()) if '=' in x)
            assert actual['AIC_INSERTION_EXECUTE_STEPS']==str(v['execute'])
            assert actual['AIC_SC_CLOSE_GRASP_CHECKPOINT']==close['checkpoint']
            assert actual['AIC_SC_CLOSE_GRASP_APPROVAL']==close['approval']
            run([py,repo/'scripts/extract_moveit_training_stream.py',audit,audit/'scored_stream.json'],audit/'extraction.log',repo,ros=True)
            run([py,repo/'scripts/score_shared150mm_rollout.py','--run',audit,'--case',casefile,'--arm',v['name']],audit/'scoring.log',repo,ros=True)
            row=json.loads((audit/'shared150mm_metrics.json').read_text());trace=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
            row['plan_continuity']=discontinuity(trace)
            steps=[x for x in trace if x['event']=='visual_servo_step']
            if row['actor_ran']:raise ValueError('Servo diagnostic unexpectedly ran the learned actor')
            row['rollout_type']='image_only_rule_insertion_not_BC_or_RL'
            row['servo_control_step_ms_p50_p95_p99']=np.percentile([x['complete_control_step_ms'] for x in steps],[50,95,99]).tolist() if steps else None
            finished=[x for x in trace if x['event']=='visual_servo_finished']
            row['close_grasp_events']=[x for x in trace if x['event'].startswith('close_grasp_')]
            rejected=[x for x in trace if x['event']=='close_grasp_update_rejected']
            row['servo_terminal_reason']=finished[-1]['reason'] if finished else rejected[-1]['reason'] if rejected else 'upstream_transport_or_acquisition_failure'
            row['learned_actor_success_claim']=False
            row['recovery_mode']=v['recovery']
            row['recovery_reference_reseeds']=[t for t in trace if t['event']=='backoff_reference_reseeded']
            row['loaded_stall_triggers']=[t for t in trace if t['event']=='loaded_servo_stall_recovery']
            row['history_seed_events']=[t for t in trace if t['event']=='insertion_history_seeded']
            if row['actor_ran']:
                seed=next(t for t in reversed(row['history_seed_events']) if t['samples']>0)
                if seed['last_sim_time']-seed['first_sim_time']<1.:
                    raise ValueError('Transport history failed to survive actor takeover')
            locks=[t for t in trace if t['event']=='shared_visual_lock']
            row['connector_orientation_events']=[t for t in trace if t['event']=='connector_rotation_acquisition']
            row['full_connector_grasp_locks']=[t for t in trace if t['event']=='full_connector_grasp_bound']
            if row['actor_ran']:
                if len(row['full_connector_grasp_locks'])!=1:raise ValueError('Missing unique full connector lock')
                fixed=np.asarray(row['full_connector_grasp_locks'][0]['rotation_tcp'])
                for t in trace:
                    if t['event']=='locked_local_perception':
                        if not t['tip_orientation_input'] or not np.allclose(t['estimated_connector_rotation_in_tcp'],fixed,atol=1e-7):
                            raise ValueError('Actor lost its fixed image-only connector orientation')
            row['pose_lock_count']=len(locks);row['near_handoff_ready']=any(t['event']=='near_handoff_ready' for t in trace)
            results.append(row);save(a.output/'progress.json',dict(completed=len(results),planned=len(cases)*len(variants),results=results,final_scenes_opened=False))
            run([py,repo/'scripts/render_transport_five_camera_video.py',audit,'--fps','20','--grid','--active-only'],audit/'video.log',repo)
    summary=dict(completed=len(results),results=results,full=sum(r['physical_full_insertion'] for r in results),
        qualified=sum(r['qualified'] for r in results),no_automatic_promotion=True,new_RL=False,final_scenes_opened=False)
    save(a.output/'summary.json',summary)
    lines=['# Full-connector insertion and jerk probe','',
        '| Case | Variant | Full insertion | Qualified | Near handoff |','| --- | --- | --- | --- | --- |']
    for r in results:lines.append(f"| {r['case']} | {r['arm']} | {r['physical_full_insertion']} | {r['qualified']} | {r['near_handoff_ready']} |")
    lines+=['','Explicit SC controller diagnostics, not BC or RL success. Coverage is limited to the frozen manifest.','Per-run geometry, force, history seeds, command discontinuity and videos remain beside raw bags.']
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

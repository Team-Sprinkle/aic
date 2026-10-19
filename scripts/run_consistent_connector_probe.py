#!/usr/bin/env python3
"""Matched complete-target BC controller-context diagnostic, no recovery."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
    if a.output.exists():raise FileExistsError('Preserve previous live comparison')
    a.output.mkdir(parents=True)
    prior=Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1')
    plan=json.loads((prior/'scenario_bank/plan.json').read_text())
    cases=[next(c for c in plan['cases'] if c['family']==f and c['replicate']==2 and c['card_count']==3 and c['port']==1
        and (f=='sc' or c['requested_card']==2)) for f in ['sc','sfp']]
    if (a.root/'evaluation_cases.json').exists():
        cases=json.loads((a.root/'evaluation_cases.json').read_text())['cases']
    variants=json.loads((a.root/'evaluation_variants.json').read_text())['variants']
    modeldir=a.output/'models';modeldir.mkdir()
    for v in variants:
        source=Path(v['checkpoint'])
        summary_path=source.parent/'summary.json'
        if summary_path.exists():
            training=json.loads(summary_path.read_text())
            if training.get('selection_metric')=='two_family_insertion':
                def objective(report):
                    return max(value for family in ['sc','sfp'] for value in [
                        report[family]['near_decoded_tcp_lateral_mm']['p95']/.5,
                        report[family]['near_first_orientation_deg']['p95']/3.,
                        report[family+'_far_approach']['first_body_command_mm']['p95']/3.,
                        report[family]['axial_chunk_span_error_mm']['p95']])
                initial_score=objective(training['initialization_validation']);trained_score=objective(training['selected_validation'])
                v['training_updates_completed']=training['updates_completed']
                v['initial_validation_objective']=initial_score;v['trained_validation_objective']=trained_score
                v['selected_initialization']=initial_score<trained_score
                if v['selected_initialization']:
                    kind='masked' if training['masked_state_indices'] else 'public'
                    initial=source.parent.parent/'context_initializations'/(kind+'.pt')
                    if not initial.exists():raise ValueError('Better initialization missing; do not silently select a regressed fit')
                    source=initial
        v['runtime_selected_source']=str(source)
        dest=modeldir/(v['name']+'.pt');shutil.copyfile(source,dest);v['checkpoint']=str(dest)
    files=[repo/'aic_example_policies/aic_example_policies/ros'/n for n in [
        'RunConsistentConnectorInsertion.py','RunContinuousConnectorInsertionV2.py','RunContinuousConnectorInsertion.py','RunFullConnectorInsertion.py','RunHistoryChunkedInsertion.py','RunSample25mmThenInsertion.py','RunSampleThenSharedInsertion.py',
        'RunRuleThenSharedRecovery.py','RunLockedPortLocal.py','RunSharedPortTCP.py']]
    files += [repo/'scripts'/n for n in ['chunked_port_actor.py','run_chunked_insertion_eval.sh','run_consistent_connector_probe.py','insertion_reference_governor_v2.py','insertion_reference_governor.py','native_connector_pose_observation.py','train_native_connector_pose.py','full_connector_geometry.py']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in modeldir.iterdir()})
    import torch
    for variant in variants:
        saved=torch.load(variant['checkpoint'],map_location='cpu',weights_only=False)
        fingerprints=saved.get('connector_estimator_fingerprints')
        if not fingerprints:raise ValueError('Missing matched estimator metadata')
        for record in fingerprints.values():
            p=Path(record['path']);actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=record['sha256']:raise ValueError('Changed selected estimator')
            hashes[str(p)]=actual
    for p in files:shutil.copyfile(p,a.output/p.name)
    save(a.output/'plan.json',dict(cases=cases,variants=variants,source_model_hashes=hashes,
        scenes='Explicit controlled development cases; evaluation_cases.json selects reused25mm integration starts. No official/final scenes.',
        gate='Matched masked/unmasked context arms on explicit reused development cases. Preserve failures; expand only after physical insertion improves. No final-scene tuning.',
        recovery='none; history/codec/chunk comparison isolates actor control',new_RL=False))
    selected=json.loads((prior/'frozen/selection.json').read_text());spec=json.loads((prior/'frozen/transport_spec.json').read_text())
    results=[]
    for c in cases:
        for v in variants:
            assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in hashes.items())
            name=c['case']+'__'+v['name'];audit=a.output/name;casefile=a.output/(name+'.case.json');save(casefile,c)
            env=environment(repo,Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1'),spec['bundle'],c,
                policy='aic_example_policies.ros.RunConsistentConnectorInsertion',domain='99',actor=Path(v['checkpoint']),recovery='none')
            aux=selected['auxiliary'];env.update(AIC_TRANSPORT_EVAL_GPU='5',AIC_TRANSPORT_EVAL_POLICY_GPU='5',
                AIC_RULE_TRANSPORT_TEMPLATE=str(prior/'frozen/transport_spec.json'),AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'],
                AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'],
                AIC_INSERTION_EXECUTE_STEPS=str(v['execute']))
            command=['bash',repo/'scripts/run_chunked_insertion_eval.sh',name,
                repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt',audit]
            save(a.output/'current_job.json',dict(case=c,variant=v,audit=str(audit)))
            save(a.output/(name+'.command.json'),dict(command=list(map(str,command)),environment={k:x for k,x in env.items() if k.startswith('AIC_')}))
            run(command,a.output/(name+'.log'),repo,env)
            effective_environment(audit,env)
            actual=dict(x.split('=',1) for x in json.loads((audit/'container_environment.json').read_text()) if '=' in x)
            assert actual['AIC_INSERTION_EXECUTE_STEPS']==str(v['execute'])
            run([py,repo/'scripts/extract_moveit_training_stream.py',audit,audit/'scored_stream.json'],audit/'extraction.log',repo,ros=True)
            run([py,repo/'scripts/score_shared150mm_rollout.py','--run',audit,'--case',casefile,'--arm',v['name']],audit/'scoring.log',repo,ros=True)
            row=json.loads((audit/'shared150mm_metrics.json').read_text());trace=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
            row['plan_continuity']=discontinuity(trace)
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
    lines+=['','Explicit reused development starts form a diagnostic screen, not a generalization result.','Per-run geometry, force, history seeds, command discontinuity and videos remain beside raw bags.']
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

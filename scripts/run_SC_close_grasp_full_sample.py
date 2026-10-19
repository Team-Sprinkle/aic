#!/usr/bin/env python3
"""One unchanged three-trial public sample run of an explicitly selected repaired candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from shared_recovery_eval_environment import environment, effective_environment


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--domain', default='100')
    parser.add_argument('--policy', default='RunVisualConnectorServoRecoveryV21')
    parser.add_argument('--close-selection', type=Path, required=True)
    parser.add_argument('--sfp-orientation-selection', type=Path)
    parser.add_argument('--transport-spec', type=Path)
    parser.add_argument('--empty-row-selection', type=Path)
    parser.add_argument('--recovery',choices=['local','hierarchical'],required=True)
    parser.add_argument('--experimental-diagnostic', action='store_true')
    parser.add_argument('--successful-pilot-gate', type=Path)
    args = parser.parse_args()
    repo = Path.cwd()
    if not args.policy.isidentifier() or not args.policy.startswith('RunVisualConnectorServo'):
        raise ValueError('Expected a local explicit controller class')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    prior = Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1')
    model_root = Path('/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1')
    sample = repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml'
    assert digest(sample) == '21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d'
    scene = root/'scene'
    shutil.copytree(prior/'configuration_scenes/exact_sample_full_r0', scene)
    assert digest(scene/'eval_config.yaml') == digest(sample)
    # This template retains the original sample robot home, not a near-port reset.
    config = yaml.safe_load(sample.read_text())
    robot_text = (scene/'ur_llb_start.xacro').read_text()
    for joint, value in config['robot']['home_joint_positions'].items():
        assert f'{joint}={value:.7f}' in robot_text
    frozen = root/'frozen'
    frozen.mkdir()
    actor = frozen/'full_connector_h4_e4.pt'
    shutil.copyfile(model_root/'full_connector_bc_h4/best.pt', actor)
    spec = json.loads((prior/'frozen/transport_spec.json').read_text())
    selection = json.loads((prior/'frozen/selection.json').read_text())
    case = dict(case='unchanged_sample_full_connector', scene=str(scene), domain='public_sample_complete')
    env = environment(repo, Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1'),
                      spec['bundle'], case, policy='aic_example_policies.ros.'+args.policy,
                      domain=args.domain, actor=actor, recovery=args.recovery)
    aux = selection['auxiliary']
    env.update(AIC_TRANSPORT_EVAL_REPO=str(repo), AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT='1',
               AIC_TRANSPORT_EVAL_GPU=args.gpu, AIC_TRANSPORT_EVAL_POLICY_GPU=args.gpu,
               AIC_RULE_TRANSPORT_TEMPLATE=str(prior/'frozen/transport_spec.json'),
               AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'], AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],
               AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'], AIC_INSERTION_EXECUTE_STEPS='4')
    repaired_sfp_paths=[]
    if args.sfp_orientation_selection:
        sfp=json.loads(args.sfp_orientation_selection.read_text())
        for key in ('checkpoint','approval'):
            path=frozen/('SFP_orientation_'+Path(sfp[key]).name)
            shutil.copyfile(sfp[key],path)
            sfp[key]=str(path);repaired_sfp_paths.append(path)
        env.update(AIC_SFP_ORIENTATION_REPAIR_CHECKPOINT=sfp['checkpoint'],
                   AIC_SFP_ORIENTATION_REPAIR_APPROVAL=sfp['approval'])
    if args.empty_row_selection:
        empty=json.loads(args.empty_row_selection.read_text())
        for key in ('checkpoint','approval'):
            path=frozen/('empty_row_'+Path(empty[key]).name)
            shutil.copyfile(empty[key],path);empty[key]=str(path);repaired_sfp_paths.append(path)
        env.update(AIC_EMPTY_ROW_CHECKPOINT=empty['checkpoint'],AIC_EMPTY_ROW_APPROVAL=empty['approval'])
    if args.transport_spec:
        transport_copy=frozen/'candidate_transport_spec.json'
        shutil.copyfile(args.transport_spec,transport_copy)
        env['AIC_RULE_TRANSPORT_TEMPLATE']=str(transport_copy)
        repaired_sfp_paths.append(transport_copy)
    close=json.loads(args.close_selection.read_text())
    approval=json.loads(Path(close['approval']).read_text())
    if args.experimental_diagnostic:
        pilot=json.loads(args.successful_pilot_gate.read_text())
        if (not pilot['explicit_trial_pass'] or
                digest(Path(pilot['raw_metrics_path']))!=pilot['raw_metrics_sha256'] or
                not approval.get('experimental_trial_only') or approval.get('pose_gate_passed') is not False or
                approval.get('allowed_development_case')!='unchanged_sample_full_connector'):
            raise ValueError('Experimental full sample requires an unchanged physical pilot pass and exact scope')
    elif not approval['pose_gate_passed']:
        raise ValueError('Unqualified model requires an explicitly scoped diagnostic')
    if not approval['visual_inspection_completed']:
        raise ValueError('Actual model inspection required')
    extra_close_paths=[]
    if approval.get('prediction_kind') in ('composite_pose','experimental_composite_pose'):
        source_bundle=Path(close['approval']).parent
        frozen_bundle=frozen/'SC_close_pose_bundle'
        shutil.copytree(source_bundle,frozen_bundle)
        close_checkpoint=frozen_bundle/Path(close['checkpoint']).name
        close_approval=frozen_bundle/Path(close['approval']).name
        extra_close_paths=[p for p in frozen_bundle.iterdir() if p.is_file()]
    else:
        close_checkpoint=frozen/'SC_close_grasp.pt';close_approval=frozen/'SC_close_grasp_approval.json'
        shutil.copyfile(close['checkpoint'],close_checkpoint);shutil.copyfile(close['approval'],close_approval)
    assert digest(close_checkpoint)==approval['checkpoint_sha256']
    env.update(AIC_SC_CLOSE_GRASP_CHECKPOINT=str(close_checkpoint),AIC_SC_CLOSE_GRASP_APPROVAL=str(close_approval),
               AIC_SC_CLOSE_DIAGNOSTIC_TRIAL='1' if args.experimental_diagnostic else '0')
    launcher_text = subprocess.check_output(['bash', str(repo/'scripts/run_close_grasp_insertion_eval.sh'),
                                             '--print-generated-launcher'], cwd=repo, text=True)
    assert launcher_text.count('deadline=$((SECONDS+450))') == 1
    launcher_text = launcher_text.replace('deadline=$((SECONDS+450))', 'deadline=$((SECONDS+1800))')
    wait='while ! test -f "$audit/engine.exit" && (( SECONDS < deadline )); do\n'
    assert launcher_text.count(wait)==1
    launcher_text=launcher_text.replace(wait,wait+
        ' if test -f "$audit/policy.exit" && test "$(cat "$audit/policy.exit")" != 0; then\n'
        '  echo failed_policy_exit > "$audit/launcher.exit"; exit 1\n'
        ' fi\n')
    launcher = frozen/'sample_launcher.sh'
    launcher.write_text(launcher_text)
    source_paths = [repo/'aic_example_policies/aic_example_policies/ros'/name for name in (
        args.policy+'.py', *[f'RunVisualConnectorServoRecoveryV{v}.py' for v in [17,15,14,13,12,10,8,7,6,5]], 'hierarchical_recovery.py', 'RunVisualConnectorServoDiagnosticV4.py', 'RunVisualConnectorServoDiagnosticV3.py', 'RunVisualConnectorServoDiagnosticV2.py', 'RunContinuousConnectorInsertionV2.py', 'RunContinuousConnectorInsertion.py', 'RunFullConnectorInsertion.py', 'RunHistoryChunkedInsertion.py', 'RunSample25mmThenInsertion.py',
        'RunSampleThenSharedInsertion.py', 'RunRuleThenSharedRecovery.py', 'RunLockedPortLocal.py',
        'RunSharedPortTCP.py')]
    source_paths += [repo/'scripts'/name for name in ('close_grasp_consensus.py', 'sc_spatial_grasp_observation.py','sc_qualified_close_orientation_observation.py','sc_qualified_close_composite_observation.py','train_sc_native_keypoints.py', 'sc_native_grasp_observation.py', 'run_close_grasp_insertion_eval.sh', 'full_connector_geometry.py', 'chunked_port_actor.py',
                     'fine_connector_orientation.py','native_connector_pose_observation.py','train_native_connector_pose.py', 'insertion_reference_governor_v2.py', 'insertion_reference_governor.py', 'correlated_lateral_retry.py', 'causal_force_reference.py', 'shared_grasp_training_jpeg.py','sfp_repaired_pose_observation.py','empty_row_observation.py','empty_scene_receipt.py','approach_reference.py', 'run_SC_close_grasp_full_sample.py')]
    if args.policy in ('RunVisualConnectorServoRecoveryV26','RunVisualConnectorServoRecoveryFastV27'):
        source_paths.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoRecoveryV21.py')
    if args.policy=='RunVisualConnectorServoRecoveryFastV27':
        source_paths.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoRecoveryV26.py')
    if args.policy in ('RunVisualConnectorServoRecoveryFastV34','RunVisualConnectorServoSFPGraspV35'):
        source_paths.extend(repo/'aic_example_policies/aic_example_policies/ros'/name for name in [
            'RunVisualConnectorServoStiffV33.py','RunVisualConnectorServoStiffV32.py',
            'RunVisualConnectorServoBiasSeedV31.py','RunVisualConnectorServoAlignedV29.py',
            'RunVisualConnectorServoContextV28.py','RunVisualConnectorServoRecoveryV26.py',
            'RunVisualConnectorServoRecoveryV21.py'])
    if args.policy=='RunVisualConnectorServoSFPGraspV35':
        source_paths.append(repo/'aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoRecoveryFastV34.py')
    if args.policy in ('RunVisualConnectorServoRecoveryV18', 'RunVisualConnectorServoRecoveryV19', 'RunVisualConnectorServoRecoveryV20', 'RunVisualConnectorServoRecoveryV22'):
        source_paths += [repo/'aic_example_policies/aic_example_policies/ros'/f'RunVisualConnectorServoRecoveryV{v}.py'
                         for v in ([17] if args.policy.endswith('V18') else [17, 18] if args.policy.endswith('V19') else [17, 18, 19])]
    # Freeze local ROS imports recursively, including new experiment subclasses.
    # This avoids silently missing a parent when a new candidate is selected.
    import ast
    ros_dir=repo/'aic_example_policies/aic_example_policies/ros'
    pending=[ros_dir/(args.policy+'.py')]
    visited=set()
    while pending:
        path=pending.pop()
        if path in visited:continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                dependency=ros_dir/(node.module.rsplit('.',1)[-1]+'.py')
                if dependency.exists():pending.append(dependency)
    source_paths=list(dict.fromkeys([*source_paths,*sorted(visited)]))
    # RGB estimators import additional local script modules. Freeze that code
    # too, rather than only the ROS inheritance chain and a hand-kept subset.
    pending=list(source_paths);all_local=set()
    while pending:
        path=pending.pop()
        if path in all_local:continue
        all_local.add(path)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                   [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                for base in (ros_dir,repo/'scripts'):
                    dependency=base/(name.rsplit('.',1)[-1]+'.py')
                    if dependency.exists() and dependency not in all_local:pending.append(dependency)
    source_paths=sorted(all_local)
    model_paths = [*repaired_sfp_paths, *extra_close_paths, close_checkpoint, close_approval, actor, model_root/'sc_orientation_jitter_v2/best.pt', model_root/'sfp_direct_pose/best.pt']
    model_paths += [Path(v) for v in spec['bundle'].values()]
    model_paths += [Path(v) for v in aux.values()]
    calibration=repo/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json'
    calibration_copy=frozen/calibration.name
    shutil.copyfile(calibration,calibration_copy)
    model_paths.extend([calibration,calibration_copy])
    hashes = {str(p): digest(p) for p in [*source_paths, *model_paths, sample, launcher]}
    for path in source_paths:
        shutil.copyfile(path, frozen/path.name)
    plan = dict(schema='full_connector_sample_configuration/v1', sample_sha256=digest(sample),
                expected_trials=list(config['trials']), maximum_score=100*len(config['trials']),
                policy=args.policy, recovery=args.recovery, recovery_scope='SC only; SFP uses its source-pinned image-controller seating dwell with no recovery', training=False, learned_actor_inference=False, insertion_controller='Explicit frozen RGB/public-state alignment and advance control; NOT BC or RL',
                experimental_trial_only=args.experimental_diagnostic,
                preceding_pilot_gate=str(args.successful_pilot_gate) if args.experimental_diagnostic else None,
                model_globally_qualified=approval['pose_gate_passed'],
                prediction_steps=4, execution_steps=4, step_ms=50,
                starts='Original YAML robot home and all three trials in original order',
                physical_gpu=int(args.gpu), reserved_final_scenes_opened=False, inputs_sha256=hashes,
                stopping_rule='One complete sample YAML, no policy tuning during evaluation',
                whole_container_watchdog_s=1800, official_task_time_limits_changed=False)
    save(root/'plan.json', plan)
    audit = root/'rollout'
    command = ['bash', str(launcher), case['case'],
               str(repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'), str(audit)]
    save(root/'command.json', dict(command=command, environment={k:v for k,v in env.items() if k.startswith('AIC_')}))
    save(root/'state.json', dict(stage='running_unchanged_sample', planned_trials=len(config['trials'])))
    with (root/'launcher.log').open('w') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        save(root/'state.json', dict(stage='infrastructure_or_execution_failure_requires_audit', returncode=result.returncode))
        raise RuntimeError(f'Launcher failed; preserve artifacts: {root}')
    actual = effective_environment(audit, env)
    assert actual['AIC_SHARED_RECOVERY_MODE'] == args.recovery
    container=dict(x.split('=',1) for x in json.loads((audit/'container_environment.json').read_text()) if '=' in x)
    assert container['AIC_SC_CLOSE_GRASP_CHECKPOINT']==str(close_checkpoint)
    assert container['AIC_SC_CLOSE_GRASP_APPROVAL']==str(close_approval)
    assert digest(audit/'eval_config.yaml') == digest(sample)
    scoring = yaml.safe_load((audit/'results/scoring.yaml').read_text())
    assert set(scoring)-{'total'} == set(config['trials']), 'Incomplete sample score'
    assert all(digest(path) == checksum for path, checksum in hashes.items()), 'Frozen input changed'
    save(root/'summary.json', dict(score=scoring, maximum_score=plan['maximum_score'],
         sample_unchanged=True, complete_configuration=True, recovery=args.recovery,
         physical_insertion_audit='pending separate causal per-trial analysis', training=False,
         reserved_final_scenes_opened=False, effective_environment=actual))
    save(root/'state.json', dict(stage='complete_score_requires_physical_and_video_review',
                                total=scoring['total'], maximum_score=plan['maximum_score']))
    print(json.dumps(scoring, indent=2), flush=True)


if __name__ == '__main__':
    main()

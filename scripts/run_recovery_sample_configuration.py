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
    parser.add_argument('--policy', choices=['RunVisualConnectorServoRecoveryV5'], required=True)
    parser.add_argument('--recovery',choices=['local','hierarchical'],required=True)
    args = parser.parse_args()
    repo = Path.cwd()
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
                      domain='100', actor=actor, recovery=args.recovery)
    aux = selection['auxiliary']
    env.update(AIC_TRANSPORT_EVAL_REPO=str(repo), AIC_TRANSPORT_EVAL_LAUNCH_SNAPSHOT='1',
               AIC_TRANSPORT_EVAL_GPU='5', AIC_TRANSPORT_EVAL_POLICY_GPU='5',
               AIC_RULE_TRANSPORT_TEMPLATE=str(prior/'frozen/transport_spec.json'),
               AIC_LOCKED_GRASP_CHECKPOINT=aux['sc_grasp'], AIC_SFP_GRASP_CHECKPOINT=aux['sfp_grasp'],
               AIC_SFP_GRASP_APPROVAL=aux['sfp_grasp_approval'], AIC_INSERTION_EXECUTE_STEPS='4')
    launcher_text = subprocess.check_output(['bash', str(repo/'scripts/run_chunked_insertion_eval.sh'),
                                             '--print-generated-launcher'], cwd=repo, text=True)
    assert launcher_text.count('deadline=$((SECONDS+450))') == 1
    launcher_text = launcher_text.replace('deadline=$((SECONDS+450))', 'deadline=$((SECONDS+1800))')
    launcher = frozen/'sample_launcher.sh'
    launcher.write_text(launcher_text)
    source_paths = [repo/'aic_example_policies/aic_example_policies/ros'/name for name in (
        args.policy+'.py', 'hierarchical_recovery.py', 'RunVisualConnectorServoDiagnosticV4.py', 'RunVisualConnectorServoDiagnosticV3.py', 'RunVisualConnectorServoDiagnosticV2.py', 'RunContinuousConnectorInsertionV2.py', 'RunContinuousConnectorInsertion.py', 'RunFullConnectorInsertion.py', 'RunHistoryChunkedInsertion.py', 'RunSample25mmThenInsertion.py',
        'RunSampleThenSharedInsertion.py', 'RunRuleThenSharedRecovery.py', 'RunLockedPortLocal.py',
        'RunSharedPortTCP.py')]
    source_paths += [repo/'scripts'/name for name in ('full_connector_geometry.py', 'chunked_port_actor.py',
                     'native_connector_pose_observation.py', 'insertion_reference_governor_v2.py', 'insertion_reference_governor.py', 'correlated_lateral_retry.py', 'causal_force_reference.py', 'run_recovery_sample_configuration.py')]
    model_paths = [actor, model_root/'sc_orientation_jitter_v2/best.pt', model_root/'sfp_direct_pose/best.pt']
    model_paths += [Path(v) for v in spec['bundle'].values()]
    model_paths += [Path(v) for v in aux.values()]
    hashes = {str(p): digest(p) for p in [*source_paths, *model_paths, sample, launcher]}
    for path in source_paths:
        shutil.copyfile(path, frozen/path.name)
    plan = dict(schema='full_connector_sample_configuration/v1', sample_sha256=digest(sample),
                expected_trials=list(config['trials']), maximum_score=100*len(config['trials']),
                policy=args.policy, recovery=args.recovery, recovery_scope='SC only; SFP uses the frozen2s-seat image controller with no recovery', training=False, learned_actor_inference=False, insertion_controller='Explicit frozen RGB/public-state alignment and advance control; NOT BC or RL',
                prediction_steps=4, execution_steps=4, step_ms=50,
                starts='Original YAML robot home and all three trials in original order',
                physical_gpu=5, reserved_final_scenes_opened=False, inputs_sha256=hashes,
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

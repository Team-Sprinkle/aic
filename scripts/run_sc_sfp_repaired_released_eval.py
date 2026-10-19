#!/usr/bin/env python3
"""Run the repaired policy on the publicly released qualification YAML.

This released configuration is exposed. Current user authorization allows
diagnosing its local videos and traces. Never call it sealed generalization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

import yaml


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--domain', default='143')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--sc-route', choices=['sc_skip_back','sc_level_shift','sc_hover','sc_marker_scout','sc_high_shift'], default='sc_skip_back')
    parser.add_argument('--sfp-lift-m', type=float, default=0.)
    args = parser.parse_args()
    repo = Path.cwd()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)

    prior = Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
    selected = json.loads((prior / 'selected_public_sample_candidate.json').read_text())
    assert selected['cycle'] == 18 and selected['policy'] == 'RunVisualConnectorServoQuietContextV56'
    baseline = Path(selected['artifact_root'])
    baseline_plan = json.loads((baseline / 'plan.json').read_text())
    baseline_hashes = baseline_plan['inputs_sha256']
    policy_dir = repo / 'aic_example_policies/aic_example_policies/ros'
    route_source = policy_dir / 'RunSampleConservativeTransport.py'
    recovery_source = policy_dir / 'RunSampleTransportRecovery.py'
    close_source = policy_dir / 'RunVisualConnectorServoContextV28.py'
    sfp_source = policy_dir / 'RunObservedEarlyVisualAcquireV46.py'
    shared_source = policy_dir / 'RunSampleThenSharedInsertion.py'
    helper = repo / 'scripts/transport_local_recovery.py'
    marker_helper = repo / 'scripts/observed_board_marker.py'
    changed = sorted(path for path, old in baseline_hashes.items()
                     if not Path(path).is_file() or digest(path) != old)
    expected = sorted([str(route_source), str(recovery_source),str(close_source),str(sfp_source),str(shared_source)])
    assert changed == expected, f'Unexpected change from frozen V56: {changed}'

    frozen = root / 'frozen'
    shutil.copytree(baseline / 'frozen', frozen)
    for file in (route_source, recovery_source, close_source, sfp_source, shared_source, helper, marker_helper):
        shutil.copy2(file, frozen / file.name)

    launcher = frozen / 'sample_launcher.sh'
    content = launcher.read_text()
    needle = ' -e AIC_RULE_TRANSPORT_TEMPLATE="${AIC_RULE_TRANSPORT_TEMPLATE:-}" \\\n'
    assert content.count(needle) == 1
    content = content.replace(needle, needle +
                              ' -e AIC_SC_TRANSPORT_ROUTE="${AIC_SC_TRANSPORT_ROUTE:-sc_old}" \\\n'
                              ' -e AIC_SC_EARLY_TILT="${AIC_SC_EARLY_TILT:-0}" \\\n'
                              ' -e AIC_SFP_FALLBACK_LIFT_M="${AIC_SFP_FALLBACK_LIFT_M:-0}" \\\n')
    needle = "keys=(*keys,'AIC_INSERTION_EXECUTE_STEPS')"
    assert content.count(needle) == 1
    content = content.replace(needle, needle +
                              "\nkeys=(*keys,'AIC_SC_TRANSPORT_ROUTE','AIC_SC_EARLY_TILT','AIC_SFP_FALLBACK_LIFT_M')")
    launcher.write_text(content)

    spec_path = frozen / 'candidate_transport_spec.json'
    spec = json.loads(spec_path.read_text())
    original_spec = json.loads((baseline / 'frozen/candidate_transport_spec.json').read_text())
    assert spec == original_spec
    assert spec['families']['sc']['bypass']['outside_board_x_m'] == -0.27
    spec['families']['sc']['bypass']['outside_board_x_m'] = 0.12
    assert spec['families']['sfp'] == original_spec['families']['sfp']
    assert spec['bundle'] == original_spec['bundle']
    assert spec['control'] == original_spec['control']
    spec_path.write_text(json.dumps(spec, indent=2) + '\n')

    url = 'https://raw.githubusercontent.com/intrinsic-dev/aic/main/aic_engine/config/eval_config.yaml'
    official = frozen / 'official_eval_config.yaml'
    with urllib.request.urlopen(url, timeout=60) as response:
        official.write_bytes(response.read())
    scene = root / 'scene'
    shutil.copytree(Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/configuration_scenes/exact_released_eval_full_r0'), scene)
    assert digest(scene / 'eval_config.yaml') == digest(official)

    source = json.loads((baseline / 'command.json').read_text())
    environment = os.environ.copy()
    environment.update(PATH=str(Path.home() / 'bin') + ':' + environment['PATH'],
                       DOCKER_HOST='unix:///run/user/1008/docker.sock',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    old_frozen = str(baseline / 'frozen')
    environment.update({key: value.replace(old_frozen, str(frozen))
                        for key, value in source['environment'].items()})
    environment.update(AIC_TRANSPORT_EVAL_GPU=args.gpu,
                       AIC_TRANSPORT_EVAL_POLICY_GPU=args.gpu,
                       AIC_TRANSPORT_EVAL_ROS_DOMAIN=args.domain,
                       AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene),
                       AIC_RULE_TRANSPORT_TEMPLATE=str(spec_path),
                       AIC_SC_TRANSPORT_ROUTE=args.sc_route,
                       AIC_SC_EARLY_TILT='1',
                       AIC_SFP_FALLBACK_LIFT_M=str(args.sfp_lift_m))
    assert environment['AIC_TRANSPORT_EVAL_POLICY'].endswith('.' + selected['policy'])
    assert environment['AIC_SHARED_RECOVERY_MODE'] == 'local'

    frozen_hashes = {str(path): digest(path) for directory in (frozen, scene)
                     for path in directory.rglob('*') if path.is_file()}
    source_hashes = {str(path): digest(path) for path in (route_source, recovery_source, close_source, sfp_source, shared_source, helper, marker_helper)}
    save(root / 'plan.json', dict(
        schema='repaired_sc_sfp_released_diagnostic/v1',
        scope='Exposed released-YAML local diagnostic; video and public-observation trace analysis allowed by user',
        repeats=args.repeats, model=selected['policy'],
        official_eval_config_sha256=digest(official),
        prior_sample_candidate=str(baseline),
        prior_changed_sources=changed,
        source_hashes=source_hashes,
        frozen_hashes=frozen_hashes,
        only_spec_delta=dict(path='families.sc.bypass.outside_board_x_m',
                             previous=-0.27, candidate=0.12),
        route=args.sc_route, early_tilt=True, sfp_fallback_lift_m=args.sfp_lift_m,
        sfp_spec_unchanged=True, model_weights_unchanged=True,
        reserved_final_scenes_opened=False, released_scene_exposed=True))
    save(root / 'command_environment.json', {key: value for key, value in environment.items()
                                             if key.startswith('AIC_')})
    checkpoint = repo / 'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
    results = []
    for repeat in range(1, args.repeats + 1):
        assert all(digest(path) == checksum for path, checksum in
                   {**frozen_hashes, **source_hashes}.items())
        run = root / f'repeat_{repeat}'
        run.mkdir()
        command = ['bash', str(launcher), f'sc_early_tilt_official_repeat_{repeat}',
                   str(checkpoint), str(run / 'rollout')]
        save(run / 'command.json', command)
        save(root / 'state.json', dict(stage='running', repeat=repeat,
                                      completed=len(results), released_scene_exposed=True))
        with (run / 'launcher.log').open('w') as log:
            proc = subprocess.run(command, cwd=repo, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            save(root / 'state.json', dict(stage='execution_failure', repeat=repeat,
                                          returncode=proc.returncode, completed=len(results)))
            raise RuntimeError(f'Repeat {repeat} failed; raw log preserved')
        actual = {item.split('=', 1)[0]: item.split('=', 1)[1]
                  for item in json.loads((run / 'rollout/container_environment.json').read_text())
                  if '=' in item}
        assert actual.get('AIC_SC_TRANSPORT_ROUTE') == args.sc_route
        assert actual.get('AIC_SC_EARLY_TILT') == '1'
        assert float(actual.get('AIC_SFP_FALLBACK_LIFT_M','nan')) == args.sfp_lift_m
        assert actual.get('AIC_RULE_TRANSPORT_TEMPLATE') == str(spec_path)
        assert digest(run / 'rollout/eval_config.yaml') == digest(official)
        score = yaml.safe_load((run / 'rollout/results/scoring.yaml').read_text())
        assert set(score) == {'trial_1', 'trial_2', 'trial_3', 'total'}
        item = dict(repeat=repeat, score=score, container_route=args.sc_route,
                    container_early_tilt=True, official_yaml_unchanged=True)
        results.append(item)
        save(run / 'score_summary.json', item)
        save(root / 'progress.json', dict(completed=len(results), planned=args.repeats,
                                         results=results, released_scene_exposed=True))
        print(json.dumps(item), flush=True)
    assert all(digest(path) == checksum for path, checksum in
               {**frozen_hashes, **source_hashes}.items())
    save(root / 'summary.json', dict(completed=args.repeats, planned=args.repeats, results=results,
                                    official_yaml_unchanged=True, released_scene_exposed=True,
                                    reserved_final_scenes_opened=False))
    save(root / 'state.json', dict(stage='complete', completed=args.repeats,
                                  released_scene_exposed=True))


if __name__ == '__main__':
    main()

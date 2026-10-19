#!/usr/bin/env python3
"""Full SC insertion on fresh sample-derived cases with a frozen V56 policy.

Use only sample-derived development scenes. Preserve every run and score; do
not point this diagnostic at the post-release qualification configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--family',choices=['sc','sfp'],default='sc')
    parser.add_argument('--sc-route',choices=['sc_old','sc_skip_back','sc_level_shift','sc_hover','sc_marker_scout','sc_ascend','sc_high_shift'],default='sc_skip_back')
    parser.add_argument('--sfp-lift-m',type=float,default=0.)
    args = parser.parse_args()
    repo = Path.cwd()
    plan = json.loads((args.root / 'plan.json').read_text())
    assert plan['sample_sha256'] == '21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d'
    cases = {row['case']: row for row in plan['cases'] if row['family'] == args.family}
    selected = [cases[name] for name in args.case]
    assert len(set(args.case)) == len(args.case)
    assert all(row['domain'] == 'sample_heldout' and not row['final_scene'] for row in selected)
    spec = json.loads(args.spec.read_text())
    assert spec['sample_sha256'] == plan['sample_sha256']
    assert spec['families']['sc']['bypass']['outside_board_x_m'] == .12
    baseline = Path('/home/nas/chmin/aic_archives/2026-10-19_sc_early_tilt_official_three_v1')
    frozen = baseline / 'frozen'
    assert (frozen / 'sample_launcher.sh').is_file()
    source = json.loads((Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1') /
                         'selected_public_sample_candidate.json').read_text())
    source = json.loads((Path(source['artifact_root']) / 'command.json').read_text())
    old_frozen = str(Path(source['environment']['AIC_RULE_TRANSPORT_TEMPLATE']).parent)
    environment = os.environ.copy()
    environment.update(PATH=str(Path.home() / 'bin') + ':' + environment['PATH'],
                       DOCKER_HOST='unix:///run/user/1008/docker.sock',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.update({key: value.replace(old_frozen, str(frozen))
                        for key, value in source['environment'].items()})
    environment.update(AIC_TRANSPORT_EVAL_GPU=args.gpu,
                       AIC_TRANSPORT_EVAL_POLICY_GPU=args.gpu,
                       AIC_TRANSPORT_EVAL_ROS_DOMAIN='145',
                       AIC_RULE_TRANSPORT_TEMPLATE=str(args.spec),
                       AIC_SC_TRANSPORT_ROUTE=args.sc_route,
                       AIC_SC_EARLY_TILT='1',
                       AIC_SFP_FALLBACK_LIFT_M=str(args.sfp_lift_m))
    assert environment['AIC_TRANSPORT_EVAL_POLICY'].endswith('.RunVisualConnectorServoQuietContextV56')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    launcher=root/'sample_launcher.sh'
    launcher_text=(frozen/'sample_launcher.sh').read_text()
    needle=' -e AIC_SC_EARLY_TILT="${AIC_SC_EARLY_TILT:-0}" \\\n'
    assert launcher_text.count(needle)==1
    launcher_text=launcher_text.replace(needle,needle+
        ' -e AIC_SFP_FALLBACK_LIFT_M="${AIC_SFP_FALLBACK_LIFT_M:-0}" \\\n')
    launcher.write_text(launcher_text)
    source_paths = [repo / 'aic_example_policies/aic_example_policies/ros' / file
                    for file in ('RunSampleConservativeTransport.py',
                                 'RunSampleTransportRecovery.py',
                                 'RunVisualConnectorServoContextV28.py',
                                 'RunObservedEarlyVisualAcquireV46.py',
                                 'RunSampleThenSharedInsertion.py')]
    source_paths += [repo / 'scripts/transport_local_recovery.py',
                     repo / 'scripts/observed_board_marker.py', args.spec,
                     launcher]
    hashes = {str(path): digest(path) for path in source_paths}
    save(root / 'plan.json', dict(schema='sc_sample_full_speed_diagnostic/v1',
         cases=args.case, scene_seed=plan['scene_seed'], source_hashes=hashes,
         policy='RunVisualConnectorServoQuietContextV56',
         route=args.sc_route, early_tilt=True,sfp_fallback_lift_m=args.sfp_lift_m,
         control=spec['control'], released_scenes_used=False,
         reserved_final_scenes_opened=False))
    checkpoint = repo / 'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
    results = []
    for row in selected:
        assert all(digest(path) == value for path, value in hashes.items())
        run = root / row['case']
        run.mkdir()
        environment['AIC_TRANSPORT_EVAL_SCENE_DIR'] = row['scene']
        command = ['bash', str(launcher), row['case'],
                   str(checkpoint), str(run / 'rollout')]
        save(run / 'command.json', dict(command=command,
             environment={key: value for key, value in environment.items() if key.startswith('AIC_')}))
        save(root / 'state.json', dict(stage='running', case=row['case'], completed=len(results)))
        with (run / 'launcher.log').open('w') as log:
            completed = subprocess.run(command, cwd=repo, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            save(root / 'state.json', dict(stage='execution_failure', case=row['case'],
                                          returncode=completed.returncode))
            raise RuntimeError(f'Run failed: {run}')
        audit = run / 'rollout'
        assert digest(audit / 'eval_config.yaml') == digest(Path(row['scene']) / 'eval_config.yaml')
        actual = {entry.split('=', 1)[0]: entry.split('=', 1)[1]
                  for entry in json.loads((audit / 'container_environment.json').read_text()) if '=' in entry}
        assert actual['AIC_SC_TRANSPORT_ROUTE'] == args.sc_route
        assert actual['AIC_SC_EARLY_TILT'] == '1'
        assert actual['AIC_SFP_FALLBACK_LIFT_M'] == str(args.sfp_lift_m)
        assert actual['AIC_RULE_TRANSPORT_TEMPLATE'] == str(args.spec)
        score = yaml.safe_load((audit / 'results/scoring.yaml').read_text())
        trial='trial_3' if args.family=='sc' else 'trial_1'
        assert set(score) == {trial, 'total'}
        item = dict(case=row['case'], card_count=row['card_count'], port=row['port'],
                    score=score['total'], inserted=score[trial]['tier_3']['score'] == 75,
                    scoring=score[trial], run=str(run))
        save(run / 'summary.json', item)
        results.append(item)
        save(root / 'progress.json', dict(results=results, completed=len(results),
                                         planned=len(selected)))
        print(json.dumps({key: item[key] for key in ('case', 'score', 'inserted')}), flush=True)
    assert all(digest(path) == value for path, value in hashes.items())
    save(root / 'summary.json', dict(results=results, completed=len(results),
                                    inserted=sum(item['inserted'] for item in results),
                                    released_scenes_used=False,
                                    reserved_final_scenes_opened=False))
    save(root / 'state.json', dict(stage='complete', completed=len(results)))


if __name__ == '__main__':
    main()

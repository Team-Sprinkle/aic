#!/usr/bin/env python3
"""Matched released-YAML diagnostic after the SC visual-timeout retreat fix.

Exactly one selected source file changes. This is an exposed-scene comparison,
not a new qualification or a tuning loop. Preserve the original result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import urllib.request

import yaml


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='4')
    p.add_argument('--domain',default='143')
    a=p.parse_args()
    repo=Path.cwd();root=a.output;root.mkdir(parents=True,exist_ok=False)
    prior=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
    selected=json.loads((prior/'selected_public_sample_candidate.json').read_text())
    assert selected['cycle']==18 and selected['policy']=='RunVisualConnectorServoQuietContextV56'
    baseline=Path(selected['artifact_root'])
    original=json.loads((baseline/'plan.json').read_text())
    baseline_hashes=original['inputs_sha256']
    repaired_source=repo/'aic_example_policies/aic_example_policies/ros/RunSampleTransportRecovery.py'
    assert str(repaired_source) in baseline_hashes
    changed=[path for path,checksum in baseline_hashes.items() if digest(path)!=checksum]
    assert changed==[str(repaired_source)],f'Expected only the intended recovery repair: {changed}'
    source_text=repaired_source.read_text()
    assert "self.rule_failure=='sample_stage_no_measured_progress'" in source_text
    assert "self.rule_failure in ('sample_stage_no_measured_progress','sample_visual_acquisition_budget')" not in source_text
    hashes=dict(baseline_hashes)
    hashes[str(repaired_source)]=digest(repaired_source)
    save(root/'source_delta.json',dict(changed_files=changed,old_sha256=baseline_hashes[str(repaired_source)],
        new_sha256=hashes[str(repaired_source)],hypothesis='Visual acquisition timeout must not trigger full SC route reversal',
        expected_trace='No transport_route_backtrack_requested after sample_visual_acquisition_budget',
        other_159_inputs_unchanged=True))
    save(root/'selected_candidate.json',selected)
    frozen=root/'frozen';shutil.copytree(baseline/'frozen',frozen)
    shutil.copyfile(repaired_source,frozen/'RunSampleTransportRecovery.py')
    # Verify against authoritative upstream without using its scene values to
    # construct controls or adjust priors. Scene template already exists.
    url='https://raw.githubusercontent.com/intrinsic-dev/aic/main/aic_engine/config/eval_config.yaml'
    official=frozen/'official_eval_config.yaml'
    with urllib.request.urlopen(url,timeout=60) as response:
        official.write_bytes(response.read())
    scene=root/'scene'
    shutil.copytree(Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/configuration_scenes/exact_released_eval_full_r0'),scene)
    assert digest(scene/'eval_config.yaml')==digest(official),'Existing scene template differs from authoritative released evaluation'
    config=yaml.safe_load(official.read_text())
    assert len(config['trials'])==3
    robot=(scene/'ur_llb_start.xacro').read_text()
    for joint,value in config['robot']['home_joint_positions'].items():
        assert f'{joint}={value:.7f}' in robot
    # Preserve exact selected AIC settings, except evaluation scene/output and
    # frozen-file locations. No policy, gain, threshold or recovery changes.
    source=json.loads((baseline/'command.json').read_text())
    env=os.environ.copy()
    env.update(PATH=str(Path.home()/'bin')+':'+env['PATH'],DOCKER_HOST='unix:///run/user/1008/docker.sock',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    remap=lambda value:value.replace(str(baseline/'frozen'),str(frozen))
    env.update({key:remap(value) for key,value in source['environment'].items()})
    env.update(AIC_TRANSPORT_EVAL_GPU=a.gpu,AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
               AIC_TRANSPORT_EVAL_ROS_DOMAIN=a.domain,AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene))
    assert env['AIC_TRANSPORT_EVAL_POLICY'].endswith('.'+selected['policy'])
    assert env['AIC_SHARED_RECOVERY_MODE']=='local'
    close=json.loads(Path(env['AIC_SC_CLOSE_GRASP_APPROVAL']).read_text())
    assert close['pose_gate_passed'] is False
    save(root/'evaluation_only_scope.json',dict(
        authorization='User explicitly requested corrected recovery and released official YAML rerun',
        official_qualification_scenes_exposed=True,reserved_internal_final_scenes_opened=False,
        tuning_allowed=False,training=False,SC_model_globally_qualified=False,
        original_approval_unmodified=True,diagnostic_extension='One source-change matched released-YAML comparison only',
        runtime_true_geometry=False))
    local_hashes={str(path):digest(path) for directory in [frozen,scene] for path in directory.rglob('*') if path.is_file()}
    plan=dict(original,schema='corrected_transport_released_qualification_comparison/v1',
              evaluation_config_source=url,evaluation_config_sha256=digest(official),
              evaluation_protocol='Released qualification YAML; previously exposed, matched source-change diagnostic only',
              sample_sha256=None,inputs_sha256=hashes,local_frozen_hashes=local_hashes,
              original_sample_baseline=str(baseline),model_globally_qualified=False,
              experimental_trial_only=True,explicit_user_authorized_scope_extension=True,
              reserved_final_scenes_opened=False,stopping_rule='One complete corrected-source official YAML; no result-driven changes or repeats')
    save(root/'plan.json',plan)
    command=['bash',str(frozen/'sample_launcher.sh'),'corrected_V56_released_official_comparison',
             str(repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'),str(root/'rollout')]
    save(root/'command.json',dict(command=command,environment={k:v for k,v in env.items() if k.startswith('AIC_')}))
    save(root/'state.json',dict(stage='running_frozen_official_evaluation',trials=3,tuning_allowed=False))
    with (root/'launcher.log').open('w') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        save(root/'state.json',dict(stage='execution_failure_requires_audit',returncode=result.returncode))
        raise RuntimeError('Preserve failed evaluation; no automatic rerun')
    assert all(digest(path)==checksum for path,checksum in {**hashes,**local_hashes}.items())
    assert digest(root/'rollout/eval_config.yaml')==digest(official)
    score=yaml.safe_load((root/'rollout/results/scoring.yaml').read_text())
    assert set(score)-{'total'}==set(config['trials'])
    save(root/'summary.json',dict(score=score,maximum_score=300,official_yaml_unchanged=True,
        complete_configuration=True,policy=selected['policy'],training=False,RL=False,
        reserved_final_scenes_opened=False,physical_insertion_audit='pending causal analysis'))
    save(root/'state.json',dict(stage='score_complete_physical_review_pending',total=score['total']))
    prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
    for script,ros in [('analyze_full_connector_sample_configuration.py',True),('render_insertion_cycle_review.py',False),('analyze_public_cycle_gaps.py',False)]:
        cmd=[str(repo/'.pixi/envs/default/bin/python'),str(repo/'scripts'/script),str(root)]
        if ros:cmd=['bash','-c',prefix+shlex.join(cmd)]
        with (root/(script+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(root/'state.json',dict(stage='complete_requires_manual_recorded_image_review',total=score['total'],
        strict_insertions=json.loads((root/'analysis_summary.json').read_text())['strict_physical_insertions'],
        evaluation_only=True,reserved_internal_final_scenes_opened=False))
    print(json.dumps(score),flush=True)


if __name__=='__main__':
    main()

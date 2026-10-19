#!/usr/bin/env python3
"""Replay the repaired, frozen controller on the exact public sample YAML."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--domain', default='181')
    args = parser.parse_args()
    repo = Path.cwd()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)

    selected = Path('/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker')
    sample = Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/cycle_18/scene')
    public = repo / 'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml'
    assert digest(sample / 'eval_config.yaml') == digest(public)
    scene = root / 'scene'
    shutil.copytree(sample, scene)
    assert digest(scene / 'eval_config.yaml') == digest(public)
    frozen = selected / 'frozen'
    original_plan = json.loads((selected / 'plan.json').read_text())
    assert all(digest(path) == checksum for path, checksum in original_plan['frozen_hashes'].items())

    environment = os.environ.copy()
    environment.update(json.loads((selected / 'command_environment.json').read_text()))
    environment.update(PATH=str(Path.home() / 'bin') + ':' + environment['PATH'],
                       DOCKER_HOST='unix:///run/user/1008/docker.sock',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       AIC_TRANSPORT_EVAL_GPU=args.gpu,
                       AIC_TRANSPORT_EVAL_POLICY_GPU=args.gpu,
                       AIC_TRANSPORT_EVAL_ROS_DOMAIN=args.domain,
                       AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene))
    assert environment['AIC_SC_TRANSPORT_ROUTE'] == 'sc_marker_scout'
    assert environment['AIC_SFP_FALLBACK_LIFT_M'] == '0.06'
    assert environment['AIC_TRANSPORT_EVAL_POLICY'].endswith('.RunVisualConnectorServoQuietContextV56')
    command = ['bash', str(frozen / 'sample_launcher.sh'), 'repaired_public_sample_video',
               str(repo / 'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'),
               str(root / 'rollout')]
    write(root / 'plan.json', dict(schema='repaired_public_sample_video/v1',
          sample_config_sha256=digest(public), sample_yaml_unchanged=True,
          selected_frozen=str(frozen), selected_plan_sha256=digest(selected / 'plan.json'),
          command=command, gpu=args.gpu, ros_domain=args.domain,
          model_weights_unchanged=True, runtime_oracle_geometry=False))
    write(root / 'environment.json', {k: v for k, v in environment.items() if k.startswith('AIC_')})
    with (root / 'launcher.log').open('w') as log:
        process = subprocess.run(command, cwd=repo, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT)
    if process.returncode:
        write(root / 'state.json', dict(stage='launcher_failed', returncode=process.returncode))
        raise RuntimeError(f'Launcher failed ({process.returncode}); inspect launcher.log')
    rollout = root / 'rollout'
    assert digest(rollout / 'eval_config.yaml') == digest(public)
    score = yaml.safe_load((rollout / 'results/scoring.yaml').read_text())
    assert set(score) == {'trial_1', 'trial_2', 'trial_3', 'total'}
    write(root / 'score_summary.json', score)

    subprocess.run([str(repo / '.pixi/envs/default/bin/python'),
                    str(repo / 'scripts/render_transport_five_camera_video.py'),
                    str(rollout), '--fps', '20', '--grid'], check=True, cwd=repo)
    video = rollout / 'five_cameras_20fps_grid.mp4'
    events = [json.loads(line) for line in (rollout / 'runtime_trace.jsonl').open()]
    terminals = [event for event in events if event.get('event') == 'terminal_before_return'
                 and str(event.get('reason', '')).startswith('shared_pipeline_')]
    assert [event.get('episode_index') for event in terminals] == [1, 2, 3]
    clips = {}
    start = 0.0
    for index, terminal in enumerate(terminals, 1):
        end = float(terminal['sim_time'])
        clip = rollout / f'sample_trial_{index}_five_cameras_20fps.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                        '-ss', str(start), '-i', str(video), '-t', str(end - start),
                        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)], check=True)
        clips[f'trial_{index}'] = str(clip)
        start = end
    write(root / 'summary.json', dict(score=score, complete_video=str(video),
          trial_videos=clips, sample_yaml_sha256=digest(public),
          official_released_config_used=False))
    write(root / 'state.json', dict(stage='complete', trials=3))
    print(json.dumps(dict(score=score['total'], video=str(video), clips=clips)))


if __name__ == '__main__':
    main()

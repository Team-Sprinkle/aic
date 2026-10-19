#!/usr/bin/env python3
"""Run all three pinned public trials with the final observation-only policy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

IMAGE = 'ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
ROOT = Path(__file__).resolve().parents[2]
HASHES = {'released': '9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5',
          'sample': '21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, help='Required for mounted-runtime evaluation')
    p.add_argument('--runtime-env', type=Path, help='Required for mounted-runtime evaluation')
    p.add_argument('--policy-image', help='Test a self-contained policy image in a separate container; no model/code mounts')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True, help='One free host GPU; inspect active jobs first')
    p.add_argument('--domain', type=int, default=202)
    p.add_argument('--config', choices=HASHES, default='released')
    p.add_argument('--timeout', type=float, default=1800)
    args = p.parse_args()
    if not 0 <= args.domain <= 232:
        p.error('ROS domain must be in [0, 232]')
    artifacts = args.artifacts.resolve() if args.artifacts else None
    runtime = args.runtime_env.resolve() if args.runtime_env else None
    output = args.output.resolve()
    if not args.policy_image:
        if runtime is None or not (runtime/'bin/python').is_file():
            p.error('Prepare and specify --runtime-env first, or use --policy-image')
        if artifacts is None or not (artifacts/'port/backbone.pt').is_file():
            p.error('Unpack and specify --artifacts first, or use --policy-image')
    environment = os.environ.copy()
    environment.setdefault('DOCKER_HOST', f'unix:///run/user/{os.getuid()}/docker.sock')
    def docker(*command, **kw):
        return subprocess.run(['docker', *command], env=environment, check=True, **kw)
    info = json.loads(docker('info', '--format', '{{json .SecurityOptions}}', capture_output=True, text=True).stdout)
    if not any('rootless' in x for x in info):
        raise RuntimeError('This launcher requires rootless Docker')
    config = 'eval_config.yaml' if args.config == 'released' else 'sample_config.yaml'
    scene = ROOT/'configs/final_evaluation'
    digest = hashlib.sha256((scene/config).read_bytes()).hexdigest()
    if digest != HASHES[args.config]:
        raise RuntimeError('Pinned evaluation YAML changed')
    locks = []
    for label in (f'gpu-{args.gpu}', f'domain-{args.domain}'):
        lock = open(Path(os.environ.get('AIC_LOCK_DIR', '/tmp'))/f'aic-final-{label}.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    output.mkdir(parents=True, exist_ok=False)
    name = f'aic-final-eval-{os.getpid()}-{int(time.time())}'
    model_name = name+'-model'
    manifest = dict(config=args.config, config_sha256=digest, simulator_image=IMAGE,
                    policy='aic_example_policies.final.FinalInsertionPolicy',
                    oracle_runtime_input=False, action_model='explicit_visual_controller',
                    recovery_mode='local', project_gpu_count=1)
    if args.policy_image:
        image = json.loads(docker('image', 'inspect', args.policy_image, capture_output=True, text=True).stdout)[0]
        manifest.update(policy_image=args.policy_image, policy_image_id=image['Id'],
                        self_contained_policy=True, host_code_or_model_mounts_in_policy=False)
    source_files = list((ROOT/'aic_example_policies/aic_example_policies/final').rglob('*.py'))
    source_files += list((ROOT/'aic_example_policies/aic_example_policies/final/config').glob('*.json'))
    manifest['runtime_source_sha256'] = {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(source_files)}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    created = False
    model_created = False
    network_created = False
    try:
        docker('network', 'create', name, capture_output=True)
        network_created = True
        mounts = ['-v', f'{ROOT}:/release:ro', '-v', f'{output}:/audit']
        if not args.policy_image:
            mounts += ['-v', f'{artifacts}:/opt/aic/artifacts:ro', '-v', f'{runtime}:{runtime}:ro']
        docker('run', '-d', '--name', name, '--gpus', f'device={args.gpu}', '--shm-size=4g',
               '--network', name, '--network-alias', 'eval', '--entrypoint', '/bin/bash',
               '-e', f'ROS_DOMAIN_ID={args.domain}', '-e', f'GZ_PARTITION={name}',
               '-e', 'NVIDIA_DRIVER_CAPABILITIES=all', '-e', 'NVIDIA_CONTAINER_CLI_NO_CGROUPS=true',
               '-e', 'AIC_ARTIFACT_ROOT=/opt/aic/artifacts', '-e', 'AIC_DEVICE=cuda:0',
               '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
               *mounts,
               '-w', '/release', IMAGE, '-lc', 'sleep infinity', capture_output=True)
        created = True
        setup = 'source /ws_aic/install/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp; '
        engine = setup + f'''export CUDA_VISIBLE_DEVICES=0 AIC_RESULTS_DIR=/audit/results;
ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id aic_world --ros-args -p use_sim_time:=true > /audit/world_frame.log 2>&1 &
/entrypoint.sh ground_truth:=false start_aic_engine:=true headless:=true aic_engine_config_file:=/release/configs/final_evaluation/{config} world_file:=/release/configs/final_evaluation/world_audit.sdf ros_gz_bridge_config_file:=/release/configs/final_evaluation/bridge_audit.yaml description_file:=/release/configs/final_evaluation/ur_llb_start.xacro > /audit/engine.log 2>&1
printf '%s\\n' "$?" > /audit/engine.exit'''
        docker('exec', '-d', name, 'bash', '-lc', engine)
        deadline = time.monotonic()+args.timeout
        # Wait for the engine/router, before configuring the public policy node.
        while time.monotonic() < deadline:
            log = output/'engine.log'
            if log.exists() and 'aic_engine' in log.read_text(errors='replace'):
                break
            if (output/'engine.exit').exists():
                raise RuntimeError('Engine exited before policy startup; inspect engine.log')
            time.sleep(1)
        if args.policy_image:
            docker('run', '-d', '--name', model_name, '--gpus', f'device={args.gpu}',
                   '--network', name, '--shm-size=1g',
                   '-e', f'ROS_DOMAIN_ID={args.domain}',
                   '-e', 'NVIDIA_DRIVER_CAPABILITIES=all', '-e', 'NVIDIA_CONTAINER_CLI_NO_CGROUPS=true',
                   '-e', 'AIC_ROUTER_ADDR=eval:7447', '-e', 'AIC_DEVICE=cuda:0',
                   '-e', 'AIC_SHARED_PORT_TCP_TRACE=/audit/runtime_trace.jsonl',
                   '-v', f'{output}:/audit', args.policy_image, capture_output=True)
            model_created = True
        else:
            prefix = shlex.quote(str(runtime))
            policy = setup + f'''export LD_LIBRARY_PATH={prefix}/lib:$LD_LIBRARY_PATH;
export LD_PRELOAD={prefix}/lib/libjpeg.so.8:{prefix}/lib/libpng16.so.16;
export PYTHONPATH={prefix}/lib/python3.12/site-packages/cv2/python-3.12:/release/aic_example_policies:/release/aic_model:{prefix}/lib/python3.12/site-packages:$PYTHONPATH;
export AIC_SHARED_PORT_TCP_TRACE=/audit/runtime_trace.jsonl;
/usr/bin/python3 -m aic_model.aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.final.FinalInsertionPolicy > /audit/policy.log 2>&1
printf '%s\\n' "$?" > /audit/policy.exit'''
            docker('exec', '-d', name, 'bash', '-lc', policy)
        while time.monotonic() < deadline:
            score = output/'results/scoring.yaml'
            log = (output/'engine.log').read_text(errors='replace')
            if score.is_file() and 'All Trials Processed!' in log:
                break
            if (output/'policy.exit').is_file() and (output/'policy.exit').read_text().strip() != '0':
                raise RuntimeError('Policy failed; inspect policy.log')
            if model_created:
                state = json.loads(docker('inspect', '--format', '{{json .State}}', model_name, capture_output=True, text=True).stdout)
                if not state['Running'] and state['ExitCode'] != 0:
                    raise RuntimeError('Packaged policy failed; inspect policy.log')
            time.sleep(2)
        else:
            raise TimeoutError('Evaluation exceeded wall-clock budget; logs retained')
        # Read the toolkit's output; never substitute a controller success proxy.
        result = docker('exec', name, 'python3', '-c',
            'import json,yaml; print(json.dumps(yaml.safe_load(open("/audit/results/scoring.yaml"))))',
            capture_output=True, text=True)
        score = json.loads(result.stdout)
        if set(score) != {'total', 'trial_1', 'trial_2', 'trial_3'}:
            raise RuntimeError('Incomplete toolkit scoring output')
        summary = dict(total=score['total'], trial_totals={k: sum(v['score'] for v in score[k].values()) for k in ('trial_1','trial_2','trial_3')},
                       insertions=sum(score[k]['tier_3']['score'] >= 75 for k in ('trial_1','trial_2','trial_3')),
                       evaluation_config=args.config, config_sha256=digest, raw_score='results/scoring.yaml')
        if args.policy_image:
            summary.update(policy_image=args.policy_image, policy_image_id=manifest['policy_image_id'],
                           self_contained_policy=True, host_code_or_model_mounts_in_policy=False)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if model_created:
            logs = subprocess.run(['docker','logs',model_name],env=environment,capture_output=True)
            (output/'policy.log').write_bytes(logs.stdout+logs.stderr)
            subprocess.run(['docker','stop',model_name],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(['docker','rm',model_name],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if created:
            subprocess.run(['docker','stop',name],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(['docker','rm',name],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if network_created:
            subprocess.run(['docker','network','rm',name],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()

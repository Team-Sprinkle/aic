#!/usr/bin/env python3
"""Run the whole frozen matrix sequentially, preserving ordinary failures."""
import argparse
from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from prepare_frozen_pipeline_coverage import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--max-new-runs',type=int,default=1000)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());out=a.plan.parent;root=Path.cwd()
    old=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1')
    python=str(root/'.pixi/envs/default/bin/python');base=root/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
    models=plan['models'];done=[];new=0
    for case in plan['cases']:
        for arm in case['arms']:
            run=out/'runs'/f'{case["case"]}__{arm}';metrics=run/'coverage_metrics.json'
            if metrics.exists():
                done.append(json.loads(metrics.read_text()));continue
            if new>=a.max_new_runs:break
            for path,sha in plan['runtime_hashes'].items():
                if digest(path)!=sha:raise ValueError(f'Frozen runtime changed: {path}')
            if not run.exists():
                gpu=None
                while gpu is None:
                    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                    free=[int(row.split(',')[0]) for row in memory.splitlines() if int(row.split(',')[0]) in [5,6] and int(row.split(',')[1])<256]
                    if free:gpu=free[0]
                    else:time.sleep(30)
                env=os.environ.copy()
                for k in list(env):
                    if k.startswith(('AIC_SC_','AIC_SFP_','AIC_LOCAL_','AIC_RULE_','AIC_TEMPORAL_','AIC_ROUTE_','AIC_TRANSPORT_','AIC_TEACHER_','AIC_OFFICIAL_','AIC_LOCKED_','AIC_ORACLE_')) or k=='AIC_DOMAIN86_LOCK_HELD':env.pop(k)
                env.update(DOCKER_HOST='unix:///run/user/1008/docker.sock',AIC_TRANSPORT_EVAL_GPU=str(gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(gpu),
                    AIC_TRANSPORT_EVAL_SCENE_DIR=case['scene'],AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                    AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='115',AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
                    AIC_TRANSPORT_EVAL_ORACLE_CONTACTS='true',AIC_AUDIT_CAPTURE_HZ='20' if case['replicate']==0 else '5')
                if case['family']=='sc':
                    env.update(AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.'+('RunVisualApproachThenLockedTipLocal' if arm=='near_approach' else 'RunGuardedRuleThenLockedTipLocal'),
                        AIC_RULE_TRANSPORT_TEMPLATE=models['template']['path'],AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS='[240,240,240,50,50,50]',
                        AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT=models['local_actor']['path'],AIC_LOCKED_GRASP_CHECKPOINT=models['grasp']['path'])
                else:
                    env.update(AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',AIC_TRANSPORT_EVAL_CONTROL_HZ='10',
                        AIC_TEMPORAL_TRANSPORT_CHECKPOINT='/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt',
                        AIC_TEMPORAL_TRANSPORT_IMAGENET='/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth',
                        AIC_ROUTE_DIFFUSION_CHECKPOINT=models['transport_actor']['path'],AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS='1',
                        AIC_ROUTE_DIFFUSION_PREWARM='1',AIC_ROUTE_DIFFUSION_SAMPLE_STEPS='12',AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK='1')
                run.parent.mkdir(parents=True,exist_ok=True)
                cmd=['bash',str(root/'scripts/run_clean_transport_eval.sh'),case['case'],str(base),str(run)]
                (out/'current_job.json').write_text(json.dumps({'case':case['case'],'arm':arm,'command':cmd,'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':gpu},indent=2)+'\n')
                with run.with_suffix('.launcher.log').open('w') as log:
                    process=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
                    capture=None
                    if case['family']=='sfp':
                        name='aic_clean_transport_eval_'+run.name
                        for _ in range(120):
                            if process.poll() is not None:
                                raise RuntimeError('Launch ended before native scoring recorder')
                            if subprocess.run(['docker','inspect',name],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
                                break
                            time.sleep(1)
                        capture=subprocess.Popen(['docker','exec',name,'bash','-c',
                            'source /ws_aic/install/setup.bash; timeout 480 stdbuf -oL gz topic -e -t /world/aic_world/dynamic_pose/info > /audit/physical_pose_proto.txt'],
                            env=env,stdout=log,stderr=log)
                    rc=process.wait()
                    if capture is not None:
                        capture.wait(timeout=30)
                (run/'batch_command.json').write_text(json.dumps({'command':cmd,'returncode':rc,'environment':{k:v for k,v in env.items() if k.startswith('AIC_')},'capture_throttles_disk_only':True},indent=2)+'\n')
                if rc:raise RuntimeError(f'Launcher invalid {run}; inspect before resuming')
            if not (run/'scored_stream.json').exists():
                command=f'''source .pixi/envs/default/setup.bash
export PYTHONPATH={str(old/'host_reader_deps')!r}:"${{PYTHONPATH:-}}"
{python} scripts/extract_moveit_training_stream.py {run} {run/'scored_stream.json'}'''
                with (run/'extraction.log').open('w') as log:
                    rc=subprocess.run(['bash','-c',command],stdout=log,stderr=log).returncode
                if rc:raise RuntimeError(f'Extraction failed {run}')
            with (run/'coverage_scoring.log').open('w') as log:
                rc=subprocess.run([python,'scripts/score_frozen_pipeline_coverage.py','--plan',str(a.plan),'--case',case['case'],'--arm',arm,'--run',str(run)],stdout=log,stderr=log).returncode
            if rc:raise RuntimeError(f'Invalid score {run}; diagnose; do not count a reset boundary as a policy failure')
            result=json.loads(metrics.read_text());done.append(result);new+=1
            summary={'schema':'aic_frozen_coverage_progress/v1','completed':len(done),'planned':plan['rollouts'],'results':done,'models_frozen':True,'final_scenes_opened':False}
            (out/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({'completed':len(done),'planned':plan['rollouts'],'case':case['case'],'arm':arm,'acquired':result['rule_visual_acquisition'],'approached':result['near_approach_completed'],'inserted':result['SC_full_physical_insertion'],'terminal_axial_mm':result['terminal']['axial_mm']}),flush=True)
        if new>=a.max_new_runs:break
    if len(done)==plan['rollouts']:
        (out/'batch_finished.json').write_text(json.dumps({'completed':len(done),'models_frozen':True,'final_scenes_opened':False},indent=2)+'\n')


if __name__=='__main__':main()

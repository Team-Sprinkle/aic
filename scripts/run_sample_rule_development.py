#!/usr/bin/env python3
"""Bounded sample-only rule development; preserve each version and failed trace."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from acquisition_gate import qualified_acquisition
from score_frozen_pipeline_coverage import score
from shared_recovery_eval_environment import environment,effective_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--gpu',default='5')
    p.add_argument('--case');p.add_argument('--spec',type=Path)
    a=p.parse_args();repo=Path.cwd();plan=json.loads((a.root/'plan.json').read_text())
    if a.output.exists():raise FileExistsError('Preserve earlier development version')
    a.output.mkdir(parents=True)
    files=[Path(__file__),repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',a.spec or Path(plan['rule_spec'])]
    frozen={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (a.output/'source_hashes.json').write_text(json.dumps(frozen,indent=2)+'\n')
    for f in files: (a.output/f.name).write_bytes(f.read_bytes())
    cases=[c for c in plan['cases'] if not a.case or c['case']==a.case]
    if a.limit:cases=cases[:a.limit]
    if not cases:raise ValueError('No requested sample cases')
    results=[]
    for c in cases:
        for f,h in frozen.items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Development source changed during version: '+f)
        env=environment(repo,Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1'),plan['bundle'],c,
            policy='aic_example_policies.ros.RunSampleConservativeTransport',domain='91')
        env.update(AIC_TRANSPORT_EVAL_GPU=a.gpu,AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
            AIC_RULE_TRANSPORT_TEMPLATE=str(a.spec or Path(plan['rule_spec'])),AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='150')
        run=a.output/c['case'];cmd=['bash',str(repo/'scripts/run_shared_recovery_eval.sh'),c['case'],
            str(repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'),str(run)]
        (a.output/'current_job.json').write_text(json.dumps(dict(case=c,run=str(run)),indent=2)+'\n')
        run.with_suffix('.command.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('AIC_')}),indent=2)+'\n')
        with run.with_suffix('.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=effective_environment(run,env)
        cmd=[sys.executable,str(repo/'scripts/extract_moveit_training_stream.py'),str(run),str(run/'scored_stream.json')]
        prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
        with (run/'extraction.log').open('w') as log:subprocess.run(['bash','-c',prefix+shlex.join(cmd)],stdout=log,stderr=subprocess.STDOUT,check=True)
        x=score(run,c,c['arm']);trace=[json.loads(line) for line in (run/'runtime_trace.jsonl').open()]
        timing=[r['inference_ms'] for r in trace if r['event']=='sample_active_perception_timing' and r['active']]
        x['all_tick_latency']=x['inference_p50_p95_p99_ms']
        x['inference_p50_p95_p99_ms']=np.percentile(timing,[50,95,99]).tolist() if timing else None
        x['qualified']=qualified_acquisition(x);x['effective_runtime_settings']=actual
        (run/'transport_metrics.json').write_text(json.dumps(x,indent=2)+'\n');results.append(x)
        report=dict(completed=len(results),planned=len(cases),results=results,final_scenes_opened=False,
            source='Predeadline sample/spec only; existing fixed perception weights, no new training.',insertion_actor_enabled=False)
        (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(case=c['case'],qualified=x['qualified'],acquired=x['rule_visual_acquisition'],failure=x.get('rule_failure'),terminal=x['terminal'])),flush=True)
    report['passed']=all(x['qualified'] for x in results)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

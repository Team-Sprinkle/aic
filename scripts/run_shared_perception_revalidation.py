#!/usr/bin/env python3
"""Bounded live revalidation after fixing omitted Docker environment settings."""
import argparse,json,shlex,subprocess,sys,math
from pathlib import Path
import numpy as np
from acquisition_gate import qualified_acquisition
from score_frozen_pipeline_coverage import score
from shared_recovery_eval_environment import environment,effective_environment


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--sc-checkpoint',type=Path);p.add_argument('--sc-rail-grid20',action='store_true')
 a=p.parse_args();repo=Path.cwd()
 if a.output.exists():raise FileExistsError('Preserve earlier revalidation')
 a.output.mkdir(parents=True);bundle=json.load((a.root/'shared_perception_bundle_approved_v1.json').open())
 sc=json.load((a.root/'SC_fresh60_acquisition_plan_v1.json').open())['cases']
 sfp=json.load((a.root/'SFP_acquisition_gap_plan_v1.json').open())['cases']
 cases=[c for c in sc if c['domain']=='primary' and c['replicate']==0]+[dict(c,family='sfp') for c in sfp if c['split']=='validation']
 if a.sc_rail_grid20:cases=[c for c in sc if c['domain']=='primary' and c['card_count'] in (1,5)]
 if a.sc_checkpoint:bundle['sc_native_refiner']=str(a.sc_checkpoint.resolve())
 assert len(cases)==20
 (a.output/'plan.json').write_text(json.dumps(dict(cases=cases,gate='>=90% per represented family; zero wrong/harm; strict acquisition and activep95<300ms. Rail-grid also>=4/5per count/port.',sc_checkpoint=bundle['sc_native_refiner'],budget='20fixed matched development episodes; no model or scene tuning; stop onwrong/harm/infra',reason='Corrected container settings; optional separate SCrail-refiner comparison',final_scenes_opened=False),indent=2)+'\n')
 results=[]
 for c in cases:
  run=a.output/c['case'];policy='aic_example_policies.ros.'+('RunNative5RequestedRuleTransport' if c['family']=='sc' else 'RunSfpRequestedRuleTransport')
  env=environment(repo,a.root,bundle,c,policy=policy)
  cmd=['bash',repo/'scripts/run_shared_recovery_eval.sh',c['case'],repo/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt',run]
  run.with_suffix('.command.json').write_text(json.dumps(dict(command=list(map(str,cmd)),environment={k:v for k,v in env.items() if k.startswith('AIC_')}),indent=2)+'\n')
  with run.with_suffix('.log').open('w') as f:subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  actual=effective_environment(run,env)
  prefix='source .pixi/envs/default/setup.bash\nexport PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
  cmd=[sys.executable,repo/'scripts/extract_moveit_training_stream.py',run,run/'scored_stream.json']
  with (run/'extraction.log').open('w') as f:subprocess.run(['bash','-c',prefix+' '.join(shlex.quote(str(x)) for x in cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
  x=score(run,c,'corrected_intended_family_bundle');trace=list(map(json.loads,(run/'runtime_trace.jsonl').open()))
  active={t['sim_time'] for t in trace if t['event'] in ('rule_visual_inspection','SFP_rule_visual_inspection')}
  times=[t['inference_ms'] for t in trace if t['event']=='proposal' and t['sim_time'] in active]
  x['all_proposals_p50_p95_p99_ms']=x['inference_p50_p95_p99_ms']
  x['active_perception_p50_p95_p99_ms']=np.percentile(times,[50,95,99]).tolist() if times else None
  # Strict acquisition helper uses this latency field; preserve the all-proposal
  # numbers separately rather than accidentally certifying inactive rule ticks.
  x['inference_p50_p95_p99_ms']=x['active_perception_p50_p95_p99_ms']
  x['qualified']=qualified_acquisition(x);x['effective_runtime_settings']=actual;results.append(x)
  report=dict(completed=len(results),planned=20,results=results,final_scenes_opened=False,insertion_actor_enabled=False)
  (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(case=c['case'],qualified=x['qualified'],wrong=x['wrong_accepted_port'],active_latency=x['active_perception_p50_p95_p99_ms'])),flush=True)
  if x['wrong_accepted_port'] is True or not x['no_scored_harm']:break
 groups={f:dict(episodes=sum(x['family']==f for x in results),qualified=sum(x['family']==f and x['qualified'] for x in results)) for f in sorted({c['family'] for c in cases})}
 cell_pass=not a.sc_rail_grid20 or all(sum(x['qualified'] for x in results if x['card_count']==n and x['port']==p)>=4 for n in (1,5) for p in (0,1))
 report['family_groups']=groups;report['passed']=len(results)==20 and all(g['qualified']>=math.ceil(.9*sum(c['family']==f for c in cases)) for f,g in groups.items()) and cell_pass and all(x['wrong_accepted_port'] is False and x['no_scored_harm'] for x in results)
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

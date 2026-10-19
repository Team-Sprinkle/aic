#!/usr/bin/env python3
"""Freeze fresh public-range development starts for the150mm shared actor.

Scene YAML is simulator input and offline scoring only, never actor input.
Build on the existing scene generator; replace its historical model/arm plan
with the current supervised insertion protocol. Reserved final scenes unopened.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import yaml


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2026101603);a=p.parse_args()
 if a.output.exists():p.error('Preserve existing frozen development plan')
 subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_frozen_pipeline_coverage.py')),'--output',str(a.output),'--seed',str(a.seed)],check=True)
 historical=a.output/'plan.json';historical.rename(a.output/'scene_generator_historical_plan.json')
 source=json.loads((a.output/'scene_generator_historical_plan.json').read_text());cases=source['cases']
 for c in cases:c['arms']=['fresh_BC','visual_warm_BC'];c['purpose']='full normal start: rule transport -> fresh RGB lock -> shared actor from150mm'
 # Generator hashes scene YAML (including randomized cable/start/rail/board).
 # Independently verify no identical physical reset in any admitted BC split.
 corpus=Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/shared_full_causal_grasp_actor_rows_v1')
 known={r['scene'] for r in map(json.loads,(corpus/'rows.jsonl').open())}
 physical=[]
 for c in cases:
  cfg=yaml.safe_load((Path(c['scene'])/'eval_config.yaml').read_text());trial=next(iter(cfg['trials'].values()))
  # Same canonical key as native provenance and BC admission: changing the
  # public requested target does not create a different physical reset.
  identity=dict(scene=trial['scene'],robot=cfg.get('robot'))
  physical.append(hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest())
 overlap=known & set(physical)
 if overlap:raise ValueError('Fresh development scenes overlap admitted corpus')
 pilot=[]
 for f in ('sc','sfp'):
  pilot.append(next(c['case'] for c in cases if c['family']==f and c['domain']=='primary' and c['card_count']==3 and c['port']==0 and c['replicate']==0))
 plan=dict(schema='shared150mm_autonomous_development/v1',seed=a.seed,cases=cases,pilot_cases=pilot,final_scenes_opened=False,training_geometry_input=False,BC_physical_scene_overlap=sorted(overlap),
  scope='Development only. Freeze settings before evaluating either trained arm. Full physical insertion, not merely handoff or teacher-forced error.',
  gates=dict(depth_mm={'sc':15.5,'sfp':44.},lateral_mm=.5,orientation_deg=1.,official_tier3=75,no_scored_harm=True,p95_live_inference_ms=300,final_promotion='>=90% qualified in each family, at least5episodes per primary card-count/port cell, no wrong-port handoff, no scored harm. Report all failures, phases and latency.'),
  stages=[dict(name='integration_pilot',cases=pilot,arms=['fresh_BC','visual_warm_BC'],recovery='none',gate='Actual container settings and public-only inputs verified, causal pre-return raw TF and native RGB retained. Ordinary BC failures are evidence, not infrastructure passes.'),
   dict(name='matched20',cases=[c['case'] for c in cases if c['domain']=='primary' and c['replicate']==0],arms=['fresh_BC','visual_warm_BC'],recovery='none',gate='Both completed matched fits; same20starts. Diagnose any failed phase from trace/video before routine training.'),
   dict(name='recovery_comparison',cases=[c['case'] for c in cases if c['domain']=='primary' and c['replicate']==0],arms=['none','local','hierarchical'],gate='One frozen BC actor; same20starts, identical whole-task budget. Supervisor exploration is not RL; require full-depth gain without wrong/harm/latency regressions.'),
   dict(name='coverage',cases=[c['case'] for c in cases],gate='Selected fixed pipeline,100primary+20alternate-public-board trials; no tuning on reservedfinals.')],
  stopping_rule='Stop a branch on wrong-port acquisition, harmful behavior or infrastructure defect; preserve evidence, target the identified cause. If neither BC arm inserts autonomously, one bounded diagnosis precedes further data/training. A failed gate stays open.',
  resources='Rootless Docker, shared3/4 cooperative locks; training5, legacy0. No active-job interruption, <=4projectGPUs.',
  forbidden='No scene YAML, true port/plug TF, oracle contacts or scoring topics in deployment decisions. No privileged near-port reset in autonomous trials. No new RL until supervised gate.')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(scene_count=len(cases),pilot=pilot,output=str(a.output))))
if __name__=='__main__':main()

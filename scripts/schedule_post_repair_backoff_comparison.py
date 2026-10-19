#!/usr/bin/env python3
"""Wait for current frozen comparisons, then run the approved recovery screen."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    output=root/'post_repair_backoff_comparison_v1'
    output.mkdir(exist_ok=False)
    state=output/'scheduler_state.json'
    fits=['context_masked_current_pose_v1','context_public_current_pose_v1',
          'sc_nominal_joint_calibrated_v3','sc_nominal_orientation_calibrated_v3',
          'sc_spatial_grasp_pixel_v6','sc_spatial_grasp_joint_v6']
    waits=[root/name/'summary.json' for name in fits]+[
        root/'context_live_current_pose_v1/runs/summary.json',
        root/'visual_servo_fresh_SFP_coverage_v2/runs/summary.json']
    while True:
        missing=[str(path) for path in waits if not path.exists()]
        save(state,dict(stage='waiting_for_current_repairs_and_frozen_comparisons',
                        missing=missing,active_jobs_interrupted=False,
                        current_models_unchanged=True,final_scenes_opened=False))
        if not missing:break
        time.sleep(45)
    for name in fits:
        report=json.loads((root/name/'summary.json').read_text())
        if report['stop_reason']!='validation_plateau' or report['final_lr']>1.00001e-6:
            raise RuntimeError('Training incomplete, do not bypass convergence gate: '+name)
    frozen=json.loads((root/'context_live_current_pose_v1/runs/plan.json').read_text())
    # Select by the existing validation objective, never by final/evaluation
    # outcomes or the future recovery screen. Preserve the exact runtime copy.
    selected=min(frozen['variants'],key=lambda v:min(v['initial_validation_objective'],v['trained_validation_objective']))
    checkpoint=Path(selected['checkpoint'])
    save(output/'evaluation_cases.json',dict(cases=frozen['cases']))
    save(output/'evaluation_variants.json',dict(variants=[dict(name='frozen_BC_'+mode,
        checkpoint=str(checkpoint),execute=selected['execute'],recovery=mode)
        for mode in ['none','local','hierarchical']]))
    save(output/'selection.json',dict(source=frozen['variants'],selected=selected,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        reason='Existing held-out validation objective, not rollout success',
        new_SC_estimator_deployed=False,SC_component_limitations_retained=True,
        actor_or_explicit_controller='trained_BC_actor',new_training=False,new_RL=False))
    cmd=[str(Path.cwd()/'.pixi/envs/default/bin/python'),
         'scripts/run_consistent_connector_backoff_probe.py','--root',str(output),
         '--output',str(output/'runs')]
    save(output/'command.json',dict(command=cmd,physical_gpu=5,
        rollouts=len(frozen['cases'])*3,maximum_project_GPUs=4,
        budget='Same original episode deadline and180s insertion simulation budget per arm;12rollouts maximum',
        promotion='Strict full insertion improves with no wrong port or harm; no automatic promotion'))
    save(state,dict(stage='running_frozen_backoff_comparison',new_RL=False,final_scenes_opened=False))
    with (output/'runner.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=json.loads((output/'runs/summary.json').read_text())
    arms={mode:[r for r in results['results'] if r['recovery_mode']==mode]
          for mode in ['none','local','hierarchical']}
    table={mode:dict(episodes=len(rows),strict_full=sum(r['physical_full_insertion'] for r in rows),
        backoffs=sum(r['backoff_observations'] for r in rows),lateral_retries=sum(r['lateral_retries'] for r in rows))
        for mode,rows in arms.items()}
    save(output/'comparison.json',dict(results_by_arm=table,promotion=False,
        interpretation='Review force/contact, wrong-port outcomes and videos before promotion. Backoff alone does not prove alignment accuracy.',
        final_scenes_opened=False,new_RL=False))
    save(state,dict(stage='complete_needs_physical_trace_and_video_review',results=table,automatic_promotion=False))


if __name__=='__main__':main()

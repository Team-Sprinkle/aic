#!/usr/bin/env python3
"""Train the approved matched chunk/continuity variants, then test live.

Two training GPUs plus the sequential Gazebo GPU and untouched legacyGPU0
stay within the four-project-GPU ceiling. No official or reserved final data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(p, d):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
    import fcntl
    lock=(a.output/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=Path('/home/nas/chmin/aic_archives/2026-10-17_recorded_horizon_actor_rows_v1')
    while not (data/'summary.json').exists():time.sleep(20)
    summary=json.loads((data/'summary.json').read_text())
    assert summary['matched_rows']>50000 and not summary['example_disagreements']
    models=[dict(name='h4',horizon=4,consistency=0.,gpu=3),
            dict(name='h8',horizon=8,consistency=0.,gpu=4),
            dict(name='h12',horizon=12,consistency=0.,gpu=3),
            dict(name='h8_continuity',horizon=8,consistency=.1,gpu=4)]
    common=[py,repo/'scripts/train_chunked_port_local_actor.py','--native-image-cache',
        '/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/shared_full_actor_image_cache_v1',
        '--imagenet','/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth',
        '--decoder-kind','transformer','--coordinate-heads-fp32','--control-path-fp32',
        '--local-history','--history-offsets-s','1.2','.6','.2','0','--mask-command-feedback',
        '--tip-input-mode','translation','--sc-action-reference','plug_tip_translation_tcp_orientation',
        '--tip-action-families','sc','sfp','--phase-reference','predicted_tip','--contact-aware-phase',
        '--require-verified-command-contract','--selection-metric','two_family_insertion',
        '--orientation-loss-weight','1','--near-lateral-loss-weight','.5','--chunk-motion-loss-weight','.1',
        '--min-updates','10000','--stale-evaluations','60','--updates','50000','--minutes','360',
        '--full-validation','--batch-size','64','--workers','4']
    save(a.output/'training_plan.json',dict(models=models,common_command=list(map(str,common)),
        data=str(data),matched_anchors=summary['matched_rows'],
        stop='Validation plateau after>=10000updates and60stale100-update checks;50k/6h safety cap is not convergence.',
        comparison='Same data/split/initialization/optimizer opportunity. Converged step counts may differ and will be reported.',
        rollout_pairs=[['h4',4],['h8',2],['h8',4],['h12',4],['h12',6],['h8_continuity',4]],
        reserved_final_scenes_opened=False,RL=False))

    def worker(gpu):
        for m in [m for m in models if m['gpu']==gpu]:
            folder=a.output/('train_'+m['name'])
            if (folder/'summary.json').exists():continue
            if folder.exists():raise RuntimeError('Partial training preserved: '+str(folder))
            command=common+['--dataset',data/f"h{m['horizon']}",'--action-horizon',m['horizon'],
                '--overlap-consistency-weight',m['consistency'],'--reset-action-head','--output',folder]
            save(a.output/(m['name']+'.command.json'),dict(command=list(map(str,command)),GPU=gpu))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='8')
            with (a.output/(m['name']+'.log')).open('w') as f:
                subprocess.run(list(map(str,command)),cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            s=json.loads((folder/'summary.json').read_text())
            save(a.output/(m['name']+'.convergence.json'),dict(stop_reason=s['stop_reason'],updates=s['updates_completed'],
                best_step=s['best_step'],stale_evaluations=s['stale_evaluations_at_end'],
                plateau=s['stop_reason']=='validation_plateau',
                cap_is_not_convergence=s['stop_reason']!='validation_plateau'))
    save(a.output/'state.json',dict(stage='training_matched_horizon_and_continuity_variants',GPU_training=[3,4],
        GPU_live=5,legacy_GPU0_untouched=True,new_RL=False))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu) for gpu in [3,4]]
        for future in futures:future.result()
    # Do not call a hard cap a plateau or automatically promote a checkpoint.
    reviews=[json.loads((a.output/(m['name']+'.convergence.json')).read_text()) for m in models]
    save(a.output/'convergence_review.json',dict(models=reviews,all_plateau=all(r['plateau'] for r in reviews)))
    while not (a.output/'runtime_hooks_ready.json').exists():time.sleep(20)
    variants=[dict(name='old_checkpoint_history_codec',checkpoint='/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/models/visual_warm.pt',execute=4)]
    for name,execute in [('h4',4),('h8',2),('h8',4),('h12',4),('h12',6),('h8_continuity',4)]:
        variants.append(dict(name=name+'_e'+str(execute),checkpoint=str(a.output/('train_'+name)/'best.pt'),execute=execute))
    save(a.output/'evaluation_variants.json',dict(variants=variants,model_parameters_frozen=True,
        caps_not_plateaus=[m['name'] for m,r in zip(models,reviews) if not r['plateau']],
        initial_gate='Two predetermined new development scenes per variant. No broad sweep/training extension without a concrete diagnosed benefit.',
        final_scenes_opened=False))
    save(a.output/'state.json',dict(stage='frozen_near_port_live_comparison',variants=len(variants),new_RL=False))
    with (a.output/'live_evaluation.log').open('w') as f:
        subprocess.run([str(py),str(repo/'scripts/run_history_chunk_live_comparison.py'),
            '--root',str(a.output),'--output',str(a.output/'live_comparison')],cwd=repo,
            stdout=f,stderr=subprocess.STDOUT,check=True)
    save(a.output/'decision.json',dict(status='bounded_training_and_live_comparison_complete',
        no_automatic_promotion=True,new_RL=False,final_scenes_opened=False,
        next='Inspect full insertion, plan discontinuity, pose-lock constancy and history-seed counts before any larger sweep.'))


if __name__=='__main__':main()

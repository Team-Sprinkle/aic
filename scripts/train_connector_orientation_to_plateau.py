#!/usr/bin/env python3
"""Retained orientation trainer specialization with LR-stage convergence and resume."""
from pathlib import Path
import sys

base=Path(__file__).with_name('train_native_connector_orientation.py')
source=base.read_text()
# Queued fits wait for their assigned GPU instead of competing with active work.
source=source.replace('fcntl.LOCK_EX | fcntl.LOCK_NB','fcntl.LOCK_EX')
replacements=[
 ("plan=dict(schema='native_connector_full_pose/v1',family=a.family,", "plan=dict(schema='native_connector_full_pose/v1',family=a.family,balanced_new_angles=a.balance_new_angles,"),
 ("    groups = list(groups.values())", "    groups = list(groups.values())\n    expanded_groups=[g for g in groups if rows[g[0]].get('source_role')]"),
 ("        model.train();", "        if a.balance_new_angles and expanded_groups:\n            selected[:8]=[expanded_groups[g][rng.integers(len(expanded_groups[g]))] for g in rng.integers(len(expanded_groups),size=8)]\n        model.train();"),
 ("    val = np.flatnonzero(~fit)", "    val = np.flatnonzero(~fit)\n    expanded_val=np.array([i for i in val if rows[i].get('source_role')],dtype=int)"),
 ("            metric,_=evaluate(val)", "            metric,_=evaluate(val)\n            expanded_metric=evaluate(expanded_val)[0] if len(expanded_val) else None"),
 ("            history.append(dict(step=step,metrics=metric,score=score,wall_seconds=time.monotonic()-started))", "            if expanded_metric is not None:score=max(score,expanded_metric['orientation_p50_p95_deg'][1],0 if a.rotation_only else expanded_metric['translation_p50_p95_mm'][1]/.4)\n            history.append(dict(step=step,metrics=metric,expanded_validation=expanded_metric,score=score,wall_seconds=time.monotonic()-started))"),
 ("    summary=dict(plan=plan,metrics=metric,train_subsample_metrics=train_metric,", "    summary=dict(plan=plan,metrics=metric,expanded_development_validation=evaluate(expanded_val)[0] if len(expanded_val) else None,train_subsample_metrics=train_metric,"),
 ("p.add_argument('--crop-jitter', type=int, default=0)","p.add_argument('--crop-jitter', type=int, default=0)\n    p.add_argument('--resume', action='store_true')\n    p.add_argument('--balance-new-angles', action='store_true')"),
 ("a.output.mkdir(parents=True, exist_ok=False)","a.output.mkdir(parents=True, exist_ok=a.resume)"),
 ("history=[];best=float('inf');best_step=0;reason='update_cap';started=time.monotonic()\n    for step in range(1,a.updates+1):",
 """history=[];best=float('inf');best_step=0;reason='update_cap';started=time.monotonic()
    lr_events=[];last_progress=0;first_step=1
    if a.resume:
        resumed=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False)
        if resumed['plan']['manifest_sha256']!=plan['manifest_sha256']:raise ValueError('Resume dataset changed')
        model.load_state_dict(resumed['model']);opt.load_state_dict(resumed['optimizer'])
        state=resumed['training_state'];best=state['best'];best_step=state['best_step']
        lr_events=state['lr_events'];last_progress=state['last_progress'];first_step=resumed['step']+1
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        history=json.loads((a.output/'history.json').read_text())
    for step in range(first_step,a.updates+1):"""),
 ("checkpoint=dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=step)",
 "checkpoint=dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=step,optimizer=opt.state_dict())"),
 ("best=score;best_step=step;torch.save(checkpoint,a.output/'best.pt')",
 "best=score;best_step=step;last_progress=step;torch.save(checkpoint,a.output/'best.pt')"),
 ("if step>=a.min_updates and step-best_step>=a.patience:\n                reason='validation_plateau';break",
 """if step>=a.min_updates and step-last_progress>=a.patience:
                rate=opt.param_groups[0]['lr']
                if rate<=1.00001e-6:
                    reason='validation_plateau';break
                next_rate=max(1e-6,rate*.3)
                selected_checkpoint=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(selected_checkpoint['model']);opt.load_state_dict(selected_checkpoint['optimizer'])
                for group in opt.param_groups:group['lr']=next_rate
                lr_events.append(dict(step=step,previous_lr=rate,new_lr=next_rate,restored_best_step=best_step))
                last_progress=step
                save(a.output/'learning_rate_events.json',lr_events)"""),
 ("torch.save(dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=step),a.output/'last.pt')",
 """torch.save(dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=step,
                    optimizer=opt.state_dict(),training_state=dict(best=best,best_step=best_step,
                    lr_events=lr_events,last_progress=last_progress,numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state())),a.output/'last.pt')"""),
 ("        wall_seconds=time.monotonic()-started)","        wall_seconds=time.monotonic()-started,learning_rate_events=lr_events,final_lr=opt.param_groups[0]['lr'],convergence_requires_lr_floor_plateau=True)"),
]
for before,after in replacements:
 if source.count(before)!=1:raise ValueError('Unexpected base trainer change: '+before[:100])
 source=source.replace(before,after)
source=source.replace("stop='Joint translation/orientation validation score plateau, or explicit time/update cap; best and last kept.'", "stop='Validation plateau after LR reductions from1e-4to1e-6, restoring best model/optimizer each reduction; time/update cap is incomplete, resume from last.pt.'")
if '--help' in sys.argv:
 exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=str(base)))
 raise SystemExit(0)
output=Path(sys.argv[sys.argv.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True)
snapshot=output.with_suffix('.converged_trainer_snapshot.py')
if snapshot.exists() and '--resume' not in sys.argv:raise FileExistsError(snapshot)
if not snapshot.exists():snapshot.write_text(source)
else:source=snapshot.read_text()  # Resume the exact immutable trainer, not newer source.
exec(compile(source,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))

#!/usr/bin/env python3
"""Full recorded-target BC with LR-floor validation convergence, matched context arms."""
import ast
from pathlib import Path
import sys

repo=Path(__file__).resolve().parent
source=(repo/'train_predicted_port_local_actor.py').read_text()
module=ast.parse((repo/'train_full_connector_actor.py').read_text())
full=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replacements' for t in n.targets))
for before,after in full:
 if source.count(before)==0:raise ValueError('Base full-pose contract changed')
 source=source.replace(before,after)
patches=[
 ("best=float('inf');best_step=0;stale=0;history=[];started=time.monotonic();reason='update_cap'", "best=float('inf');best_step=0;stale=0;history=[];started=time.monotonic();reason='update_cap';lr_events=[]"),
 ("                torch.save({'model':model.state_dict(),'architecture':'shared_port_tcp_local_predicted_pose',", "                torch.save({'optimizer_state':opt.state_dict(),'model':model.state_dict(),'architecture':'shared_port_tcp_local_predicted_pose',"),
 ("if step>=a.min_updates and stale>=a.stale_evaluations and not a.fixed_update_budget:\n                reason='validation_plateau';break", """if step>=a.min_updates and stale>=a.stale_evaluations and not a.fixed_update_budget:
                rate=opt.param_groups[0]['lr']
                if rate<=1.00001e-6:
                    reason='validation_plateau';break
                rate=max(1e-6,rate*.3)
                selected_checkpoint=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(selected_checkpoint['model']);opt.load_state_dict(selected_checkpoint['optimizer_state'])
                for group in opt.param_groups:group['lr']=rate
                lr_events.append(dict(step=step,new_lr=rate,restored_best_step=best_step))
                (a.output/'learning_rate_events.json').write_text(json.dumps(lr_events,indent=2)+'\\n')
                stale=0"""),
 ("'best_selection_metric':best,'best_update':best_step,", "'best_selection_metric':best,'best_update':best_step,'learning_rate_events':lr_events,'final_lr':opt.param_groups[0]['lr'],"),
 ("'stop_reason':reason,'selected_validation':selected,", "'stop_reason':reason,'selected_validation':selected,'learning_rate_events':lr_events,'final_lr':opt.param_groups[0]['lr'],'convergence_requires_lr_floor_plateau':True,"),
]
for before,after in patches:
 if source.count(before)!=1:raise ValueError('Unexpected trainer change: '+before[:100])
 source=source.replace(before,after)
source=source.replace("'true_port_geometry_runtime_input':False,","'connector_estimator_fingerprints':json.loads((a.dataset/'summary.json').read_text()).get('orientation_checkpoints'), 'connector_position_modes':json.loads((a.dataset/'summary.json').read_text()).get('position_modes',{'sc':'retained','sfp':'retained'}),'retained_translation_estimator_fingerprints':json.loads((a.dataset/'summary.json').read_text()).get('retained_translation_estimator_fingerprints'),'true_port_geometry_runtime_input':False,")
# Preserve update0 as a valid candidate for warm-start comparisons. A fit
# that regresses at every update must not displace its better initialization.
source=source.replace("'tip_orientation_input':False,", "'tip_orientation_input':a.tip_input_mode=='full_pose',")
compiled=ast.parse(source)
saves=[n for n in ast.walk(compiled) if isinstance(n,ast.Call)
       and isinstance(n.func,ast.Attribute) and n.func.attr=='save'
       and len(n.args)>1 and isinstance(n.args[0],ast.Dict)
       and isinstance(n.args[1],ast.BinOp) and isinstance(n.args[1].right,ast.Constant)
       and n.args[1].right.value=='best.pt']
if len(saves)!=1:raise ValueError('Expected unique best-checkpoint schema')
initial=saves[0].args[0]
for i,key in enumerate(initial.keys):
 if isinstance(key,ast.Constant) and key.value=='update':initial.values[i]=ast.Constant(value=0)
initial_save=ast.unparse(initial)
anchor_loop='    for step,batch in enumerate(loader,start=1):'
if source.count(anchor_loop)!=1:raise ValueError('Unexpected training loop')
initial_block="""    if a.selection_metric=='two_family_insertion':
        baseline_metric=max(value for family in families for value in [
            baseline_validation[family]['near_decoded_tcp_lateral_mm']['p95']/.5,
            baseline_validation[family]['near_first_orientation_deg']['p95']/3.,
            baseline_validation[family+'_far_approach']['first_body_command_mm']['p95']/3.,
            baseline_validation[family]['axial_chunk_span_error_mm']['p95']])
        best=baseline_metric;best_step=0
        history=[dict(step=0,train_loss=None,selection_mm=baseline_metric,validation=baseline_validation)]
        initial_checkpoint=INITIAL_SAVE
        torch.save(initial_checkpoint,a.output/'initial.pt')
        torch.save(initial_checkpoint,a.output/'best.pt')
""".replace('INITIAL_SAVE',initial_save)
source=source.replace(anchor_loop,initial_block+anchor_loop)
if '--help' in sys.argv:
 exec(compile(source,str(repo/'train_predicted_port_local_actor.py'),'exec'),dict(__name__='__main__',__file__=str(repo/'train_predicted_port_local_actor.py')))
 raise SystemExit(0)
output=Path(sys.argv[sys.argv.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True)
snapshot=output.with_suffix('.trainer_snapshot.py')
if snapshot.exists():raise FileExistsError(snapshot)
snapshot.write_text(source)
exec(compile(source,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))

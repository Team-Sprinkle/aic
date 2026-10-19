#!/usr/bin/env python3
"""One frozen-data fine-feature ablation from a completed coarse warm model."""
import ast
from pathlib import Path
import sys

root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
base = root/'spacefilled_near_fused_orientation_warm.trainer_snapshot.py'
source = base.read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConnectorPose')
lines = source.splitlines(keepends=True)
source = ''.join(lines[:node.lineno-1])+('from fine_connector_orientation import FineConnectorPose as ConnectorPose\n')+''.join(lines[node.end_lineno:])
before = "model.load_state_dict(warm['model'],strict=True)"
assert source.count(before) == 1
source = source.replace(before, "model.load_coarse_warm_start(warm['model'])")
source = source.replace("p.add_argument('--native-warm-start', type=Path)",
    "p.add_argument('--freeze-coarse', action='store_true')\n    p.add_argument('--native-warm-start', type=Path)")
before = "    params=[x for x in model.parameters() if x.requires_grad]"
assert source.count(before) == 1
source = source.replace(before, """    if a.freeze_coarse:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('fine_project.', 'fine_head.')))
    plan['freeze_coarse']=a.freeze_coarse
    save(a.output/'plan.json',plan)
"""+before)
source = source.replace('        if a.freeze_encoder:model.landmark.eval()',
                        '        if a.freeze_encoder or a.freeze_coarse:model.landmark.eval()')
# A warm model is itself a candidate: do not lose it if the first updates
# worsen validation. Existing running snapshots are not changed.
before = "    lr_events=[];last_progress=0;first_step=1"
assert source.count(before) == 1
source = source.replace(before, before+"""
    if not a.resume:
        initial_metric,_=evaluate(expanded_val)
        best=initial_metric['orientation_p50_p95_deg'][1]
        torch.save(dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,
                        step=0,optimizer=opt.state_dict()),a.output/'best.pt')
        save(a.output/'initial_candidate.json',dict(score=best,step=0,
             retained_as_candidate=True,metrics=initial_metric))
""")
before = "plan['normalization_source']='Retain complete warm RGB head normalization; no output rescaling on load'"
assert source.count(before) == 1
source = source.replace(before, before+"\n        plan['architecture']='fine_spatial_orientation_v1'\n        plan['hypothesis']='Retain layer2 12x12 spatial features instead of using only layer3 pooled6x6; same frozen data/splits/augmentation/selection/budget, zero-initialized fine branch preserves warm predictions.'\n        plan['fine_model_source_sha256']=hashlib.sha256(Path(__file__).with_name('fine_connector_orientation.py').read_bytes()).hexdigest()")
# The snapshot lives on NAS, so hash the actual repository module explicitly.
source = source.replace("Path(__file__).with_name('fine_connector_orientation.py')", repr(str(Path(__file__).with_name('fine_connector_orientation.py').resolve())))
source = source.replace("hashlib.sha256("+repr(str(Path(__file__).with_name('fine_connector_orientation.py').resolve()))+".read_bytes())", "hashlib.sha256(Path("+repr(str(Path(__file__).with_name('fine_connector_orientation.py').resolve()))+").read_bytes())")
output = Path(sys.argv[sys.argv.index('--output')+1])
snapshot = output.with_suffix('.trainer_snapshot.py')
if snapshot.exists() and '--resume' not in sys.argv:
    raise FileExistsError(snapshot)
if not snapshot.exists():
    snapshot.write_text(source)
else:
    source = snapshot.read_text()
exec(compile(source, str(snapshot), 'exec'), dict(__name__='__main__', __file__=str(snapshot)))

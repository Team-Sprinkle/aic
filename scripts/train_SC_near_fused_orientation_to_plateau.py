#!/usr/bin/env python3
"""One existing fused-view RGB orientation model, selected for25mm use only."""
import ast
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('train_connector_orientation_to_plateau.py')
tree=ast.parse(wrapper.read_text())
replacements=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='replacements' for n in x.targets)))
base=Path(__file__).with_name('train_native_connector_orientation.py')
source=base.read_text()
source=source.replace('fcntl.LOCK_EX | fcntl.LOCK_NB','fcntl.LOCK_EX')
for before,after in replacements:
    if source.count(before)!=1:raise ValueError('Unexpected trainer source: '+before[:80])
    source=source.replace(before,after)
source=source.replace("expanded_groups=[g for g in groups if rows[g[0]].get('source_role')]", "expanded_groups=[g for g in groups if rows[g[0]].get('source_role','').startswith('Near25mm')]")
source=source.replace("expanded_val=np.array([i for i in val if rows[i].get('source_role')],dtype=int)", "expanded_val=np.array([i for i in val if rows[i].get('source_role','').startswith('Near25mm')],dtype=int)\n    assert len({rows[i]['episode'] for i in expanded_val})>=4,'Four independent heldoutnear episodes required'")
before="if expanded_metric is not None:score=max(score,expanded_metric['orientation_p50_p95_deg'][1],0 if a.rotation_only else expanded_metric['translation_p50_p95_mm'][1]/.4)"
after="assert a.rotation_only and expanded_metric is not None\n            score=expanded_metric['orientation_p50_p95_deg'][1]"
assert source.count(before)==1;source=source.replace(before,after)
source=source.replace("selection='max(translation_p95/0.4mm,orientation_p95/1degree)'", "selection='Heldoutnear orientationp95 only; no far/home model replacement. Translation remains a separately estimated RGB component.'")
source=source.replace("stop='Joint translation/orientation validation score plateau, or explicit time/update cap; best and last kept.'", "stop='Nearvalidationplateau atLRfloor1e-6after restoring best model/optimizer each reduction; caps are incomplete.'")
# Optional future paired initialization check. The default path is unchanged.
source=source.replace("p.add_argument('--family',", "p.add_argument('--native-warm-start', type=Path)\n    p.add_argument('--family',")
before="    model=ConnectorPose(saved['model'],mean,scale).cuda()"
after="""    model=ConnectorPose(saved['model'],mean,scale).cuda()
    if a.native_warm_start is not None:
        warm=torch.load(a.native_warm_start,map_location='cpu',weights_only=False)
        if not warm['plan'].get('rotation_only') or warm['plan']['family']!='sc':raise ValueError('SC RGB rotation-only warm start required')
        if not np.array_equal(np.asarray(warm['plan']['crop_origins_native']),origins):raise ValueError('Warm-start native crop mapping differs')
        model.load_state_dict(warm['model'],strict=True)
        # Retain the pretrained head's normalization, so copying weights does
        # not silently change the physical prediction before the first update.
        mean=model.pose_mean.detach().cpu().numpy().copy()
        scale=model.pose_scale.detach().cpu().numpy().copy()
        plan['native_warm_start']=str(a.native_warm_start)
        plan['native_warm_start_sha256']=hashlib.sha256(a.native_warm_start.read_bytes()).hexdigest()
        plan['normalization_source']='Retain complete warm RGB head normalization; no output rescaling on load'
        save(a.output/'plan.json',plan)"""
assert source.count(before)==1;source=source.replace(before,after)
output=Path(sys.argv[sys.argv.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True)
snapshot=output.with_suffix('.trainer_snapshot.py')
if snapshot.exists() and '--resume' not in sys.argv:raise FileExistsError(snapshot)
if not snapshot.exists():snapshot.write_text(source)
else:source=snapshot.read_text()
exec(compile(source,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))

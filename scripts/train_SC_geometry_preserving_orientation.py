#!/usr/bin/env python3
"""Bounded no-spatial-jitter test; retain colour augmentation and warm baseline."""
from pathlib import Path
import sys

root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
source = (root/'spacefilled_near_fused_orientation_warm.trainer_snapshot.py').read_text()
before = "            gain=rng.uniform(.85,1.15,size=(len(ids),3,1,1,3))"
assert source.count(before) == 1
source = source.replace(before, "        if jitter:\n"+before)
before = "    lr_events=[];last_progress=0;first_step=1"
assert source.count(before) == 1
source = source.replace(before, before+"""
    if not a.resume:
        assert a.crop_jitter==0
        plan['hypothesis']='Unknown independent per-camera crop translations can obscure subpixel multiview geometry. Preserve calibrated crop coordinates; retain the same colour augmentation, data, model, split and plateau budget.'
        plan['augmentation']='Colour only; native camera crop origins are unchanged'
        save(a.output/'plan.json',plan)
        initial_metric,_=evaluate(expanded_val)
        best=initial_metric['orientation_p50_p95_deg'][1]
        torch.save(dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,
                        step=0,optimizer=opt.state_dict()),a.output/'best.pt')
        save(a.output/'initial_candidate.json',dict(score=best,step=0,retained_as_candidate=True))
""")
assert source.count('if score<best*.995:') == 1
source = source.replace('if score<best*.995:', 'if score<best*.995 or (score<=1<best):')
output = Path(sys.argv[sys.argv.index('--output')+1])
snapshot = output.with_suffix('.trainer_snapshot.py')
if snapshot.exists() and '--resume' not in sys.argv:
    raise FileExistsError(snapshot)
if not snapshot.exists():
    snapshot.write_text(source)
else:
    source = snapshot.read_text()
exec(compile(source, str(snapshot), 'exec'), dict(__name__='__main__', __file__=str(snapshot)))

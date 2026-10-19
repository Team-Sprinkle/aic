#!/usr/bin/env python3
"""Same SC data/model, contact-sensitive orientation loss and explicit angle gate."""
from pathlib import Path
import sys
old=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
base=old/'spacefilled_near_fused_orientation_warm.trainer_snapshot.py';source=base.read_text()
before="        loss=angle if a.rotation_only else position+angle"
after="""        relative=r.transpose(1,2)@rr[selected]
        contact=((10.8*relative[:,2,0]).square()+(3.65*relative[:,2,1]).square()).mean()/(.05**2)
        loss=angle+contact if a.rotation_only else position+angle+contact"""
assert source.count(before)==1;source=source.replace(before,after)
before='        return metrics(result,truth[ids]),result'
after="""        report=metrics(result,truth[ids])
        relative=np.swapaxes(result[:,:3,:3],1,2)@truth[ids,:3,:3]
        corner=10.8*np.abs(relative[:,2,0])+3.65*np.abs(relative[:,2,1])+2.25*(1-relative[:,2,2])
        report['front_corner_axial_excess_p50_p95_mm']=np.percentile(corner,[50,95]).tolist()
        return report,result"""
assert source.count(before)==1;source=source.replace(before,after)
source=source.replace("            score=expanded_metric['orientation_p50_p95_deg'][1]", "            score=max(expanded_metric['orientation_p50_p95_deg'][1],expanded_metric['front_corner_axial_excess_p50_p95_mm'][1]/.05)")
before='    lr_events=[];last_progress=0;first_step=1'
after=before+"""
    if not a.resume:
        initial_metric,_=evaluate(expanded_val)
        best=max(initial_metric['orientation_p50_p95_deg'][1],initial_metric['front_corner_axial_excess_p50_p95_mm'][1]/.05)
        torch.save(dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=0,optimizer=opt.state_dict()),a.output/'best.pt')
        save(a.output/'initial_candidate.json',dict(score=best,metrics=initial_metric,step=0))
"""
assert source.count(before)==1;source=source.replace(before,after)
out=Path(sys.argv[sys.argv.index('--output')+1]);snapshot=out.with_suffix('.trainer_snapshot.py');assert not snapshot.exists();snapshot.write_text(source)
exec(compile(source,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))

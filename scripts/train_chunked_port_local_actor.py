#!/usr/bin/env python3
"""Retained specialization of the existing trainer for matched horizon tests.

Changes are explicit and fail if the base trainer no longer matches. Existing
training/checkpoints remain intact; the generated full trainer is archived.
"""
from pathlib import Path
import hashlib
import json
import sys


source = Path(__file__).with_name('train_predicted_port_local_actor.py')
text = source.read_text()
replacements = [
    ('from train_shared_port_tcp_bc import SharedPortTCPActor,body_translation_from_poses,rotation_from_pose9',
     'from train_shared_port_tcp_bc import body_translation_from_poses,rotation_from_pose9\n'
     'from chunked_port_actor import ChunkedPortActor as SharedPortTCPActor\n'
     'from paired_local_actor_rows import PairedLocalRows,overlapping_plan_loss'),
    ("p.add_argument('--dataset',type=Path,required=True)",
     "p.add_argument('--dataset',type=Path,required=True)\n"
     "    p.add_argument('--action-horizon',type=int,choices=(4,8,12),required=True)\n"
     "    p.add_argument('--overlap-consistency-weight',type=float,default=0.)"),
    ("r.get('action_target_offsets_s')!=[.05,.1,.15,.2]",
     "r.get('action_target_offsets_s')!=[round((j+1)*.05,8) for j in range(a.action_horizon)]"),
    ('action_target_offsets_s=[.05,.1,.15,.2],',
     'action_target_offsets_s=[round((j+1)*.05,8) for j in range(a.action_horizon)],'),
    ('model=SharedPortTCPActor(stats,pretrained=False,',
     'model=SharedPortTCPActor(stats,action_horizon=a.action_horizon,pretrained=False,'),
    ('DataLoader(LocalRows(train,', 'DataLoader(PairedLocalRows(train,'),
    ('output[j,3,2]', 'output[j,-1,2]'),
    ('target[j,3,2]', 'target[j,-1,2]'),
    ("errors[group]['fourth_target_mm'].append(float((output[j,3,:3]-target[j,3,:3]).norm()*1000))",
     "errors[group]['last_target_mm'].append(float((output[j,-1,:3]-target[j,-1,:3]).norm()*1000))"),
    ('loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()',
     '''if a.overlap_consistency_weight:
            pair=batch[11]
            pi,ps,pt,pp=pair[0].cuda(non_blocking=True),pair[1].cuda(non_blocking=True),pair[2].cuda(non_blocking=True),pair[3].cuda(non_blocking=True)
            ph=pair[7].cuda(non_blocking=True);tip2=pair[8].cuda(non_blocking=True);mask2=pair[9].cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                second,_=model(pi,ps,pt,external_pose=pp,external_mask=torch.ones(len(pp),device='cuda',dtype=torch.bool),local_history=ph,physical_tip=tip2,tip_mask=mask2)
            decoded2=decoded_tcp_targets(second,pp,tip2,pt,model.sc_action_reference,model.tip_action_families)
            consistency=overlapping_plan_loss(decoded,decoded2,batch[12].cuda(non_blocking=True),batch[13].cuda(non_blocking=True),batch[14].cuda(non_blocking=True),rotation_from_pose9)
            loss+=a.overlap_consistency_weight*consistency
        if not torch.isfinite(loss):raise ValueError('Nonfinite combined continuity loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()'''),
]
changes = []
for before, after in replacements:
    if text.count(before) != 1:
        raise ValueError('Base trainer changed: '+before)
    text = text.replace(before, after)
    changes.append(before)
# Metadata is present in selected and final checkpoints. Both must rebuild
# the exact query count, rather than infer it from a convenient filename.
anchor = "'trained_families':list(families),"
assert text.count(anchor) == 2
text = text.replace(anchor, anchor+"'action_horizon':a.action_horizon,'overlap_consistency_weight':a.overlap_consistency_weight,")
out = Path(sys.argv[sys.argv.index('--output')+1])
out.parent.mkdir(parents=True, exist_ok=True)
snapshot = out.with_suffix('.trainer_snapshot.py')
if snapshot.exists():
    raise FileExistsError(snapshot)
snapshot.write_text(text)
out.with_suffix('.trainer_specialization.json').write_text(json.dumps(dict(
    source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot_sha256=hashlib.sha256(text.encode()).hexdigest(), changes=changes,
    original_trainer_modified=False), indent=2)+'\n')
# Execute in this process: the trainer's existing GPU reservation still
# applies, and all paths resolve through the original scripts import path.
exec(compile(text, str(snapshot), 'exec'), dict(__name__='__main__', __file__=str(snapshot)))

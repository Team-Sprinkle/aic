#!/usr/bin/env python3
"""Retained full-SE3 specialization; the legacy trainer remains unchanged."""
from pathlib import Path
import hashlib
import json
import sys

source=Path(__file__).with_name('train_predicted_port_local_actor.py')
text=source.read_text()
replacements=[
    ("('none','disabled','constant_translation','translation')","('none','disabled','constant_translation','translation','full_pose')"),
    ("('tcp','plug_tip_translation_tcp_orientation')","('tcp','plug_tip_translation_tcp_orientation','full_connector_pose')"),
    ("a.tip_input_mode!='translation'","a.tip_input_mode not in ('translation','full_pose')"),
    ("model.masked_tip_indices=list(range(3,9)) if include_tip else []",
     "model.masked_tip_indices=list(range(3,9)) if include_tip and a.tip_input_mode!='full_pose' else []"),
    ("    offset=(rotation_from_pose9(output.float()) @ body[:,None,:,None]).squeeze(-1)",
     """    output_rotation=rotation_from_pose9(output.float())
    if reference=='full_connector_pose':
        grasp_rotation=current_rotation.transpose(-1,-2)@rotation_from_pose9(tip.float())
        tcp_rotation=output_rotation@grasp_rotation[:,None].transpose(-1,-2)
        offset=(tcp_rotation@body[:,None,:,None]).squeeze(-1)
        xyz=output[...,:3].float()-offset
        return torch.cat((xyz,tcp_rotation[...,:,:2].transpose(-1,-2).reshape(*xyz.shape[:-1],6)),dim=-1)
    offset=(output_rotation @ body[:,None,:,None]).squeeze(-1)"""),
]
changes=[]
for before,after in replacements:
    count=text.count(before)
    if count==0:raise ValueError('Base trainer changed: '+before)
    text=text.replace(before,after);changes.append(dict(before=before,occurrences=count))
# Keep H4 to isolate full-object representation, not another horizon change.
out=Path(sys.argv[sys.argv.index('--output')+1]);out.parent.mkdir(parents=True,exist_ok=True)
snapshot=out.with_suffix('.trainer_snapshot.py')
if snapshot.exists():raise FileExistsError(snapshot)
snapshot.write_text(text)
out.with_suffix('.trainer_specialization.json').write_text(json.dumps(dict(
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot_sha256=hashlib.sha256(text.encode()).hexdigest(),changes=changes,
    original_trainer_modified=False,action_horizon=4),indent=2)+'\n')
exec(compile(text,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))

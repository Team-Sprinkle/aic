#!/usr/bin/env python3
"""Run the existing native trainer with SC-domain checkpoint selection.

The original trainer is frozen by an active controlled evaluation. Generate
and retain a one-line selection-family specialization rather than editing it.
Architecture, sampler, losses and original two-family validation are unchanged.
"""
import hashlib,json,sys
from pathlib import Path

source=Path(__file__).with_name('train_native_requested_pose_keypoints.py')
text=source.read_text();old="keys=[f'sfp/{a.selection_domain}/'+k for k in ('centre_px','reference_px')]"
assert text.count(old)==1
text=text.replace(old,"keys=[f'sc/{a.selection_domain}/'+k for k in ('centre_px','reference_px')]")
output=Path(sys.argv[sys.argv.index('--output')+1])
cache=Path(sys.argv[sys.argv.index('--cache')+1]);qa=json.load((cache/'label_QA_v1.json').open())
if qa.get('visual_review_pending') is not False or not qa.get('projection_consistency_check_passed') or not qa.get('physical_split_disjoint'):
 raise ValueError('SC extension must pass independent label/crop/split QA before fitting')
snapshot=output.with_suffix('.trainer_snapshot.py')
if snapshot.exists() or output.exists():raise FileExistsError('Preserve previous fit and its trainer')
snapshot.write_text(text)
output.with_suffix('.source.json').write_text(json.dumps(dict(base_source=str(source.resolve()),base_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),only_change='Checkpoint selection reads sc/SC_rail_expansion instead of sfp/domain; all family metrics and losses retained.',command=sys.argv,final_scenes_opened=False),indent=2)+'\n')
exec(compile(text,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))

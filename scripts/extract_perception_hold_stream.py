#!/usr/bin/env python3
"""Read perception-only holds even if teardown prevented an official score.

No synthetic score is assigned. Complete bag metadata and retained public
terminal observations are required. This output is not BC expert admission.
"""
from pathlib import Path
import sys

base=Path(__file__).with_name('extract_moveit_training_stream.py');source=base.read_text()
old="score=yaml.safe_load((a.audit/'results/scoring.yaml').read_text());assert trial_id in score"
new="""score_path=a.audit/'results/scoring.yaml'
 score=yaml.safe_load(score_path.read_text()) if score_path.exists() else None
 trace=[json.loads(l) for l in (a.audit/'runtime_trace.jsonl').open()]
 assert any(t['event']=='grasp_capture_terminal' for t in trace),'Missing retained perception terminal'
 if score is not None:assert trial_id in score"""
assert source.count(old)==1;source=source.replace(old,new)
old="official_tier3=score[trial_id]['tier_3']['score'],";new="official_tier3=score[trial_id]['tier_3']['score'] if score is not None else None,perception_only=True,BC_expert=False,official_score_available=score is not None,teardown_score_unavailable=score is None,";assert source.count(old)==1;source=source.replace(old,new)
exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=str(base)))

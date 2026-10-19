#!/usr/bin/env python3
"""Publish completion facts only; no log-driven tuning or job interruption."""
from pathlib import Path
from datetime import datetime,timezone
import json
import time


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    repo=Path.cwd();root=Path('/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1')
    previous=None
    for _ in range(2880):
        state=read(root/'state.json');decision=read(root/'decision.json')
        models={name:{k:v for k,v in read(root/('train_'+name)/'summary.json').items()
                      if k in ['updates_completed','best_step','stop_reason','stale_evaluations_at_end']}
                for name in ['h4','h8','h12','h8_continuity']}
        live=read(root/'live_comparison/progress.json')
        snapshot=dict(stage=state.get('stage'),completed_models=sum(bool(v) for v in models.values()),models=models,
            live_completed=live.get('completed',0),live_planned=14,decision=decision,new_RL=False,final_scenes_opened=False)
        pidfile=root/'coordinator.pid'
        if pidfile.exists() and not decision:
            pid=pidfile.read_text().strip();stat=Path('/proc')/pid/'stat'
            if not stat.exists() or stat.read_text().split()[2]=='Z':snapshot['stage']='stopped_requires_diagnosis'
        encoded=json.dumps(snapshot,sort_keys=True)
        if encoded!=previous:
            previous=encoded;snapshot['updated_utc']=datetime.now(timezone.utc).isoformat()
            (root/'live_progress.json').write_text(json.dumps(snapshot,indent=2)+'\n')
            start='<!-- history-horizon-live:start -->';end='<!-- history-horizon-live:end -->'
            block='\n'.join([start,'## Live history/chunk experiment','',
                f"- Stage: `{snapshot['stage']}`. Completed training variants: **{snapshot['completed_models']}/4**.",
                f"- New trained-model live comparison: **{snapshot['live_completed']}/14** complete; no insertion promotion inferred from training.",
                '- Runtime preserves measured transport history; accepted port pose stays fixed during the actor attempt. New acquisition uses fresh pose-window consensus.',
                '- Same60,671anchor samples for4/8/12step targets; recorded commands only. No newRL or reserved-final testing.',
                f'- Artifacts: `{root}`.',end])
            for name in ['docs/STATUS.md','docs/EXPERIMENTS.md','docs/experiments/2026-10-17-history-horizon-poseinsert-continuation.md','outputs_README.md']:
                p=repo/name;text=p.read_text()
                if start in text:
                    before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+block+after
                else:text=block+'\n\n'+text
                p.write_text(text)
        if decision or snapshot['stage']=='stopped_requires_diagnosis':return
        time.sleep(60)


if __name__=='__main__':main()

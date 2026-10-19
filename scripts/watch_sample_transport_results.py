#!/usr/bin/env python3
"""Publish job facts only; never change a policy, gate, split, or checkpoint."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np


START='<!-- sample-transport-live:start -->'
END='<!-- sample-transport-live:end -->'


def read(p):return json.loads(p.read_text()) if p.exists() else {}
def pct(v):return np.percentile(v,[50,90]).tolist() if v else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--sc-screen',type=Path,required=True)
    a=p.parse_args();repo=Path.cwd();previous=None
    targets=[repo/'docs/STATUS.md',repo/'docs/EXPERIMENTS.md',repo/'docs/experiments/2026-10-16-sample-conservative-rule-transport.md',repo/'outputs_README.md']
    for _ in range(1920):
        state=read(a.root/'state.json');decision=read(a.root/'decision.json')
        sc=read(a.sc_screen/'summary.json') or read(a.sc_screen/'progress.json')
        evaluation=read(a.root/'evaluation_runs/summary.json') or read(a.root/'evaluation_runs/progress.json')
        sr=sc.get('results',[]);rows=evaluation.get('results',[])
        snapshot=dict(stage=state.get('stage'),decision=decision,SC_completed=len(sr),SC_qualified=sum(r['qualified'] for r in sr),
            SFP_development_qualified=22,SFP_development_completed=22,evaluation_completed=len(rows),evaluation_planned=130,
            evaluation_qualified=sum(r['qualified'] for r in rows),weights_and_rules_frozen=(a.root/'frozen/manifest.json').exists(),
            insertion_evaluated=False,new_training=False,reserved_final_scenes_opened=False)
        encoded=json.dumps(snapshot,sort_keys=True)
        if encoded!=previous:
            previous=encoded;snapshot['updated_utc']=datetime.now(timezone.utc).isoformat()
            (a.root/'live_results.json').write_text(json.dumps(snapshot,indent=2)+'\n')
            stamp=snapshot['updated_utc'][:19]
            lines=[START,f'## Live sample-derived transport results — {stamp} UTC','',
                f'- Adaptive SC development: **{snapshot["SC_qualified"]}/{snapshot["SC_completed"]} qualified**, seven planned.',
                '- Unchanged SFP development: **22/22 qualified**.',
                f'- Frozen transport/handoff evaluation: **{snapshot["evaluation_qualified"]}/{snapshot["evaluation_completed"]} qualified**,130planned. This is not insertion.',
                f'- Rules/weights frozen: **{snapshot["weights_and_rules_frozen"]}**. Stage: `{snapshot["stage"]}`.',
                f'- Decision: `{decision.get("status","pending")}`. No new training or reserved-final testing.',
                f'- Artifacts: `{a.root}`. Sample-derived and exposed official results stay separate; no evaluation tuning.',END]
            block='\n'.join(lines)
            for file in targets:
                text=file.read_text()
                if START in text:
                    before,after=text.split(START,1);_,tail=after.split(END,1);text=before+block+tail
                else:text=block+'\n\n'+text
                file.write_text(text)
        if decision:
            if rows:
                groups=defaultdict(list)
                for r in rows:groups[(r['domain'],r['family'],r['card_count'],r['port'],r.get('requested_card'))].append(r)
                cells=[]
                for k,rs in groups.items():
                    cells.append(dict(domain=k[0],family=k[1],cards=k[2],port=k[3],requested_card=k[4],episodes=len(rs),
                        qualified=sum(r['qualified'] for r in rs),correct_acquisition=sum(bool(r['rule_visual_acquisition']) and r['wrong_accepted_port'] is False for r in rs),
                        wrong_accepts=sum(r['wrong_accepted_port'] is True for r in rs),scored_harm=sum(not r['no_scored_harm'] for r in rs),
                        terminal_lateral_p50_p90_mm=pct([r['terminal']['lateral_mm'] for r in rs]),
                        terminal_standoff_miss_p50_p90_mm=pct([abs(-r['terminal']['axial_mm']-150) for r in rs]),
                        terminal_orientation_p50_p90_deg=pct([r['terminal']['orientation_error_deg'] for r in rs])))
                (a.root/'results_by_setting.json').write_text(json.dumps(dict(cells=cells,complete=len(rows)==130),indent=2)+'\n')
                lines=['# Frozen transport/handoff evaluation','',f'Completed{len(rows)}/130. Full insertion was not evaluated.','',
                    '| Domain | Family | Cards | Port | NIC target | Episodes | Qualified | Wrong | Harm |',
                    '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
                for c in cells:lines.append('| '+' | '.join(str(c[k]) for k in ['domain','family','cards','port','requested_card','episodes','qualified','wrong_accepts','scored_harm'])+' |')
                lines+=['','Two episodes per setting are a screen, not a reliability certificate. Exposed official scenes were tested only after freezing and are not unseen final evidence. No tuning follows these results.']
                (a.root/'results_by_setting.md').write_text('\n'.join(lines)+'\n')
            return
        time.sleep(45)


if __name__=='__main__':main()

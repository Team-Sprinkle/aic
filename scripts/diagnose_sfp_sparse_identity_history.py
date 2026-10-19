#!/usr/bin/env python3
"""Read-only sparse-history check on frozen SFP predictions; never uses labels to decide."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous diagnostic')
    args.output.mkdir(parents=True)
    labels=[json.loads(s) for s in (args.source/'labels_all_ten/rows.jsonl').open()]
    matched={(r['episode'],round(r['time_s'],3)):r for r in labels}
    decisions=[];summary={}
    for model in ('allport','structured','selected','crop'):
        rows=[json.loads(s) for s in (args.source/'all_ten_model_scores'/f'{model}.jsonl').open()]
        groups=defaultdict(list)
        for r in rows:groups[r['episode']].append(r)
        for needed in (1,2,3):
            for movement_mm in (0,10):
                if needed==1 and movement_mm:continue
                outcomes=[]
                for episode,items in sorted(groups.items()):
                    items.sort(key=lambda r:r['time_s'])
                    t0=items[0]['time_s']
                    first=None
                    for end in range(needed-1,len(items)):
                        window=items[end-needed+1:end+1]
                        if not all(r['accepted'] and r['predicted_base_m'] is not None for r in window):continue
                        xyz=np.asarray([r['predicted_base_m'] for r in window],float)
                        if np.max(np.linalg.norm(xyz-xyz.mean(axis=0),axis=1))*1000>5:continue
                        if needed>1 and movement_mm:
                            r0=matched.get((episode,round(window[0]['time_s'],3)))
                            r1=matched.get((episode,round(window[-1]['time_s'],3)))
                            if r0 is None or r1 is None:continue
                            s0=np.asarray(r0['state'][:3]);s1=np.asarray(r1['state'][:3])
                            if np.linalg.norm(s1-s0)*1000<movement_mm:continue
                        # True nearest port is consulted only *after* decision.
                        end_label=matched.get((episode,round(window[-1]['time_s'],3)))
                        if end_label is None:raise ValueError('Missing posthoc label row')
                        estimate=xyz.mean(axis=0)
                        truth=min(end_label['targets'],
                                  key=lambda t:np.linalg.norm(estimate-np.asarray(t['position_base_m'])))['identity']
                        first=dict(model=model,episode=episode,needed=needed,movement_mm=movement_mm,
                                   time_s=window[-1]['time_s'],delay_s=window[-1]['time_s']-t0,
                                   requested=window[-1]['requested'],nearest_identity_label_only=truth,
                                   wrong=truth!=window[-1]['requested'],
                                   predicted_base_m=xyz.mean(axis=0).tolist())
                        decisions.append(first);break
                    outcomes.append(first)
                key=f'{model}_n{needed}_move{movement_mm}mm'
                summary[key]=dict(episodes=len(groups),any_accept=sum(x is not None for x in outcomes),
                                  early_3s=sum(x is not None and x['delay_s']<=3 for x in outcomes),
                                  wrong_first=sum(x is not None and x['wrong'] for x in outcomes))
    with (args.output/'decisions.jsonl').open('w') as f:
        for r in decisions:f.write(json.dumps(r,separators=(',',':'))+'\n')
    report=dict(schema='sfp_sparse_identity_history_diagnostic/v1',
                input=str(args.source),rules='consecutive accepted, positions within 5 mm; optional measured TCP movement',
                no_label_in_decision=True,counts=summary,final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

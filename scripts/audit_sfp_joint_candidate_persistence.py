#!/usr/bin/env python3
"""Apply a fixed causal two-frame stability gate to joint camera candidates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-step-mm',type=float,default=10.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    rows=[json.loads(s) for s in a.predictions.open()]
    byepisode=defaultdict(list)
    for row in rows:byepisode[row['episode']].append(row)
    reviewed=[];errors=[];wrong=0;early=0;anygroup=0
    for ep,items in byepisode.items():
        items.sort(key=lambda r:r['time_s']);previous=None
        for row in items:
            chosen=row['selected'] if row['accepted'] else None
            xyz=np.asarray(chosen['position_base_m']) if chosen is not None else None
            step_mm=(float(np.linalg.norm(xyz-previous)*1000)
                     if xyz is not None and previous is not None else None)
            stable=step_mm is not None and step_mm<=a.max_step_mm
            rec=dict(row);rec['accepted_before_persistence']=row['accepted']
            rec['accepted']=stable;rec['step_mm']=step_mm
            if stable:
                errors.append(row['error_mm_label_only'])
                wrong+=row['nearest_identity_label_only']!=row['requested']
            reviewed.append(rec)
            previous=xyz
        first=next((r for r in reviewed if r['episode']==ep and r['accepted']),None)
        anygroup+=first is not None
        early+=first is not None and first['time_s']-items[0]['time_s']<=3
    summary={'schema':'sfp_joint_candidate_persistence/v1','episodes':len(byepisode),
             'rows':len(rows),'accepted':len(errors),'wrong':wrong,'early_groups':early,
             'any_groups':anygroup,'accepted_3d_p95_mm':float(np.quantile(errors,.95)) if errors else None,
             'max_step_mm':a.max_step_mm,'required_consecutive':2,
             'source':str(a.predictions),'runtime_true_geometry_input':False,
             'final_scenes_opened':False}
    a.output.mkdir(parents=True)
    with (a.output/'predictions.jsonl').open('w') as f:
        for row in reviewed:f.write(json.dumps(row,separators=(',',':'))+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

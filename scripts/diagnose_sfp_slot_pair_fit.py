#!/usr/bin/env python3
"""Read-only, label-only diagnosis of a frozen SFP slot/pair checkpoint."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_sfp_slot_pair_identity import CAMERAS,SlotPairModel,Views


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--init',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous diagnosis')
    args.output.mkdir(parents=True)
    rows=[json.loads(s) for s in (args.labels/'rows.jsonl').open()]
    indices=[(i,c) for i,r in enumerate(rows) if r['fit_split']=='validation' for c in range(3)]
    data=DataLoader(Views(rows,indices,False),batch_size=12,num_workers=0)
    model=SlotPairModel(args.init).cuda()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    details=defaultdict(dict)
    with torch.inference_mode():
        for image,meta,points,mask,ri,ci in data:
            xy,logits=model(image.cuda().float()/255,meta.cuda())
            xy=xy.cpu().numpy();prob=logits.sigmoid().cpu().numpy()
            for bi in range(len(ri)):
                i=int(ri[bi]);c=CAMERAS[int(ci[bi])];r=rows[i]
                card=int(r['selected_identity'].split('_')[2]);pair=[5+2*card,6+2*card]
                required=[card,*pair]
                report=[]
                for j in required:
                    target=points[bi,j].numpy();inframe=bool(mask[bi,j].item())
                    report.append(dict(index=j,label_inframe=inframe,prob=float(prob[bi,j]),
                                       pixel_error=float(np.linalg.norm((xy[bi,j]-target)*[1151,1023]))
                                       if inframe else None))
                details[i][c]=report
    records=[]
    for i,v in sorted(details.items()):
        r=rows[i];card=int(r['selected_identity'].split('_')[2]);req=[card,5+2*card,6+2*card]
        counts={str(j):sum(v[c][k]['prob']>=.5 for c in CAMERAS)
                for k,j in enumerate(req)}
        records.append(dict(row=i,episode=r['episode'],request=r['selected_identity'],
                            time_s=r['time_s'],three_view_count=counts,per_camera=v))
    total=Counter()
    for r in records:
        counts=list(r['three_view_count'].values())
        if counts[0]<3:total['missing_slot_flag']+=1
        if min(counts[1:])<3:total['missing_pair_flag']+=1
        if min(counts)==3:total['all_flags_high']+=1
    errors={}
    for label,index in (('slot',0),('port0',1),('port1',2)):
        vals=[r['per_camera'][c][index]['pixel_error'] for r in records for c in CAMERAS
              if r['per_camera'][c][index]['pixel_error'] is not None]
        errors[label]=dict(count=len(vals),median_px=float(np.median(vals)),p95_px=float(np.quantile(vals,.95)),
                           above_20px=sum(v>20 for v in vals))
    with (args.output/'rows.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    summary=dict(schema='sfp_slot_pair_fit_diagnosis/v1',checkpoint_step=checkpoint['step'],
                 validation_rows=len(records),flag_counts=dict(total),pixel_errors=errors,
                 labels_used_posthoc_only=True,final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

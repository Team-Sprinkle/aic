#!/usr/bin/env python3
"""Cache RGB-only selected and alternate port points for a fixed grouped split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_two_family_port_locator import Locator, Views, CAMERAS, NAMES, WIDTH, HEIGHT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--features-output',type=Path,
                   help='Optional matching frozen 768D feature cache for each view')
    p.add_argument('--all-candidates',action='store_true',
                   help='Retain absent-candidate outputs so orientation input matches runtime')
    p.add_argument('--family',choices=('sc','sfp'))
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(l) for l in (a.labels/'rows.jsonl').open()]
    indices=[(i,c) for i,r in enumerate(rows) if a.family is None or r['family']==a.family for c in range(3)]
    loader=DataLoader(Views(rows,a.sfp_cache,indices,False),batch_size=96,shuffle=False,
                      num_workers=4,pin_memory=True)
    model=Locator(a.imagenet).cuda().eval()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.features_output:
        if a.features_output.exists():p.error('Preserve existing feature cache')
        a.features_output.parent.mkdir(parents=True,exist_ok=True)
        features=np.lib.format.open_memmap(a.features_output,mode='w+',dtype=np.float16,
                                           shape=(len(rows),3,768))
    with a.output.open('w') as file, torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for image,meta,_,_,_,_,_,index,cam in loader:
            pred,flag,feature=model(image.cuda(non_blocking=True).float()/255.,
                                    meta.cuda(non_blocking=True),return_features=True)
            pred=pred.float().cpu().numpy();flag=flag.float().sigmoid().cpu().numpy()
            if a.features_output:
                features[index.numpy(),cam.numpy()]=feature.float().cpu().numpy().astype(np.float16)
            for j in range(len(index)):
                i=int(index[j]);c=CAMERAS[int(cam[j])]
                item={'row':i,'episode':rows[i]['episode'],'scene_group':rows[i]['scene_group'],
                      'split':rows[i]['split'],'family':rows[i]['family'],'camera':c,
                      'selected_identity':rows[i]['selected_identity'],
                      'predicted':{name:{'xy':(pred[j,NAMES.index(name)]*
                              [WIDTH-1,HEIGHT-1]).tolist(),
                          'projected_inframe_probability':float(flag[j,NAMES.index(name)])}
                          for name in (NAMES if a.all_candidates else
                                       [t['identity'] for t in rows[i]['targets']])},
                      'runtime_geometry_input':False,'visibility_certified':False}
                file.write(json.dumps(item,separators=(',',':'))+'\n')
    if a.features_output:features.flush()
    print(json.dumps({'rows':len(indices),'episodes':len({rows[i]['episode'] for i,_ in indices}),
                      'checkpoint':str(a.checkpoint),'output':str(a.output),
                      'features_output':str(a.features_output) if a.features_output else None}))


if __name__=='__main__':main()

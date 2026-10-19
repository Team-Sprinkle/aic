#!/usr/bin/env python3
"""Cache frozen RGB-only multi-view features for an orientation diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_two_family_port_locator import Locator,Views


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(l) for l in (a.labels/'rows.jsonl').open()]
    indices=[(i,c) for i in range(len(rows)) for c in range(3)]
    loader=DataLoader(Views(rows,a.sfp_cache,indices,False),batch_size=96,shuffle=False,
                      num_workers=4,pin_memory=True)
    model=Locator(a.imagenet).cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out=np.lib.format.open_memmap(a.output,mode='w+',dtype=np.float16,shape=(len(rows),3,768))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for image,_,_,_,_,_,_,index,cam in loader:
            image=image.cuda(non_blocking=True).float()/255.
            x=(image-model.mean)/model.std
            b=model.backbone
            x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
            x=b.layer1(x);x=b.layer2(x);x=b.layer3(x)
            early=x.mean((-1,-2));late=b.layer4(x).mean((-1,-2))
            features=torch.cat((early,late),dim=-1).float().cpu().numpy().astype(np.float16)
            out[index.numpy(),cam.numpy()]=features
    out.flush()
    print(json.dumps({'rows':len(rows),'views':len(indices),'feature_dim_per_view':768,
                      'output':str(a.output),'runtime_geometry':False}))


if __name__=='__main__':main()

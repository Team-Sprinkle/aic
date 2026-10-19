#!/usr/bin/env python3
"""Cache frozen observation-only selected-port offsets for native-image SFP rows."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from sfp_selected_port_observation import SelectedSfpPort


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--crop',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous cache')
    if not 1<=a.batch_size<=32:p.error('Batch-size bound is 1..32')
    torch.set_num_threads(4)
    labels={}
    for line in a.labels.open():
        r=json.loads(line)
        if 'images' in r:labels[(int(r['episode_index']),int(r['frame_index']))]=r
    rows=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    values=np.zeros((len(rows),3),np.float32)
    valid=np.zeros(len(rows),bool)
    expected=Counter();covered=Counter();missing=[];pending=[]
    for i,r in enumerate(rows):
        if r['family']!='sfp':continue
        key=(int(r['episode']),int(r['frame']))
        label=labels.get(key)
        if label is None:continue
        group=r['split'];expected[group]+=1
        paths=label['images']
        if not all(Path(paths[c]).is_file() for c in ('center','left','right')):
            missing.append(key);continue
        pending.append((i,label))
    if missing or not pending:raise ValueError(f'Missing native RGB: {len(missing)}; pending={len(pending)}')
    model=SelectedSfpPort(a.board,a.crop)
    model.warmup()
    for start in range(0,len(pending),a.batch_size):
        batch=pending[start:start+a.batch_size];items=[]
        for _,r in batch:
            images={c:cv2.imread(r['images'][c],cv2.IMREAD_COLOR)
                    for c in ('center','left','right')}
            if any(v is None for v in images.values()):raise FileNotFoundError(r['images'])
            items.append((images,np.asarray(r['state'],float),
                          int(r['selected_card']),int(r['selected_port'])))
        outputs=model.predict_batch(items)
        for (i,r),out in zip(batch,outputs):
            if out['accepted']:
                state=np.asarray(r['state'],float)
                relative=Rotation.from_quat(state[3:7]).inv().apply(
                    out['predicted_base_m']-state[:3])/.2
                if not np.isfinite(relative).all():raise ValueError('Nonfinite selected-port offset')
                values[i]=relative.astype(np.float32);valid[i]=True
                covered[r['split']]+=1
        if (start//a.batch_size)%500==0:
            print(json.dumps({'processed':min(start+a.batch_size,len(pending)),
                              'pending':len(pending),'accepted':int(valid.sum())}),flush=True)
    a.output.mkdir(parents=True)
    np.save(a.output/'values.npy',values)
    np.save(a.output/'accepted.npy',valid)
    result=dict(schema='aic_sfp_selected_port_route_values/v1',rows=len(rows),
                native_sfp_rows=len(pending),expected_by_split=dict(expected),
                accepted_by_split=dict(covered),accept_fraction_by_split={k:covered[k]/v for k,v in expected.items()},
                scale_m=.2,frame='measured_tcp_body',invalid_value=[0,0,0],
                checkpoint_sha256={'board':sha(a.board),'crop':sha(a.crop)},
                source_sha256={'cache_rows':sha(a.cache/'rows.jsonl'),'labels':sha(a.labels)},
                runtime_geometry_input=False,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

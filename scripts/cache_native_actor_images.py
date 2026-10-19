#!/usr/bin/env python3
"""Cache exact native actor pixels on NAS, preserving the original JPEGs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from native_actor_images import actor_image,image_key,NativeActorImageCache,CONTRACT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    unique={}
    for row in map(json.loads,(a.dataset/'rows.jsonl').open()):
        if row.get('image_kind')!='native_1152x1024_rgb_jpeg':raise ValueError('This cache is native-only')
        unique.setdefault(image_key(row),row)
    rows=list(unique.values());cache=np.lib.format.open_memmap(a.output/'images.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,3,224,224))
    old=NativeActorImageCache(a.reuse) if a.reuse else None;reused=0
    def materialize(row):
        if old is not None and image_key(row) in old.index:return old[row],True
        result=[]
        for c in ('center','left','right'):
            im=cv2.imread(row['images'][c])
            if im is None:raise FileNotFoundError(row['images'][c])
            result.append(actor_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),row['family'],already_native_jpeg=True))
        return np.stack(result),False
    with (a.output/'manifest.jsonl').open('w') as writer,ThreadPoolExecutor(max_workers=8) as pool:
        # Submit bounded chunks: executor.map over all rows would retain
        # gigabytes of completed image results behind a slow NAS read.
        for start in range(0,len(rows),128):
            for index,(pixels,used) in enumerate(pool.map(materialize,rows[start:start+128]),start):
                cache[index]=pixels;reused+=used
                writer.write(json.dumps(dict(index=index,image_key=image_key(rows[index]),family=rows[index]['family'],images=rows[index]['images']))+'\n')
            if start%1024==0:cache.flush();print(json.dumps(dict(completed=min(start+128,len(rows)),planned=len(rows),reused=reused)),flush=True)
    cache.flush();summary=dict(contract=CONTRACT,unique_triplets=len(rows),reused_triplets=reused,source_dataset=str(a.dataset),
        source_rows_sha256=hashlib.sha256((a.dataset/'rows.jsonl').read_bytes()).hexdigest(),original_images_modified=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

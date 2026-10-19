"""Exact native-image contract shared by new BC caches and live actors."""
import hashlib
import json
import cv2
import numpy as np
from PIL import Image

CONTRACT='native_JPEG86_then_area288x256_then_SFP_JPEG95_then_PIL224/v1'


def image_key(row):
    return hashlib.sha256(json.dumps([row['family'],*[row['images'][c] for c in ('center','left','right')]],separators=(',',':')).encode()).hexdigest()


def actor_image(rgb,family,*,already_native_jpeg=False):
    if rgb.shape!=(1024,1152,3) or rgb.dtype!=np.uint8:raise ValueError('Original RGB required')
    if family not in ('sc','sfp'):raise ValueError('Unknown family')
    if not already_native_jpeg:
        ok,encoded=cv2.imencode('.jpg',rgb[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
        if not ok:raise RuntimeError('Native actor codec failed')
        rgb=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    small=cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA)
    if family=='sfp':
        ok,encoded=cv2.imencode('.jpg',small[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:raise RuntimeError('SFP small-image codec failed')
        small=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(small).resize((224,224),Image.Resampling.BILINEAR)).transpose(2,0,1).copy()


class NativeActorImageCache:
    def __init__(self,directory):
        from pathlib import Path
        root=Path(directory);summary=json.loads((root/'summary.json').read_text())
        if summary['contract']!=CONTRACT:raise ValueError('Actor image cache contract mismatch')
        self.index={r['image_key']:r['index'] for r in map(json.loads,(root/'manifest.jsonl').open())}
        self.images=np.load(root/'images.npy',mmap_mode='r')

    def __getitem__(self,row):
        key=image_key(row)
        if key not in self.index:raise ValueError('Native input cache does not cover this row')
        return np.array(self.images[self.index[key]],copy=True)

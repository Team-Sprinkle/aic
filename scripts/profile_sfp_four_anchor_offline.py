#!/usr/bin/env python3
"""Profile the four frozen RGB anchor networks on saved original camera files.

Includes JPEG decoding, resize, native crop and forward passes. Excludes ROS,
controller, camera transport, triangulation and other live policy components.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from train_two_family_port_locator import CAMERAS, LOOKUP, Locator
from train_target_conditioned_port_locator import ConditionalLocator
from train_sfp_observed_rail_crop import crop_native


def load(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    return model.cuda().eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--proposals', type=Path, required=True)
    p.add_argument('--allport', type=Path, required=True)
    p.add_argument('--structured', type=Path, required=True)
    p.add_argument('--selected', type=Path, required=True)
    p.add_argument('--crop', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=20)
    a = p.parse_args()
    if a.output.exists():p.error('Preserve previous profile')
    torch.set_num_threads(4)
    models = {
        'allport': load(Locator(None), a.allport),
        'structured': load(Locator(None), a.structured),
        'selected': load(ConditionalLocator(None), a.selected),
        'crop': load(ConditionalLocator(None), a.crop),
    }
    labels = {(r['episode'], r['time_s']):r for line in (a.labels/'rows.jsonl').open()
              if (r := json.loads(line))['split'] == 'validation'}
    proposal_rows = [json.loads(line) for line in (a.proposals/'rows.jsonl').open()]
    rows = [r for r in proposal_rows if r['crop']][:a.samples]
    if not rows:raise ValueError('No crop proposals')

    def one(row):
        label = labels[(row['episode'], row['time_s'])]
        t0 = time.perf_counter()
        originals = [cv2.imread(label['images'][cam], cv2.IMREAD_COLOR) for cam in CAMERAS]
        if any(image is None for image in originals):raise FileNotFoundError(label['images'])
        meta = np.zeros((3,5), np.float32);meta[:,1]=1
        for ci in range(3):meta[ci,2+ci]=1
        task = torch.full((3,), LOOKUP[row['requested']], device='cuda', dtype=torch.long)
        meta = torch.from_numpy(meta).cuda()
        original = [cv2.cvtColor(im,cv2.COLOR_BGR2RGB) for im in originals]
        sizes = {'allport':(288,256),'structured':(576,512),'selected':(576,512)}
        with torch.inference_mode():
            for name in ('allport','structured','selected'):
                w,h=sizes[name]
                batch = np.stack([cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA) for im in original])
                inp = torch.from_numpy(batch.copy()).cuda().permute(0,3,1,2).float()/255.
                if name=='selected':models[name](inp,meta,task)
                else:models[name](inp,meta)
            crops = np.stack([crop_native(originals[ci],row['crop']['center_px'][cam])
                              for ci,cam in enumerate(CAMERAS)])
            inp = torch.from_numpy(crops.copy()).cuda().permute(0,3,1,2).float()/255.
            models['crop'](inp,meta,task)
        torch.cuda.synchronize()
        return (time.perf_counter()-t0)*1000

    one(rows[0])
    times=[one(row) for row in rows]
    report={'schema':'sfp_four_anchor_offline_profile/v1','samples':len(times),
            'gpu':torch.cuda.get_device_name(),'jpeg_decode_resize_crop_and_four_models_ms':{
                'p50':float(np.quantile(times,.5)),'p95':float(np.quantile(times,.95)),
                'p99':float(np.quantile(times,.99)),'max':max(times)},
            'excludes':['ROS','camera transport','triangulation','controller','transport actor'],
            'complete_live_latency_gate_passed':False,
            'final_scenes_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Benchmark decoded native RGB → learned crop → two-view SC pose on one GPU.

ROS transport, sensor acquisition, actor, and command conversion are excluded.
No projected label or TF field is read in the inference loop.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_sc_full_crop_fusion import FullCropFusion
from train_sc_native_locator import Locator, CAMERAS, WIDTH, HEIGHT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--locator", type=Path, required=True)
    p.add_argument("--fusion", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--pretrained-resnet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(8)
    rows = [r for r in map(json.loads, args.labels.open()) if r["split"] == "validation"][:args.samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator = Locator().to(device).eval()
    locator_saved = torch.load(args.locator, map_location="cpu", weights_only=False)
    locator.load_state_dict(locator_saved["model"], strict=True)
    base_saved = torch.load(args.base, map_location="cpu", weights_only=False)
    model = FullCropFusion(base_saved, args.pretrained_resnet).to(device).eval()
    model.load_state_dict(torch.load(args.fusion, map_location="cpu", weights_only=False)["model"], strict=True)
    transform = transforms.Compose([
        transforms.Resize((base_saved["image_size"], base_saved["image_size"])),
        transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    times=[]
    with torch.inference_mode():
        for row in rows:
            if device.type == "cuda": torch.cuda.synchronize()
            start=time.perf_counter()
            raw=[];coarse=[];full=[]
            for camera in CAMERAS:
                with Image.open(row["images"][camera]) as im:
                    rgb=im.convert("RGB")
                    if rgb.size != (WIDTH,HEIGHT): raise ValueError("Native image size mismatch")
                    raw.append(rgb.copy())
                    coarse.append(torch.from_numpy(np.asarray(rgb.resize((288,256),Image.Resampling.BILINEAR)).copy()).permute(2,0,1))
                    full.append(transform(rgb.resize((288,256),Image.Resampling.BOX)))
            pixel=locator(torch.stack(coarse).to(device).float()/255).cpu()
            crops=[]
            for image,points in zip(raw,pixel):
                xy=(points*torch.tensor([WIDTH-1,HEIGHT-1]*2)).reshape(2,2).mean(0)
                left=round(float(xy[0])-112);top=round(float(xy[1])-112)
                crops.append(transform(image.crop((left,top,left+224,top+224))))
            state=(torch.tensor(row["state"],dtype=torch.float32)-base_saved["state_mean"])/base_saved["state_std"]
            state=torch.cat((state,torch.tensor(row["task_vector"],dtype=torch.float32)))
            model(torch.stack(full)[None].to(device),torch.stack(crops)[None].to(device),state[None].to(device))
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    reported=times[10:] if len(times)>20 else times
    result={"schema":"sc_full_crop_offline_latency/v1",
            "scope":"JPEG decode, global resize, RGB-only locator, raw crop, pose model; excludes ROS acquisition, actor, and command conversion",
            "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
            "samples":len(times),"warmup_excluded":len(times)-len(reported),
            "p50_ms":float(np.percentile(reported,50)),
            "p95_ms":float(np.percentile(reported,95)),
            "p99_ms":float(np.percentile(reported,99)),
            "max_ms":float(max(reported))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()

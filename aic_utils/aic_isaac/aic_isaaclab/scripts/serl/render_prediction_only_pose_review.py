#!/usr/bin/env python3
"""Render a label-free audit of predicted plug/opening pose and cross-view consistency."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    value = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(value); return value


opening = module("opening_prediction_review", "train_opening_landmark_pose_probe.py")
pretrained = module("pretrained_prediction_review", "calibrate_pretrained_opening_landmarks.py")
CAMERAS = ("center_camera", "left_camera", "right_camera")


def file_id(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20): h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def raw_rows(replay, allowed):
    """Read only observation fields; do not parse simulator object-pose labels."""
    transitions = torch.load(replay, map_location="cpu", weights_only=False)["transitions"]
    rows = []
    for transition_index, transition in enumerate(transitions):
        metadata = transition.get("metadata") or {}
        episode_id = (metadata.get("causal_episode") or {}).get("episode_id")
        if episode_id not in allowed: continue
        cameras = ((metadata.get("highres_observation") or {}).get("cameras") or {})
        views = []
        for camera_name in CAMERAS:
            item = cameras.get(camera_name) or {}
            required = ("path", "width", "height", "intrinsic_matrix", "camera_position_world", "camera_orientation_wxyz_ros")
            if any(item.get(key) is None for key in required): break
            # Deliberately omit locator_supervision_xy, opening_corner_supervision_xy,
            # causal_insertion_geometry, and all_body_insertion_geometry.
            views.append({key: item[key] for key in required})
        feature = (transition.get("obs") or {}).get("world_feature")
        if len(views) == 3 and isinstance(feature, torch.Tensor) and feature.numel() == 384:
            rows.append({"episode_id": episode_id, "transition_index": transition_index,
                         "views": views, "feature": feature.reshape(-1).float()})
    return rows


def qmul(a, b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array((w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2))


def qconj(q): return np.array((q[0], -q[1], -q[2], -q[3]), np.float64)


def qrot(q, vector):
    q = np.asarray(q, np.float64); q /= np.linalg.norm(q)
    return qmul(qmul(q, np.r_[0.0, vector]), qconj(q))[1:]


def calibration(view):
    return {"K": np.asarray(view["intrinsic_matrix"], np.float64),
            "position": np.asarray(view["camera_position_world"], np.float64),
            "quat": np.asarray(view["camera_orientation_wxyz_ros"], np.float64),
            "width": int(view["width"]), "height": int(view["height"])}


def project(point_world, camera):
    camera_point = qrot(qconj(camera["quat"]), np.asarray(point_world) - camera["position"])
    if camera_point[2] <= 1e-9: return None
    pixel = camera["K"] @ (camera_point / camera_point[2])
    return np.asarray(pixel[:2])


def triangulate(per_view_xy, calibrations):
    packed = []
    for point, camera in zip(per_view_xy, calibrations):
        packed.append([point[0] / (camera["width"] - 1), point[1] / (camera["height"] - 1)])
    return opening.tri.triangulate_one(torch.tensor(packed, dtype=torch.float32), calibrations, 0)


def circle(draw, point, color, radius=5, width=3):
    x,y = map(float, point); draw.ellipse((x-radius,y-radius,x+radius,y+radius), outline=color, width=width)


def cross(draw, point, color, radius=6, width=3):
    x,y = map(float, point); draw.line((x-radius,y,x+radius,y), fill=color, width=width); draw.line((x,y-radius,x,y+radius), fill=color, width=width)


def arrow(draw, start, end, color, width=3):
    start=np.asarray(start,float); end=np.asarray(end,float); draw.line((*start,*end),fill=color,width=width)
    delta=end-start; length=np.linalg.norm(delta)
    if length<1e-6:return
    unit=delta/length; side=np.array((-unit[1],unit[0])); tip=end
    for sign in (-1,1):
        wing=tip-unit*10+sign*side*5;draw.line((*tip,*wing),fill=color,width=width)


def predict(rows, checkpoint, auxiliary, crop_size, device):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    locator=opening.crop.Locator().to(device).eval();locator.load_state_dict(saved["locator"])
    landmark=pretrained.MobileNetLandmarks().to(device).eval();landmark.load_state_dict(saved["landmark"])
    outputs=[]
    with torch.inference_mode():
        for row in rows:
            images=[Image.open(v["path"]).convert("RGB") for v in row["views"]]
            full=torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2,0,1) for image in images]).to(device).float()/255
            coarse=locator(F.interpolate(full,size=(256,288),mode="bilinear",align_corners=False)).cpu().reshape(3,2,2)
            crops=[]; boxes=[]
            for image,view,points in zip(images,row["views"],coarse):
                pixel=points*torch.tensor([view["width"]-1,view["height"]-1]);center=pixel.mean(0)
                left=int(round(float(center[0])-crop_size/2));top=int(round(float(center[1])-crop_size/2))
                crops.append(torch.from_numpy(np.asarray(image.crop((left,top,left+crop_size,top+crop_size))).copy()).permute(2,0,1));boxes.append((left,top))
            logits=landmark(torch.stack(crops).to(device).float()/255);local=opening.decode(logits).cpu()
            probability=torch.softmax(logits.flatten(-2),-1);h,w=logits.shape[-2:];yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=device),torch.linspace(0,1,w,device=device),indexing="ij");grid=torch.stack((xx.flatten(),yy.flatten()),-1);mean=(probability[...,None]*grid).sum(-2);variance=(probability[...,None]*(grid-mean[...,None,:]).square()).sum((-2,-1)).cpu()
            pixel_landmarks=[];pixel_plug=[]
            for view_index,(view,box) in enumerate(zip(row["views"],boxes)):
                pixel_landmarks.append(local[view_index]*(crop_size-1)+torch.tensor(box))
                pixel_plug.append(coarse[view_index,0]*torch.tensor([view["width"]-1,view["height"]-1]))
            pixel_landmarks=torch.stack(pixel_landmarks);pixel_plug=torch.stack(pixel_plug);calibrations=[calibration(v) for v in row["views"]]
            plug_world=triangulate(pixel_plug,calibrations);corners_world=np.stack([triangulate(pixel_landmarks[:,i],calibrations) for i in range(2,6)]);center_world=corners_world.mean(0)
            reprojection=np.empty((3,6,2));residual=[]
            worlds=[plug_world,center_world,*corners_world]
            raw=torch.cat((pixel_plug[:,None],pixel_landmarks[:,1:]),1).numpy()
            for vi,camera in enumerate(calibrations):
                for pi,point in enumerate(worlds):reprojection[vi,pi]=project(point,camera)
                residual.extend(np.linalg.norm(reprojection[vi]-raw[vi],axis=1).tolist())
            x_axis=corners_world[1]-corners_world[0];x_axis/=max(np.linalg.norm(x_axis),1e-9)
            y_axis=corners_world[3]-corners_world[0];y_axis/=max(np.linalg.norm(y_axis),1e-9)
            normal=np.cross(x_axis,y_axis);normal/=max(np.linalg.norm(normal),1e-9)
            outputs.append({"coarse":coarse,"landmarks":pixel_landmarks,"boxes":boxes,"variance":variance,
                "raw":raw,"reprojection":reprojection,"reprojection_rms_px":float(np.sqrt(np.mean(np.square(residual)))),
                "plug_world":plug_world,"center_world":center_world,"corners_world":corners_world,
                "axes":np.stack((x_axis,y_axis,normal)),"delta_world_mm":(plug_world-center_world)*1000})
    aux=torch.load(auxiliary,map_location="cpu",weights_only=False);models=[opening.world.Probe(384) for _ in aux["feature_members"]]
    for model,state in zip(models,aux["feature_members"]):model.load_state_dict(state)
    reg,variance,_,_=opening.world.predict(models,[{"feature":r["feature"]} for r in rows],"feature",aux["target_mean"],aux["target_std"],device)
    for item,pose,var in zip(outputs,reg,variance):item["predicted_rotation_vector_deg"]=pose[3:].numpy();item["predicted_rotation_std_deg"]=torch.sqrt(var[3:]).numpy()
    return outputs


def select(rows, predictions, count):
    residual=np.asarray([p["reprojection_rms_px"] for p in predictions]);unc=np.asarray([float(p["variance"][:,2:6].mean()) for p in predictions]);chosen={0,len(rows)-1,int(np.argmin(residual)),int(np.argmax(residual)),int(np.argmax(unc))}
    for q in (.25,.5,.75,.9):chosen.add(int(np.argmin(np.abs(residual-np.quantile(residual,q)))))
    if len(chosen)<count:
        for index in np.linspace(0,len(rows)-1,count,dtype=int):
            chosen.add(int(index))
            if len(chosen)>=count: break
    return sorted(chosen,key=lambda i:(rows[i]["episode_id"],rows[i]["transition_index"]))


def render(rows,predictions,indices,output,crop_output,crop_size):
    full_rows=[];crop_rows=[]
    for index in indices:
        row=rows[index];pred=predictions[index];panels=[];crop_panels=[]
        for vi,view in enumerate(row["views"]):
            image=Image.open(view["path"]).convert("RGB");draw=ImageDraw.Draw(image);raw=pred["raw"][vi];rep=pred["reprojection"][vi]
            circle(draw,raw[0],"magenta",6);center=raw[1];cross(draw,center,"yellow",7);corners=raw[2:6]
            draw.line([tuple(x) for x in np.r_[corners,corners[:1]]],fill="cyan",width=3)
            for point in corners:circle(draw,point,"cyan",4,2)
            arrow(draw,raw[0],center,"lime",3)
            for point in rep:circle(draw,point,"white",3,2)
            camera=calibration(view);axis_colors=("red","lime","deepskyblue")
            for axis,color in zip(pred["axes"],axis_colors):
                endpoint=project(pred["center_world"]+.003*axis,camera)
                if endpoint is not None:arrow(draw,center,endpoint,color,2)
            label=(f"pred only | {CAMERAS[vi]} | reproj RMS {pred['reprojection_rms_px']:.2f}px")
            draw.rectangle((0,0,min(image.width,430),22),fill="black");draw.text((5,5),label,fill="white")
            panels.append(image.resize((288,256)))
            left,top=pred["boxes"][vi];crop=image.crop((left,top,left+crop_size,top+crop_size)).resize((240,240));crop_panels.append(crop)
        strip=Image.new("RGB",(864,300),"black")
        for vi,panel in enumerate(panels):strip.paste(panel,(288*vi,0))
        rot=pred["predicted_rotation_vector_deg"];delta=pred["delta_world_mm"]
        text=(f"episode={row['episode_id']} transition={row['transition_index']} | predicted delta_world mm=[{delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f}] | "
              f"predicted relative rotvec deg=[{rot[0]:+.2f}, {rot[1]:+.2f}, {rot[2]:+.2f}]")
        ImageDraw.Draw(strip).text((6,268),text,fill="white");full_rows.append(strip)
        crop_strip=Image.new("RGB",(720,270),"black")
        for vi,panel in enumerate(crop_panels):crop_strip.paste(panel,(240*vi,0))
        ImageDraw.Draw(crop_strip).text((5,245),f"predictions only | {row['episode_id']} t={row['transition_index']} | center / left / right",fill="white");crop_rows.append(crop_strip)
    sheet=Image.new("RGB",(864,300*len(full_rows)),"black");csheet=Image.new("RGB",(720,270*len(crop_rows)),"black")
    for i,item in enumerate(full_rows):sheet.paste(item,(0,300*i))
    for i,item in enumerate(crop_rows):csheet.paste(item,(0,270*i))
    sheet.save(output);csheet.save(crop_output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--replay",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--auxiliary-checkpoint",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--crop-size",type=int,default=160);p.add_argument("--sample-count",type=int,default=9);p.add_argument("--device",default="cuda");a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    ids={x["episode_id"] for x in json.loads(a.manifest.read_text())["development"]};rows=raw_rows(a.replay,ids);predictions=predict(rows,a.checkpoint,a.auxiliary_checkpoint,a.crop_size,torch.device(a.device));indices=select(rows,predictions,a.sample_count)
    full=a.output_dir/"prediction_only_full_views.png";crops=a.output_dir/"prediction_only_native_crops.png";render(rows,predictions,indices,full,crops,a.crop_size)
    records=[]
    for i in indices:
        records.append({"episode_id":rows[i]["episode_id"],"transition_index":rows[i]["transition_index"],"selection_signals":{"reprojection_rms_px":predictions[i]["reprojection_rms_px"],"mean_corner_heatmap_variance":float(predictions[i]["variance"][:,2:6].mean())},"predicted_delta_world_mm":predictions[i]["delta_world_mm"].tolist(),"predicted_rotation_vector_deg":predictions[i]["predicted_rotation_vector_deg"].tolist()})
    reprojection=np.asarray([x["reprojection_rms_px"] for x in predictions]);position_jitter=[];normal_jitter=[]
    for episode_id in sorted({r["episode_id"] for r in rows}):
        episode=[i for i,r in enumerate(rows) if r["episode_id"]==episode_id]
        centers=np.stack([predictions[i]["center_world"] for i in episode]);reference=np.median(centers,axis=0)
        position_jitter.extend((np.linalg.norm(centers-reference,axis=1)*1000).tolist())
        normals=np.stack([predictions[i]["axes"][2] for i in episode]);normal_reference=np.median(normals,axis=0);normal_reference/=max(np.linalg.norm(normal_reference),1e-9)
        normal_jitter.extend((np.arccos(np.clip(np.abs(normals@normal_reference),0,1))*180/math.pi).tolist())
    report={"schema_version":1,"status":"prediction_only_visual_audit","claim":"Shows model output plausibility and cross-view geometric/temporal consistency; does not establish metric accuracy without labels.","label_exclusion":{"object_pose_labels_read":False,"excluded_fields":["locator_supervision_xy","opening_corner_supervision_xy","causal_insertion_geometry","all_body_insertion_geometry","translation_mm","rotation_deg"],"inputs":["RGB images","camera intrinsics/extrinsics","frozen observed feature","trained model weights"]},"prediction_only_self_consistency":{"row_count":len(rows),"episode_count":len({r['episode_id'] for r in rows}),"cross_view_reprojection_rms_px":{"median":float(np.median(reprojection)),"p95":float(np.quantile(reprojection,.95)),"maximum":float(np.max(reprojection))},"stationary_opening_position_jitter_mm":{"definition":"distance of predicted 3D opening center from its per-episode coordinate-wise median","median":float(np.median(position_jitter)),"p95":float(np.quantile(position_jitter,.95))},"stationary_opening_normal_jitter_deg":{"definition":"unsigned angle of predicted opening normal from its per-episode median normal","median":float(np.median(normal_jitter)),"p95":float(np.quantile(normal_jitter,.95))}},"legend":{"magenta":"predicted plug point","cyan":"predicted opening corners and boundary","yellow":"predicted opening center","lime_arrow":"predicted image-plane correction from plug to opening","white_hollow":"reprojection of triangulated predictions","red_green_blue":"predicted opening x, y, normal axes"},"selection":"first/last plus quantiles and extremes of prediction-only reprojection residual and heatmap variance; no error labels","samples":records,"artifacts":{"full_views":file_id(full),"native_crops":file_id(crops),"checkpoint":file_id(a.checkpoint),"replay":file_id(a.replay)}}
    (a.output_dir/"prediction_only_review.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"rows":len(rows),"selected":len(indices),"output":str(a.output_dir),"reprojection_rms_px":{"median":float(np.median([x['reprojection_rms_px'] for x in predictions])),"p95":float(np.quantile([x['reprojection_rms_px'] for x in predictions],.95))}},indent=2))


if __name__=="__main__":main()

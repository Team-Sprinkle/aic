#!/usr/bin/env python3
"""Audit cable/gripper occlusion coverage around supervised opening landmarks."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

CAMERAS=("center_camera","left_camera","right_camera");CORNERS=("top_left","top_right","bottom_right","bottom_left")
def file_id(path):
 h=hashlib.sha256();
 with path.open("rb") as f:
  while b:=f.read(8<<20):h.update(b)
 return {"path":str(path),"bytes":path.stat().st_size,"sha256":h.hexdigest()}
def category(path):
 p=path.lower()
 if "/rope/" in p:return "rope"
 if "/sfp_module/" in p:return "plug"
 if any(x in p for x in ("gripper","wrist_","ati_base")):return "gripper"
 if "/nic_card/" in p:return "target"
 if "/robot/" in p:return "robot_other"
 return "other"
def quantiles(values):
 a=np.asarray(values,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.quantile(a,.95)),"maximum":float(a.max())}
def roi_mask(width,height,corners,pad=4):
 mask=Image.new("1",(width,height));draw=ImageDraw.Draw(mask);center=np.mean(corners,axis=0)
 expanded=[]
 for point in corners:
  delta=point-center;norm=np.linalg.norm(delta);expanded.append(tuple(point+(pad*delta/max(norm,1e-6))))
 draw.polygon(expanded,fill=1);return np.asarray(mask,dtype=bool)
def patch_fraction(classes,point,name,radius=7):
 x,y=np.rint(point).astype(int);y0=max(0,y-radius);y1=min(classes.shape[0],y+radius+1);x0=max(0,x-radius);x1=min(classes.shape[1],x+radius+1)
 return float(np.mean(classes[y0:y1,x0:x1]==name)) if y1>y0 and x1>x0 else 0.

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--replay",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(a.manifest.read_text());allowed={x["episode_id"] for x in manifest["train"]};payload=torch.load(a.replay,map_location="cpu",weights_only=False);rows=[];rejected=Counter()
 sequence_by_transition={};sequence_index=-1;previous_episode=None;previous_terminal=True
 for ti,t in enumerate(payload["transitions"]):
  m=t.get("metadata") or {};ep=(m.get("causal_episode") or {}).get("episode_id")
  if previous_terminal or ep!=previous_episode:sequence_index+=1
  sequence_by_transition[ti]=f"reset-{sequence_index:06d}:{ep}";previous_episode=ep;previous_terminal=bool(m.get("terminated") or m.get("truncated"))
 for ti,t in enumerate(payload["transitions"]):
  m=t.get("metadata") or {};ep=(m.get("causal_episode") or {}).get("episode_id")
  if ep not in allowed:rejected["excluded_episode"]+=1;continue
  global_episode_index=m.get("global_episode_index");global_episode_index=(m.get("causal_episode") or {}).get("global_episode_index") if global_episode_index is None else global_episode_index
  sequence=f"global-{global_episode_index}:{ep}" if global_episode_index is not None else sequence_by_transition[ti];depth=float((m.get("causal_insertion_geometry") or {}).get("signed_depth_m_env0",float("nan")));phase="approach" if depth<-.003 else "alignment"
  cameras=((m.get("highres_observation") or {}).get("cameras") or {})
  for camera in CAMERAS:
   view=cameras.get(camera) or {};labels=view.get("locator_supervision_xy") or {};corner_map=view.get("opening_corner_supervision_xy") or {};mask_path=view.get("instance_segmentation_path")
   if not mask_path or labels.get("plug") is None or any(corner_map.get(x) is None for x in CORNERS):rejected["missing_mask_or_labels"]+=1;continue
   instances=np.load(mask_path)["instance"];mapping=(view.get("instance_segmentation_info") or {}).get("idToLabels") or {};lookup={int(k):category(v) for k,v in mapping.items()};classes=np.empty(instances.shape,dtype=object);classes[:]= "other"
   for identity,name in lookup.items():classes[instances==identity]=name
   corners=np.asarray([corner_map[x] for x in CORNERS],float);region=roi_mask(int(view["width"]),int(view["height"]),corners);counts={name:float(np.mean(classes[region]==name)) for name in ("rope","plug","gripper","target","robot_other","other")}
   center=corners.mean(0);half=80;x0=max(0,int(center[0])-half);x1=min(classes.shape[1],int(center[0])+half);y0=max(0,int(center[1])-half);y1=min(classes.shape[0],int(center[1])+half);crop=classes[y0:y1,x0:x1];rope=np.argwhere(crop=="rope")
   centroid=[None,None] if not len(rope) else [float(rope[:,1].mean()+x0-center[0]),float(rope[:,0].mean()+y0-center[1])]
   rows.append({"transition_index":ti,"episode_id":ep,"sequence_id":sequence,"global_episode_index":m.get("global_episode_index"),"camera":camera,"phase":phase,"signed_depth_m":depth,"image_path":view["path"],"mask_path":mask_path,"corners":corners.tolist(),"plug_point":labels["plug"],"opening_rope_fraction":counts["rope"],"opening_plug_fraction":counts["plug"],"opening_gripper_fraction":counts["gripper"],"opening_target_fraction":counts["target"],"opening_other_fraction":counts["other"]+counts["robot_other"],"plug_patch_plug_fraction":patch_fraction(classes,np.asarray(labels["plug"]),"plug"),"rope_crop_fraction":float(np.mean(crop=="rope")),"rope_centroid_dx_px":centroid[0],"rope_centroid_dy_px":centroid[1]})
 if not rows:raise RuntimeError("no accepted rows")
 fields=[k for k in rows[0] if k not in ("corners","plug_point")]
 with (a.output_dir/"occlusion_rows.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
 coverage={}
 for camera in CAMERAS:
  selected=[r for r in rows if r["camera"]==camera];coverage[camera]={"row_count":len(selected),"sequence_count":len({r["sequence_id"] for r in selected}),"opening_rope_fraction":quantiles([r["opening_rope_fraction"] for r in selected]),"opening_plug_fraction":quantiles([r["opening_plug_fraction"] for r in selected]),"opening_gripper_fraction":quantiles([r["opening_gripper_fraction"] for r in selected]),"plug_patch_plug_fraction":quantiles([r["plug_patch_plug_fraction"] for r in selected]),"rope_present_in_opening_gt_1pct":sum(r["opening_rope_fraction"]>.01 for r in selected),"rope_heavy_in_opening_gt_15pct":sum(r["opening_rope_fraction"]>.15 for r in selected),"plug_landmark_visible_proxy_gt_5pct":sum(r["plug_patch_plug_fraction"]>.05 for r in selected)}
 by_config={}
 for ep in sorted(allowed):
  selected=[r for r in rows if r["episode_id"]==ep];by_config[ep]={"rows":len(selected),"sequence_ids":sorted({r["sequence_id"] for r in selected}),"rope_opening_mean":None if not selected else float(np.mean([r["opening_rope_fraction"] for r in selected])),"rope_crop_mean":None if not selected else float(np.mean([r["rope_crop_fraction"] for r in selected]))}
 # Keep clear, median and heavily rope-occluded samples for each view.
 panels=[];selected_records=[]
 for camera in CAMERAS:
  candidates=[r for r in rows if r["camera"]==camera];values=np.asarray([r["opening_rope_fraction"] for r in candidates])
  for label,index in (("clear",int(np.argmin(values))),("median",int(np.argmin(np.abs(values-np.median(values))))),("rope_max",int(np.argmax(values)))):
   row=candidates[index];image=Image.open(row["image_path"]).convert("RGB");draw=ImageDraw.Draw(image);corners=row["corners"];draw.line([tuple(x) for x in corners+[corners[0]]],fill="cyan",width=4);x,y=row["plug_point"];draw.ellipse((x-6,y-6,x+6,y+6),outline="magenta",width=4);draw.rectangle((0,0,500,26),fill="black");draw.text((5,6),f"{camera} {label} rope={row['opening_rope_fraction']:.3f} plug={row['opening_plug_fraction']:.3f}",fill="white");panels.append(image.resize((288,256)));selected_records.append({"camera":camera,"selection":label,"episode_id":row["episode_id"],"transition_index":row["transition_index"],"opening_rope_fraction":row["opening_rope_fraction"],"image_path":row["image_path"]})
 sheet=Image.new("RGB",(864,768),"black")
 for i,image in enumerate(panels):sheet.paste(image,((i%3)*288,(i//3)*256))
 sheet.save(a.output_dir/"occlusion_coverage_examples.png")
 report={"schema_version":1,"status":"complete","method":{"visibility_source":"instance-segmentation proxy; no depth buffer saved","mask_sampling":"collector retained masks at decision 1 and every 100 decisions; all RGB frames and projected labels remain available to perception training","opening_region":"projected physical four-corner polygon expanded four pixels","categories":"USD prim-path classification","limitations":["Instance masks cannot resolve subpixel visibility or depth-order at a geometric point.","Opening interior legitimately contains background and later the plug; low target fraction is not itself occlusion.","Manifest varies TCP starts but declares no independent cable seed or cable-shape parameter.","The sparse mask subset describes observed occlusion examples but is too small to prove full-replay prevalence."]},"data":{"accepted_camera_rows":len(rows),"transition_count":len({r['transition_index'] for r in rows}),"configuration_count":len({r['episode_id'] for r in rows}),"sequence_count":len({r['sequence_id'] for r in rows}),"rejected":dict(rejected)},"coverage":coverage,"by_configuration":by_config,"cable_diversity_decision":{"independent_cable_seed_present":False,"adequate_for_cable_generalization":False,"reason":"Only TCP axial/lateral starts vary explicitly; apparent cable occlusion changes over rollout but is not independently randomized across cable reset states."},"montage_samples":selected_records,"sources":{"replay":file_id(a.replay),"manifest":file_id(a.manifest)}}
 (a.output_dir/"occlusion_coverage.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(a.output_dir),"data":report["data"],"coverage":coverage,"cable_diversity":report["cable_diversity_decision"]},indent=2))
if __name__=="__main__":main()

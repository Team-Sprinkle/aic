#!/usr/bin/env python3
"""Merge phase complete guide trajectories with episode grouped splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

CAMERAS=("center_camera","left_camera","right_camera")
CORNER_ORDER=("top_left","top_right","bottom_right","bottom_left")


def polygon(shape,corners):
    image=Image.new("1",(shape[1],shape[0]),0)
    ImageDraw.Draw(image).polygon([(int(round(x)),int(round(y))) for x,y in corners],fill=1)
    return np.asarray(image,dtype=bool)


def visibility_label(view):
    path=view.get("instance_segmentation_path")
    if not path or not Path(path).exists():return None
    instance=np.load(path)["instance"]
    labels=(view.get("instance_segmentation_info") or {}).get("idToLabels") or {}
    plug_ids=[int(k) for k,v in labels.items() if "/Robot/cable/sfp_module/" in str(v)]
    rope_ids=[int(k) for k,v in labels.items() if "/Rope/Rope/link_" in str(v)]
    plug=np.isin(instance,plug_ids);rope=np.isin(instance,rope_ids)
    plug_xy=(view.get("locator_supervision_xy") or {}).get("plug")
    corners=view.get("opening_corner_supervision_xy") or {}
    if plug_xy is None or any(corners.get(k) is None for k in CORNER_ORDER):return None
    x,y=map(float,plug_xy);yy,xx=np.ogrid[:instance.shape[0],:instance.shape[1]]
    disk=(xx-x)**2+(yy-y)**2<=36;visible=np.argwhere(plug)
    nearest=float(np.sqrt(np.min((visible[:,1]-x)**2+(visible[:,0]-y)**2))) if len(visible) else float("inf")
    opening=polygon(instance.shape,[corners[k] for k in CORNER_ORDER])
    rope_fraction=float((rope&opening).sum()/max(1,opening.sum()))
    return {"plug_visible":bool(nearest<=4),"plug_nearest_mask_px":nearest,
            "plug_disk_fraction":float((plug&disk).sum()/max(1,disk.sum())),
            "opening_clear":bool(rope_fraction<.03),"rope_opening_fraction":rope_fraction,
            "plug_mask_pixels":int(plug.sum()),"rope_mask_pixels":int(rope.sum())}


def digest(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while block:=f.read(8<<20): h.update(block)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":h.hexdigest()}


def scalar_geometry(source):
    if not source:return None
    return {key:source.get(key) for key in
            ("signed_depth_m_env0","lateral_error_m_env0","orientation_error_rad_env0","success_geometry_by_env")}


def compact_transition(item):
    """Drop duplicated tensors unused by frozen perception or RPDP labels."""
    item=dict(item)
    for key in ("obs","next_obs"):
        source=item.get(key) or {}
        item[key]={name:source[name] for name in ("state","world_feature") if name in source}
    if item.get("terminal_observation"):
        terminal=dict(item["terminal_observation"]);terminal.pop("images",None)
        item["terminal_observation"]=terminal
    return item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit",type=Path,required=True)
    p.add_argument("--fit-extra",type=Path,action="append",default=[],
                   help="Additional fit-only replay; complete strict-success sequences are appended.")
    p.add_argument("--include-unsuccessful-fit-extra",action=argparse.BooleanOptionalAction,default=False,
                   help="Retain complete fit-extra sequences with recorded teacher targets even when the blended rollout failed.")
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--development",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    merged=[];splits={};audit={};all_ids=set()
    split_sources={"fit":[a.fit,*a.fit_extra],"calibration":[a.calibration],"development":[a.development]}
    for name,paths in split_sources.items():
        source_transitions=[];sequences=[]
        for path_index,path in enumerate(paths):
            path_transitions=torch.load(path,map_location="cpu",weights_only=False)["transitions"]
            source_transitions.extend(path_transitions);current=[]
            is_fit_extra=name=="fit" and path_index>0
            for item in path_transitions:
                ep=((item.get("metadata") or {}).get("causal_episode") or {}).get("episode_id")
                previous_ep=(((current[-1].get("metadata") or {}).get("causal_episode") or {}).get("episode_id")
                             if current else ep)
                if current and ep!=previous_ep:
                    sequences.append((current,is_fit_extra));current=[]
                current.append(item)
                m=item.get("metadata") or {}
                if m.get("terminated") or m.get("truncated"):
                    sequences.append((current,is_fit_extra));current=[]
            if current:sequences.append((current,is_fit_extra))
        successful=[];rejected_incomplete=rejected_unsuccessful=retained_labeled_unsuccessful=0
        for sequence,is_fit_extra in sequences:
            last=sequence[-1];m=last.get("metadata") or {}
            if not (m.get("terminated") or m.get("truncated")):
                rejected_incomplete+=1;continue
            terminal=last.get("terminal_observation") or {};g=terminal.get("insertion_geometry") or {}
            if not bool((g.get("success_geometry_by_env") or [False])[0]):
                target_complete=all(item.get("guide_action") is not None for item in sequence)
                if not (a.include_unsuccessful_fit_extra and is_fit_extra and target_complete):
                    rejected_unsuccessful+=1;continue
                retained_labeled_unsuccessful+=1
            successful.extend(sequence)
        transitions=successful
        ids=[];terminal=success=terminal_retained=terminal_reset_mismatch=0;phases=Counter()
        labeled=[];visibility_derived=visibility_placeholder=0
        for item in transitions:
            m=item.get("metadata") or {};ep=(m.get("causal_episode") or {}).get("episode_id")
            if not ep:raise RuntimeError(f"{name}: transition missing causal episode identity")
            ids.append(ep);g=m.get("causal_insertion_geometry") or {}
            views=((m.get("highres_observation") or {}).get("cameras") or {})
            labels={camera:visibility_label(views.get(camera) or {}) for camera in CAMERAS}
            provenance={}
            for camera,value in list(labels.items()):
                if value is None:
                    # The frozen visibility head is used for inference only;
                    # these placeholders merely keep the frozen-perception
                    # loader from discarding RGB rows. They are never used to
                    # fit or evaluate that already frozen head.
                    labels[camera]={"plug_visible":True,"opening_clear":True,
                                    "rope_opening_fraction":0.,"plug_nearest_mask_px":float("nan"),
                                    "plug_disk_fraction":float("nan"),"plug_mask_pixels":-1,"rope_mask_pixels":-1}
                    provenance[camera]="placeholder_not_used_for_training_or_metrics";visibility_placeholder+=1
                else:
                    provenance[camera]="instance_segmentation";visibility_derived+=1
            item=dict(item);m=dict(m);m["offline_visibility_labels"]=labels
            m["offline_visibility_label_provenance"]=provenance;item["metadata"]=m;labeled.append(item)
            force=torch.as_tensor(m.get("causal_force_xyz_n") or [0.,0.,0.],dtype=torch.float32).norm().item()
            depth=float(g.get("signed_depth_m_env0",-999))
            phases["contact" if force>=10 else "insertion" if depth>=0 else "alignment" if depth>=-.003 else "approach"]+=1
            done=bool(m.get("terminated") or m.get("truncated"))
            if done:
                terminal+=1;retained=item.get("terminal_observation") or {};tg=retained.get("insertion_geometry")
                if not tg:raise RuntimeError(f"{name}: terminal transition missing retained observation")
                terminal_retained+=1;success+=bool((tg.get("success_geometry_by_env") or [False])[0])
                pg=m.get("post_step_insertion_geometry") or {}
                if pg and abs(float(pg.get("signed_depth_m_env0",0))-float(tg["signed_depth_m_env0"]))>1e-6:
                    terminal_reset_mismatch+=1
        transitions=[compact_transition(item) for item in labeled]
        unique=sorted(set(ids));overlap=all_ids.intersection(unique)
        if overlap:raise RuntimeError(f"episode leakage across splits: {sorted(overlap)}")
        all_ids.update(unique);start=len(merged);merged.extend(transitions)
        splits[name]={"episode_ids":unique,"episode_count":len(unique),"transition_count":len(transitions)}
        audit[name]={"sources":[digest(path) for path in paths],"source_transition_count":len(source_transitions),
                     "source_sequence_count":len(sequences),"rejected_incomplete_sequences":rejected_incomplete,
                     "rejected_unsuccessful_sequences":rejected_unsuccessful,
                     "retained_teacher_labeled_unsuccessful_sequences":retained_labeled_unsuccessful,
                     "merged_index_range":[start,len(merged)],"terminal_count":terminal,
                     "retained_terminal_count":terminal_retained,"successful_terminal_count":success,
                     "ordinary_post_step_differs_from_terminal_count":terminal_reset_mismatch,
                     "visibility_labels_from_instance_masks":visibility_derived,
                     "visibility_loader_placeholders":visibility_placeholder,
                     "phase_counts":dict(phases)}
    replay=a.output_dir/"filtered_replay.pt"
    torch.save({"capacity":len(merged),"size":len(merged),"filter":"phase_complete_privileged_guide",
                "saved_size":len(merged),"transitions":merged},replay)
    report={"schema_version":1,"status":"prepared","reserved_final_opened":False,
            "split_policy":{"unit":"complete episode/reset configuration","transition_random_split":False},
            "selection":{"teacher":"Isaac cheatcode_transform","actor_receives_privileged_geometry":False,
                         "terminal_endpoint":"terminal_observation before auto reset",
                         "include_unsuccessful_fit_extra":bool(a.include_unsuccessful_fit_extra),
                         "unsuccessful_fit_extra_semantics":"DAgger state with recorded teacher target, not a successful expert episode"},
            "splits":splits,"audit":audit,"artifacts":{"filtered_replay":str(replay)},
            "sources":{name:[digest(path) for path in paths] for name,paths in split_sources.items()}}
    (a.output_dir/"dataset_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

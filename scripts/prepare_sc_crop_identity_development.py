#!/usr/bin/env python3
"""Freeze independent one-port SC development resets before crop-head fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml

CASES = [
    ("crop_development_p0_a", 0, .143, -.219, 3.108, -.032, (0, 1, 2, 3, 4)),
    ("crop_development_p0_b", 0, .178, -.189, 2.997, .025, (0, 2, 4)),
    ("crop_development_p1_a", 1, .147, -.174, 3.166, .037, (0, 1, 3, 4)),
    ("crop_development_p1_b", 1, .164, -.224, 3.022, -.026, (1, 2, 3)),
]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-port0",type=Path,required=True)
    parser.add_argument("--source-port1",type=Path,required=True)
    parser.add_argument("--prior-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Output exists; preserve prior scenes")
    old=json.loads(args.prior_manifest.read_text())
    seen={r["complete_reset_sha256"] for r in old["scenes"]}
    args.output.mkdir(parents=True)
    records=[]
    for name,port,x,y,yaw,rail,cards in CASES:
        source=args.source_port0 if port==0 else args.source_port1
        dest=args.output/name;dest.mkdir()
        for filename in ("world_audit.sdf","bridge_audit.yaml"):
            shutil.copy2(source/filename,dest/filename)
        cfg=yaml.safe_load((source/"eval_config.yaml").read_text())
        trial=next(iter(cfg["trials"].values()))
        task=next(iter(trial["tasks"].values()))
        assert task["target_module_name"]==f"sc_port_{port}"
        board=trial["scene"]["task_board"]
        board["pose"].update(x=x,y=y,yaw=yaw)
        board[f"sc_rail_{1-port}"]={"entity_present":False}
        board[f"sc_rail_{port}"]["entity_pose"]["translation"]=rail
        for card in range(5):
            if card not in cards:board[f"nic_rail_{card}"]={"entity_present":False}
        identity=json.dumps({"scene":trial["scene"],"robot":cfg.get("robot"),"task":task},sort_keys=True)
        digest=hashlib.sha256(identity.encode()).hexdigest()
        if digest in seen:raise ValueError(f"Duplicate reset: {name}")
        seen.add(digest)
        (dest/"eval_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
        records.append({"name":name,"scene_dir":str(dest),"split":"development",
                        "selected_port":port,"other_port_present":False,"cards":list(cards),
                        "board_xy_yaw":[x,y,yaw],"rail_translation_m":rail,
                        "complete_reset_sha256":digest})
    manifest={"schema":"sc_crop_identity_development/v1","frozen_before_fitting":True,
              "prior_manifest":str(args.prior_manifest),"final_scenes_opened":False,
              "scenes":records}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"scenes":len(records),"output":str(args.output)}))
if __name__=="__main__":main()

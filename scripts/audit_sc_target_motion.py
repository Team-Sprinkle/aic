#!/usr/bin/env python3
"""Compare scored SC teacher setpoints with measured next observation motion.

A far setpoint and small step do not prove a bad demonstration; this audit
identifies where target tracking and contact must be inspected before BC.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(values):
    if not values:return {"count":0}
    ordered=sorted(values)
    return {"count":len(values),"median":statistics.median(values),"p95":ordered[min(len(values)-1,int(.95*len(values)))]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    by_episode=defaultdict(list)
    for line in a.labels.open():
        row=json.loads(line);by_episode[row["episode_index"]].append(row)
    result={"schema":"aic_sc_teacher_target_motion_audit/v1","source":str(a.labels),
            "interpretation":"Teacher target is a controller setpoint, not necessarily the TCP/plug motion realized in the next 50 ms. A small step alone is not a failure label.",
            "phases":{}}
    for phase in ("all","near_port"):
        gaps=[];projections=[];small_motion_large_setpoint=0;negative=0;episodes=set()
        for episode,rows in by_episode.items():
            rows.sort(key=lambda r:r["frame"])
            for current,following in zip(rows,rows[1:]):
                c=current["observed_sc_tip_pose_port_frame"][:3]
                target=current["teacher_sc_tip_target_poses_port_frame"][0][:3]
                after=following["observed_sc_tip_pose_port_frame"][:3]
                if phase=="near_port" and abs(c[2]+.01564)>=.03:continue
                episodes.add(episode)
                direction=[target[i]-c[i] for i in range(3)]
                move=[after[i]-c[i] for i in range(3)]
                gap=math.dist(c,target)
                gaps.append(gap*1000)
                if gap>0.005:
                    projected=sum(x*y for x,y in zip(direction,move))/gap
                    projections.append(projected*1000)
                    negative+=projected<0
                    small_motion_large_setpoint+=math.dist(c,after)<0.001
        result["phases"][phase]={"episodes":len(episodes),"setpoint_gap_mm":summarize(gaps),
            "projected_next_motion_mm_for_setpoint_gt5mm":summarize(projections),
            "fraction_negative_projected_motion_for_setpoint_gt5mm":negative/max(len(projections),1),
            "fraction_next_motion_lt1mm_for_setpoint_gt5mm":small_motion_large_setpoint/max(len(projections),1)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["phases"]["near_port"]))


if __name__=="__main__":main()

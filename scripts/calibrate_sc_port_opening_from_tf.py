#!/usr/bin/env python3
"""Record the Gazebo SC opening relative to its scored receptacle base TF."""

import argparse
import json
from pathlib import Path

from calibrate_sc_tcp_tip_from_tf import compose, inverse, world_pose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--port-index",type=int,choices=(0,1),required=True)
    a=p.parse_args()
    source=json.loads(a.snapshot.read_text())
    tf=source["transforms_at_first_event"]
    base=f"task_board/sc_port_{a.port_index}/sc_port_base_link"
    opening=f"task_board/sc_port_{a.port_index}/sc_port_base_link_entrance"
    xyz,quat=compose(inverse(world_pose(tf,base)),world_pose(tf,opening))
    result={"schema":"aic_sc_port_opening_gazebo_tf/v1","source_bag":source["bag"],
            "source_snapshot":str(a.snapshot),"port_index":a.port_index,
            "base_frame":base,"opening_frame":opening,
            "base_to_opening":{"xyz_m":xyz,"quat_xyzw":quat},
            "training_label_only":True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(a.output)


if __name__=="__main__":main()

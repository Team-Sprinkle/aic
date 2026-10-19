#!/usr/bin/env python3
"""Extract fixed SC board and camera calibration TF from scored source bags.

The output is for *offline scoring* of recorded actor images. The autonomous
actor receives none of these TF values.
"""
import argparse
import json
from pathlib import Path

from inspect_sc_port_tf import extract_edges


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-unreadable',action='store_true')
    a=p.parse_args()
    source={e['case']:e for e in json.loads(a.dataset.read_text())['episodes']}
    rows=[];failures=[]
    for case in a.cases:
        e=source[case]
        try:
            edges=extract_edges(Path(e['scored_bag']),first_seconds=3,include_camera_chain=True)
        except Exception as exc:
            if not a.skip_unreadable:raise
            failures.append({'case':case,'bag':e['scored_bag'],'error':str(exc)})
            print('SKIP',case,str(exc),flush=True)
            continue
        if len(edges)<15:raise ValueError(f'Insufficient TF edges for {case}: {len(edges)}')
        rows.append({'trial':case,'target_module_name':f'sc_port_{e["selected_port"]}',
                     'scene_sha256':e['scene_reset_group'],'source_split':e['split'],
                     'bag':e['scored_bag'],'edges':edges})
        print(case,len(edges),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_sc_actor_pose_edges/v1','rows':rows,'failures':failures,
        'source':'scored source bags, offline labels only','runtime_geometry_input':False},indent=2)+'\n')


if __name__=='__main__':main()

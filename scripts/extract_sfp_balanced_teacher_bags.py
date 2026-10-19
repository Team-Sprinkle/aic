#!/usr/bin/env python3
"""Extract scored ROS streams from completed, rootless SFP teacher controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_sfp_cheatcode_target_pilot import IMAGE, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--progress',type=Path,required=True)
    p.add_argument('--follow',action='store_true',help='Poll completed trials until the plan is done')
    a=p.parse_args()
    planned={Path(r['source_trial']).stem for r in json.loads(a.plan.read_text())['scene_trials']}
    env={**os.environ,'DOCKER_HOST':f'unix:///run/user/{os.getuid()}/docker.sock'}
    while True:
        current=json.loads(a.progress.read_text())['results'] if a.progress.exists() else {}
        for tid in sorted(planned & current.keys()):
            row=current[tid]
            if row['official_score'].get('tier3')!=75:continue
            audit=Path(row['audit']);dest=audit/'scored_stream.json'
            if dest.exists():continue
            cmd=['docker','run','--rm','--net=none','-v',f'{ROOT}:/repo:ro',
                 '-v',f'{audit}:/audit','-w','/repo','--entrypoint','bash',IMAGE,'-lc',
                 'source /ws_aic/install/setup.bash; '
                 'python3 scripts/extract_moveit_training_stream.py /audit /audit/scored_stream.json']
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
            scored=json.loads(dest.read_text())
            if scored['official_tier3']!=75 or len(scored['commands'])<100:
                raise ValueError(f'Invalid scored stream: {tid}')
            print(json.dumps(dict(trial=tid,commands=len(scored['commands']),
                                  cameras={k:len(v) for k,v in scored['cameras'].items()})),flush=True)
        if not a.follow or planned<=current.keys():break
        time.sleep(15)
    print(json.dumps(dict(planned=len(planned),scored_results=len(planned & current.keys()),
                          extracted=sum(Path(current[t]['audit'],'scored_stream.json').exists()
                                        for t in planned & current.keys()))),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Host-extract completed public-anchor scored bags while collection continues."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--follow',action='store_true')
    a=p.parse_args()
    plan=json.loads((a.collection/'plan.json').read_text())
    names={r['name'] for r in plan['scenes']}
    env=os.environ.copy();env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
    env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
    done=set()
    while True:
        try:progress=json.loads((a.collection/'progress.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):progress={}
        for name,item in progress.items():
            if name not in names or name in done or item.get('returncode')!=0:continue
            audit=Path(item['audit']);destination=audit/'scored_stream.json'
            if not destination.exists():
                cmd=[str(ROOT/'.pixi/envs/default/bin/python'),
                     str(ROOT/'scripts/extract_moveit_training_stream.py'),
                     str(audit),str(destination)]
                subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            done.add(name)
            print(json.dumps(dict(extracted=name)),flush=True)
        if not a.follow or names<=progress.keys():break
        time.sleep(15)


if __name__=='__main__':main()

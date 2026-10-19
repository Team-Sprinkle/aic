#!/usr/bin/env python3
"""Host-extract completed scored SFP paired-control bags, optionally following collection."""
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
    planned={r['name'] for r in plan['scenes']}
    env=os.environ.copy()
    env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
    env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
    while True:
        progress_path=a.collection/'progress.json'
        progress=json.loads(progress_path.read_text())['results'] if progress_path.exists() else {}
        for name,item in progress.items():
            if name not in planned or item['official_score'].get('tier3',0)<75:continue
            audit=Path(item['audit']);stream=audit/'scored_stream.json'
            if stream.exists():continue
            cmd=[str(ROOT/'.pixi/envs/default/bin/python'),
                 str(ROOT/'scripts/extract_moveit_training_stream.py'),str(audit),str(stream)]
            subprocess.run(cmd,cwd=ROOT,env=env,check=True,stdout=subprocess.DEVNULL)
            print(json.dumps(dict(name=name,extracted=str(stream))),flush=True)
        if not a.follow or planned<=progress.keys():break
        time.sleep(15)
    print(json.dumps(dict(planned=len(planned),completed=len(planned & progress.keys()),
        extracted=sum((Path(progress[x]['audit'])/'scored_stream.json').exists()
                      for x in planned & progress.keys()))),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Extract causal images, state and commands from safe corrective teacher bags."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_sfp_cheatcode_target_pilot import IMAGE, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--expect', type=int, choices=(10,20), required=True)
    p.add_argument('--follow', action='store_true')
    a = p.parse_args()
    env = {**os.environ,'DOCKER_HOST':f'unix:///run/user/{os.getuid()}/docker.sock'}
    while True:
        progress_path = a.collection / 'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        for row in progress.values():
            if not row['accepted']:
                continue
            audit = Path(row['audit'])
            dest = audit / 'scored_stream.json'
            if dest.exists():
                continue
            command = ['docker','run','--rm','--net=none',
                       '-v',f'{ROOT}:/repo:ro','-v',f'{audit}:/audit',
                       '-w','/repo','--entrypoint','bash',IMAGE,'-lc',
                       'source /ws_aic/install/setup.bash; '
                       'python3 scripts/extract_moveit_training_stream.py /audit /audit/scored_stream.json']
            subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
            stream=json.loads(dest.read_text())
            if len(stream['commands'])<100:
                raise ValueError(f'Too few recorded commands: {audit}')
            print(json.dumps(dict(trial=row['trial'],commands=len(stream['commands']),
                                  cameras={k:len(v) for k,v in stream['cameras'].items()})),flush=True)
        if not a.follow or len(progress)>=a.expect:
            break
        time.sleep(15)
    print(json.dumps(dict(completed=len(progress),accepted=sum(v['accepted'] for v in progress.values()),
                          extracted=sum(Path(v['audit'],'scored_stream.json').is_file()
                                        for v in progress.values() if v['accepted']))),flush=True)


if __name__=='__main__':
    main()

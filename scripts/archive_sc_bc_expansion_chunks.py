#!/usr/bin/env python3
"""Extract SC scored edges and archive completed collection groups as they arrive."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(command):
    env = dict(os.environ, DOCKER_HOST='unix:///run/user/1008/docker.sock')
    subprocess.run(command, check=True, env=env)


def process(row, manifest_dir, archive, container):
    root = Path(row['root'])
    if row['complete']:
        for script, output in (
            ('extract_sc_native_batch_edges.py', root / 'edges_joined.json'),
            ('extract_sc_dynamic_grasp_calibration.py', root / 'dynamic_grasp_calibration.json'),
        ):
            if output.exists():
                continue
            args = (f'python3 scripts/{script} {root} {output}' if 'batch_edges' in script
                    else f'python3 scripts/{script} --edges {root}/edges_joined.json --output {output}')
            run(['docker', 'exec', container, 'bash', '-lc',
                 'source /opt/ros/kilted/setup.bash && '
                 'source /ws_aic/install/setup.bash && '
                 'cd /data1/chmin/yj/ws_aic/src/aic && ' + args])
    manifest = manifest_dir / (root.name + '-mcap-archive.json')
    if not manifest.exists():
        run(['python3', 'scripts/archive_sc_scored_bags.py',
             '--root', str(root / 'eval_collection'), '--archive', str(archive),
             '--manifest', str(manifest), '--remove-original-after-verification'])
    return str(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--container', default='aic_act_all_eval_20260918')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    while True:
        path = args.source / 'chunk_results.json'
        try:
            rows = json.loads(path.read_text()) if path.exists() else []
        except json.JSONDecodeError:
            rows = []
        for row in rows:
            if row['root'] in seen:
                continue
            try:
                manifest = process(row, args.manifest_dir, args.archive, args.container)
            except (subprocess.CalledProcessError, OSError) as error:
                print(json.dumps({'root': row['root'], 'error': str(error)}), flush=True)
                continue
            print(json.dumps({'root': row['root'], 'archive': manifest,
                              'complete': row['complete']}), flush=True)
            seen.add(row['root'])
        if not args.follow or (args.source / 'collection_done').exists():
            break
        time.sleep(30)


if __name__ == '__main__':
    main()

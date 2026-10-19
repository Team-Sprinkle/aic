#!/usr/bin/env python3
"""Alias saved nonfinal SFP RGB pairs without exposing scene/target identities."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sim-seconds', type=float, default=11.0)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists; preserve the prior blind audit')
    episodes = sorted(x for x in args.source.glob('teacher_card*') if (x/'frame_manifest.csv').is_file())
    if len(episodes) != 10:
        raise ValueError(f'Expected 10 saved development scenes, found {len(episodes)}')
    random.Random(614802).shuffle(episodes)
    args.output.mkdir(parents=True)
    aliases = []
    blind = []
    for index, episode in enumerate(episodes):
        alias = f'case_{index:02d}'
        frames = list(csv.DictReader((episode/'frame_manifest.csv').open()))
        images = {}
        for camera in ('center', 'right'):
            nearest = min((r for r in frames if r['camera'] == camera),
                          key=lambda r: abs(int(r['sim_ns'])/1e9-args.sim_seconds))
            if abs(int(nearest['sim_ns'])/1e9-args.sim_seconds) > 0.1:
                raise ValueError(f'{episode} {camera}: no 11-second frame')
            source = episode/'frames'/nearest['file']
            if not source.is_file():
                raise FileNotFoundError(source)
            target = args.output/f'{alias}_{camera}.jpg'
            target.symlink_to(source)
            images[camera] = {'alias': target.name, 'sim_ns': int(nearest['sim_ns']),
                              'source': str(source),
                              'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        aliases.append({'alias': alias, 'episode': episode.name, 'images': images})
        blind.append({'alias': alias, 'center': f'{alias}_center.jpg',
                      'right': f'{alias}_right.jpg'})
    (args.output/'sealed_mapping.json').write_text(json.dumps(aliases, indent=2)+'\n')
    (args.output/'blind_manifest.json').write_text(json.dumps(blind, indent=2)+'\n')
    print(json.dumps({'cases': len(blind), 'blind_manifest': str(args.output/'blind_manifest.json'),
                      'mapping_sealed': True}))


if __name__ == '__main__':
    main()

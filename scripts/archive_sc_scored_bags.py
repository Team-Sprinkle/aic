#!/usr/bin/env python3
"""Archive completed scored Gazebo MCAPs with verified lossless restoration."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--remove-original-after-verification', action='store_true')
    a = p.parse_args()
    bags = sorted(a.root.rglob('*.mcap'))
    if not bags:
        raise ValueError(f'No MCAPs under {a.root}')
    a.archive.mkdir(parents=True, exist_ok=True)
    files = []
    for source in bags:
        target = a.archive / (source.name + '.zst')
        if target.exists():
            raise FileExistsError(target)
        original = digest(source)
        with target.open('wb') as out:
            subprocess.run(['zstd', '-T2', '-1', '-c', str(source)], check=True, stdout=out)
        restored = hashlib.sha256()
        stream = subprocess.Popen(['zstd', '-dc', str(target)], stdout=subprocess.PIPE)
        for chunk in iter(lambda: stream.stdout.read(1024 * 1024), b''):
            restored.update(chunk)
        if stream.wait() or restored.hexdigest() != original:
            target.unlink(missing_ok=True)
            raise RuntimeError(f'Archive verification failed: {source}')
        files.append({'original_path': str(source), 'original_sha256': original,
                      'compressed': str(target), 'compressed_bytes': target.stat().st_size,
                      'restore_command': f"zstd -dc '{target}' > '{source}'"})
        if a.remove_original_after_verification:
            source.unlink()
    report = {'schema': 'sc_scored_mcap_archive/v1',
              'verified_before_source_removal': True, 'files': files}
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'archived': len(files), 'manifest': str(a.manifest)}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze posthoc SFP wrong-identity examples before a bounded repair."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def load(path):
    return {(r['episode'],r['time_s']):r for line in path.open() if (r:=json.loads(line))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--prediction',action='append',nargs=2,metavar=('NAME','JSONL'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve frozen failure manifest')
    labels=load(args.labels/'rows.jsonl')
    rows=[];counts=Counter()
    for name,path in args.prediction:
        for key,pred in load(Path(path)).items():
            if not pred['accepted']:continue
            label=labels[key]
            requested=label['selected_identity']
            nearest=pred['nearest_identity_label_only']
            target=next(t for t in label['targets'] if t['identity']==requested)
            tcp=np.asarray(label['state'][:3]);port=np.asarray(target['position_base_m'])
            distance=float(np.linalg.norm(tcp-port)*1000)
            phase='far' if distance>100 else 'approach' if distance>40 else 'near'
            record=dict(model=name,episode=key[0],time_s=key[1],requested=requested,
                        nearest_identity_label_only=nearest,wrong=nearest!=requested,
                        error_mm=pred['error_mm'],reprojection_px=pred.get('reprojection_px'),
                        tcp_to_requested_port_mm=distance,phase_label_only=phase,
                        projected_inframe_label_only={c:bool(v[3]) for c,v in target['cameras'].items()},
                        original_images=label['images'],scene_group=label['scene_group'])
            rows.append(record)
            counts[(name,phase,'wrong' if record['wrong'] else 'correct')]+=1
    args.output.mkdir(parents=True)
    with (args.output/'accepted_predictions.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    failures=[r for r in rows if r['wrong']]
    with (args.output/'wrong_predictions.jsonl').open('w') as f:
        for r in failures:f.write(json.dumps(r,separators=(',',':'))+'\n')
    summary=dict(schema='sfp_fresh_identity_failure_manifest/v1',accepted=len(rows),
                 wrong=len(failures),counts={'/'.join(k):v for k,v in sorted(counts.items())},
                 wrong_by_request=dict(Counter(r['requested'] for r in failures)),
                 projected_inframe_is_not_occlusion_visibility=True,
                 labels_used_posthoc_only=True,final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()

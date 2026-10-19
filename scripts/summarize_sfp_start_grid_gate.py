#!/usr/bin/env python3
"""Summarize the bounded SFP requested-code/actual-start transport gate."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads((a.collection/'plan.json').read_text())
    progress=json.loads((a.collection/'progress.json').read_text())
    labels=json.loads((a.labels/'manifest.json').read_text())
    by_name={x['name']:x for x in plan['scenes']}
    admitted={x['name']:x for x in labels['admitted']}
    attempts=[]
    for name,outcome in progress.items():
        scene=by_name[name]
        if scene['family']!='sfp':raise ValueError('SFP-only summary')
        start=outcome.get('start_variant_audit',{})
        row=dict(name=name,split=scene['split'],board_anchor=scene['official_anchor'],
                 card_count=scene['card_count'],target_card=scene['target_card'],
                 target_port=scene['target_port'],official_tier3=outcome['official_tier3'],
                 full_insert=outcome['official_tier3']==75,
                 actual_start_valid=bool(start.get('valid')),
                 transport_admitted=name in admitted,
                 transport_chunks=admitted.get(name,{}).get('transport_chunks',0),
                 transport_only_prefix=admitted.get(name,{}).get('transport_only_prefix',False))
        attempts.append(row)
    codes={f"{x['target_card']}/{x['target_port']}" for x in attempts if x['transport_admitted']}
    passed=(len(attempts)>=20 and sum(x['transport_admitted'] for x in attempts)>=16
            and len(codes)==10 and all(x['actual_start_valid'] for x in attempts))
    summary=dict(schema='aic_sfp_start_grid_gate/v1',attempts=len(attempts),
                 admitted=sum(x['transport_admitted'] for x in attempts),
                 full_insertions=sum(x['full_insert'] for x in attempts),
                 partial_or_no_insertions=sum(not x['full_insert'] for x in attempts),
                 valid_actual_starts=sum(x['actual_start_valid'] for x in attempts),
                 admitted_request_codes=sorted(codes),gate_passed=passed,
                 gate='>=16/20 admitted safe prefixes, all 10 card/port codes, all actual starts verified',
                 final_scenes_opened=False,attempt_details=attempts)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='attempt_details'}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Aggregate the fixed SC/SFP development handoff gate without fitting thresholds."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sc-decision',type=Path,required=True)
    p.add_argument('--sc-replay-root',type=Path,required=True)
    p.add_argument('--sfp-old',type=Path,required=True)
    p.add_argument('--sfp-new',type=Path,required=True)
    p.add_argument('--sfp-validation',type=Path,required=True)
    p.add_argument('--sc-hidden',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    decision=json.loads(a.sc_decision.read_text())
    sc=[]
    for r in decision['sc_frozen_traces']:
        stop=r['visual']['stop_s']
        if stop is None:continue
        name=r['trace']
        file=(a.sc_replay_root/('wrong_route_stop_replay_v1.json' if 'v7_fastresize' in name else
                               'valid_route_stop_replay_v1.json' if 'v11_recent_stop' in name else
                               f'{name}_stop_replay_v1.json'))
        replay=json.loads(file.read_text())
        sc.append({'trace':name,'posthoc_false_stop':r['visual']['false_handoff'],
                   'posthoc_orientation_error_deg':r['visual']['orientation_deg'],
                   'posthoc_unsafe_route':r['visual']['unsafe_route'],
                   'identity_accept_frames':replay['accepted'],
                   'identity_total_frames':replay['total'],
                   'identity_accept_near_stop':bool(replay['rows'] and replay['rows'][-1]['all_three_accept']),
                   'replay':str(file)})
    old=json.loads(a.sfp_old.read_text());new=json.loads(a.sfp_new.read_text())
    val=json.loads(a.sfp_validation.read_text());hidden=json.loads(a.sc_hidden.read_text())
    sc_bad=[r for r in sc if r['posthoc_false_stop'] and r['identity_accept_near_stop']]
    valid=[r for r in sc if not r['posthoc_false_stop']]
    result={'schema':'aic_frozen_selected_port_handoff_gate/v1',
        'decision':'fail_do_not_start_shadow_handoff',
        'gate_rule':'zero accepted known unsafe proposals and >=95% clear-arrival retention',
        'thresholds_frozen':True,'sc':{'visual_stops':sc,
            'known_unsafe_identity_accepted':len(sc_bad),
            'valid_stop_identity_retained':sum(r['identity_accept_near_stop'] for r in valid),
            'valid_stops':len(valid),
            'provisional_hidden_triplets':hidden['counts']['triplets'],
            'provisional_hidden_combined_accept':hidden['counts']['combined_accept']},
        'sfp':{'old_saved_proposals':old['counts'],
               'new_saved_proposals':new['counts'],
               'grouped_validation':val['metrics']['validation']},
        'blockers':['sc_wrong_posture_accepted','sfp_neighbor_point_false_accepts',
                    'uncertified_occlusion_labels','sfp_orientation_above_3deg',
                    'complete_live_latency_unmeasured'],
        'runtime_ground_truth_input':False,
        'truth_for_posthoc_scoring_only':True,'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'decision':result['decision'],'sc_unsafe_accept':len(sc_bad),
        'sc_valid_retained':result['sc']['valid_stop_identity_retained'],
        'sc_valid_stops':len(valid),'sfp_old':old['counts'],'sfp_new':new['counts']},indent=2))


if __name__=='__main__':main()

"""Strict posthoc acquisition gate; never deployment input or an insertion score."""
import math


def qualified_acquisition(result):
    t=result.get('terminal') or {}
    latency=result.get('inference_p50_p95_p99_ms')
    values=[result.get('accepted_opening_error_mm'),t.get('axial_mm'),
            t.get('lateral_mm'),t.get('orientation_error_deg')]
    if (any(v is None or not math.isfinite(v) for v in values) or
            not latency or len(latency)!=3 or not math.isfinite(latency[1])):
        return False
    return bool(result.get('rule_visual_acquisition') and
                result.get('wrong_accepted_port') is False and
                result.get('no_scored_harm') is True and
                values[0]<=1 and -180<=values[1]<=-130 and
                values[2]<=25 and values[3]<=15 and latency[1]<300)


def summarize(results, cases, *, minimum_fraction=.9, minimum_per_group=4):
    if len({r['case'] for r in results})!=len(results):
        raise ValueError('Duplicate evaluation case; do not count twice')
    by_name={c['case']:c for c in cases}
    if any(r['case'] not in by_name for r in results):raise ValueError('Unknown evaluation case')
    groups={}
    for r in results:
        c=by_name[r['case']];key=f"{c['domain']}/n{c['card_count']}/p{c['port']}"
        group=groups.setdefault(key,dict(completed=0,qualified=0))
        group['completed']+=1;group['qualified']+=qualified_acquisition(r)
    qualified=sum(qualified_acquisition(r) for r in results)
    wrong=sum(r.get('wrong_accepted_port') is True for r in results)
    harm=sum(r.get('no_scored_harm') is not True for r in results)
    expected_groups={f"{c['domain']}/n{c['card_count']}/p{c['port']}" for c in cases}
    per_group_pass=all(groups.get(k,{}).get('qualified',0)>=minimum_per_group for k in expected_groups)
    return dict(completed=len(results),planned=len(cases),qualified=qualified,wrong=wrong,
                harm_or_missing_score=harm,groups=groups,
                passed=len(results)==len(cases) and qualified>=math.ceil(minimum_fraction*len(cases))
                and wrong==0 and harm==0 and per_group_pass,
                all_group_counts_pass=per_group_pass,final_scenes_opened=False,
                scope='Rule transport and causal requested-port acquisition only; insertion not evaluated')

# Handoff improvement work: October 8, 2026

I tested transport arrival and visual approval separately, using new nonfinal SC Gazebo scenes and saved SFP autonomous routes. All runs were handoff-only: no insertion actor, RL, privileged control input, or final scenes.

## What worked

- The frozen SC visual handoff approved the **requested port on 4/4 new two-port starts** spanning one, four and five cards and both SC ports. Posthoc opening errors were **0.092–0.241 mm**, and none had scored contact. Three runs first rejected a real wrong tool orientation, corrected it, and then approved the handoff.
- On six saved SFP autonomous routes, only **one** entered the intended stop corridor. On that route, requested-port position plus independently predicted orientation identified both sampled corridor visits, with no false stop candidates on the other five routes. This is a useful shadow result, not enough arrival coverage for an automatic switch.

## What failed

One SC one-port run got within **0.931 mm** of its desired standoff but the visual checker rejected every inspection because its close-view board-line orientation estimate was unavailable. I added a guarded option to reuse an earlier, request-bound board orientation. It remains **off by default**. On two independent one-port scenes, the one scene that reached inspection passed with the option both off and on; both approvals used a *fresh* board estimate. The other scene never reached inspection in either run, finishing **281 mm / 187 mm** from the standoff. Thus the new memory path has **no demonstrated live benefit**. A saved-image replay showed its safety checks run, but that is not a live repair result.

A simple SFP distance-to-saved-port stop is unsafe: it would have falsely proposed three stops on a wrong-lateral route. The predicted port-frame lateral check removed those proposals in this small replay. Full camera-to-command latency with the future insertion actor is still unmeasured; only separate model and live decision times were measured.

## Decision and next bounded test

Automatic handoff stays disabled. SC two-port visual approval is promising; its remaining one-port orientation miss needs a deterministic native-image replay that proves a corrected decision **using** the saved orientation without approving an absent port. Limit that replay to the existing recordings and about **two hours** of diagnosis; run new SC arrivals only if the fixed sequence changes the decision safely. For SFP, prioritize requested-port transport arrival before further stop-head training: one corridor visit among six routes cannot establish stop recall. A later autonomous development gate should require at least **2/3 safe buffered arrivals**, no increase in contact, and correct requested-port approval with complete p95 inference below **300 ms**. Do not open final scenes or start RL while these gates remain open.

The [full experiment record](../experiments/2026-10-08-handoff-only-shadow-and-memory.md), [machine decision](../experiments/2026-10-08-handoff-only-shadow-and-memory.decision.json), [SC matched results](/home/nas/chmin/aic_archives/2026-10-08_handoff_only_new_sc_v1/independent_four_summary.json), and [SFP shadow predictions](/home/nas/chmin/aic_archives/2026-10-08_handoff_only_sfp_anchor_orientation_v1.json) contain the commands, configurations, metrics and preserved failures. Python compilation and shell syntax checks passed. The preexisting Isaac job was left untouched; project usage stayed within four GPUs.

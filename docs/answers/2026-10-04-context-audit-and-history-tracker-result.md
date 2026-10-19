# Result: did the models have enough context?

I ran the proposed audit on the **ten existing, nonfinal SFP teacher-view episodes** (581 synchronized three-camera observations). I reviewed representative original images without truth markers, then replayed the frozen perception predictions with an observation-only, causal history rule. No model was retrained and no new rollout or final scene was used.

The raw views expose two different problems. In the first card-1/port-0 view, the center camera shows only part of the card row, so a close-up alone does not establish the card's number. In the card-4/port-1 scene only three cards are installed, in slots **1, 2 and 4**; counting visible cards cannot identify “card 4.” But the first card-2/port-0 view shows all five cards, and the frozen predictor still struggles. Thus both missing global context **and** failure to use available context matter. These are visual-review examples, not certified visibility labels; simulator projection merely says where an opening would fall in the picture.

For the replay, I allowed an earlier accepted prediction from the frozen **selected-port** model to establish an anchor. Later frozen models could confirm that same 3D point within 5 mm, using measured TCP motion and expiring stale anchors after 150 mm without confirmation. Simulator geometry was used only **after** decisions, to score them.

| Check | Episodes acquired at any time | Acquired within 3 s | Accepted frames | Wrong accepted frames |
| --- | ---: | ---: | ---: | ---: |
| Current-frame selected-port model | 5/10 | 3/10 | 101 | 0 |
| Same model plus causal anchor tracking | 5/10 | 3/10 | 153 | 0 |

History preserved a previously identified port through **52 additional later frames**, but it found **no new target** and did not improve early coverage. The zero wrong accepts are confined to these ten reused development scenes; they are not a deployment guarantee. This confirms our distinction: history is useful once the requested port has been identified reliably, but cannot solve a missing or mistaken initial identity. A looser earlier test that merely required consecutive agreement retained a stable wrong-card decision.

**Decision:** keep automatic transport-to-insertion handoff off. The next bounded repair is to identify numbered **board slots**, including empty ones, from a full original-resolution view, then use local crops for precise port position. First verify in raw RGB whether a board/slot reference is actually visible. Evaluate any trained repair on new complete nonfinal scene groups, requiring zero wrong accepted ports and useful early coverage (the previous target was at least 9/10), then measure complete live p95 below 300 ms. If the visual evidence cannot establish the slot, change the observation posture rather than asking a tracker to guess.

The [full experiment record](../experiments/2026-10-04-sfp-context-sufficiency-audit.md) contains the exact command, source episodes, visual-review details and artifact paths. The replay source is [audit_sfp_selected_anchor_tracking.py](../../scripts/audit_sfp_selected_anchor_tracking.py). The machine results are `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/selected_anchor_tracking_audit_v1/summary.json` and `decisions.jsonl`.

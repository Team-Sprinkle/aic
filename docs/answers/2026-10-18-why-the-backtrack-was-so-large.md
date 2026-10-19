# Why did the robot backtrack so far?

You are right to question this. We have **two separate recovery mechanisms**, and I described them as if they were one:

1. **Near the port:** after a measured contact or stall, the insertion supervisor is configured to retrace about **12 mm** of measured TCP motion (possibly farther if the plug still appears inside the opening), try a **3 mm sideways move**, and then resume the insertion controller. This is the local behavior closest to your short-backoff idea. It is not strictly limited to sideways motion forever: the controller must advance again to insert. The frozen official run did not activate this mechanism at all.
2. **During SC transport:** a different rule controller can reverse the **entire recorded route** to a checkpoint and try the opposite corridor. In the official SC trial it reversed **322 measured waypoints**, consuming about **44.75 simulated seconds**. This was not a few-millimeter local retreat. Its purpose was originally a possible cable or card route obstruction, where a few millimeters would not change the cable's path around the cards.

## What actually triggered the large retreat

The frozen trace shows `sample_visual_acquisition_budget` at simulation time **86.15 s**, immediately before `transport_route_backtrack_requested`. The requested port was not visually acquired within the configured time. The code incorrectly counted that timeout as physical blockage and logged the generic reason `force_or_measured_blockage`. An offline contact record exists nearby, but it does **not** prove contact triggered the retreat. My earlier description of this as a blockage-triggered reset was wrong.

I removed the visual-acquisition timeout from the full-route retreat trigger in the live source. The frozen result stays frozen. A subsequent matched released-YAML rerun confirms **zero route backtracks**, but still scores only **98.856989/300 with 1/3 insertions**; SC cannot acquire a valid requested-port lock. See the [October 19 experiment](../experiments/2026-10-19-visual-timeout-route-retreat-fix.md). A visual timeout now leaves the robot at the conservative transport hold and ends that attempt; a different bounded visual-acquisition repair should be developed on fresh starts. Force stop or genuine measured no-progress still has the route-reset path, because a cable caught around cards can require changing the route itself.

## The behavior to test next

For a plug blocked at the opening, use a measured short retreat sufficient to clear the plug, then a small lateral search and reapproach. For a missed visual lock, retry observation or adjust the view locally; do not reverse the entire SC route. For a persistent cable/card snag, first unload safely, then consider a deeper measured retreat and a different corridor only if the local attempt cannot free it. That escalation should be bounded and logged with its real trigger. The exact short retreat distance should be set from measured clearance, not guessed as a fixed few millimeters when the plug is already partly inserted.

Evidence: [frozen official evaluation](../experiments/2026-10-18-frozen-V56-official-evaluation.md), `rollout/runtime_trace.jsonl` in `/home/nas/chmin/aic_archives/2026-10-18_V56_official_qualification_evaluation_v1/`, and the current `RunSampleTransportRecovery.py`, `hierarchical_recovery.py`, and `RunRuleThenSharedRecovery.py` implementations.

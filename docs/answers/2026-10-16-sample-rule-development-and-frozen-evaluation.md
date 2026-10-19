# Sample-based transport changes and evaluation

The new rules use the pinned public sample and pre-release specification. No post-release evaluation placements were used to tune them.

## What changed

- Sample-derived SC bypass and SFP staged transport, with early requested-port inspection and strict fresh visual handoff near150mm.
- Bounded viewpoint search when precise pose checks fail; no relaxed pose-acceptance threshold.
- For SC force/blockage failures, measured-path backtracking to an earlier checkpoint, followed by an opposite-side route and fresh visual acquisition. More backward clearance or a lower transfer alone did not solve the cable blockage.

I also fixed a real backtracking progress bug: the robot had reversed but the bookkeeping failed to recognize progress past tiny waypoints. A subsequent live test completed the retreat and handed off at155mmstand-off/2mmlateral, without scored harm. This is rule recovery, not RL or insertion.

## Development results

| Controller/family | Qualified handoffs |
| --- | --- |
| SFP | **22/22** |
| SC, original fixed side | **6/7** |
| SC, opposite fixed side | **4/7** |
| SC, adaptive controller | **7/7** |

Backtracking did not trigger in the seven-case adaptive SC screen. Its functionality was exercised in the separate successful recovery trial, so the7/7 result alone must not be attributed to backtracking or treated as a population reliability guarantee.

## Full evaluation is running

Rules and vision weights are frozen for **130 episodes**:

- **24 SC:** 0–5 cards × two ports × two repetitions.
- **100 SFP:** 1–5 cards × five NIC target identities × two ports × two repetitions.
- **Six exposed official-scene repeats**, evaluated only after freezing and kept separate from development.

The first two completed held-out runs yielded one pass and one failure. The failure remains in the results; no rules are changed during the sweep. **Overall reliability is not established yet.** This evaluates transport/handoff, with no insertion actor or new model training. Reserved internal final scenes remain sealed.

The job continues without further input and updates STATUS.md and per-setting reports automatically.

Live artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3/`.

Recovery video: `/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/SC_reverse_progress_repair_retest_v7/development_sc_n1_p0_r0__94/five_cameras_20fps_grid_active.mp4`.

[Detailed experiment record](../experiments/2026-10-16-sample-conservative-rule-transport.md).

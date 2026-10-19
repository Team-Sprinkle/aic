# Four-rollout insertion and jerk results

Ran **four Gazebo rollouts**: two repeats each of one SC and one SFP development configuration. No RL, backtracking or privileged policy inputs; no additional training.

## Insertion results

| Episode | Depth | Strict full insertion |
| --- | ---: | --- |
| SC1 |1.95mm|No|
| SC2 |1.99mm|No|
| SFP1 |46.75mm|**Yes**|
| SFP2 |37.43mm|No: stopped short|

Successful SFP1 also achieved **0.48mm lateral error and0.46° orientation error**, with no scored harm. This is promising, but **1/4 is not reliable performance**, and these repeats cover only two configurations.

**Simulator caveat:** Gazebo awarded completion to both shallow SC contacts and froze the robot. I did not count these as physical full insertions or use their stationary endings as evidence of smooth control. Earlier whole-episode smoothness numbers overstated the SC improvement.

## Does jerk remain?

**Yes, although SFP's large back-and-forth motion is substantially reduced.**

- During the first15seconds, before freezing, SFP plan-boundary target-jump p95 fell from **15.4mm to3.2–4.4mm** versus the previous matched H4 run.
- Subsequent plans still occasionally jump roughly **5–9mm** across the four runs.
- A larger discontinuity occurs at handoff: the last visual-controller target and first actor target differ by **14–19mm forSC**, **4–8mm forSFP**.
- SFP can still advance rapidly after a contact stall, reaching measured TCP speeds around **54–59mm/s**. Smaller command oscillations do not establish jerk-free physical motion.

SC early smoothness was inconsistent before simulator freezing. Its image-estimated grasp orientation was also wrong by about **2.5°**; the actual grasp angle lay outside the estimator's training range. That is a confirmed perception gap, not a proven sole cause of SC failure.

## Suggested fixes

1. **Make the handoff and consecutive plans continuous.** Bridge from the previous commanded reference to the new complete target with velocity/acceleration limits. Preserving observation history alone did not preserve command continuity. This is my first bounded test, with actor weights fixed.
2. **Make forward execution aware of contact and measured progress.** In SFP1, the commanded point stayed roughly40mm ahead while the plug was blocked. Bound that accumulating demand while preserving steady force compensation, so release does not produce a sudden advance.
3. **Repair SC grasp-angle coverage.** Include the missing allowed rotations and verify live accuracy before expecting reliable pre-contact alignment. More history or smoothing alone cannot correct a consistently wrong grasp estimate.

SFP2's early stop additionally warrants checking teacher phase/controller-bias context before another actor retrain.

## Videos

- [SFP full insertion](/home/nas/chmin/aic_archives/2026-10-17_full_connector_live_probe_v1/videos/sfp_1_full_insertion.mp4)
- [SFP partial insertion](/home/nas/chmin/aic_archives/2026-10-17_full_connector_live_probe_v1/videos/sfp_2_partial_insertion.mp4)
- [SC shallow contact before simulator freeze](/home/nas/chmin/aic_archives/2026-10-17_full_connector_live_probe_v1/videos/sc_1_before_freeze.mp4)

All four videos, measurements and proposed fixes are indexed in the [experiment record](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-17-full-connector-insertion-jerk-probe.md). No further jobs are queued.

Saved response: [four-rollout results](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-17-four-rollout-insertion-and-jerk-results.md).

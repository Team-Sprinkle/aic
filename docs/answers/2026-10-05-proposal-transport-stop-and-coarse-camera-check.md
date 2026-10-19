# Proposal: improve transport and the coarse camera check together

**Yes—work on both, but measure their jobs separately.** Transport must move toward the *requested* port. Its learned head may propose “pause and inspect.” The coarse camera check must decide whether that proposal really concerns the requested port and whether the current view is good enough for precise inspection. A positive coarse check would still not authorize insertion; the original-image refiner and final posture/force check come afterward.

```text
Transport actor ──movement──> new camera views
       │
       └──pause proposal──> coarse requested-port check
                                  ├── reject: resume / seek another view
                                  └── accept: hold and run precise refiner
```

## 1. Repair what each model is asked to learn

For **transport**, use original camera images with spatial card/port information and the requested task code. The present actor sees globally pooled image vectors; a separate locator could find the port in all three recent sparse-card routes, yet the actor missed the approach region in all three. I would compare a matched actor that receives an observation-derived requested-target cue—such as selected spatial tokens or a coarse relative-port vector with an explicit valid/invalid mask—against the present actor. It must still work when the cue is invalid. Train on the same episode-grouped teacher decisions and update budget; evaluate causal early route choices and full autonomous arrival, not imitation loss alone. Do not provide true simulator port pose to the actor.

For the **transport probability head**, define its target carefully. I recommend that it predict **“is this a useful, safe place to pause for a fresh camera inspection soon?”**, not “is insertion safe now?” The previous SFP head was trained separately on a frozen actor and was timely in only 2/10 teacher episodes under its three-decision rule. Merely adding more updates or making its output probabilistic is unlikely to fix narrow or poorly timed labels. First audit the actual safe pause interval in teacher and failed actor routes, including wrong-route, wrong-port, misoriented and high-force states. Label a sufficiently long *inspect opportunity window* where appropriate, plus explicit unsafe negatives. A sigmoid score is only a ranking signal until episode-level calibration and a frozen threshold are checked. The broad base-coordinate prior can veto impossible locations but cannot turn the score into a safe handoff.

For the **coarse camera check**, use current original-resolution images, the requested card/port code, measured TCP and distance-sampled visual history. It should output a requested-port region **or abstain**, and a separate accept/reject decision for the pause proposal. Train against reviewed clear, clipped-board, occluded, empty-slot and neighboring-port cases. Geometric projection can supervise location, but it must not be treated as a true visibility label. The middle-card result is the immediate target: six of six card-1/2/3 requests missed early even though their projected openings were inside all three camera frames. A full-row slot representation is more justified than another crop-only loss or threshold adjustment.

## 2. Run the two development tracks in parallel, then connect them

The actor/stop-head and coarse camera check can be developed at the same time on separate GPUs, while respecting the four-GPU total. They should **not** be assessed as one fused success until each individual gate passes:

| Component | Initial pass criterion on new episode-grouped development starts |
| --- | --- |
| Spatial transport actor | Better causal early request-dependent actions than a matched baseline, then at least **2/3** safe requested-port buffered arrivals without increased contact; extend to **9/10** before handoff promotion. |
| Pause-proposal head | Timely on at least **9/10** physically valid inspection opportunities, with no proposal accepted as safe on frozen wrong-route/wrong-port/high-force negatives. Report misses and early proposals separately. |
| Coarse camera check | **Zero accepted wrong ports**, at least **9/10** early correct requested-port acquisitions across all ten requests, and explicit abstention when the target is unidentifiable. Report accept rate, not just accuracy on accepted frames. |
| Connected pause/inspect flow | New autonomous starts: safe hold, correct requested-port identity, original-image refiner agreement, acceptable posture/force and complete inference p95 **<300 ms**. |

These are development gates, not estimates of production success. Keep the four reserved final configurations sealed. The stop head may be trained jointly as an auxiliary output of the improved transport network, but its threshold and episode-level decisions must be evaluated independently. If the actor reaches the target but the head misses, repair the head/labels. If the head fires while the actor is far away, repair transport or reject the proposal in the camera check. If the camera check cannot identify the requested slot from a safe view, change the observation posture rather than ask it to guess.

## Recommended order and bounded budget

1. Audit the saved middle-card and autonomous early images and create reviewed visibility/slot and safe-pause labels. Use existing runs first; do not collect routine episodes. **Stop** if human review cannot consistently identify a requested slot from those views; test a safe view change instead.
2. Fit **one** spatial coarse candidate against the frozen selected-port locator: at most **2,000 updates on one GPU**. Freeze model and threshold, then check ten new nonfinal requests. Stop at any accepted wrong port or fewer than 9/10 early acquisitions.
3. Fit **one** spatial transport actor and a matched current-feature control: at most **3,000 updates each on one GPU**, using the same native-image subset and complete episode split. The pause head can be trained as an auxiliary output with reviewed labels. Stop before live testing unless early request-dependent action quality improves by the preset margin.
4. If the offline gates pass, run at most **three new nonfinal autonomous starts** for transport and shadow pause/inspect, then a ten-start development gate. No automatic insertion from a coarse “yes.”

The important change from the previous attempt is that we would train the head on a clear **inspection opportunity** and separately ask the camera check to verify **identity and view quality**. We should not interpret the head's raw probability as calibrated confidence or let it bypass the camera check.

Supporting results: [transport continuation](../experiments/2026-10-05-transport-route-handoff-continuation.md), [coarse-camera failures](../experiments/2026-10-04-coarse-shared-history-execution.md), and [spatial prerequisite](../experiments/2026-10-05-spatial-transport-prerequisite.md).

# Five-step decision tree after the original-resolution crop test

**Current state (October 3, 2026):** The automatic transport-to-insertion handoff is disabled. The SC orientation head improved but missed the independent 3° gate (4.65° p95 on four new resets). A known SC wrong-posture stop was accepted by identity alone. SFP still has same-card neighboring-port confusion. There is no demonstrated reliable autonomous full insertion from normal starts, and the reserved final scenes remain sealed.

The next five steps are **decision points**, not a promise to run five training jobs. A failed safety or information gate can stop the branch. Each test uses complete held-out reset configurations; simulator geometry is allowed for training labels and *after-run scoring*, never as an evaluation crop, actor input, or handoff decision.

```text
1. Test native-image requested-port crop
   |-- passes both families and latency ----------> 3. Test complete handoff
   `-- fails -------------------------------------> 2. Isolate cause; one targeted repair
                                                   |-- repair passes --> 3
                                                   `-- still fails --> stop switch; revise sensing/view

3. Test transport -> pause -> perception -> accept/reject, without insertion
   |-- passes both families ----------------------> 4. Test insertion actor
   `-- fails --> fix the responsible transport / perception / pause-loop component;
                  repeat 3 once, then stop promotion if still unsafe

4. Test insertion actor from real, verified handoff states
   |-- passes ------------------------------------> 5. Full autonomous development rollouts
   `-- fails --> one matched actor/contact repair; repeat 4 once;
                  if still failing, diagnose insertion rather than start RL

5. Full episodes across SC and SFP
   |-- reliable insertion, safe contacts, <300 ms --> independent final evaluation,
   |                                               then consider RL for residual recoveries
   `-- not reliable --> classify failure by phase; repair that phase and repeat
```

## 1. Native-image crop comparison

Use the existing image-based locator to pick a patch from the original 1152×1024 camera image. Train a small requested-port crop head, while retaining the full-image context and task code. Compare with the frozen current head on identical held-out resets and saved good/bad stops. The fixed gate is SC near-port orientation p95 **≤3°**, no known unsafe stop accepted, **≥95%** clear-stop retention, resolution of held-out SFP adjacent-port confusion, and complete live p95 **<300 ms**. The existing five good and one bad SC stops are mandatory replay cases, but too few to prove reliability alone. Audit existing images first; collect at most 12 targeted SC train resets and two development resets only if coverage is missing. Fit/replay budget: four GPU-hours, plus one day of image review.

If **SC passes but SFP fails**, keep SFP handoff off and diagnose its same-card port choice. If **SFP passes but SC fails**, keep SC handoff off and diagnose orientation. Neither family gets automatic insertion merely because the other passes. If **both pass**, go to step 3.

## 2. If the crop fails, find the cause with one bounded diagnostic

Review the worst original images, task requests, image-selected crop boxes, predictions, per-camera timestamps, and scored poses. As a *diagnostic upper bound only*, center a crop using the scored port location offline. This oracle crop may identify the source of error; it can never be the deployed method.

| Observation | What it means | One targeted response; stopping rule |
| --- | --- | --- |
| Image-selected crop misses the requested opening, but an oracle-centered crop works | Global locator or requested-port identity is at fault | Repair the image-only locator/request matching on the affected family; keep the fine crop head fixed. Stop if new reset groups still select the wrong port. |
| Both crops include a clear port; oracle crop also predicts the wrong angle/neighbor | Fine representation or supervision is at fault | First audit camera calibration, timestamp pairing, port-frame convention, and label projections. If correct, try one explicit port-edge/keypoint or multi-view orientation head on the same grouped split. Stop if independent p95 or wrong-port gate still fails. |
| Errors concentrate in a particular board pose or one-port scene | Training coverage is too narrow | Add only targeted missing configurations within the step-1 collection cap; compare per-scene, not just pooled error. Stop if the new configurations do not improve held-out groups. |
| Requested opening is genuinely hidden or visually indistinguishable | The current image cannot justify a precise pose | Train/test an **invalid** decision; hold or seek a new view rather than guess. If no safe observation-only view makes it identifiable, redesign sensing or approach. Do not hand off. |
| Accuracy passes but full runtime exceeds 300 ms p95 | Computational problem | Profile image transfer, locator, crop encoder, and controller; make one bounded optimization. Stop promotion if complete live p95 remains too high. |
| Labels or scoring disagree with inspected images/robot pose | Measurement problem | Correct that specific label/calibration error and repeat the frozen comparison. Do not claim a model gain from the bad labels. |

This step is a **single diagnosis and one justified repair**, not a broad architecture sweep. Its proposed cap is two additional GPU-hours and no general episode collection. A failure after that is a reason to pause the automatic handoff design and change the view or sensor evidence, rather than proceed to actor or RL training.

## 3. Complete handoff test, with no insertion motion

Run normal transport from episode start. Its arrival head proposes a pause; the robot holds its measured TCP pose; perception checks the **requested** port and safe standoff; then it accepts or rejects. Test SC and SFP on new development reset groups spanning card counts, selected ports, and known confusing configurations. Replay the known wrong SC posture and SFP wrong-port cases. Score false accepted handoffs, missed valid arrivals, repeated pause loops, lateral/axial/orientation error, force, and complete latency. A pause proposal is allowed to be wrong **only if the verifier rejects it safely and transport can continue**.

**Proposed bounded screen:** 12 new complete starts per family, with at least two card layouts and both requested ports represented where applicable. Require zero false accepted handoffs and unsafe contacts, at least 10/12 valid handoffs per family, no episode trapped in a pause loop, and p95 latency below 300 ms. The 10/12 threshold is a screening target, not a statistical reliability claim; step 5 supplies the larger confirmation. Freeze this manifest and rule before running.

Branch by failure: missed approach or unsafe route → transport; correct arrival but wrong-port/angle acceptance → perception/verifier; endless reject-and-resume → handoff state logic. Make one phase-specific repair and repeat the same gate on independent development groups. No insertion actor is attached until both families pass, known false stops are rejected, and p95 runtime is below 300 ms.

## 4. Test the insertion actor from actual verified handoffs

Use the real distribution of actor-generated, visually verified stopping poses, including the variation around the target. Compare the frozen insertion BC actor with **one matched supervised repair** using the same scenes and update budget. Give it the vision-estimated requested-port frame, RGB, measured state, task code, and force; never the true port pose. Check first-command direction, lateral drift, orientation, insertion depth, blocking contact, and whether a correct pose estimate is lost during the handoff. Include both SC and SFP. A teacher prefix that starts perfectly aligned is only a diagnostic; it is not this gate.

**Proposed bounded screen:** At least 10 distinct real handoff states per family, held out by complete reset. Require at least 8/10 full insertions per family with no unsafe-force event and live p95 below 300 ms before a larger full-episode test. Record partial insertions and blocked contact separately; neither counts as full insertion. Freeze the actor comparison's training episodes and update budget before fitting.

If insertion still fails despite an accurate, safe handoff, inspect whether the problem is action-frame conversion, distribution shift in starting poses, contact response, or inadequate recovery examples. Try one repair matching that trace, then repeat. If it still fails, keep RL and final evaluation closed. The current evidence includes insertion-actor failures even when perception looked good in a narrow scene, so passing perception alone is insufficient.

## 5. Full autonomous development rollouts, then choose whether RL is justified

Run complete normal-start episodes across both task families, card layouts, requested ports, and starts **with no guide, privileged geometry, or teacher actions at runtime**. Record videos, scored insertion depth, lateral/axial/orientation errors, force/contact, handoff decisions, recovery events, and p50/p95/p99 latency. Use new episode-grouped development starts; keep the four reserved final configurations sealed until a predeclared development success and safety gate passes.

**Proposed confirmation gate:** 20 new normal-start development episodes per family, at least 16/20 full insertions in **each** family, zero wrong-port insertions or unsafe-force events, and complete p95 inference below 300 ms. Report per-card-count and per-port results so an easy subgroup cannot mask a failure. This is an engineering target for calling the supervised system *ready for final evaluation*, not an official challenge threshold or proof of production reliability.

Classify any failure by its first causal phase: route/obstacle, wrong pause, wrong port or pose, insertion control, or force/contact recovery. Repair that phase and rerun a matched comparison. **Only if supervised full insertion is already reliable** should a bounded model-free RL comparison test whether it improves residual contact/cable recovery over the same BC policy, with and without the existing backtracking rule. RL is not a substitute for an unsafe visual handoff. World-model imagination and Seer remain parked.

**Practical stop rule:** If the native crop and one targeted repair cannot establish a reliable, identifiable port pose, stop the two-actor promotion and choose better viewing/sensing or a safer approach trajectory first. If perception passes but autonomous insertion does not, work on the insertion actor and contact data. Do not open final scenes or claim full insertion from perception metrics alone.

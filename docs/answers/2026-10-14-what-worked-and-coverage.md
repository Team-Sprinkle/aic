# What we tried, what worked, and how widely we tested it

Status: the broader frozen evaluation completed on October 15. See [the final explanation](2026-10-15-frozen-coverage-explanation.md) and [full results](../experiments/2026-10-14-frozen-pipeline-coverage.md). The earlier results below are historical and do not override the failed broader gate.

## The important correction

No: the earlier evaluation was not five episodes for every card-count/port/family setting. The newest11mm insertion was **one reused SC-to-SC episode**. Older6/6 results pooled different normal-board SC starts and tested transport/visual acquisition, with insertion disabled. They do not establish full insertion reliability or SFP coverage.

## What was tried

| Experiment | What actually changed | Result and strength of evidence |
| --- | --- | --- |
| Repair timestamp-based transport labels and stop supervision | Same learned transport architecture and budget; corrected causal future actions/terminal masks | Offline late-command p95 improved11.11→6.50mm, but autonomous stable acquisition stayed1/6 versus1/6. No demonstrated live improvement |
| Rule-based SC transport | Replace learned transport actions with the Low Lateral Bypass sequence; use measured stage completion |6/6 reused then6/6 fresh normal-board acquisition screens. Useful transport feasibility, not all settings or full insertion |
| Bounded visual docking after learned transport | Same learned actor plus an explicit camera-based final docking controller |6/6 reused and6/6 fresh supported normal-board acquisitions. A hybrid/controller improvement, not an improved learned actor |
| Correct clipped board-orientation estimation | Expand guarded image-line orientation search; reject boundary saturation | Matched orientation error1.37°→0.017° on one case; later stricter fresh screen5/6. Targeted perception improvement, limited scope |
| Complete plug-tip action targets | Change the local BC output reference, with matching inverse conversion to TCP commands | Near-port offline command p95 improved1.010→0.629mm, but the live actor stalled about130mm before opening |
| Corrective teacher continuations from actor states | Add two qualified training continuations to existing mixed-task BC data; keep matched control | Offline held-out continuation near error0.900→0.420mm. Both autonomous actors still stalled about141mm away,0/1 insertion each |
| Decoder and robot-history repairs | Matched6000-update decoder tests, then masked/current versus6second robot-state history | Failed frozen offline command/motion gates. No live reliability claim; these models were not promoted |
| Observation-only approach before the existing insertion actor | Keep rule transport, visual models and trained actor fixed; explicit feedback moves from roughly155mm to30mm before opening | One reused SC case: reached29.12mm stand-off/0.148mm lateral, then actor inserted11.07mm. Meaningful approach progress, but failed14mm physical depth gate |

A privileged controller diagnostic reached15.57mm depth. It used true geometry and cannot count as autonomous success.

## What runs in the latest SC hybrid

```text
public SC task/port code + measured robot state
             |
             v
rule-based transport around the cards
             |
             v
three-camera coarse/identity check -> original-image crop refinement
             |
             v
correct requested port + stable camera acquisition approved
             |
             v
image-based held-plug position estimate
             |
             v
explicit RGB/proprio approach to30mm before opening
             |
             v
EXISTING trained BC insertion actor
             |
             v
four complete plug-tip/TCP-orientation targets,50ms each; execute all four
```

**Transport and the extra approach controller are rules/controllers, not newly trained models.** Coarse perception, identity checking, refinement and the held-plug position helper are trained models. The final insertion actor is also trained: the same frozen corrective-data checkpoint is reused. Its image backbone is ResNet18 with a transformer action head; it is not the transport diffusion model. No RL, backtracking or ground-truth simulator pose is used in this autonomous pipeline. Simulator geometry is recorded separately for teacher diagnostics and posthoc scoring only.

The actor predicts full targets in the estimated port frame. The runtime converts them to TCP targets and then the robot API's relative movement. This is not a learned delta correction added to a frozen base actor.

The latest hybrid is SC-only. SFP has a shared trained transport actor, and separate perception/near-port experiments exist, but the selected strict handoff/local runtime is not integrated for SFP. The current shared transport checkpoint's supervised stop head explicitly supports SC only. We must not present SFP transport rollouts as full SFP insertion tests.

## New evaluation requested here

[Exact frozen protocol](../experiments/2026-10-14-frozen-pipeline-coverage.md).

-50 SC hybrid episodes:1–5cards ×2ports ×5fresh replicates.
-20 matched SC controls without the extra approach:1/5cards ×2ports ×5same scenes.
-50 SFP learned-transport-only episodes:1–5cards ×2ports ×5fresh replicates, cycling requested card locations.
-20 additional alternate-public-board stress episodes:3cards ×2ports ×2families ×5fresh replicates.

Total140rollouts on120fresh development scenes. Rail positions, card subsets, actual xacro robot starts and small board/grasp perturbations vary. All ordinary failures count; models and thresholds are frozen. This tests the declared matrix, not every possible subset/start or board-orientation cross-product. Five episodes per cell remain too few for a strong statistical reliability claim.

SC results separate transport/visual approval, near approach, actor execution, actual depth/lateral/orientation and contacts. SFP reports transport geometry; its absent full-pipeline handoff/insertion is N/A, not an invented insertion success rate.

## Artifacts

New batch:`/home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1/`.

`plan.json` freezes scenes/weights/code;`progress.json` records each completed result; each run retains raw bags, logs, images, causal terminal scoring and all failures. Models are not updated from these evaluation scenes. Final scenes remain sealed.


### Important implementation distinction: selected coarse model

The live SC diagnostic uses the older `joint_locator_v1` / `SCShadowHandoffVerifier` stack. Its coarse network predicts12internal **2D** candidate slots (2SC and10SFP), plus projected-in-frame flags. The public requested SC code selects one slot. A separate request-conditioned spatial identity head receives current image features, camera identity, measured7D TCP pose and requested SC port code. The selected original-image crop is refined, then three-view geometry/board orientation supplies one requested-port3D pose.

This is **not** the later proposed target-only temporal coarse model. There is no full camera-image history in this selected identity head; the guarded rule pipeline can retain observation-derived board orientation, which is different. Its in-frame flags are projection-presence labels, not certified occlusion visibility or calibrated pose confidence. The current matrix tests this older frozen stack because that is what the reported latest hybrid actually used. Only the requested candidate is passed downstream; internal12slot2D proposals must not be described as predicting12reliable3Dposes. Earlier user-facing explanations did not make this version distinction sufficiently clear.

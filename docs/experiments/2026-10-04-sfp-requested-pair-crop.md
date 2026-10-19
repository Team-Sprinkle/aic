# Bounded requested-card paired-port crop experiment

**Date:** 2026-10-04. **Status:** design and code drafted; no pair-model training or evaluation started. Work deferred while clarifying the handoff design. Automatic handoff stays disabled.

**Handoff scope:** this pair model addresses the *final* precise camera check after a held inspection. Coarse requested-target perception must separately reject an obvious wrong route/card and decide whether a pause is worth inspecting. A coarse positive is not permission to insert. Neither check is promoted by this isolated crop test.

## Why this experiment

The current pipeline uses an all-port low-resolution predictor to propose an observation-only crop, then a native-image refiner predicts **one requested port** in each view. The corrected replay enumerates all ten SFP channels without simulator presence filtering: on ten nonfinal teacher-view scenes, the true requested opening lay inside all three crops for 362/370 proposals, yet twelve accepted crop predictions chose a wrong port. A causal selected-port anchor preserved identity in 53 extra later frames but did not acquire any new episode. The older 335/346, eleven-wrong and 52-extra counts came from a presence-filtered offline candidate list and are superseded. A posthoc oracle board-row filter alone left same-card port swaps; board-row identity is not a sufficient repair.

## Fixed test and stopping rule

Train **one** model on crops from original 1152×1024 RGB, using the existing observation-derived 480 px proposals. It predicts the **two openings of the requested card jointly** for each camera, with an explicit card/request input. Labels are projected opening points and presence flags from training scenes; simulator geometry is never a runtime input or crop selector. At evaluation, triangulate each predicted opening from the three measured/calibrated camera views and require plausible pair spacing and consistent port-0→port-1 order. Return only the requested opening if all checks pass; otherwise abstain. Test target-in-crop coverage and negative same-card swaps separately. Do not treat projected-in-frame as certified visual visibility.

Use the 59 existing complete training reset groups, including nine previously diagnosed fresh scenes, and ten disjoint older development groups for checkpoint selection. At most **1,200 updates or 60 minutes**, whichever occurs first, on **one GPU**. Preserve all checkpoints and predictions. The development gate is zero accepted wrong identities, at least **8/10** episodes acquired within 3 s of first sampled image, and accepted 3D position p95 at most **10 mm**. If this fails, stop this architecture instead of relaxing thresholds or collecting routine scenes. If it passes, freeze the checkpoint/rule and evaluate on ten **new nonfinal reset groups** balanced across requested card/port codes, with zero wrong accepted, at least **9/10** early acquisitions, and accepted p95 at most **20 mm**. Measure complete live p95 below 300 ms only after identity passes. Reserved final scenes remain sealed. No transport actor, insertion actor or RL promotion follows a failed gate.

The baseline is the frozen selected-only camera check (zero wrong but 3/10 early on the earlier fresh scenes) and frozen one-port crop refiner (12 wrong accepts on the corrected all-channel replay). Development scenes are not independent evidence of promotion. All comparisons must report complete episode counts, wrong accepts, abstentions and accepted error; training accuracy alone is insufficient.

## Interpretation

This is a **port-pair disambiguation** test. It does not by itself establish numbered-card identity when only part of the board is visible, nor does it resolve the existing unreliable SFP occlusion labels. If port-pair errors disappear but wrong-card or low-coverage failures remain, next test a visually grounded board/slot reference on raw full views, with an explicit abstain path. An oracle board-row filter is diagnostic only and cannot enter autonomous inference.

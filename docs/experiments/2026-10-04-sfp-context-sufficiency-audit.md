# SFP requested-port context and causal-anchor audit

**Date:** 2026-10-04. **Decision:** history improves continuity after a trustworthy anchor, but does not establish missing card/port identity. Automatic handoff remains disabled. No training, new simulator scenes, final scenes, or RL were used.

**Correction:** The first replay below used crop candidates filtered by simulator port-presence labels. The [all-channel corrected replay](2026-10-04-sfp-fresh-anchor-shadow.md#corrected-all-channel-crop-replay-october-4) keeps the same ten scenes and frozen models, enumerates all ten SFP port channels, and finds 154 anchored accepted frames rather than 153: 53 additional later frames rather than 52. Early coverage remains 3/10 with zero wrong accepts on this set. Corrected artifact: `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/selected_anchor_tracking_all_channels_v2/summary.json`. The original table and command below are retained for audit history.

## Question and existing data

Does a camera check lack information about the requested SFP card/port, and can causal history repair it? Reuse the ten frozen, nonfinal `card0_port0` through `card4_port1` CheatCode observation episodes from [the fresh-anchor check](2026-10-04-sfp-fresh-anchor-shadow.md). They contain 581 synchronized original-resolution three-camera triplets and measured TCP poses. These are **teacher camera trajectories**, not autonomous transport successes. Four frozen perception-head outputs and posthoc simulator geometry labels are already saved. No checkpoint, scene, or acceptance threshold was fitted for this audit.

## Raw RGB review and information limits

I inspected the original, unmarked center images at the first saved sample and a middle sample for card 1/port 0, card 2/port 0, card 3/port 0, and card 4/port 1; I also inspected the first left/right images for card 1/port 0 and the existing unmarked three-camera side-view grids for card 1/port 0 and card 2/port 0. No truth markers were used for the visual review. The sample is deliberately small, so these are qualitative findings, not certified per-camera visibility labels.

| Scene | What the first raw view shows | Posthoc projection upper bound, first triplet | Consequence |
| --- | --- | --- | --- |
| Card 1 / port 0, four installed cards | Center shows only two card faces; left/right also show a partial row. The card-row origin and empty slots are not established by that center image alone. | Projected occupied slot indices: center 1,2; left 1,2,3; right 1,2. | A close face can be visible while its **numbered identity** is underdetermined without a board/slot reference or earlier better view. |
| Card 2 / port 0, five installed cards | All five faces and both apertures per face are apparent in the first center image. The middle view brings the row closer but partly occludes the lower faces with the tool. | All five indices project inside all three images. | The failure here cannot be explained simply as the entire row being outside the image. The existing model/cross-view association fails despite apparent global context. |
| Card 3 / port 0, five installed cards | The complete row is apparent in the first center image. | All five indices project inside all three images. | Tests ability to use visible ordinal card context; projection alone does not certify port-face visibility. |
| Card 4 / port 1, three installed cards | Only three cards are installed; a naive “fourth visible card” interpretation is impossible. | Occupied slot indices 1,2,4 project in all three views. | The task's card code names a **slot**, so counting installed cards is insufficient when slots are empty. |

These projection flags come from simulator geometry and are used only **after** image capture to characterize the diagnostic. They say an opening would fall inside image bounds; they do not establish that the opening is visually identifiable. The earlier SFP auxiliary depth capture was unreliable, so this audit does **not** claim a verified occlusion label. Representative raw files are the `teacher_card1_port0/frames/` and `teacher_card2_port0/frames/` paths in the frozen archive; the side-view unmarked grids are in `/home/nas/chmin/aic_archives/2026-10-04_sfp_side_view_v1/original_rgb_grids_v1/`.

## Controlled causal replay

The fixed baseline is the frozen **selected-only** predictor at each current image. The diagnostic tracker can establish an episode/request-specific 3D base-frame anchor **only** from an accepted selected-only prediction. At later observations it accepts a frozen all-port, structured, or native-crop prediction only if that predictor's own current-frame gate accepts and its base-frame point is within 5 mm of the saved anchor. The anchor expires after 150 mm of measured TCP travel without a confirming selected-only observation, matching the earlier SC shadow-history distance scale. A conflicting selected-only prediction clears the anchor. Simulator target geometry is read only after decisions are saved, to score the requested identity. This is a replay, not a live deployment verifier.

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_selected_anchor_tracking.py \
  --source /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/selected_anchor_tracking_audit_v1
```

| Frozen replay | Episodes with any accepted request | Episodes accepted within 3 s | Accepted observation frames | Wrong accepted frames |
| --- | ---: | ---: | ---: | ---: |
| Selected-only current frame | 5/10 | 3/10 | 101 | 0 |
| Selected anchor plus current matching heads | 5/10 | 3/10 | 153 | 0 |

The tracker added **52** accepted later frames when the selected-only head rejected the current frame, including 22 on card 0/port 1 and seven on card 1/port 0. It added **zero newly acquired episodes** and did not improve early acquisition. The prior, looser same-model sparse-agreement test also failed to ensure safety: the structured head still had a wrong first anchor with two consistent frames, while requiring 10 mm tool motion cut early coverage to 3/10. Thus history has a plausible role in maintaining a *verified* identity through temporary bad views, but it is not an identity source by itself. The observed zero wrong accepts here relies on the selected-only head's zero wrong accepts on **these ten development scenes**; it is not a general safety guarantee.

## Decision and next bounded test

Do not attach this tracker to the live handoff. Its episode acquisition gate remains 3/10 early against the earlier ≥9/10 target, true visibility is unverified, and complete live latency has not been measured. Reserved final scenes stay sealed.

The next repair hypothesis should be **slot-aware identity from a full original-resolution board view, paired with local port crops**, using the requested card/port code and an explicit possibility of empty slots. This directly addresses the information lost by a close crop and the card-4 scene's nonconsecutive occupancy. First review image/occlusion evidence and establish a board/slot reference in the existing RGB; if card identity is not observable, test a genuinely more informative observation posture rather than fitting another head. If a specific visual representation is then trained, freeze its rules before new nonfinal episode-grouped evaluation; stop unless it achieves zero wrong accepted ports and useful early request coverage (the prior target is ≥9/10), and only then measure complete live p95 against 300 ms. Do not claim autonomous insertion from these teacher-view diagnostics.

**Artifacts:** [replay script](../../scripts/audit_sfp_selected_anchor_tracking.py), `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/selected_anchor_tracking_audit_v1/summary.json`, and the complete `decisions.jsonl` in that directory. The original frozen predictions and labels remain in `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/`.

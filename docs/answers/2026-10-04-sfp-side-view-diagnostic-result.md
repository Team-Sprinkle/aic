# SFP side-view diagnostic result

I ran the agreed camera-view test on four existing, nonfinal SFP development configurations: card 1/port 0, card 2/port 0, card 0/port 1 and card 4/port 1. A geometry-guided CheatCode controller brought the tool to an 80 mm plug-tip standoff **for this diagnostic only**. It then paused at the original view and moved roughly 15 mm to each side. The visual models were unchanged and received images, measured robot state, calibration and the public request code; they did not receive simulator target coordinates. The four reserved final scenes stayed sealed.

The first safety runs stopped early because an absolute 12 N wrist-force limit misread the approximately 20 N raw sensor baseline, and because a blanket contact rule treated a light cable brush against the board as unsafe. I preserved those runs. Oracle traces showed cable/board pair forces below 0.1 N in the aborted cases, with no significant wrist-force rise. The corrected guard allowed that light cable contact, stopped for a force rise above 5 N or absolute force above 35 N, rejected other collision pairs, and checked that the measured TCP actually reached each side offset. All four final sweeps completed. Their largest force rise was 0.97 N; measured side offsets were about +12 to +14 mm and −14 to −15 mm. The subsequent successful insertions were made by **CheatCode**, not the learned policy.

I saved 12 matched three-camera image triplets, one at each of the three view holds per scene. The raw grids have **no ground-truth markers**. I inspected those first, then made separate posthoc-marked copies to check where the requested and neighboring port openings lie. The exact camera-to-event join was within 18 ms. The raw views show multiple port faces, but I cannot certify exact numbered-card visibility in every image from this review alone.

| Unchanged model: accepted requests out of four | Original view | +15 mm | −15 mm |
| --- | ---: | ---: | ---: |
| All-port | 2 | 2 | 2 |
| Structured all-port | 2 | 1 | 2 |
| Selected-only | 3 | 1 | 1 |
| Native crop | 3 | 1 | 3 |

The hard **card-2/port-0** request was accepted by none of the models at any view. The crop head already accepted the other three requests at the original view; the side move did not add one. For card 2, coarse predicted opening points were often 70–265 pixels away from the posthoc opening projections. The one crop proposal at the negative view contained the true opening, but its camera predictions disagreed, so the consistency gate rejected it. There were no accepted wrong-port predictions in these 12 triplets, a sample too small and reused to establish a false-accept rate.

**Decision:** The 15 mm side move is feasible under the corrected contact rule, but it did not improve the remaining identity failure. I did not train a view-conditioned model or connect transport, insertion or RL. The next targeted check is actual requested-opening visibility and per-camera occlusion on these saved hard images; that should determine whether to repair ordinal card/port association or seek a more informative viewpoint. Further training needs a separate episode-grouped split and fixed stop rule.

The full [experiment record](../experiments/2026-10-04-sfp-side-view-diagnostic.md) includes commands, safety revisions, exact artifacts and the matched per-scene results. The [machine summary](../experiments/2026-10-04-sfp-side-view-diagnostic.summary.json) captures the decision. Raw grids are in `/home/nas/chmin/aic_archives/2026-10-04_sfp_side_view_v1/original_rgb_grids_v1/`; posthoc-marked copies are in `posthoc_truth_grids_v1/` beside them. Main status, experiment ledger, handoff plan and artifact guide were updated.

# Robust two-view SFP rail correspondence check

The slot-local occupied/empty head passed its bounded test, but the first 3D rail-to-port verifier obtained a usable multi-camera rail reference in only 2/10 saved nonfinal teacher routes. Reason-coded traces show that 392/581 frames violated a strict 5 px reprojection residual and 185/581 lacked two in-frame views of both endpoints. A bad third camera can spoil triangulation even if two views agree.

**Frozen hypothesis:** fit rail 0 and rail 4 from the *same best pair* among the three RGB cameras, require predicted span 120–200 mm and pair reprojection residual ≤5 px, and remember up to the last three valid causal references. Use the existing RGB-predicted five-bit occupancy vote and old detector's three-view-consistent requested-port proposal. The robot's measured TCP and fixed camera calibration are allowed; simulator rail/port geometry is for posthoc scoring only. The precise rail-index tolerance stays ±0.5 slot. No learned weights or score thresholds change.

Run on the same ten nonfinal teacher-route groups, with all 581 frames and the checkpoint frozen. The bounded diagnostic passes only if it obtains a reference in **at least 8/10 groups**, accepts **zero wrong-port candidates**, and keeps at least **3/10 early correct groups** (the old detector's early coverage). If it fails, do not attach it or tune the pair residual on those scenes; the next repair needs better multi-view endpoint predictions or direct requested-port correspondence with reviewed visual labels. Even passing this replay would require new nonfinal autonomous starts and complete latency before handoff. Final scenes stay sealed.

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_rail_candidate_3d.py \
  --rail-predictions /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_slot_local_v1/predictions.jsonl \
  --candidates /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_candidates_all_channels_v2.jsonl \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten/rows.jsonl \
  --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --reference-mode best_two \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_rail_candidate_best_two_v1
```

## Best-pair result and bounded geometric follow-up

The frozen best-pair replay obtained a valid reference in **10/10** groups and rejected all 27 wrong accepted rows of the old detector, but had only **2/10 early correct groups**, below the predeclared 3/10 gate. [Summary and every causal frame](</home/nas/chmin/aic_archives/2026-10-05_sfp_rail_candidate_best_two_v1/summary.json>). In the missing card-0 case, the chosen pair triangulated a 130 mm rail with about **62 mm vertical change** between endpoints. It passed the loose 120–200 mm span check but is physically incompatible with the task board: the 70 train reset configurations' CAD rail vector has **zero vertical component**, 157–160 mm positive board-row displacement and at most about 30 mm transverse displacement. This diagnosis uses train CAD and saved model predictions, not the target scene's ground-truth board pose in inference.

One more bounded read-only repair will fit both rail endpoints **jointly** as a horizontal, fixed 160 mm line to each camera pair, rather than triangulating them independently. Yaw is restricted to 1.2–1.8 rad, a conservative envelope around the training board directions; require the same ≤5 px pair reprojection residual and the unchanged causal occupancy, ±0.5-slot and existing port checks. The frozen gate remains **≥8/10 board-reference groups, zero wrong accepted port candidates and ≥3/10 early correct groups**. If it fails, stop geometric postprocessing on these reused scenes; train a visual landmark with cross-view/temporal consistency or directly learn requested-port correspondence using reviewed labels, then evaluate on new autonomous development scenes. No scene-specific true geometry enters the fit.

**Result:** The joint horizontal fit failed: only **4/10** groups had a valid reference and **1/10** had an early correct accepted candidate, although it accepted zero wrong candidates. [Frozen results](</home/nas/chmin/aic_archives/2026-10-05_sfp_rail_candidate_horizontal_fit_v1/summary.json>). The very low coverage with fixed 160 mm horizontal geometry confirms that the current 2D endpoint predictions do not meet the strict cross-camera precision needed for a 3D board-pose verifier. Stop threshold postprocessing on these reused scenes. A distinct slot-conditioned native-image port-crop feasibility check follows in a separate record; no handoff is enabled.

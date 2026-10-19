# Sparse SC approach history for selected-port handoff

**Date:** October 3, 2026. **Status:** bounded offline and live shadow diagnostic complete; automatic handoff gate remains open. No insertion actor or RL was enabled. Final IK scenes remain sealed. [Machine summary](2026-10-03-sc-sparse-history-handoff.summary.json).

## Hypothesis

The frozen spatial selected-port head recognizes the requested SC opening earlier on an approach but may reject the same physical port after the wrist cameras rotate near the standoff. Keep a sparse, request-specific, observation-derived 3D opening anchor from an earlier clear view. At the stop, an original-resolution crop and all three calibrated cameras must place the fresh selected opening within 5 mm of that anchor, with ≤2 px reprojection disagreement. This can support port *identity*; force, posture, standoff, board orientation, and wrong-port checks remain separate. The anchor is never derived from simulator truth at runtime.

## Existing-data audit

I used saved complete-reset SC development recordings: four earlier one-port crop scenes, a one-port p0 baseline, two p0 orientation-correction scenes, and four two-port route scenes. The RGB classifier and locator checkpoints stayed frozen. Three-camera images were grouped by nearest timestamp within 75 ms; some saved camera streams are offset by 50 ms, making this a JPEG offline proxy rather than a native-buffer certificate. Simulator projections were used only after inference to count in-frame examples and score pose error; **in-frame is not reviewed visibility**. The four reserved final IK configurations were not opened.

The coarse 288×256 locator alone was unsuitable for anchoring: three-view reprojection residuals were typically 2.7–4.1 px and sometimes much larger. Applying the existing **160×160 original-RGB crop refiner before triangulation** reduced typical residuals below 1–2 px on most groups. This is a targeted repair of the measured failure, not a new trained model.

With a measured-motion keyframe refresh spacing of 20 mm, the offline tracker used **4–9 identity-confirmed keyframes per one-port trajectory**. Trying 10 and 30 mm changed the keyframe count but did not improve accepted-frame counts on these 1 Hz saved trajectories, so 20 mm is the provisional sampling interval. The current frame is still inspected with original-resolution crops at handoff. An anchor is created only after all three view identity scores pass and the refined geometry is consistent. New accepted keyframes can update it only when within 5 mm of the old anchor; rejected frames cannot move it. A candidate expires after 150 mm of measured travel without refresh. This is a conservative offline prototype, not yet deployment code.

Across the seven one-port trajectories, a frozen single-frame all-view identity check accepted **235/367** target-projected triplets. A combined decision using the same current identity check **or** a causally established, geometry-consistent anchor accepted **363/367**. On the two previously failing p0 correction recordings, this was **51/154 → 151/154**. These are correlated frames from seven scenes, not 367 independent trials, and projected-in-frame examples are not certified identifiable. The alternate physically absent request had **0/444** all-view single-frame accepts; it cannot inherit the requested port's anchor because history is request-specific. This is not yet a hard-negative handoff safety result.

In four two-port scenes, *both* requested-port codes can legitimately be accepted when both ports are visible. Independent refined positions for the two candidate openings were separated by approximately **68–83 mm** in the matched saved views, far beyond the 5 mm history match and the verifier's 15 mm standoff check. This supports geometric separation on these scenes but does not prove general wrong-port rejection.

One further saved one-port port-1 route, not used to choose the 20 mm interval, gave **22/24** target-projected triplets accepted by the original-or-history rule and **0/39** physically absent alternate-request identity accepts. I reviewed the two projected misses: at 21.75 s the port was visible at the edge of the center image but largely covered by the gripper in the left image; the next frame was similar. They are not clean all-view visibility positives. The tracker correctly did not invent an earlier anchor in those views. [Frozen independent-p1 trace](../../outputs/experiments/2026-10-03_sparse_history/independent_p1_track_20mm_v1.json).

At 30 saved two-port frames where the selected opening's observation-derived standoff was ≤15 mm, the *other* visible port's estimated standoff was at least **68.4 mm**. The task request and geometry therefore distinguished the two on these four routes even though the identity head gave high scores to both present ports. This is a posthoc audit of observation-derived estimates, not a universal separation bound.

The crop-refined near-standoff pose on the two corrected p0 runs (posthoc true-standoff distance ≤15 mm) had port-frame **lateral p95 0.102 and 0.245 mm**, axial p95 0.607 and 0.398 mm over 52 and 50 saved samples. The offline anchor was about **0.55–0.80 mm** from the independent fresh live verifier's opening position at the old postrotation inspections. At those timestamps the tracker had a causal anchor and current geometry match while the frozen single-frame identity head rejected the port. This is evidence for the identity mechanism, not autonomous insertion or a certified 0.5 mm overall pose guarantee.

Two inspection montages were reviewed: [known p0](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/p0_known_stale_orientation/review/all_inspections.jpg) and [new p0](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/new_p0_orientation_correction/review/all_inspections.jpg). In both, the blue requested SC port remains visible in the center, left, and right review crops before and after rotation. Broader visibility labels are still needed.

## Live shadow diagnostic

An off-by-default `AIC_SC_SHADOW_HISTORY_AUDIT=1` mode samples approach RGB/TCP at 10 mm before an anchor and 20 mm afterward, or after a 5° view turn. It logs eligible anchors and at a held fresh inspection logs whether history **would** satisfy identity plus unchanged posture, standoff, force, and image-geometry checks. It does **not** change control or the original verifier's accept/reject outcome. The flag is passed through rootless Docker by `run_clean_transport_eval.sh` and recorded in the eval manifest. The first run omitted the flag inside Docker due to launcher wiring; it completed as a frozen baseline. I fixed that pass-through before the next two runs. The later code also binds the anchor to the requested port and expires it after 150 mm of unrefreshed travel; that small defensive revision was made **after** the completed shadow runs and has not been promoted.

All three live runs used the same one-port development reset and frozen model checkpoints; the active policy received no simulator port geometry. The baseline reached **0.249 mm** true closest standoff and made three inspections, all rejected by the original identity head after the existing posture correction. The first history-enabled run traveled a different route and never proposed a stop; its closest true standoff was **48.25 mm**. Its TCP was already **20.5 mm** from the baseline at 10 s, before history inference began at 30 s, so this cannot be attributed to the new history computation. Its logged observation anchor was about 0.29–0.91 mm from the true opening on the sampled approach, but it provided no handoff test.

The prespecified **one bounded repeat** reached **0.788 mm** true closest standoff and made three fresh inspections. At the first, posture was wrong by **6.86°** and the unchanged posture gate correctly rejected it. The existing held-state orientation correction reduced error to **0.187°** and **0.130°** at the next two inspections. The original selected-port identity still rejected both; the shadow history diagnosis matched the fresh opening to its earlier observation anchor by **0.975** and **0.889 mm**, with reprojection residuals **0.606** and **0.713 px**, standoff errors **0.828** and **0.842 mm**, and forces about **20.96** and **20.78 N** against the unchanged 40 N bound. It logged `would_accept_with_history=true` twice, but **did not alter the actual rejection**. A separate counterfactual request for the other port was rejected on all three inspections by view geometry. Posthoc true geometry placed the fresh predicted opening about **0.38/0.37 mm** from the real one at the two candidate inspections; it was not an inference input.

The repeat policy plus periodic history processing logged **196.25 ms p95** over 331 decisions; the three held fresh inspections took **90.96/95.45/100.37 ms** each. These measurements do **not** include a connected PoseInsert actor or complete ROS-to-first-insertion-command timing. No scored insertion occurred. The live run's native-camera handoff evidence is encouraging, but the number of independent scenes and hard occlusions is insufficient for automatic promotion.

## Gate and next action

Do **not** attach PoseInsert yet. The next targeted check is request-specific history on new complete development resets containing **reviewed clear-to-occluded approaches and coherent wrong-port proposals**. Require zero accepted known absent/wrong ports, at least 95% recall on reviewed clear selected-port arrivals, correct 3D and orientation estimates at handoff, and full ROS-to-decision p95 under 300 ms. The existing 20 provisionally hidden frames were rejected 0/20 by the fresh three-view geometry rule, but they do not contain causal earlier anchors and therefore do not close this new history gate. If the live anchor disagrees with fresh native geometry or adds false accepts, diagnose that exact failure before further training. If the SC gate passes, complete the SFP-specific arrival/identity gate before connecting the shared PoseInsert actor. Then run the planned start-offset training and isolated correction-envelope tests. RL remains gated.

## Commands and artifacts

The saved audit commands are reproducible with:

```bash
PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/audit_sc_sparse_history.py \
  /home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/crop_development_*/projection_state_v2/rows.jsonl \
  /home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/new_p0_baseline/projection_state_v2/rows.jsonl \
  --locator outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --identity /var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt \
  --output <new-audit.json> --device cuda:1

PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/evaluate_sc_sparse_history_track.py \
  <new-audit.json> --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --refiner outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_v1/best.pt \
  --keyframe-spacing-mm 20 --output <new-track.json> --device cuda:1
```

Exact full JSON outputs, including every prediction, timestamp, anchor, and pose error, are under [the sparse-history artifacts](../../outputs/experiments/2026-10-03_sparse_history/). Live rootless artifacts are under [the NAS experiment folder](/home/nas/chmin/aic_archives/2026-10-03_sparse_history_v1/).

The live repeat command was:

```bash
AIC_SC_SHADOW_HISTORY_AUDIT=1 AIC_SC_SHADOW_DIRECT_NATIVE_RGB=1 \
 AIC_SC_SHADOW_ALLOW_STALE_PAUSE=1 AIC_SC_SHADOW_ORIENTATION_ON_IDENTITY_MISS=1 \
 AIC_SC_SHADOW_COUNTERFACTUAL_REQUEST=1 \
 AIC_SC_SHADOW_SCENE_ROOT=/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1 \
 AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC=75 AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC=200 \
 bash scripts/run_sc_shadow_handoff_pilot.sh \
 /home/nas/chmin/aic_archives/2026-10-03_sparse_history_v1/live_known_p0_history_repeat development_p0_a
```

# SC selected-port original-RGB crop identity ablation

**Date:** October 3, 2026. **Decision:** reject this crop identity checkpoint. Keep the existing SC handoff and insertion switch disabled. This was a supervised perception diagnostic, not insertion or RL training. The four reserved final IK configurations were not opened.

## Question and fixed gate

The preceding native-RGB shadow run reached a real port-0 transport standoff but rejected the selected port in its center view. I tested whether a 160×160 crop from the **original 1152×1024 camera image**, together with frozen full-image features, measured TCP and requested-port code, would rescue that arrival without accepting another or absent port. The crop center came from the existing RGB locator's **predicted** point. Simulator projection was a label and review aid only; it did not choose an inference crop.

Before fitting, [four new complete one-port reset configurations](/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/manifest.json) were fixed, two per SC target. Their reset hashes do not overlap the prior one-port manifest. The old eight one-port fitting groups and 83 two-port training groups supplied the 2,952 sampled image rows (all three cameras, each scored for both requested ports); the four old one-port development groups were excluded from fitting. The new four resets were also excluded. The selected-port threshold remained 0.5 in **all three views**. Promotion required the saved port-0 miss to become a *safe accepted* handoff, preserved earlier good controls, no accepted absent or hidden request, and at least 95% reviewed clear-arrival recall. Live insertion and complete latency were conditional on that gate.

The training labels are projected-in-frame positives and physically absent-port negatives. Projection does **not** certify visibility. The final camera crops from the new resets were visually inspected as a small qualitative audit: [port-0 clear example](/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/manual_review_panels/crop_development_p0_a_4.jpg), [port-0 partly blocked example](/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/manual_review_panels/crop_development_p0_b_4.jpg), and [port-1 example](/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/manual_review_panels/crop_development_p1_a_4.jpg). These are **oracle-centered for human review only**; the model's crop was selected by its image locator. The partly blocked port-0 frame is projected inside all views, demonstrating that the numeric projection recall below is only a proxy.

## Matched results

The frozen spatial identity model and new crop model were scored on the exact same 144 synchronized triplets from the four new reset groups. There were 135 triplets where the real selected opening projected inside all cameras.

| Frozen check on new groups | Real projected all-view recall | Physically absent port accepted in all views |
| --- | ---: | ---: |
| Existing spatial identity | **134/135** | **0/144** |
| New original-RGB crop identity | **102/135** | **25/144** |

The new crop missed 26/33 projected triplets in one port-0 reset and accepted the physically absent port on 21/36 triplets in one port-1 reset. Both are complete-reset failures, not random-frame holdout noise. On the older saved JPEG proxy replay, it retained 4/4 good two-port handoffs and rejected 4/4 swapped requests, 4/4 early frames and 20/20 provisionally hidden frames. Those controls did not predict the failure on new reset groups.

The crop model raised the known missed port-0 identity scores to 0.746/0.754/0.747, but the **complete verifier still rejected it**: the measured TCP posture differed by **6.71°** from the image-derived selected-port orientation, above the existing 3° safety threshold. Its measured TCP standoff discrepancy was 1.29 mm. Thus even the one identity rescue did not create a safe handoff. A previously accepted port-1 control remained accepted.

The failure has a concrete representation cause. In the new port-1 reset with 21 absent-port identity accepts, the frozen locator put the real and absent requested crop centers only **2.3, 10.3 and 4.6 pixels apart** in center, left and right images at the final sampled instant. Both 160-pixel crops therefore showed almost the same physical opening. The global branch of this small crop model pooled away spatial layout; it did not recover the selected-port distinction that the existing spatial head preserves. This is an inference from those observations, not a proof that all crop representations would fail.

## Decision and next controlled repair

The fixed recall and absent-port gates failed decisively. No checkpoint was promoted, no new model commanded motion, and no live or attached-actor latency claim is made. The opt-in crop path stays off by default. The previously frozen verifier remains the comparator, but the overall transport-to-insertion switch still has the older port-0 and SFP gates open.

The next narrow repair should keep the **spatial identity head** for port selection and use native crops only for precise pose refinement, as the existing verifier already does. Diagnose why the known port-0 proposal has a 6.71° image-derived/measured TCP posture discrepancy: replay the exact board-angle estimate, commanded TCP orientation and measured orientation through the final transport decisions, then test one transport-orientation correction on that development reset and one new matched reset. Stop if the correction fails to make a safe, stable handoff or harms the four preserved controls. Do not train another generic crop classifier or attach insertion/RL from this failed result.

## Reproduction and artifacts

```bash
# Rootless Docker; one Gazebo GPU (2) and one offline fit GPU (1).
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
root=/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1
.pixi/envs/default/bin/python scripts/prepare_sc_crop_identity_development.py \
  --source-port0 /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1 \
  --source-port1 /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1 \
  --prior-manifest /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1/manifest.json \
  --output "$root"
# For each of crop_development_p0_a, p0_b, p1_a and p1_b:
AIC_AUDIT_GPU=2 AIC_AUDIT_SMOOTH_CAPTURE=1 AIC_AUDIT_DEADLINE_SEC=180 \
  bash scripts/run_recovery_gazebo_diagnostic.sh "$root/<scene>" stock
# Extract scored_stream.json in the pinned rootless aic_eval image, then:
.pixi/envs/default/bin/python scripts/build_sc_original_rgb_projection_audit.py \
  "$root/<scene>/scored_stream.json" --output "$root/<scene>/projection_state_v2" --stride-s 1

prior=/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1
PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/train_sc_observation_crop_identity.py \
  --two-port-labels outputs/experiments/2026-10-02_two_family_perception/projection_labels/rows.jsonl \
  --one-port-train "$prior"/train_*/projection_state_v2/rows.jsonl \
  --one-port-development "$prior"/development_*/projection_state_v2/rows.jsonl \
  --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --output "$root/fit_v1" --updates 900 --device cuda:1
PYTHONPATH=scripts .pixi/envs/default/bin/python scripts/evaluate_sc_observation_crop_identity.py \
  --projection "$root"/crop_development_*/projection_state_v2/rows.jsonl \
  --locator outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --baseline /var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt \
  --crop-checkpoint "$root/fit_v1/crop_identity.pt" \
  --output "$root/independent_four_reset_comparison_v1.json" --device cuda:1
```

The [NAS artifact directory](/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/) contains the exact manifest, four scene configurations, scored bags and camera captures, projection rows, checkpoint, full per-triplet comparison, saved p0/p1 and good/hidden replay JSON, and manual-review panels. The [machine summary](2026-10-03-sc-original-rgb-crop-identity.summary.json) records the decision and counts. `python -m py_compile`, `bash -n`, and reset-hash uniqueness checks passed. The old p0/p1 and hidden replay images were JPEG proxies; the four new Gazebo captures were original-resolution RGB JPEGs, so a passing gate would still require fresh native-buffer replay.

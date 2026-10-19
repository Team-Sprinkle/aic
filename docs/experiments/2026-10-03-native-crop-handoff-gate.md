# Original-resolution requested-port crop gate

**Date:** 2026-10-03. **Decision:** Step 1 fails; the one bounded Step 2 diagnosis and SC representation repair also fail. Therefore the automatic handoff remains **disabled**, and Steps 3–5 (integrated handoff, insertion actor, full rollouts) were **not run**. No Gazebo data were collected, no training job was interrupted, no RL ran, and the reserved final scenes remained sealed. The supervised branch needs a revised visual orientation/identity method before another handoff attempt.

## Hypothesis and frozen tests

The previous fixed gate found an SC wrong-posture stop that selected-port identity accepted. Its measured TCP was 6.7° from the scored teacher attitude. A one-port-aware frozen RGB orientation head improved independent new-reset near-port p95 to 4.65° but missed the 3° gate. SFP's residual grouped point errors confused the two ports on one card. We tested whether keeping original camera pixels around the requested port fixes these errors. The precise crop **was chosen by RGB model predictions**, never scored port coordinates. Scored geometry supplied training labels and posthoc accuracy/crop audits only. All splits use complete reset configurations.

The fixed Step 1 gate was SC near-port orientation p95 ≤3°, zero known unsafe accepted stops, ≥95% clear-stop retention, resolved SFP adjacent-port confusion, and complete live inference p95 <300 ms. Accuracy and identity failed before stop replay or live integration, so the latter conditions were not assessed with the new heads. A complete latency claim is **not** made.

## SC: 160-pixel native crop orientation

The source had 83 original SC training reset groups and 21 original validation groups with 1152×1024 RGB, plus eight requested one-port training resets and four independent one-port development resets. Seven one-port train groups supplied usable synchronized triplets in this subsample. The frozen RGB locator proposed crop centers. The opening fell inside the 160×160 crop in **96.1%** of 882 sampled one-port camera views; the median/p95 crop-center error was **4.32/26.97 px**. On the old ordinary SC source, the corresponding values were **2.22/5.94 px** and 100% capture. The [reviewed crop montage](/home/nas/chmin/aic_archives/2026-10-03_native_crop_handoff_gate_v1/chmin_aic_20261003_sc_native_crop_audit_v1/montage.jpg) shows the occasional black/off-frame crop as well as clear ports.

We froze the previous orientation head and the existing native SC keypoint backbone, then trained only a small crop-feature orientation residual on **1,524** grouped fitting examples. The matched four-one-port-development sample has **98 triplets**. This is a smaller sample than the prior exact-timestamp 149 near-port triplets, so the numbers below are compared only with the frozen baseline on these **same 98** rows.

| SC orientation on matched 98 held-out one-port triplets | p95 |
| --- | ---: |
| Frozen orientation head | **4.62°** |
| Original-resolution crop with pooled features, selected at update 700/800 | **3.70°** |
| Same pooled head with *oracle-centered crops*, diagnostic only | **3.97°** |
| Targeted spatial-feature repair, selected at update 300/800 | **3.95°** |

The pooled crop helped but missed 3°. Its held-out `development_p0_b`/`development_p1_b` p95 values were **4.32°/4.11°**. Both failures persisted with the spatial-feature repair. The scored opening was inside all `p0_b` crops, with a **10.0 px** crop-center p95, and oracle centering made aggregate p95 worse. Thus crop **centering is not the demonstrated cause**. Reconstructed base-frame port orientation labels were constant within every one-port reset (p95 spread below printed precision), so a moving label does not explain the result. The remaining cause is not proven; fine visual orientation may be too subtle for the frozen pooled/spatial features or inadequately covered by the usable one-port examples. One train reset supplied only two synchronized triplets in this sampler. No further generic fitting sweep was run.

The old original-SC validation sample also improved from **2.79° to 2.27° p95** with the pooled crop, which reinforces that the new one-port reset generalization gap is the deciding result. The spatial variant scored **3.95°** on the same held-out set, so the planned single representation repair did not close it. The known wrong-stop replay was not used to select this checkpoint or claimed as passed.

## SFP: 320-pixel original-image pair crop

We tested a separate same-card correspondence head on **29** native-RGB SFP episodes: 19 fitting groups, five mixed-port checkpoint-selection groups, and five further development groups. It used 870 near-port rows / 2,610 camera views, with both predicted openings on the requested card inside a 320×320 original-image crop in **98.0%** of reviewed projected cases. The 320 crop was resized to 224 for the frozen native SFP feature encoder. Each image was paired with both port requests and both image-proposed candidate points; the classifier received crop features, candidate locations, and the request code, not scored coordinates or a semantic candidate-slot ID. Labels were scored proximity to each requested/neighbor opening. Ambiguous candidate distances were excluded. These are **projection-derived correspondence labels, not certified visibility labels**.

The frozen SFP feature encoder itself had previously been checkpoint-selected on the same five groups used here for verifier selection; it was fitted on the 19 training episodes. The *further* five groups were unseen by that encoder's checkpoint selection as well as by this verifier's fitting and selection. Thus the two halves should not be pooled as ten wholly untouched groups.

The matched frozen V5 low-resolution verifier and the native-crop verifier were scored on the **same candidate points**. A complete constructed triplet accepts only when all three camera scores exceed 0.5:

| Grouped split | Verifier | Correct triplets accepted | Wrong triplets accepted |
| --- | --- | ---: | ---: |
| Five selection resets | Frozen low resolution | 41/87 | 17/423 |
| Five selection resets | Native pair crop | **66/87** | **1/423** |
| Five further development resets | Frozen low resolution | 71/159 | 6/366 |
| Five further development resets | Native pair crop | **147/159** | **1/366** |
| Selection / further development | Require *both* verifiers | 39/87 / 62/159 | 0/423 / 0/366 |

The crop model materially improves correspondence, but the **one false accepted triplet in each split** fails the zero-false criterion. Requiring both models removes these two false accepts at a large loss of valid handoffs. The [marked failure montage](/home/nas/chmin/aic_archives/2026-10-03_native_crop_handoff_gate_v1/chmin_aic_20261003_sfp_native_pair_match_v3/false_triplets_montage.jpg) was visually inspected: yellow is the image-proposed candidate, green the scored requested opening, red the scored sibling. In each false triplet, two cameras pointed near the request while the third proposed a different opening **150–243 px** away; the crop classifier nevertheless assigned that view a high match score. These are offline constructed candidate triplets, **not proven accepted full transport-to-insertion handoffs**. The original `sfp_139` low-resolution cache does not retain original camera pixels, so these native scenes test the failure class rather than replay that exact image.

## Decision at each of the five planned steps

1. **Native crop test:** failed its SC 3° requirement and SFP zero-false requirement. No collection was needed because the existing native SC/SFP sets supplied grouped tests.
2. **One bounded diagnosis/repair:** oracle SC centering did not help; static label consistency passed; a spatial-feature orientation repair failed. SFP marked images expose a per-camera wrong-point acceptance. This step therefore did not restore the gate.
3. **Integrated pause/inspect/reject test:** gated. The candidate heads are not safe enough to authorize handoff.
4. **Insertion actor test from verified autonomous stops:** gated. No such verified autonomous stop population was promoted.
5. **Full autonomous development rollouts and possible RL:** gated. There is no basis to claim new full insertion progress from these perception tests.

The next **explicit plan revision** should first make a full port-face/corner orientation observable, for example supervised corner/keypoint detection in the original images followed by calibrated multi-view pose fitting, and test whether a deliberate safe viewpoint improves the ambiguous one-port scenes. A line between two SC openings has already failed as a full-orientation proxy (5.56° p95 even with oracle pair centers); the new test must use enough independent edges/corners to recover the full frame. For SFP, test camera-consistent candidate geometry plus an invalid/abstain decision on new episode groups; do not lower the threshold simply to retain more positives. Review identifiable visibility separately, because projected-in-frame does not establish it. Predeclare independent reset manifests and the same ≤3°, zero-false, ≥95%, <300 ms gates before any new handoff or actor run. Further routine data collection, actor retraining and RL remain off.

## Commands and artifacts

The [tracked machine decision](2026-10-03-native-crop-handoff-gate.summary.json) records metrics, stage disposition, checkpoint paths, and SHA-256 hashes; an identical local copy is under `outputs/experiments/2026-10-03_native_crop_handoff_gate/`. The [archive](/home/nas/chmin/aic_archives/2026-10-03_native_crop_handoff_gate_v1) preserves manifests, checkpoints, full per-episode summaries, oracle diagnosis, point predictions, marked images and two failed implementation attempts. The three scripts are [SC crop ablation](../../scripts/run_sc_native_crop_orientation_ablation.py), [oracle-centering diagnosis](../../scripts/diagnose_sc_crop_orientation_failure.py), and [SFP pair-crop comparison](../../scripts/run_sfp_native_pair_crop_identity.py). All fitting used one physical GPU per job (SC GPU 1, SFP GPU 2); no more than two new GPU jobs ran at once. No rootless simulator container was started for this offline test.

Canonical commands, run from the repository root with the existing `.pixi` environment, are saved in the archive's `commands.sh`. The SC run used `--updates 800 --per-original-group 16 --crop-size 160`; the sole repair repeated it with `--spatial`. The SFP run used `--updates 1000 --frames-per-episode 30`. Failed setup attempts are preserved separately and did not update model weights.

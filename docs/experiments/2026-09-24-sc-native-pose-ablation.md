# SC native-image pose ablation and scene-diversity continuation

Date: 2026-09-24. Status: **step 3 remains open; no autonomous actor or RL
training from this experiment**. This is a development diagnostic, not final
evaluation. The four reserved final IK configurations remain sealed.
The [machine summary](2026-09-24-sc-native-pose-results.json) records
checkpoint and label hashes, arm metrics, latency, and the failed gate.

## Why these images exist

The 21 verified SC Gazebo episodes saved only 288×256 wrist JPEGs. The scored
bags did not retain camera topics. We added optional capture of the original
1152×1024 RGB image every ten recorded frames and checked a scored smoke on
`trial_000516`: official Tier 3 = 75, total 90.464, 59 native triplets out of
587 observations. The [smoke audit](2026-09-24-sc-native-image-smoke.json)
contains hashes, image sizes, and paths. The first smoke failed with a
collector `NameError` before a frame; its directory was retained, the task
identity bug was fixed, and the retry passed.

Two native capture batches covered 12 already known, nonfinal SC reset
configurations. Eleven scored Tier 3 = 75 in the new runs. Recollected
`trial_000515` scored 18.45 despite its earlier verified success, so its
frames were **excluded** from this successful-expert pose/BC set and its
failure was retained. See [first batch](2026-09-24-sc-native-ablation-capture.json)
and [second batch](2026-09-24-sc-native-ablation-extra-capture.json).
In this failed repeat, the connector stopped about **53.8 mm before the
opening**, while its recorded teacher target remained about 34.4 mm before
the opening for the last roughly 30 simulated seconds. Lateral tip error at
the end was only about 0.07 mm. The teacher's alignment-gate flag stayed
false, and Tier 3 reported no insertion. Off-limit contact scoring detected
none; a brief 23.18 N force peak occurred, but late recorded wrist force was
only about 1–2 N. Thus this is a **commanded-motion stall**, not yet a
verified cable snag or card collision. It is a useful failure candidate for
the later recovery inventory, but not a successful BC trajectory.
The scored bags were joined to each episode by trial ID and a checked
0–15-second bag-start-to-episode-start wall-time offset; no post-reset
geometry is attached to a terminal transition. Camera and port TF then
created **training labels only**. Two visually inspected raw center images
placed the projected SC opening within the blue port at frames 300 of
`trial_000501` and `trial_000516`; the actual files and label coordinates are
under `/var/tmp/chmin_aic_20260918_act/sc_native_ablation_20260924/`.
The source camera FOV is 0.8718 rad. Exact camera information was not saved
with these rows, so subpixel projection accuracy is not established.

## Split correction

The first nine-scene pilot inadvertently put `trial_000506` and
`trial_000516` in its validation set. Both were already training scenes of
the previous 21-episode SC model. Its scratch full-view result of 3.81 mm
near-port lateral p95 is retained as a **superseded exploratory diagnostic**
under `/var/tmp/chmin_aic_20260918_act/sc_native_ablation_20260924/pose_full/`.
It cannot support a warm-start comparison. The corrected labels take the
split directly from `outputs/trajectory_datasets/expert_verified/manifest.json`:
seven distinct train and four validation reset configurations, 658 native
frames, including 264 with tip axial distance under 30 mm. Exactly 74 of
those near-port samples are in the four validation scenes. The
[machine label manifest](2026-09-24-sc-native-ablation-labels-matched.json)
records every admitted/excluded trial and camera in-frame count. The old
21-episode model used the same validation scenes for checkpoint selection;
this comparison is **development evidence**, not an untouched test.

## Matched image and pose results

The same 12.18M-parameter ImageNet ResNet-18 pose architecture was initialized
from the prior 21-episode SC model and fine-tuned for 20 epochs on the same
seven train scene groups. Both arms use ordinary 32D robot state and 10D
task/target encoding. The full-view arm uses the collector's 288×256 images;
the crop arm uses a 224-pixel native crop. A small CNN first predicts plug
and opening pixel centers from the global RGB image. Its output, **never
simulator TF**, selects the evaluation crop. Training-time TF projected
landmarks supervise only the locator. The held-out four-scene locator errors
were 2.64/6.86 pixels median/p95 for the plug and 23.04/65.21 pixels for
the opening. Its model and metrics are under
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation_matched_20260924/locator/`.

| Development arm | Near-port lateral median / p95 | Axial median / p95 | Orientation median / p95 | Decision |
| --- | ---: | ---: | ---: | --- |
| Low-res full view | 0.59 / **1.14 mm** | 2.26 / 3.78 mm | 0.60 / 0.89° | Reference |
| Learned native crop alone | 0.66 / **1.51 mm** | 1.22 / 2.54 mm | 0.86 / 1.15° | Lateral worse; less global context |
| Scalar fusion, weights fit only on train rows | 0.53 / **1.28 mm** | 0.99 / 2.74 mm | 0.70 / 0.94° | No lateral gain at p95 |
| Full view plus learned native-crop feature encoder | 0.39 / **0.91 mm** | 1.21 / 3.45 mm | 0.60 / 0.89° | Best bounded arm; still fails 0.5 mm |
| Geometry-centered native crop alone | 0.55 / **1.52 mm** | 3.53 / 5.25 mm | 1.06 / 1.97° | **Privileged upper-bound diagnostic only**, never deployable |

The feature-fusion model froze the full-view pose estimate and trained a
native-crop encoder to refine **pose**, not action. It has one actor-independent
perception output. The crop-only geometry upper bound is no better than the
learned crop, so crop placement alone does not explain this error. The
full-plus-crop model improves lateral p95 by about 20% over the low-res
full-view reference, but its train near-port p95 is 0.18 mm versus 0.91 mm
on the four validation scenes. This large gap suggests scene generalization
and data diversity are bigger limits. In the best full-plus-crop model,
validation lateral p95 is 0.88 mm for the 45 preopening samples and 0.93 mm
for 29 samples after crossing the opening. Late occlusion is not the only
problem. No arm passes the 0.5 mm p95 gate, so none is promoted to BC/RL.
The two-view feature model has **23.82M total parameters, 8.86M trainable**
for this diagnostic. An offline latency run
over 100 validation observations measured p50/p95/p99 **49.35/52.30/54.73
ms** on one RTX A6000, including JPEG decode, global resize, learned crop,
and pose inference. Its exact report is
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation_matched_20260924/offline_latency.json`.
This is not a complete live inference measurement because ROS acquisition,
the actor, and command conversion were excluded.

### Causal fixed-port history diagnostic

Because the port stays fixed during an episode, each **image-predicted**
relative pose and measured TCP pose gives an observation-based estimate of
the port in the robot base frame. A causal sliding median of those estimates
can be transformed back into the current connector-relative pose. This uses
the known TCP-to-tip tool calibration, but never true port TF or a
geometry-selected crop. It resets with each episode and starts accumulating
only when the model itself predicts the tip within 100 mm of the opening.

On the first four development validation scenes, the same frozen 23.82M
two-view model gave **0.914 mm** lateral p95 without history, **0.599 mm**
with ten recent eligible observations, and **0.510 mm** with 40. The 40-frame
train p95 was 0.304 mm. This nearly reaches the 0.5 mm corridor but does
not pass it; it may also react slowly if the port moves, although this task
assumes a fixed port after reset. The first four validation scenes were
examined while forming the rule, so the 40-frame choice is **exploratory**.
The rule is now frozen as the longest tested history whose training p95
stays at or below 0.5 mm. The three additional original validation resets
being recollected will test this choice without further tuning. Full
candidate and per-episode metrics are in
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation_matched_20260924/causal_filter.json`.
Native captures are 0.55–0.9 seconds apart, so 40 entries span much longer
than 40 control ticks; a live implementation must keep at least 0.5 seconds
between history updates to preserve the tested timescale.

This table is a matched **development** comparison. Each arm selected its
epoch using these validation scenes; the reported p95 is optimistic for new
resets. The low-res model's four-scene metric comes from the exact paired
prediction evaluation, which differs slightly from its training-loop p95
because of percentile interpolation. Selected checkpoints, full histories,
and phase/per-scene metrics are under
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation_matched_20260924/`.
End-to-end live latency has not been measured for the new two-view arm; it is
not deployable under the 300 ms gate yet.

## Expanded verified-scene check and pose-only offset augmentation

The remaining ten original SC resets were recollected with native images. Six
scored full Tier 3; four repeated resets did not. We retain the four failures
for **pose supervision only**, since scored TF still labels their geometry,
while keeping them out of successful BC data. Together with the first 12
captures, the expanded set has 22 scored episodes from 21 distinct reset
configurations, 1,485 native frames, and 594 near-port frames. The original
episode-grouped train/validation split remains 14/7 reset groups. See the
[capture audit](2026-09-24-sc-native-recollect-capture.json) and
[label counts](2026-09-24-sc-native-expanded-pose-labels.json).

The four-scene pilot's 0.91 mm p95 did not generalize across all seven
development validation resets. On the expanded set, the low-resolution pose
model reached 2.01 mm lateral p95, and adding native crop features reached
2.01 mm. The previously chosen 40-sample causal pose filter reached 1.63 mm.
On three newly recollected validation resets, the frozen pilot model and
filter gave 3.00 mm p95 overall. The one full-success reset reached 0.265 mm;
two partial or failed resets were much harder. At one failure, true near-port
lateral offset remained around 6.6 mm while the model underestimated it by
roughly 3 mm. This points to a failure-state distribution gap, but it does
not prove occlusion is the sole cause.

A small scratch native-crop heatmap followed by calibrated three-camera ray
intersection gave 3.32 mm validation lateral p95. With **ImageNet-pretrained
ResNet-18** spatial features, the same approach improved to 1.33 mm. Direct
triangulation of the projected training-label pixels is effectively exact,
which checks the ray and frame conventions; those pixels are never an
autonomous input. The pretrained model's held-out opening pixel error is
0.36/4.53 pixels median/p95, but a few pixels can still become millimetres
after triangulation. Full details and checkpoints are under
`/var/tmp/chmin_aic_20260918_act/sc_native_expanded_20260924/keypoint_pretrained/`.

To get observations at larger lateral errors, a **privileged data-collection
teacher** was perturbed on one existing training reset. It produced 230
native frames, including 168 within 30 mm axially of the opening; near-port
lateral offsets ranged from 0.24 to 14.26 mm. This run reached only partial
insertion (Tier 3 = 38.21), so its images/TF are labeled for perception only.
Adding this one scene to the same pretrained keypoint model reduced the
unchanged seven-scene validation p95 from 1.33 to **0.87 mm**. The model still
misses the 0.5 mm gate. A frozen 40-sample median made this keypoint result
worse (1.05 mm), so it is not selected. The [perturbation capture](2026-09-24-sc-near-perturb-pilot-capture.json)
and [label audit](2026-09-24-sc-near-perturb-pilot-labels.json) identify the
source and exclusion from BC. The exact run command is in
`/var/tmp/chmin_aic_20260918_act/sc_near_perturb_pilot_20260924/run.sh`.

The three-view RGB locator, native crops, pretrained keypoints, and ray solve
measured **31.5/33.4/33.9 ms p50/p95/p99** offline on one A6000 over 100
validation observations. ROS acquisition, actor inference, and command
conversion remain unmeasured. The report is
`/var/tmp/chmin_aic_20260918_act/sc_native_offset_aug_20260924/offline_latency.json`.

Four more already-designated **training** reset configurations were collected
with the same perturbation schedule. All four were scored: two full Tier 3 =
75, and two partial/failed at 25 and 18.77. All 765 native frames (394
near-port) entered the perception-only augmentation; none of these perturbed
traces was added to BC. The offsets cover mild-to-large misalignment, with
one failed scene at 24.12 mm median lateral error within the 30 mm axial
window. The [capture](2026-09-24-sc-near-perturb-multiscene-capture.json) and
[pose-label summary](2026-09-24-sc-near-perturb-multiscene-labels.json) retain
all four scores and scene identities. The exact Gazebo command is in
`/var/tmp/chmin_aic_20260918_act/sc_near_perturb_multiscene_20260924/run.sh`.

With these four scenes plus the first perturbed scene, the same 2,200-update
ImageNet spatial refiner reached **0.079/0.260 mm** median/p95 on the same
seven development validation resets (231 near-port frames), and 0.312 mm
axial p95. This pooled metric meets 0.5 mm, but one fully inserted validation
reset, `trial_000505`, still has **1.64 mm per-episode lateral p95**. A late
center-camera frame chose a different visible blue feature while the
TF-projected lower target opening was partly behind the held plug: center
pixel error was about 121 px, versus 2.3 px in the left camera. Supplying the
normal task target-port one-hot to the heatmap refiner cut that scene's p95 to
**1.29 mm** while pooled p95 stayed **0.260 mm**. This suggests target/view
ambiguity and occlusion rather than a universal spatial resolution limit.
It is still a development-set hypothesis, not a proven causal explanation.
The conditioned checkpoint's offline path measured 35.5/40.4/42.4 ms
p50/p95/p99. Both model variants and per-scene metrics are under
`/var/tmp/chmin_aic_20260918_act/sc_native_multioffset_aug_20260924/`.

The exact conditioned ablation adds `--condition-target-port` to the
`train_sc_native_keypoints.py` command above and uses
`sc_native_multioffset_aug_20260924/{labels,crops}.jsonl`. The matching
`predict_sc_native_keypoints.py` checkpoint loads the one-hot from the normal
task vector; simulator geometry never selects the output during inference.
The seven development validation resets influenced checkpoint selection, so
the pooled 0.260 mm figure is not an independent test. A five-reset fresh
SC-only development suite generated with seed `2026092401` was captured
at `/var/tmp/chmin_aic_20260918_act/sc_fresh_dev_20260924/`. It was generated
after these checkpoints were frozen and does not use the four reserved final
IK settings.

### Fresh-scene locator failure and spatial replacement

The five new SC-only reset configurations were subsequently scored with the
frozen first-stage locator and refiner: three full insertions and two
partial/failures, 771 native frames, 411 near-port frames. The small pooled
CNN locator had **101/217 px** median/p95 target-opening error across visible
fresh views; the resulting three-camera lateral p95 was **25.37 mm**. A raw
image in `trial_900003` plainly shows the blue port above the held plug while
the locator centers the crop near the plug. This is a coarse localization
generalization failure, not evidence that the high-resolution crop itself
cannot resolve submillimetre detail. See the [fresh capture](2026-09-24-sc-fresh-dev-capture.json)
and [validation-only label summary](2026-09-24-sc-fresh-dev-labels.json).

A bounded spatial locator uses ImageNet ResNet-18 layer features and a global
heatmap, conditioned on the ordinary target-port one-hot. It was trained on
the **earlier training scenes only**; these five fresh resets were never used
for its gradients or checkpoint selection. On them, target-opening error
fell to **2.06/6.48 px** median/p95 near the port. Feeding its image-selected
224-pixel crops to the previously frozen conditioned keypoint refiner reduced
fresh lateral p95 to **0.745 mm** and axial p95 to 0.526 mm. Per-scene lateral
p95 was 1.271, 0.300, 0.586, 0.782, and 0.396 mm. This still fails the
0.5 mm target on multiple scenes. Offline perception latency was
34.5/36.4/36.6 ms p50/p95/p99 on an A6000, excluding ROS and actor work.
The checkpoint, per-scene metrics, and latency are under
`/var/tmp/chmin_aic_20260918_act/sc_native_multioffset_aug_20260924/locator_v2/`
and `/var/tmp/chmin_aic_20260918_act/sc_fresh_dev_20260924/`.

Retraining the high-resolution refiner on crops chosen by the new locator,
using only the old training scenes and equal 2,200 updates, **worsened** the
five fresh scenes to 1.37 mm p95. That arm is retained at
`sc_native_multioffset_aug_20260924/keypoint_v2crops/`; it is not selected.
The next iteration collects ten **new training** SC resets, including failed
ones for pose supervision only, to broaden the image distribution. A later
independent fresh suite will be needed after that fitting. These are
development checks; the final IK set remains sealed.

## Next controlled iteration

Keep the original manifest split and frozen checkpoints while scoring the
fresh development suite. The per-episode `trial_000505` error still exceeds
the insertion corridor despite the pooled metric. Investigate occluded or
ambiguous target views and orientation before the actor gate. Do not promote
the actor or RL branch from the pooled development metric alone.

## Exact commands and artifacts

Native batch commands are preserved in each batch's `run.sh`; scored YAMLs,
bags, episodes, collector metadata, and raw images are under their respective
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation*_20260924/` directories.
The actual host tools are:

```bash
.pixi/envs/default/bin/python scripts/audit_sc_native_capture.py BATCH AUDIT.json
# The next command runs inside the official rootless ROS container after
# sourcing /opt/ros/kilted/setup.bash and /ws_aic/install/setup.bash.
python3 scripts/extract_sc_native_batch_edges.py BATCH BATCH/edges_joined.json
.pixi/envs/default/bin/python scripts/merge_sc_native_edge_manifests.py MERGED.json BATCH1/edges_joined.json BATCH2/edges_joined.json
.pixi/envs/default/bin/python scripts/build_sc_native_pose_labels.py MERGED.json \
  --tcp-tip-calibration configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json \
  --port-opening-calibration configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json \
  --labels-output LABELS.jsonl --summary-output SUMMARY.json
```

The remaining exact model commands are the `train_sc_native_locator.py`,
`train_sc_pose_probe.py`, `evaluate_sc_pose_fusion.py`, and
`train_sc_full_crop_fusion.py` invocations in the shell history and their
`metrics.json` files. The model source and outputs should be snapshotted
together before promotion. All geometry-selected crops are marked oracle
and excluded from any autonomous evaluation.

The expanded spatial check ran inside `aic_sc_native_gpu2` using the
rootless Docker socket (`DOCKER_HOST=unix:///run/user/1008/docker.sock`) and
the Isaac Sim Python environment. Its exact core commands, with the common
root abbreviated as `RUN=/var/tmp/chmin_aic_20260918_act`, were:

```bash
python scripts/train_sc_native_keypoints.py \
  --native-labels "$RUN/sc_native_expanded_20260924/pose_labels_all_scored.jsonl" \
  --crop-labels "$RUN/sc_native_expanded_20260924/locator/pose_labels_predicted_crops.jsonl" \
  --output-dir "$RUN/sc_native_expanded_20260924/keypoint_pretrained" \
  --pretrained-resnet /sc_fix/resnet18-f37072fd.pth \
  --updates 2200 --batch-size 32 --crop-size 224
python scripts/predict_sc_native_keypoints.py \
  --crop-labels "$RUN/sc_native_expanded_20260924/locator/pose_labels_predicted_crops.jsonl" \
  --checkpoint "$RUN/sc_native_expanded_20260924/keypoint_pretrained/refiner.pt" \
  --output "$RUN/sc_native_expanded_20260924/keypoint_pretrained/all_predictions.jsonl"
python scripts/evaluate_sc_native_triangulation.py \
  --labels "$RUN/sc_native_expanded_20260924/pose_labels_all_scored.jsonl" \
  --predictions "$RUN/sc_native_expanded_20260924/keypoint_pretrained/all_predictions.jsonl" \
  --edges "$RUN/sc_native_expanded_20260924/edges_joined.json" \
  --output "$RUN/sc_native_expanded_20260924/keypoint_pretrained/triangulation.json"
```

The one-scene offset augmentation repeated the same 2,200-update keypoint
training command with paired inputs under `sc_native_offset_aug_20260924/`.
Its `labels.jsonl` and `crops.jsonl` contain the expanded 1,485 frames plus
230 pilot frames; the pilot is aliased to `trial_000601` while preserving its
original source capture directory and training scene hash. The unchanged
validation rows are identical byte-for-byte in content. The perturbation
collector's exact invocation is preserved in its `run.sh`. The offline
latency was measured with `scripts/benchmark_sc_native_keypoint_latency.py`
on the augmented checkpoint.

## Broader reset check and causal 3D estimate

Ten newly generated SC **training** resets added 1,507 native-image frames;
four scored full insertion and six yielded partial or failed trajectories.
All ten are pose-supervision only. Combined with the earlier captures and
offset repeats, the broadened manifest has 4,758 frames, 24 training and 12
development validation scene groups. Training used only the training groups;
the four reserved final IK configurations remain sealed. The exact sources,
failed trajectories, labels, and crop paths are under
`/var/tmp/chmin_aic_20260918_act/sc_native_broadened_20260924/`.

The frozen spatial locator and landmark refiner had **1.049 mm** near-port
lateral p95 before temporal filtering on the 12 development validation
scenes. On a separate five-reset suite, the raw p95 was **13.335 mm**; one
inserted scene produced wrong-port detections in the partially occluded left
view. This is a real visual failure, retained in the raw report. Three of
the five resets had near-port frames.

The target opening is fixed during an episode. The causal filter therefore
triangulates each image-predicted target pixel from three views into the base
frame, checks the three-view reprojection residual, and keeps a median of up
to 40 accepted observations. It accepts updates no faster than every 0.5 s,
only when the predicted target is within 100 mm of the measured plug tip and
within 5 mm of the current visual anchor. The **1.5-pixel** reprojection
gate was rounded above the training near-port p95 of roughly 1.03 pixels.
It uses RGB, measured TCP, and fixed tool/camera calibration; true port TF
appears only in the scoring branch. This produced near-port lateral/axial
p95 **0.081/0.223 mm** across 642 frames from 12 development scenes and
**0.087/0.137 mm** across 404 frames from three near-port episodes in the
separate suite. History already contained at least two accepted observations
for 95.5% and 100% of those near-port frames, respectively. A live actor
must wait for two accepted visual observations before relying on the filter.
This fixed-port assumption would need revalidation if the task board moves.
Exact reports are `sc_native_broadened_20260924/keypoint_conditioned/robust_causal40.json`
and `sc_independent_dev_20260924/robust_causal40.json` under the run root.

## SC port orientation: negative and positive arms

Directly regressing a quaternion from the old full view gave **3.85°**
near-port orientation p95 on fresh scenes. A broadened orientation-only head
gave **2.49°**; a direct native crop head gave **2.41°**. Five virtual
port-axis landmarks, predicted from the observation-selected native crops
and triangulated across three cameras, gave **1.85°** after a causal
orientation average. An ImageNet ResNet-18 that directly regressed the one
varying SC board yaw from three global views and measured state fit training
but generalized poorly: **8.42°** p95 on three independent near-port scenes.
The checkpoint and negative result are retained in `sc_native_broadened_20260924/board_yaw/`
and `sc_independent_dev_20260924/frozen_board_yaw/`.

Scored **training** TF shows board roll/pitch and the board-to-port mounting
rotation stay fixed across these SC scenes. We saved that fixed CAD-style
rotation in `configs/hierarchical_recovery/sc_board_to_port_training_fixed.json`.
This is valid for the toolkit's qualification-like profile, where board
roll/pitch and SC port-local yaw are fixed. The `training_broad` generator
allows those angles to vary; the yaw-only estimator must be disabled or
retrained as a full orientation estimator for that profile.
At inference, each landmark-predicted port rotation is projected onto the
single remaining board-yaw degree of freedom, then causally averaged. This
uses no scene-specific simulator orientation. On the independent three
near-port episodes, the original 2-pixel reprojection gate gave **0.893°**
p95, but across the 12 development scenes it initialized too late in one
episode and gave **1.093°** pooled p95 with a 37° per-scene outlier. A
3-pixel gate for the first two accepted observations followed by the original
2-pixel gate reduced this to **0.491°** pooled p95 and 98.0% initialized
coverage on the 12 development scenes. On the already examined independent
suite it gave **0.898°** p95 and 97.8% initialized coverage. Per-scene
development p95 still reached **0.685°**. This initialization rule was
selected after inspecting development errors, so a new yaw-edge scene suite
must check it before promotion. Frames without two accepted observations
must not be counted as ready for autonomous insertion.

The complete **offline** RGB decode, resize, spatial locator, two-landmark
translation head, five-landmark orientation head, and ray solves measured
p50/p95/p99 **39.4/43.3/46.0 ms** on one RTX A6000 across 100 native
triplets. This excludes ROS image delivery, the actor, and command conversion;
the full live 300 ms gate remains open. The exact report is
`sc_independent_dev_20260924/offline_full_pose_with_landmarks_latency.json`.
The filtered translation and yaw-only orientation are perception diagnostics,
not autonomous insertion outcomes. No shared BC or RL actor has consumed
these estimates yet.

## Yaw-edge continuation and frozen confirmation

The official qualification-like scene generator was run at both ends of its
board-yaw range. Twelve new **training** reset groups (`trial_930001`–
`trial_930012`) produced 860 native frames, including 412 near-port frames.
Eight scored full Tier 3; four were partial. All were used for pose
supervision only, regardless of score; they have not been admitted to the BC
expert manifest. [Capture audit](2026-09-24-sc-yaw-extremes-capture.json) and
[label audit](2026-09-24-sc-yaw-extremes-labels.json) retain every score and
the exact scene hashes. A matched 2,200-update retrain used 36 training and
the same 12 old validation scene groups, 5,618 native frames in all. The
older validation scenes stayed out of gradient updates. Its checkpoints,
labels, predicted crops, logs, and metrics are under
`/var/tmp/chmin_aic_20260918_act/sc_native_yaw_aug_20260924/`.

On six separately generated yaw-edge **development** resets (`trial_940001`–
`trial_940006`), four reached within 30 mm of the opening; two of six scored
full Tier 3. The old position checkpoint gave **0.503 mm lateral / 0.590 mm
axial p95** across 208 near-port frames; its causal history never initialized
in one fully inserted episode. The retrained position checkpoint gave
**0.141 / 0.106 mm p95**, with at least two accepted observations for every
near-port frame across all four episodes. The old/new orientation checkpoints
gave **0.965° / 0.804° p95** under the frozen yaw constraint, but the new
orientation history never initialized in one fully inserted episode. Its
five-landmark reprojection residual was about **11.94 pixels median** there,
even though the separately predicted port opening had initialized. The
center view visibly shows the plug and gripper housing overlapping the blue
target. Relaxing the position reprojection gate alone created bad anchors
and worsened the old development result; those negative diagnostics remain
under `sc_native_broadened_20260924/keypoint_conditioned/causal_init*.json`.

For a cold orientation history only, a bounded fallback now accepts a yaw
estimate when its landmark-predicted opening lies within **0.5 mm** of the
independent causal RGB position anchor and the board yaw lies within **0.2
rad** of nominal. Once two orientation samples are accepted, the ordinary
2-pixel landmark reprojection gate resumes. This uses neither true port TF
nor teacher commands. It gave **0.596° p95** with 100% initialized coverage on
the 12 old development scenes and **0.806° p95** with 100% coverage on the
six yaw-edge scenes' four near-port episodes. The fallback was designed after
looking at those scenes, so neither suite is untouched evidence for it.
Exact scored reports are `sc_native_yaw_aug_20260924/orientation_with_initial_position_fallback.json`
and `sc_yaw_independent_20260924/orientation_with_initial_position_fallback.json`.

Before collecting another scene, the locator, translation, orientation,
fixed board mounting rotation, filter settings, and source-file hashes were
frozen in `sc_pose_frozen_eval_20260924/frozen_bundle.json`. Eight new
qualification-like yaw-edge development resets (`trial_950001`–
`trial_950008`) are being collected with seeds 2026092408/09. They are
separate from the four reserved final IK configurations. Report their raw,
filtered, per-episode, and initialization metrics before opening shared BC.
If the position or orientation gate fails on these new starts, do not promote
the perception bundle on pooled p95 alone.

### Frozen-suite failure and board-rail yaw ablation

The eight-scene frozen suite scored three full insertions and yielded 411
near-port frames from seven episodes. The held position head passed its
diagnostic: **0.214 mm lateral / 0.167 mm axial p95**, with all seven causal
histories initialized. The held orientation head **failed**: **1.608° p95**,
driven by `trial_950008`, whose orientation history was accepted early but
then stopped updating as near-port landmark reprojection residuals rose to
about 4.74 pixels median. The three full native views at frame 760 are in
`sc_pose_frozen_eval_20260924/episodes/episode_0008_*/native_{left,center,right}_000760.jpg`.
The blue port is partly hidden by the plug and gripper, but long board rails
remain visible in all views. The exact held bundle, capture scores, and pose
reports are under `sc_pose_frozen_eval_20260924/` with tracked
[capture](2026-09-24-sc-pose-frozen-capture.json) and
[label counts](2026-09-24-sc-pose-frozen-labels.json). This result keeps the
orientation gate open.

A blue-port silhouette ridge probe was also tried and **did not help**:
2.18° near-port yaw p95 on 12 grouped development scenes. Its inputs were
observation-selected crops and robot state; the model, metrics, and source
are `scripts/probe_sc_blue_shape_yaw.py` and
`sc_native_yaw_aug_20260924/blue_shape_yaw.{json,npz}`. The negative arm is
retained.

The next, more geometric diagnostic uses the board itself. Gazebo renders
long straight black rails in the full native views. A coarse landmark yaw
selects Hough lines roughly parallel to the projected board X axis; measured
TCP and fixed training camera calibration convert each line direction into
a yaw candidate. The model takes a robust median across lines and views and
then a causal median across the fixed board's recent accepted yaw estimates.
Its inference does **not** read scored board/port TF; TF labels only measure
the resulting error. On the 12 grouped development scenes it reached
**0.064° near-port yaw p95**, 642 frames and 12/12 initialized episodes.
On the already-inspected eight-scene suite it reached **0.078° p95**, 411
frames and 7/7 initialized episodes; the previously bad `trial_950008`
was **0.031° p95**. These are still development results because that scene
motivated the method. The line stage measured **64.5/88.8/97.3 ms**
p50/p95/p99 offline for three native JPEG decodes, Canny/Hough, and ray
geometry, excluding upstream CNNs, ROS, actor, and commands. The unchanged
CNN stage measured **37.5 ms p95** separately. The separate p95 values should
not be presented as a measured complete live percentile. Source, exact
predictions, per-episode results, and timings are
`scripts/evaluate_sc_board_lines_yaw.py`,
`sc_native_yaw_aug_20260924/board_lines_yaw.json`, and
`sc_pose_frozen_eval_20260924/{board_lines_yaw,board_lines_latency}.json`.

This method assumes straight visible rails and the qualification-like fixed
board roll/pitch. It may fail on occluded rails, changed board art, or Isaac
rendering differences. A new eight-scene development suite with seeds
2026092414/15 was generated only after freezing the line rule, camera/tool
calibration, checkpoints, and code hashes in
`sc_board_lines_frozen_eval_20260924/frozen_bundle.json`. Report its
per-episode error and initialization before promoting SC orientation.

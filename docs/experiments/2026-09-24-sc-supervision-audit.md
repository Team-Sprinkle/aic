# SC-to-SC label, timing, and image audit

Date: 2026-09-24. Status: **fresh frozen full-pose check passed pooled lateral
accuracy; scene outlier, live latency, and autonomous-policy gates remain open**. This is step 3 of the
[shared actor execution plan](2026-09-24-shared-actor-sc-serl-execution.md).

## Fourth independent frozen check (September 24)

After the third suite exposed occasional rail-yaw search-boundary jumps,
`evaluate_sc_board_lines_yaw.py` gained a fixed rejection for candidates
at that boundary. The code, image checkpoints, calibration, and thresholds
were hashed in
`/var/tmp/chmin_aic_20260918_act/sc_yaw_boundary_refreeze_eval_20260924/frozen_bundle.json`
before eight new scenes (`trial_990601`–`trial_990608`, generator seeds
2026092421/22) were captured. Neither these scenes nor the four reserved
final IK starts were used to choose another threshold.

Four scenes yielded 154 near-port frames. The physical plug-to-opening
relative pose had lateral **0.156/0.294 mm median/p95**, axial
**0.201/0.309 mm**, with both RGB-based port and tip histories initialized
on every near frame. Port-only lateral p95 was **0.209 mm**; physical tip-only
lateral p95 was **0.270 mm**. Board-line yaw was **0.048/0.084°** median/p95,
initialized on all near frames; worst per-scene yaw p95 was 0.156°.
The latter scene accepted a new rail candidate on only 14.3% of near frames
because rails disappeared during contact views; its causal held estimate
remained within 0.156°. This is an offline teacher-view result, with no
privileged geometry used by the image predictions or crop locator. The
labels and scored TF are read only by the separate scorers. Complete
ROS+actor+perception inference p95 and autonomous insertion remain open.

The original scored MCAPs were archived to
`/home/nas/chmin/aic_archive_20260924/sc_yaw_boundary_refreeze_eval_20260924/`.
Each compressed bag was decompressed and SHA256 checked before deleting its
local source; exact restore commands and digests are in
[the archive manifest](2026-09-24-sc-yaw-boundary-refreeze-mcap-archive.json).
Original camera images, physical-tip labels, image-only predicted crops,
port/tip/orientation predictions, causal trajectories, and all metrics are
under the same `/var/tmp/.../sc_yaw_boundary_refreeze_eval_20260924/` root.

## Third independent frozen check (September 24)

After the original port tracker failed on a second eight-scene suite, the
observation-only pairwise tracker was fixed and its model/code/threshold hashes
recorded in `/var/tmp/chmin_aic_20260918_act/sc_port_tracker_refreeze_eval_20260924/frozen_bundle.json`
**before** generating eight new Gazebo scenes, trial 990501–990508. None of
these scenes overlaps the inspected suites or the reserved final IK scenes.
Five of eight scored full insertion. Six episodes contributed 228 near-port
frames; the other two are retained as route failures, not silently counted as
pose successes. Frozen RGB inference used all three native wrist images,
learned crop selection, measured TCP, and a camera calibration fixed on
training scenes. Scored TF was read only by the metric scripts afterwards.

| Held-out quantity | Result |
| --- | ---: |
| Predicted port lateral error, p95 | 0.181 mm (227 initialized frames) |
| Predicted physical tip lateral error, p95 | 0.304 mm (228 frames) |
| Combined physical tip-to-opening lateral error, median / p95 | 0.213 / 0.443 mm (227 frames) |
| Combined axial error, median / p95 | 0.174 / 0.278 mm |
| Combined near-frame initialization | 227/228 |
| Relative yaw error against scored opening TF, median / p95 | 0.039° / 0.672° |
| Complete offline native-image perception latency, p50 / p95 / p99 | 125.05 / 147.36 / 155.11 ms |

Pooled lateral p95 is under the 0.5 mm corridor, but trial 990501 has **1.004
mm per-episode lateral p95** and trial 990507 has **0.981° yaw p95**. The yaw
outlier's rail measurements were accepted on only 20% of near frames. A
prediction-only trace of trial 990507 shows the initial rail history near the
scored 1.64° truth (1.55°, then 1.60°), then two later accepted candidates
at -0.22° and -0.64° pulled the retained median to 0.66°; most subsequent
frames had no valid rail lines. A tempting 0.5° history-jump rejection was
tested on the **prior development suite**, where it raised yaw p95 from
0.132° to **4.393°** by locking bad early estimates in two episodes. It is
disabled by default and is not promoted. The unmodified frozen third-suite
score remains the reported result. These
are real held-out failures; do not tune on this suite. Diagnose comparable
training/development scenes and then run a newly frozen suite. Offline timing
excludes ROS transport, the shared actor, and command conversion; it is not a
live 300 ms guarantee.

Exact predictions, frame-level scores, frozen bundle, source scenes, and
captured RGB are in the root above (`port_metrics.json`, `tip_metrics.json`,
`relative_metrics.json`, `yaw_metrics.json`, `full_perception_latency.json`).
The eight MCAPs were losslessly compressed, SHA256-verified, and moved to NAS;
[restore commands](2026-09-24-sc-port-tracker-refreeze-mcap-archive.json) are
recorded. The images and extracted TF labels remain local. This check does not
establish autonomous insertion.

## Corrected geometry contract

The 21 score-verified SC Gazebo episodes have 13,349 recorded image/teacher
rows. Each row stores the gripper TCP pose and teacher absolute TCP target in
`base_link` as position plus **XYZW quaternion**. The old
`scripts/audit_sc_port_targets.py` read that quaternion as WXYZ; its
round-trip check was self-consistent and therefore passed a wrong label. The
bag-251 terminal port-frame orientation differs by **153.7 degrees** between
the old and corrected conversion. The [regenerated v2 port audit](2026-09-24-sc-port-target-audit.json)
uses XYZW explicitly. The old derived port-frame JSONL was not used for
policy training.

Scored Gazebo TF defines the selected `sc_port_base_link`. A separate
`sc_port_base_link_entrance` is the actual opening, **15.640 mm along local
-Z** from the base frame. Independent port-0 and port-1 bag snapshots agree
on that offset to about 0.00002 mm. [Opening calibration](../../configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json)
is a training-label transform. The recorded TCP is `gripper/tcp`; a scored
TF snapshot gives [TCP→SC-tip calibration](../../configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json).
Bag 252 agrees with bag 251 to 0.025 mm and 0.0003°, whereas a port-1 bag
differs by 0.223 mm and 0.018°. That residual needs investigation before
claiming sub-0.5-mm final physical plug calibration across scenes. Comparing the scored
`world`/`aic_world` chains against the bag-251 event snapshot gave under
0.00001 mm base-to-port translation discrepancy.

For a teacher TCP setpoint, the intended connector waypoint is

$$
{}^{O}T_{\mathrm{tip,target}} =
({}^{P}T_O)^{-1}({}^{B}T_P)^{-1}
{}^{B}T_{\mathrm{TCP,target}}\,{}^{\mathrm{TCP}}T_{\mathrm{tip}},
$$

where $B$ is robot base, $P$ is selected port base, and $O$ is the port
opening. The same chain converts the **observed** TCP pose into a pose
training label. The actor must receive only an observation-based estimate;
these TF transforms are never an actor input. Exact TCP-target round-trip
error of the corrected conversion is below $1.8\times10^{-16}$ m.

## Timing and camera provenance

The collection stored one row per observed camera frame. There are 8,585
adjacent 50 ms gaps and 4,655 adjacent 100 ms gaps across the SC episodes.
The older audit counted 13,016 four-command windows with consecutive indices
and allowed gaps up to 250 ms. A strict 50±10 ms rule leaves **2,508**
four-command chunks. The [strict summary](2026-09-24-sc-connector-waypoint-labels-strict-v2.json)
retains those. A second, causal held-target construction repeats the most
recent *recorded* command at an intervening 50 ms tick only when no command
index was skipped; it requires observations through the full 200 ms chunk and
limits hold age to 150 ms. This yields **12,965** chunks (8,308 train, 4,657
validation). Of those, 9,451 repeat one target tick, 1,029 repeat two, and
two repeat all three. The [opening-frame label summary](2026-09-24-sc-connector-waypoint-labels-opening-v2.json)
and ignored JSONL record source command indices and times for every waypoint.
This is a faithful setpoint schedule, not measured next-frame motion.

The [camera provenance audit](2026-09-24-sc-image-provenance.json) sampled
five frames from each episode and found three distinct RGB 288×256 JPEGs
per sampled observation, no missing files or metadata mismatch, and zero
scene-hash overlap between 14 train and seven validation episodes. The label
builder also checks the three image files for every admitted anchor. The
collector saves the cameras, state, and teacher target from the same ROS
observation; the center-camera simulation stamp is the row time. There is
still a possible simulation-camera timing issue to test by synchronized
visual/TF overlays if the pose probe underperforms.

The source camera model is configured for **1152×1024**, but
`CollectCorrectiveCheatCode._save_camera` resized each recorded JPEG to
288×256 before saving it. The scored MCAP bags retain controller, force,
joint, and TF topics but **no camera/image topic**, confirmed with `ros2 bag
info` on bag 251. Therefore the old bags cannot supply native-resolution
wrist images for a contact crop. A true high-resolution crop ablation needs
new Gazebo collection or another retained raw-image source; upscaling these
JPEGs would not restore detail. The collection can save a bounded near-port
subset of raw camera frames and project training-only labels for a learned
locator, while autonomous crop selection must remain observation-only.
The collector now accepts `AIC_CORRECTIVE_NATIVE_IMAGE_EVERY_N_FRAMES` (zero
by default); setting it to ten saves one native JPEG triplet per ten recorded
SC frames and records the original camera shape and file names. A scored
training-scene smoke on `trial_000516` **passed**: official Tier 3 was 75,
total score 90.464, and 59 of 587 recorded frames contain three native
1152×1024 RGB JPEGs with no missing files. See the
[machine audit](2026-09-24-sc-native-image-smoke.json), exact
`/var/tmp/chmin_aic_20260918_act/sc_native_smoke_retry_20260924/run.sh`,
scoring YAML, and raw images under its `episodes/` directory. The first smoke
failed before recording a frame because `_send_delta_pose_target` referred to
an undefined local `task`; this was fixed by retaining whether the current
task is SC on the collector instance. The failed run remains at
`/var/tmp/chmin_aic_20260918_act/sc_native_smoke_20260924/`.

Visual inspection of the native center, left, and right images at frame 580
confirmed more connector detail than the 288×256 copy, but the gripper hides
part of the opening close to insertion. Native resolution alone may not make
the 0.5 mm corridor observable in every view. A bounded scene-grouped native
capture across both SC ports and one to three NIC cards is running under
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation_20260924/`. It remains
privileged **teacher collection**, not autonomous policy evidence.

## Setpoint versus realized motion

The teacher's absolute target often lies well ahead of measured motion. The
[measured-motion audit](2026-09-24-sc-target-motion-audit.json) finds a
10.75 mm median teacher-target/observed-tip gap for near-port frames. Among
near-port commands more than 5 mm away, median motion in the commanded
direction at the next recorded observation is 0.009 mm; 29.5% have negative
projected motion. A large setpoint and slow next step alone do not prove a
bad demonstration: the controller may intentionally move gradually, and
contact may intervene. It does mean that teacher setpoints are **not**
measured one-tick connector waypoints. Before BC, compare the commanded,
executed, and measured trajectories at the insertion event and keep a
quality mask for clearly blocked or post-terminal commands. Do not label
every large target gap as a failure or silently convert it to a 50 ms motion.

## Reproduction

```bash
python scripts/calibrate_sc_tcp_tip_from_tf.py \
  /var/tmp/chmin_aic_20260918_act/bag251_event_tf_aic_sc_fix.json \
  configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json
python scripts/calibrate_sc_port_opening_from_tf.py \
  /var/tmp/chmin_aic_20260918_act/bag251_event_tf_aic_sc_fix.json \
  configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json --port-index 0
.pixi/envs/default/bin/python scripts/audit_sc_port_targets.py \
  --manifest outputs/trajectory_datasets/expert_verified/manifest.json \
  --port-edges docs/experiments/2026-09-24-sc-expert-port-edges.json \
  --output docs/experiments/2026-09-24-sc-port-target-audit.json \
  --labels-output outputs/experiments/2026-09-24_hierarchical_recovery/sc_port_tcp_teacher_targets.jsonl
.pixi/envs/default/bin/python scripts/build_sc_connector_waypoint_labels.py \
  --manifest outputs/trajectory_datasets/expert_verified/manifest.json \
  --port-edges docs/experiments/2026-09-24-sc-expert-port-edges.json \
  --tcp-tip-calibration configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json \
  --port-opening-calibration configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json \
  --resample-mode causal_hold \
  --labels-output outputs/experiments/2026-09-24_hierarchical_recovery/sc_connector_waypoint_chunks_opening_v2.jsonl \
  --summary-output docs/experiments/2026-09-24-sc-connector-waypoint-labels-opening-v2.json
```

## Decision

The **geometry and timing conversion is available**. A matched
[pose-probe comparison](2026-09-24-sc-pose-probe-results.json) trained a
shared ImageNet-pretrained ResNet-18 over three 288×256 wrist JPEGs, ordinary
32D robot state, and the 10D task/target vector. On seven held-out episodes,
near-port lateral median/p95 error was **0.41/1.70 mm**, compared with
**1.35/3.82 mm** for state+task alone. Orientation median/p95 was
0.28/0.70°. The visual probe has 12.18M parameters; its checkpoint is
48.8 MB. Offline JPEG decode plus inference measured p50/p95/p99 of
7.45/8.10/10.79 ms on one GPU, excluding live ROS, crop selection, actor,
and robot command conversion. The frozen old four-scene final split was not
opened. An earlier visual run omitted the task vector; its best lateral p95
was 1.35 mm, but it is not the matched task-conditioned candidate. A first
run with the rootless container's 64 MB shared memory failed in data loading;
the selected run used a separate rootless GPU-2 container with 4 GB shared
memory and eight CPU threads. All checkpoints, metrics, latency, scripts,
and hashes are under `outputs/experiments/2026-09-24_sc_pose_probe/`.

The **0.5 mm p95 gate failed**. The shared BC and RL branches remain closed.
The [bounded native-image ablation](2026-09-24-sc-native-pose-ablation.md)
used a learned RGB-only locator to crop before global resize. It improved
lateral p95 only when fused with the global view, from 1.14 to 0.91 mm on
four development scene groups; a privileged oracle-crop diagnostic did not
improve a crop-only estimator. Recollection of the remaining verified SC
resets is running to test whether the 0.18 mm train versus 0.91 mm
validation gap narrows with scene diversity.
The old 288×256 JPEGs cannot stand in for this ablation. Before BC, also
check commanded versus executed motion around insertion so that unreachable
setpoints are not silently treated as physical 50 ms waypoints. No
autonomous policy has consumed privileged TF labels.

## Fresh native-image confirmation and remaining physical-tip question

The later native-resolution pipeline uses a learned RGB target locator, an
RGB keypoint refiner, and a causal fixed-port filter. Its board yaw estimate
uses long visible rail lines in all three RGB cameras, measured TCP, and
camera/mount calibration fixed from training scenes. The earlier eight-scene
development check exposed 1.608° yaw p95 and motivated the rail estimator.
All checkpoint, calibration, evaluation-code, and threshold hashes were then
frozen in
`/var/tmp/chmin_aic_20260918_act/sc_board_lines_frozen_eval_20260924/frozen_bundle.json`
**before** generating a fresh eight-scene development suite. The hash audit
matched every entry at evaluation time. On its seven episodes reaching within
30 mm of the port (236 frames), the new suite measured port-position error
0.165 mm lateral and 0.183 mm axial p95, plus 0.109° board-yaw p95. Both
causal estimators had initialized histories on every near-port frame. The
eighth episode did not reach the near-port region and is excluded from these
conditional errors, but remains in the scored capture. See
`/var/tmp/chmin_aic_20260918_act/sc_board_lines_frozen_eval_20260924/position_metrics.json`,
`board_lines_yaw.json`, and the [capture audit](2026-09-24-sc-board-lines-frozen-capture.json).
The four reserved final IK configurations remain unopened.

These numbers establish **port opening and board yaw** accuracy relative to
the scored geometry in this development sample. The relative plug-to-port
labels still infer the plug tip from measured TCP and one fixed TCP-to-tip
transform. Scored successful episodes sometimes show a few millimeters of
fixed-proxy lateral offset at the last frame. A dynamic `/scoring/tf` audit
must distinguish cable/plug flex from frame or timestamp error before we
claim that the physical plug-to-port estimate satisfies the 0.5 mm gate.
Complete live ROS-to-action p95 latency and autonomous insertion have also
not yet been measured for this pipeline. The old low-resolution 0.5 mm gate
failure above remains an accurate historical result for that older probe;
it is not the conclusion for this newer native-resolution port estimator.

The first dynamic TF audit has now resolved the main ambiguity. On three
scored episodes, physical plug TF and the camera frame timestamps matched
exactly at every sampled frame. The fixed TCP-to-tip transform was displaced
from the physical tip by **3.32, 1.78, and 1.15 mm** at the terminal frame.
The discrepancy was already present at the first frame and remained nearly
constant through each episode, including near contact. Fast indexed bag
sampling of all 16 pilot scenes showed first-frame fixed-grasp translation
offsets of **1.15–10.58 mm** and first-to-last within-episode changes mostly
below 0.12 mm (one failed trial reached 0.207 mm). This points to a
**scene-specific grasp offset**, rather than mostly dynamic cable flex, as
the reason the fixed proxy disagrees with scored insertion. The scored
physical tip, not the fixed proxy, must supervise the plug head. The old
submillimeter numbers remain valid for **port opening** location; they must
not be described as physical plug-to-port accuracy. See
`/var/tmp/chmin_aic_20260918_act/sc_bc_matrix_pilot_20260924/dynamic_tip_subset.json`
and `dynamic_grasp_calibration.json`. The indexed sampler is
`scripts/extract_sc_dynamic_grasp_calibration.py`; it seeks by bag wall time
but verifies the scored simulation timestamp, and all scored TF stays out of
autonomous inference.

On the same fresh eight-scene suite, the frozen RGB tip head was scored
against the newly sampled **physical** tip rather than its former rigid
proxy. Its near-port tip error was **6.83 mm lateral and 3.21 mm axial p95**
over seven episodes. This explicitly **fails** the plug half of the pose
gate, despite the separate port/yaw success. The physical-tip relabeler is
`scripts/relabel_sc_physical_tip.py` and its [54-episode training-label
manifest](2026-09-24-sc-train-physical-tip-labels.json) retains the original
proxy pose for comparison. The fresh-suite [physical label manifest](2026-09-24-sc-fresh-physical-tip-labels.json)
and raw metric file at
`/var/tmp/chmin_aic_20260918_act/sc_board_lines_frozen_eval_20260924/frozen_physical_tip_metrics.json`
preserve the failed frozen baseline. A corrected RGB tip head is training on
42 scene-grouped training episodes with 12 development episodes. This is a
new, separately selected model; the fresh suite is held out from training
and threshold selection.

The follow-up uses two observation-selected crops per camera: a paired
plug/port crop for the port and orientation heads, and a separate crop
centered on the frozen global RGB locator's **predicted plug point**. The
second crop exposes the plug early in transport, when the paired crop often
falls between the distant plug and port. A tip-only ResNet-18 heatmap head
trains on the 54 relabeled episodes plus 16 additional scored pilot scenes
used for **pose supervision only**, including failed insertion episodes.
No score or physical TF selects either crop at inference. The tip-only head
and its causal stable-grasp filter reached **0.290 mm lateral / 0.286 mm
axial p95** on 12 grouped development episodes (665 near-port frames),
with 100% history initialization. On the earlier inspected eight-scene
suite, it reached **0.458 mm lateral / 0.177 mm axial p95** for the physical
tip; the combined plug-to-opening estimate reached **0.450 mm lateral /
0.174 mm axial p95** over 239 near-port frames in seven episodes, again
with both histories initialized throughout. This design was motivated by
that earlier suite's cold-start failure, so another eight-scene suite was
generated only after freezing all model/code/calibration/threshold hashes in
`/var/tmp/chmin_aic_20260918_act/sc_physical_pose_frozen_eval_20260924/frozen_bundle.json`.
The [model comparison](2026-09-24-sc-physical-tip-model-comparison.json)
preserves the failed fixed-proxy baseline, shared-crop alternatives, and
separate-crop result. An [offline complete perception benchmark](../../scripts/benchmark_sc_full_perception.py)
on 100 retained frames measured **105.8/128.3/133.1 ms p50/p95/p99** for
native JPEG decode, locator, both crops, all CNNs, ray geometry, and rail yaw.
It excludes ROS transport, the actor, and command conversion, so the
end-to-end 300 ms gate is still open.
# Second frozen physical-tip suite and corrected scoring (September 24)

The second eight-scene suite was generated with seeds 2026092417/18 after
freezing all four RGB checkpoints, camera/board calibration, thresholds, and
scoring code in `sc_physical_pose_frozen_eval_20260924/frozen_bundle.json`.
All 15 hashes matched before scoring; all eight scene hashes were new. Its
559 native frames included 283 near-port frames from seven scenes. Scored
physical TCP-to-tip drift within each episode was at most 0.045 mm.

The original causal three-view port tracker **failed** on this frozen suite:
port lateral p95 **45.706 mm**, and combined physical plug-to-opening lateral
p95 **45.535 mm**. In `trial_990207` it accepted a false fixed-port position
while the port was still outside the images. After the port became visible,
correct center/right detections were rejected because they disagreed with
the latched position; the left camera still localized the wrong opening by
about 30 pixels at frame 290. In `trial_990202` the left view was wrong by
about 47 pixels while center/right were within 1 pixel. Prediction-only
three-view residuals identified the disagreement. The separate plug crop
itself achieved **0.197 mm physical-tip lateral p95** with all near-port
histories initialized.

The first reported board yaw p95 of **1.441°** was a *scorer defect*, not an
image-model failure. The old scorer reconstructed the scored opening
orientation from the **fixed** TCP-to-tip rotation, then multiplied by the
new **physical** tip-to-opening label. Scene-specific physical grasps differ
in yaw by about 0.93–1.73° from the fixed proxy in this suite. The corrected
scorer reads the selected port-opening TF directly from the scored bag only
after producing its image estimate. With identical frozen image predictions
the yaw error is **0.132° p95** over 283 near-port frames. The corrected
scorer is `scripts/evaluate_sc_board_lines_yaw.py`; the original and corrected
reports are `sc_physical_pose_frozen_eval_20260924/yaw_metrics.json` and
`yaw_metrics_physical_truth.json`.

A development revision `scripts/evaluate_sc_pairwise_port_tracker.py` now
uses the three-view estimate when its reprojection residual is at most 1.5
pixels. Otherwise it uses left/right for selected SC port 0 or center/right
for port 1. It acquires a candidate only near the **observed TCP/proxy tip**
and after two stable image estimates, holds a fixed target through small
outliers, and can replace a grossly wrong hypothesis after two new stable
observations. It never reads scored port TF or projected label pixels. On the
already-inspected frozen suite it reached **0.421 mm port lateral p95** and
**0.395 mm combined physical plug-to-opening lateral p95**, with all seven
near-port histories initialized. On the earlier eight-scene development
suite it reached **0.164 mm port lateral p95**. These are development
figures; the selected code/settings were hashed in
`sc_port_tracker_refreeze_eval_20260924/frozen_bundle.json` **before**
generating a third eight-scene suite with seeds 2026092419/20. That suite is
the next independent confirmation. The four reserved final IK scenes remain
sealed, and complete ROS+actor latency and autonomous insertion are still
unmeasured.

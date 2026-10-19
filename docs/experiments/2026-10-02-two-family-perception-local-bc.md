# Two-family selected-port perception and local insertion BC

Date: 2026-10-02. Development work only. The four reserved final IK
configurations remain sealed. No RL was run. All runtime estimators use RGB,
measured robot state, public task code, and fixed camera calibration. Scene
geometry and target TF are supervision or scoring labels only.

## Videos to inspect offline

- [Five-card SC wrong-route transport failure](../../outputs/experiments/2026-10-01_transport_reliability/live_sc_visual_standoff_llb_0009_v7_fastresize/five_cameras_10fps.mp4)
- [Five-card SC successful transport handoff](../../outputs/experiments/2026-10-01_transport_reliability/live_sc_visual_standoff_llb_0009_v9_zfloor_repeat2/five_cameras_10fps.mp4)
- [Three-card SC successful transport handoff](../../outputs/experiments/2026-10-01_transport_reliability/live_sc_visual_standoff_llb_0122_v3_trigger80/five_cameras_10fps.mp4)

These are autonomous **transport-only** runs; they stop before local insertion.

## Data and labels

The [projection-label manifest](../../outputs/experiments/2026-10-02_two_family_perception/projection_labels/summary.json)
contains 30,579 synchronized three-camera timesteps from 268 verified SFP
episodes and 104 scored SC Low Lateral Bypass successes. Complete scene/reset
groups remain split: 317 training and 53 validation groups. Every physically
present port is labeled, including 39,498 alternate-port SFP and 12,284
alternate-port SC examples in training. Thus a visible wrong port is a
specific candidate, never silently treated as the selected target. SFP
selected-target supervision is still highly imbalanced: 258 of 268 verified
episodes target card 0, port 1. This limits generalization claims across all
SFP target choices.

The initial labels are **projected aperture centers and projected in-frame
flags**, not actual occlusion or human identifiable visibility. An invisible
but physically in-frame port can have a perfectly known training projection;
the model must not call this a visibility label. The distinction is preserved
in checkpoint metadata and every result below.

The local BC source has 83 SC and 236 SFP training episodes, plus 21 SC and
32 SFP validation episodes before checking whether a causal pose estimate is
available. The buffered SFP local portion begins at port-frame TCP axial
coordinate $z\geq-0.13$ m, about 30 mm before the usual pre-insertion TCP
standoff. SC uses the cleaned local chunks beginning near $z=-0.18$ m; the
SC plug tip is offset from TCP, so TCP-to-opening distance is not interchangeable
with plug-tip distance. The causal RGB estimate is at most 0.6 s old.

## Coarse selected-port locator

The ImageNet-initialized ResNet-18 locator predicts all present candidate
opening pixels and a projected-in-frame probability for each of the three
cameras. At inference, ordinary 10D task encoding chooses which candidate to
triangulate. [Training script](../../scripts/train_two_family_port_locator.py),
[label builder](../../scripts/build_two_family_port_projection_labels.py),
[checkpoint and full held-out metrics](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/summary.json).
Training stopped at update 2,500 after validation plateau; selected checkpoint
was update 1,700. On all held-out projected-in-frame selected views, near
selected-opening pixel p95 was 5.98 px SC and 6.84 px SFP. The alternate-port
heads are less precise, especially when a port is occluded or far away.

Three-view ray triangulation using measured TCP and fixed TCP-to-camera
calibration gives the following held-out **near TCP-to-opening <100 mm** result:

| Family | Frames | 3D lateral median / p95 | 3D axial p95 |
| --- | ---: | ---: | ---: |
| SC coarse | 920 | 0.410 / 1.052 mm | 1.305 mm |
| SFP coarse | 1,194 | 0.536 / 1.262 mm | 1.608 mm |

These are [posthoc errors](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/geometry_validation.json)
against target TF, which is never passed to the estimator. Lower camera
reprojection residual alone did not select a reliable 0.5 mm subset: for SC,
a strict 1 px residual retained only 12% of near views and still had 0.84 mm
lateral p95. A consistent wrong feature can reproject consistently.

## Native SC contact crop

The SC refiner uses the **original 1152×1024 JPEG**. The frozen RGB locator
chooses a 160×160 crop around the selected opening; no target TF picks the
crop. A second ImageNet-initialized spatial heatmap estimates the opening
within that native crop. [Training and evaluation](../../scripts/train_two_family_sc_native_refiner.py).
The first all-validation-selected model improved 7,019 held-out SC views from
5.99 to 2.02 px p95. Its near 3D lateral p95 was 0.370 mm, but the same 21
episodes influenced checkpoint choice.

A cleaner repeat used 10 validation episodes only for refiner checkpoint
selection and kept 11 additional episodes untouched by that choice. On those
11 groups, crop pixel p95 was **1.81 px**. For 364 near frames, triangulated
lateral p95 was **0.442 mm**, axial p95 **0.318 mm**, and 96.4% of frames
fell within 0.5 mm laterally. This tests refiner selection generalization;
the frozen coarse locator had previously used all 21 groups in its own
selection, so this is not a fully new-scene end-to-end test. See the
[split summary](../../outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_selection10_v2/summary.json)
and [3D frame scores](../../outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_selection10_v2/geometry_validation.json).
The SC 100–200 mm band still has larger errors, so a local actor starting at
the transport standoff must refine its estimate while descending.

The existing verified SFP recordings contain only 288×256 images, including
their source videos and aligned JPEGs. Upsampling them cannot supply the
missing native contact detail; a matched native SFP collection is needed for
the same crop test.

## Orientation and visibility

A geometry-only MLP over the RGB-predicted port points failed as a general
orientation estimator: SC p95 6.74° versus 5.56° for a constant-mean
baseline, SFP p95 7.08° versus 13.00° mean baseline. Adding frozen visual
features from the locator gave SC median/p95 **0.91°/2.56°** and SFP
**1.12°/3.70°** on held-out scenes. SC crosses the 3° orientation target;
SFP does not. These validation groups selected the checkpoint, so an
independent development check remains. [Model and baseline](../../outputs/experiments/2026-10-02_two_family_perception/orientation_visual_v2/summary.json).

An independent four-scene SC RGBD audit directly exposed the projected-flag
failure: the locator accepted **78.7% of 94 in-frame but geometrically
occluded images** at a 0.5 threshold. Its score must not be interpreted as
visibility or calibrated pose confidence. The [RGBD audit](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sc_rgbd_occlusion_audit.json)
uses ordinary RGB for the model and separately synchronized label-only depth.
A native RGB crop visibility pilot trained on three scenes and tested the
fourth in each fold; pooled false-visible rate was 5.3% with 97.8% recall,
but the port-1 five-card fold still falsely accepted 4 of 9 hidden images.
Four scenes are insufficient to promote a visibility gate; SFP has no depth
visibility labels yet. [Pilot and all folds](../../outputs/experiments/2026-10-02_two_family_perception/sc_rgbd_visibility_pilot_v1/summary.json).

One additional SC RGBD replay was captured with rootless Gazebo, and the
depth-label builder retained 130 synchronized rows from it. The teacher
replay itself failed its final insertion because an optional lateral-gate
environment value was blank; this is a perception-only run, not an expert
trajectory. A corrected replay was then recorded separately. These runs
do not alter the 104 verified expert episodes.

## Predicted-frame insertion BC

The [new local dataset builder](../../scripts/build_predicted_port_local_dataset.py)
first estimates a selected port frame from frozen RGB models and measured
TCP. It then converts each **recorded complete teacher target** from the
true port frame into that predicted frame using true TF only in the offline
label conversion. The actor input and action labels therefore share the
same imperfect predicted frame. This avoids training a policy against one
frame and deploying it in another. It does not predict absolute base-link
actions or add a delta correction to a separate actor.

The [dataset summary](../../outputs/experiments/2026-10-02_two_family_perception/local_dataset_coarse_v1/summary.json)
has 98,548 SFP and 21,125 SC training rows from 234 and 83 episodes with
causal two-view estimates, plus 13,195 SFP and 5,330 SC validation rows.
The actor gets RGB, ordinary 32D robot state, 10D public task code, and its
estimated current TCP pose in the estimated port frame. A four-token
transformer predicts four complete TCP targets in that frame. SC and SFP,
near and earlier local phases, and complete episodes are balanced for BC.

Two matched 1,600-update BC arms differ only in initialization. Scratch
starts from ImageNet vision weights; transport warm start copies only the
transport backbone, multi-view attention and state/task context. The old
body-frame action head is **not** reused for port-frame targets. Offline
held-out best-checkpoint first-command p95 was 5.48 mm SC / 9.30 mm SFP for
scratch, versus 4.60 mm SC / 8.40 mm SFP warm. Near lateral command p95 was
2.48 / 5.34 mm scratch and 2.17 / 5.29 mm warm. This is insufficient by
itself for reliable insertion; teacher-frame BC error is not autonomous
success. Both checkpoints and failures remain in
[scratch](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_scratch_v1/summary.json)
and [warm](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_transport_warm_v1/summary.json).

## Current gate and next action

Step 5 and autonomous full insertion remain open. The native SC position
result is promising, but true selected-target visibility, SFP fine position,
and independent orientation reliability need stronger evidence. The first
near-port Gazebo rollout of the warm local actor is being prepared on a
non-final teacher-transported scene. The teacher only supplies the prefix;
after handoff, the local actor reads no target TF or privileged geometry.
The rollout will report official insertion, force/contact, predicted port
stability, commands, measured TCP/tip motion, and full inference p50/p95/p99.
If local BC fails, diagnose the trace and video before further training.

## Live SC near-port diagnosis and targeted continuation

The first same-scene teacher-prefix/local-actor run was invalid because a
concurrent SFP Gazebo capture shared the Gazebo/Zenoh world endpoint. It was
not scored as a model failure. A serial retry confirmed that the local actor
loads and receives only ordinary RGB, measured robot state, public task code,
and its own RGB-estimated port pose. The teacher supplies only the route
prefix. Its handoff trace explicitly records
`teacher_geometry_passed_to_actor=false`.

The first valid serial run (`local_live_sc_llb0001_warm_v5_serial`) failed
without insertion. Complete inference p95 was about **1,020 ms** because
PyTorch and OpenCV each used the container's 128 exposed CPU threads. A
local component benchmark with four PyTorch threads and one OpenCV thread
showed individual image operations and networks taking tens of milliseconds.
The thread-limited same-checkpoint repeat
[`local_live_sc_llb0001_warm_v6_threads`](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_warm_v6_threads/)
reduced measured complete p50/p95/p99 to **322/387/476 ms**. This still
misses the 300 ms p95 gate. The five-camera
[10 fps video](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_warm_v6_threads/five_cameras_10fps.mp4)
shows the near-port stall.

The scored SC bag was extracted after the run, and its actual cable-tip TF
was compared with the selected opening before reset. At the last recorded
observation the plug tip was **6.85 mm before the opening and 7.81 mm
laterally away**. It never came closer than about 6.96 mm laterally in the
near phase. A 60–70 s precontact force baseline was subtracted from the
measured wrist wrench; its 110–150 s residual magnitude had median/p95
**3.26/3.60 N**. The official score was 0 for insertion, with no scored
off-limit contact. This is a near-opening alignment/blocked-motion failure,
not evidence of a cable snag. The RGB pose estimate was initially accurate
against label-only static scene geometry (<0.1 mm position error and about
1° orientation error) but drifted to about **1–2 mm and 17°** when the port
became occluded. The prior 0.6 s stale cutoff also aborted the v5 run.

Because the port is stationary during one task, a targeted runtime ablation
locks its pose after five mutually consistent **RGB-only** native-crop
estimates. It does not read a simulator target pose. In the matched v7
repeat the locked estimate was posthoc **0.09 mm position / 0.74° orientation**
from the simulator opening. That repeat still failed to insert: its
source-calibrated TCP-to-tip estimate placed the final tip about 12.6 mm
laterally away and 6.8 mm axially short. (The v7 tip is an approximation
from the source episode's rigid TCP-to-tip transform; v6 numbers above use
the v6 scored bag's actual tip TF.) The lock therefore fixed the observed
orientation drift without fixing the actor. The first five perception calls
and then the image/actor path gave full-run p50/p95/p99 **328/389/806 ms**;
even the locked path remained above the latency gate. The actor repeatedly
predicted TCP targets near port-frame axial $z=-0.03$ m rather than a sustained
advance. The successful source teacher's plug tip eventually reached about
zero lateral error and crossed the opening by more than 15 mm.

Two bounded continuations have now ended at validation plateaus. The
[general continuation](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_warm_continue_v2/summary.json)
selected update 600 of 1,400; the [SC near-weighted continuation](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_sc_near_continuation_v3/summary.json)
selected update 700 of 1,900. The latter reached held-out SC near
port-frame first-target lateral p95 **1.05 mm**, but worsened the SFP
first-command p95 to **9.06 mm**. Matched SC live runs v8 and v9 still
scored **0** autonomous insertions. Its [five-camera video](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_scnear_v3_cpu_actor/five_cameras_10fps.mp4)
preserves the full attempt. The v9 model's actual tip briefly came
within **0.57 mm** laterally while **17.32 mm** before the opening, then
drifted to **6.17 mm** lateral as its axial coordinate reached the opening;
see its [posthoc tip score](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_scnear_v3_cpu_actor/tip_scoring.json).
Its complete inference p50/p95 was **275/320 ms**, still outside the 300 ms
p95 limit. Routine unchanged BC continuation is therefore stopped.

A more specific failure was found by comparing the actual Gazebo SC tip TF
against the 71 near-port recorded teacher rows for the same development
case. The teacher tip orientation is effectively zero in its port frame,
while v6 and v9 actor rollouts reached roughly **77° and 84°** from that
orientation. In v9 the actual tip was aligned within about **0.2° at the
teacher-to-actor handoff**, stayed near 0–3° through roughly 100 s, and
began rotating as the learned actor approached the port. Its predicted
target was near the correct orientation initially, then rotated toward
about 76–80° when the TCP approached 35 mm axial standoff. The measured
tip followed the TCP within about 1.8 mm/3.5° of a fixed grasp transform,
so the large orientation error was not evidence of cable slip or an
incorrect handoff posture. This is a concrete actor near-port orientation
and distribution-shift failure. A targeted orientation-loss BC continuation
is running on one GPU. A separate optional teacher-only prealignment was
implemented for a diagnostic, but it is **not** the proposed main repair
after this stronger time-trace finding. True TF is used only for posthoc
diagnosis, never by the actor.

The same actor target also moves toward the **other task family's**
orientation as it approaches the SC opening. The successful SFP and SC
teacher TCP orientations differ by about **97.4°** in their respective
port-frame target conventions. In the v9 rollout, the actor target was
0.6° from the SC teacher mean at handoff and 78.8° away by 114 s; by then
it was only 25.1° from the SFP mean. The task one-hot was present and
correct, so this is a shared-output generalization failure rather than a
missing task input. A matched architecture ablation keeps one shared
vision/state/transformer actor but gives the two task families separate
orientation output heads; its training and live result must be checked
before concluding the output mixing caused the entire failure.

A task-specific orientation prior was fitted from 83 successful SC training
episodes and evaluated on 21 separate validation episodes. Their near-port
first TCP target orientation differs from the training mean by **1.95° p95**
on train and **2.21° p95** on validation; see the
[frozen prior](../../outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json).
An optional runtime clamp bounds the actor's **complete target orientation**
to this demonstrated range in the RGB-estimated port frame, without using
true TF. This is a physical action constraint on the predicted target, not
a delta-action residual. It needs a matched live test before claiming any
insertion benefit. A separate BC arm explicitly penalizes near-port
orientation error: its first continuation improved SC held-out near
orientation p95 from **3.63° to 3.01°**, while keeping near lateral target
p95 near 1.04 mm; its second continuation reached **2.42°** orientation
p95 and **1.13 mm** lateral p95 before validation plateau. The latter
remains untested live and offline error alone does not establish insertion.

The SFP depth pilots tried a world follower, a robot-attached auxiliary
camera, and a primary RGBD replacement. After motion, depth became
nonfinite and auxiliary RGB was uniform gray. Those outputs cannot label
natural occlusion. No SFP true-visibility model has been promoted. Instead,
the native RGB collection uses 30 distinct non-final resets spanning all
ten selected card/port identities at one, three, and five cards; 20 reset
groups are train and 10 validation. The first ten completed stock-CheatCode
replays all scored tier 3 = 75. Their new 1152×1024 native images and bag TF
label **all physically present ports**, including confusing alternatives;
the projected in-frame bit remains explicitly distinct from actual
visibility. The old locator's held-out selected-opening pixel p95 on two
pilot train cases and one pilot validation case was 23, 126, and 58 pixels,
respectively. This confirms the older SFP card-0-port-1 bias and motivates
the balanced native continuation. Collection and group-held-out training
remain in progress; the four reserved final IK scenes remain sealed.

A small separate SC pose-reliability head was also fit from frozen RGB
features, selected-port predictions, and measured TCP. Its target is the
probability that the **coarse 3D lateral position** is within 0.5 mm of
posthoc ground truth. This is confidence in pose accuracy, not literal
visibility. On ten episode-grouped test scenes, its Brier score was
**0.208** versus **0.250** for a constant-rate baseline. At a nominal 0.9
confidence threshold, it retained 9.1% of estimable rows and only 88.2%
were actually within 0.5 mm; the five-bin calibration gap was 0.084.
This is a useful diagnostic but is **not calibrated strongly enough to
authorize insertion by itself**. See the [complete split and metrics](../../outputs/experiments/2026-10-02_two_family_perception/sc_pose_confidence_v1/summary.json).

### Frozen SC on-policy replay and native SFP split

The [frozen RGB on-policy replay](../../outputs/experiments/2026-10-02_two_family_perception/local_sc_frozen_onpolicy_replay_v1.json)
recomputed first targets for 198 observations from the v9 scored rollout.
The replayed v3 translation matches the originally recorded v9 first target
with median/p95 error **0.105/0.352 mm**, verifying that the replay uses
effectively the same actor input and preprocessing. For observations with
estimated TCP axial distance under 40 mm, the commanded orientation's
median deviation from the successful SC teacher prior is **82.2°** for v3,
**78.3°** for the stronger angular-loss v5 checkpoint, and **76.0°** for
the shared-actor/separate-family-orientation-head checkpoint. The two
training variants improve offline held-out errors but do not remove the
large on-policy wrong rotation. This is not a live success comparison; a
bounded live prior-constraint test is the next direct diagnostic.

For native SFP, the locator training code now reserves five of the ten
new validation episodes for checkpoint selection and leaves the other five
untouched for a final model check. The native crop refiner can read that
same selection manifest so its checkpoint cannot indirectly tune on the
locator's test episodes. All physically present ports still provide
training labels; the task code selects the requested port at inference.

## Balanced SFP target identity and native crop continuation

The stock SFP teacher collection has **29 successful distinct development
episodes**, spread across the ten selected card/port identities and one,
three, and five installed cards. One further reset crashed Gazebo before
scoring and is excluded. Every successful replay scored tier 3 = 75. Its
1152×1024 RGB and scored TF yielded 2,081 synchronized three-view rows,
partitioned by scene/reset into 19 training, five checkpoint-selection,
and five untouched development-test episodes. The [five-camera example](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1/card2_port1_cards5_train/five_cameras_10fps.mp4)
shows an alternate selected target. The [native manifest](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_projection_29_hostpaths/summary.json)
and [posthoc candidate overlay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_projection_29_hostpaths/example_target_labels_center.jpg)
show that every physically present other port has its own identity and label.
The red selected overlay is a scoring/training label, never an inference crop.

For **perception supervision only**, 200 pre-existing Gazebo agent episodes
(80 easier one/two-card and 120 harder three/four/five-card scenes) added
18,969 low-resolution RGB/state rows. These are not expert trajectories;
their actions never enter actor BC. Incomplete old folders were filtered.
The [merged 229-episode manifest](../../outputs/experiments/2026-10-02_two_family_perception/sfp_perception_merged_229/summary.json)
has 21,050 rows and checks train/validation reset isolation. The older
verified SFP actor source is heavily biased toward card 0, port 1 (258/268
episodes), so a low error on that source cannot demonstrate correct selection
of arbitrary target cards. All ten native selected identities are represented,
but only about two independent native training starts per identity are
available.

Several bounded coarse-locator variants reached validation plateaus without
making the selected target reliable on all ten native development scenes.
Native-only continuation, heatmap loss, task conditioning, residual
conditioning, extra agent images, a pairwise structural loss, and a frozen
DINOv2-S encoder were each retained with summaries under
[`2026-10-02_two_family_perception/`](../../outputs/experiments/2026-10-02_two_family_perception/).
The chosen **crop proposal** checkpoint is
[`joint_locator_native_sfp_v5_heatmap`](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/summary.json):
its untouched native near-view selected-point p95 is 113.2 pixels, and its
near three-view lateral p95 is 44.5 mm. It is unsuitable as a final pose
estimator. Its strength is that the actual selected projection lies inside
a 480-pixel RGB-centered crop in 97.8% of near development views across
the ten validation scenes.

The first native RGB refiner takes that **480×480 original-pixel crop**,
resizes it to 224×224 for a small ResNet-18 spatial head, then returns a
refined center. No target TF chooses the crop. The
[`sfp_broad_crop480_v2` result](../../outputs/experiments/2026-10-02_two_family_perception/sfp_broad_crop480_v2/summary.json)
has untouched-test median/p95 **2.07/29.67 px**, from coarse **10.09/113.53
px**. Its five checkpoint-selection episodes are harder at 143.1 px p95;
the difference must remain visible. A first 160×160 second-stage refiner
reached 2.23/14.57 px median/p95 on untouched test but was selected by a
metric dominated by examples whose target lay outside that second crop.
The corrected checkpoint selector scores *inside-crop* pixels on the five
selection episodes while still reporting the complete distribution. The
[`sfp_fine_crop160_v2_inside_selection` result](../../outputs/experiments/2026-10-02_two_family_perception/sfp_fine_crop160_v2_inside_selection/summary.json)
reached **0.655/3.263 px** median/p95 on the five untouched test episodes.
Its all-view p95 on the five selection episodes remains **162.3 px** because
one target is often assigned to another port before fine cropping. The
[test image gallery](../../outputs/experiments/2026-10-02_two_family_perception/sfp_fine_crop160_v2_inside_selection/heldout_test_gallery.jpg)
shows examples from median through p99. In that gallery red is the RGB
prediction; green selected-port and orange alternate-port markers were
added only after inference. The p99 example picks the wrong side of a card.

Triangulating all three refined camera points on the untouched five native
development episodes gives **0.127/0.443 mm** near-port lateral median/p95
over 333 near frames, and **0.114/0.396 mm** near axial median/p95. The five
selection episodes have 0.165/37.37 mm lateral median/p95 before an
observation-only consistency gate because of wrong-target views. Selecting
only estimates with three-view reprojection residual below 2 px keeps
**271/335 (80.9%)** near selection frames at 0.371 mm lateral p95 and
**311/333 (93.4%)** near untouched-test frames at 0.305 mm lateral p95.
No accepted frame in either small set exceeded 0.5 mm laterally. This is a
*conditional* development result with only ten reset groups, not full-scene
reliability or true visibility calibration. The gate abstains on most
frames of the difficult card-1/port-1 selection scene (10/64 accepted).
Per-episode coverage and every frame are in the
[3D report](../../outputs/experiments/2026-10-02_two_family_perception/sfp_fine_crop160_v2_inside_selection/geometry_validation.json).
The old projected-in-frame head remains geometrical only; it is not a
trained occlusion-visibility output. SFP depth pilots failed to provide
valid occlusion labels. A calibrated pose confidence and orientation check
are still required before a live SFP handoff. The final IK split remains
sealed.

## SC recorded-prefix contact diagnosis and supervised repair

The earlier SC local actor had been handed off with the plug tip roughly
150 mm before the opening. To test the intended near-port regime, a
successful scored `llb_0001` teacher prefix was replayed only until its
recorded tip was 31 mm before the opening. The teacher's targets and port
TF are removed at handoff; the actor then uses RGB-estimated geometry and
measured state. The [unconstrained rollout video](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_v3_recorded_prefix_v1/five_cameras_10fps.mp4)
scored 0. Posthoc scored TF shows it reached 0.035 mm lateral error at
7.21 mm before the opening, then a maximum of about 10 mm into the opening,
with final 17.15° orientation drift and 2.08 mm lateral error. Its full
inference p50/p95/p99 was 222/330/740 ms, missing the 300 ms p95 limit.

A matched same-checkpoint/same-scene replay constrained the SC target
orientation to within 2° of the successful training-episode orientation,
in the **RGB-estimated** port frame. Its
[five-camera video](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_v3_recorded_prefix_guard2_v1/five_cameras_10fps.mp4)
also scored 0. At termination the scored tip was about **10.59 mm past the
opening**, **0.365 mm lateral**, and **1.28°** from successful teacher tip
orientation. The successful teacher reached about 15.57 mm past the opening
and received the insertion event. The guarded actor requested deeper TCP
targets but measured TCP motion plateaued while wrist-force residual rose
to roughly 5 N near contact. This is a local insertion/contact failure,
not evidence of a cable snag; correcting the large late rotation by itself
does not solve it. The [posthoc tip trace](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_v3_recorded_prefix_guard2_v1/tip_scoring.json)
and original scored bag preserve the distinction between commanded and
measured motion.

Comparing source teacher and actor commands at similar measured tip depth
shows a several-millimeter sideways TCP target difference, even with an
accurate RGB port position estimate (about 0.12 mm from posthoc scene TF).
This motivates a bounded on-policy supervised diagnostic rather than
another routine replay of unchanged expert BC. The
[matched-label builder](../../scripts/build_sc_onpolicy_teacher_match.py)
relabels actor observations against the *recorded complete* target of the
successful same-scene teacher at nearby tip position and orientation.
It rejects off-manifold matches (>6 mm tip separation or >5°): 41 of 159
subsampled unconstrained states and 195 of 195 guarded states are retained.
The accepted teacher matches have 0.88 mm median and 2.20 mm p95 tip
distance. Geometry selects a teacher label offline only; the resulting
actor row contains RGB, measured state, public task code, RGB-estimated
port frame, and a complete teacher target reexpressed in that frame.
These 236 rows supplement **234 SFP + 83 SC expert training episodes**;
they are not counted as independent scene configurations. A matched
continuation from the existing actor is underway with a fixed update cap
and a live same-scene stopping rule. Its existing 31 SFP and 21 SC
validation episodes remain grouped and untouched by the new rollout rows.

## October 2 continuation: complete SFP cascade and insertion diagnosis

The SFP locator's coarse RGB prediction now selects a **480-pixel native
camera crop**; a broad RGB refiner selects a **160-pixel native crop**; a fine
refiner locates the selected opening. The public task one-hot selects card
and port. The three camera predictions triangulate a base-frame position.
A separately trained RGB-feature and candidate-geometry head estimates
orientation. No true port transform enters any of these runtime calls.
The [cascade implementation](../../scripts/sfp_port_cascade.py) and
[120-frame benchmark](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_cascade_benchmark_v2.json)
preserve every selected identity, gate decision, per-frame error and latency.
On ten episode-grouped development scenes, its three-view reprojection
gate accepted 93/120 sampled observations. Accepted position lateral p95
was 0.36 mm overall, with 18.4 ms warm perception-only p95. The benchmark
does **not** include image decode, actor or control overhead. Orientation
on the accepted untouched five development scenes was 10.2° p95 with the
native-emphasized head, improved from 15.9° for the original orientation
head on the comparable accepted subset. The orientation improvement has
not yet passed a live insertion test.

A decisive counterexample appears at 7.05 s in
`card2_port0_cards3_validation`: all three crop refiners agreed geometrically
on a **wrong feature 41.35 mm away**, with only 0.36-pixel reprojection
residual. A low residual is thus a *consistency check*, not calibrated
probability that the selected port was identified. The prior frame failed
the gate, and later frames correctly localized the port. The runtime now
holds its measured TCP position until five observation-only estimates
agree within 0.5 mm and 5°; only then does it lock the static port pose
and allow actor motion. This proposed handoff still needs a scored live
test. Projected-in-frame flags remain distinct from true visibility:
the SFP RGBD pilots failed to yield valid occlusion labels. Do not call
this a trained true-visibility model.

The [29 native SFP successful expert runs](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1/)
cover ten selected card/port identities and one-, three-, and five-card
settings. All 29 scored tier 3 = 75; no failed agent rollout was used as
an action expert. The [new actor data builder](../../scripts/build_sfp_native_cascade_bc_dataset.py)
causally joins each saved image to measured 32D robot state and the **four
actual recorded Cartesian command targets**, then expresses those targets
in the frozen RGB-estimated port frame. The first derivative contains
1,073 accepted local rows from 1,168 source observations: 753 train,
158 checkpoint-selection, 162 untouched development-test rows, grouped
as 19/5/5 complete scenes. It joins the older verified 234 SFP and 83 SC
training episodes. The second derivative uses the improved orientation
head and excludes training-label-known port-position outliers over 0.5 mm;
its exact count is in its [summary](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_cascade_bc_29_v2/summary.json).
Geometry is used only to score and relabel training examples; the actor's
state contains the RGB estimate. New target identities receive an
episode-balanced sampling boost of 8. The two SFP actor checkpoints are
separately recorded under `local_actor_sfp_native_29_v1/` and the later
v2 continuation. Neither should be called a successful insertion policy
before scored near-port rollouts.

The SC on-policy matched-label diagnostic completed 900 updates at its
fixed cap. Its grouped SC near lateral p95 **worsened** from 1.05 to
1.25 mm, while fitting its 236 same-scene matched failure observations
much better (first target median 7.54→1.26 mm). The decisive new
[five-camera scored rollout](../../outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_dagger_v2_guard2_v1/five_cameras_10fps.mp4)
still scored tier 3 = 0. Posthoc tip at closest axial point was 0.01 mm
before the opening and **1.44 mm lateral**; at termination it was 0.79 mm
before and 1.19 mm lateral. Wrist force stayed near its baseline, with no
observed high-force insertion contact. The RGB port estimate was only
about 0.05 mm from posthoc true geometry after visual locking. This is
an alignment/action failure despite excellent port localization. Offline
fit to one scene is not generalization.

The successful SC teacher spends an extended period aligning with almost
zero axial command, then advances from visually similar poses. The first
on-policy match mixed both phases; its labels alone cannot teach a
single-frame actor when to advance. A distinct bounded continuation
`local_actor_sc_progress_v1` samples positive near-port axial teacher
commands five times as heavily and holds at 0.2 weight, with a new
advancing-phase grouped validation metric. It will receive one matched
scored rollout if offline progress justifies it. If it again hesitates
or drifts, add explicit phase/history or measured-progress feedback
instead of repeating similar single-frame BC.

This arm finished after 1,000 of 1,200 capped updates because the
advancing-phase grouped-validation metric plateaued. Its best checkpoint
was update 100, but the first-target p95 on actual advancing SC rows
**worsened from 2.23 to 2.50 mm** and near lateral target p95 worsened
from 1.13 to 1.58 mm compared with its initialization. It missed the
offline stopping rule, so no new Gazebo rollout was spent on it. Simple
sampling alone does not resolve phase ambiguity. [Complete run](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_sc_progress_v1/summary.json).

The discarded SFP RGBD pilot files were checksum-matched to their backup
at `/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_native_rgbd/`
and replaced locally by a symlink to free about 4.2 GB. The source
experts, scored runs, videos, checkpoints, and failure traces remain.

### Later October 2: target confusion, confidence, and insertion actor

The five-camera [SFP v1 near-port failure](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_card2port1_five_v1/five_cameras_10fps.mp4),
[later handoff failure](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_card2port1_five_v1_later_handoff/five_cameras_10fps.mp4),
and [warm actor v4 failure](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_card2port1_five_v4_warm_later_handoff_retry/five_cameras_10fps.mp4)
are scored Gazebo development runs. The teacher supplies only the approach
prefix. At handoff, the actor receives RGB, measured robot state, the public
task code, and the RGB-estimated port pose. The actor does not read teacher or
true target geometry. All three scored tier 3 = 0; none demonstrate autonomous
full-start insertion. The v4 run began with TCP about 58 mm before the port and
4.1 mm lateral. It briefly approached 2 mm lateral but ended about 56 mm
axially short. Its first targets often commanded another 10–17 mm forward
while measured axial progress stayed near zero; the controller error and wrist
wrench increased. A collision or controller blockage, rather than a pure
"held command" policy, is therefore a live hypothesis. The v4 complete
inference latency was p50/p95/p99 176/213/346 ms over 295 decisions, so the
300 ms p95 requirement passed on this run.

The 29-scene native SFP actor data use the actual recorded four-command TCP
targets and the same RGB-estimated port frame used at runtime. A matched
[warm continuation](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_sfp_native_29_v4_ensemble/summary.json)
and [scratch actor](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_sfp_native_29_v4_scratch/summary.json)
finished 1,600 update caps. Warm initialization gave native five-episode
selected first-target p95 9.44 mm and near lateral target p95 2.52 mm;
scratch gave 13.24 and 2.86 mm. Warm initialization is better offline, yet
its scored run still failed. Do not describe this as reliable insertion.

The [native-crop orientation probe](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_crop_orientation_v1/summary.json)
used 160×160 original-resolution crops chosen only by RGB. It achieved 3.67°
p95 on its five checkpoint-selection episodes but 20.18° on the other five
development episodes, especially one occluded target scene. A conservative
[crop residual on the prior RGB orientation](../../outputs/experiments/2026-10-02_two_family_perception/sfp_crop_orientation_residual_v1/summary.json)
improved p95 from 8.97° to 8.57° on selection and 6.50° to 5.79° on the other
five; these are modest, development-inspected results and do not reach a
sub-3° contact requirement. Neither crop orientation head is deployed.

The [SFP 0.5 mm accuracy confidence fit](../../outputs/experiments/2026-10-02_two_family_perception/sfp_fine_pose_confidence_v1/summary.json)
failed its calibration test. Its development-test Brier score was 0.211 versus
0.221 for a constant-mean baseline. In one card-1/port-1 scene, only 17% of
predictions were within 0.5 mm laterally while the model assigned confidence
above 0.9 to all of them. The [image diagnostic](../../outputs/experiments/2026-10-02_two_family_perception/sfp_fine_pose_confidence_v1/card1port1_wrong_view_diagnostic.jpg)
shows the left view selecting a different feature about 260 native pixels from
the true selected opening. Green is posthoc truth for diagnosis only; no such
marker is an actor input. This checkpoint must not gate insertion. The coarse
projected-in-frame head remains distinct from actual occlusion visibility.

Two targeted SFP broad-refiner tests showed that a crop including both port
sides and conditioned on the selected side is insufficient with the current
number of native starts. The original broad crop's selected-point p95 on the
five other development scenes was 29.67 px. The
[480-pixel conditioned sibling model](../../outputs/experiments/2026-10-02_two_family_perception/sfp_broad_crop480_condition_sibling_v3/summary.json)
gave 91.15 px; the [640-pixel version](../../outputs/experiments/2026-10-02_two_family_perception/sfp_broad_crop640_condition_sibling_v1/summary.json)
gave 73.27 px. Both improved the five selection scenes but degraded the other
five. They are preserved as failures and not deployed. A separate
[two-view temporal consensus diagnostic](../../outputs/experiments/2026-10-02_two_family_perception/sfp_temporal_pair_consensus_w15_r05_n5.json)
locked all ten prior development scenes at lateral p95 below 0.2 mm using RGB
predictions and known camera calibration; the existing static visual lock was
already comparably accurate, so this is not a promotion result.

The concrete data hypothesis is that roughly two independent native training
starts per SFP card/port identity leave the broad model unable to distinguish
lookalike cards and ports under some views. The
[new scene manifest](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_confusion_scenes_v2/collection_manifest.json)
excludes prior reset hashes and plans 20 new training and ten new development
starts with three/five cards. The first pilot, card 1/port 1 with five cards,
scored tier 3 = 75 and its RGB/scored stream was saved under
`/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_native_confusion_v2/`.
The first ten training starts are being collected serially on physical GPU 3;
do not use the ten new development starts for checkpoint selection. The
stopping rule is to re-fit the selected-port perception with the first ten and
check complete-scene held-out wrong-target rate and the 0.5 mm accuracy gate
before collecting the remaining twenty.

The bounded temporal-insertion hypothesis is that a single observation cannot
always separate alignment, blocked progress, and insertion. Two otherwise
matched four-observation actor continuations are running: the first preserves
per-observation estimated frames as an ablation; the second reexpresses every
past measured TCP pose in the current RGB-estimated port frame. Their history
contains only causal robot state, estimated pose, and age. They will be
compared by grouped near-port command error, then scored live only if the
offline gate improves enough to justify it. A new history checkpoint is not
an insertion result until that scored test passes.

### Orientation feature mismatch and observation-only lock

The original orientation cache omitted candidate slots for physically
absent cards, while live inference populated all 12 candidate slots from
the RGB locator. This train/runtime mismatch was repaired by caching all
12 predictions and retraining the same orientation architecture; the
[runtime-matched checkpoint](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_orientation_v3_runtime_matched/summary.json)
uses exactly the live feature layout. Its full
[722-frame benchmark](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_cascade_benchmark_v3_allframes.json)
accepted 278/355 selection and 318/367 actor-held-out test observations.
Accepted lateral p95 was 0.375/0.321 mm, axial p95 0.305/0.405 mm,
and orientation p95 10.46/6.36° respectively. The native-emphasized
older head had 10.20/10.26° orientation p95 under the same complete
runtime benchmark. Averaging their rotations on paired predictions
improved the selection orientation p95 to 8.95°, at a held-out test p95
of 7.71°; this ensemble was chosen using development metrics and is not
a clean untouched-test result. The exact ensemble is now an optional
runtime path and the `sfp_native_cascade_bc_29_v4_ensemble/` actor
dataset uses that same estimate.

The full-frame replay simulated the live rule: five accepted static-port
estimates must agree within 0.5 mm position and 5° orientation before
actor motion. All ten development scenes formed a lock. Locked position
lateral p95 was 0.158 mm and axial p95 0.175 mm; locked orientation
p95 remained 11.59° for the runtime-matched head. The isolated 41 mm
wrong-feature acceptance did not form a lock. This validates the
specific temporal consistency repair on those scenes, but a consistently
wrong port could still pass; independent scored development starts and
true occlusion labels are needed for calibrated confidence. Runtime
holds the measured pose while a lock is pending and remembers a static
port after lock through temporary occlusion.

### October 2 follow-up: new SFP starts, body-frame targets, and live contacts

The first ten **new five-card SFP-to-NIC training starts** in
[`sfp_native_confusion_scenes_v2`](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_confusion_scenes_v2/collection_manifest.json)
each scored tier 3 = 75 with stock privileged CheatCode. They cover all ten
selected card/port identities and use reset hashes distinct from the earlier
29 native experts. RGB and scored streams are under
`/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_native_confusion_v2/`;
the [projection derivative](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_confusion_labels_first10/summary.json)
has 5,045 synchronized RGB/state rows. The planned further twenty starts were
not collected after the bounded perception retraining missed its held-out gate.

Two continuations from the prior joint locator used the same old grouped
selection/test scenes. The higher native-source weight and structured-layout
loss reduced native test selected-port p95 from 113.2 to 91.5 pixels, yet left
10.2% of predictions closer to an alternate port and worsened SC p95 from 7.14
to 8.04 pixels. A lower learning rate with more old-scene replay gave native
test p95 85.5 pixels and 6.93% alternate-port confusion, but **selection**
confusion worsened from 31.4% to 36.8%. Neither is a robust selected-target
model, so neither was deployed. See [baseline](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/summary.json),
[first continuation](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_confusion_first10_v1/summary.json),
and [lower-rate continuation](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_confusion_first10_low_lr_v2/summary.json).
The RGB confidence head also failed calibration: it assigned at least 0.9
probability to all views in one held-out case where only 17% were within
0.5 mm. The five-estimate static lock remains an operational consistency
gate, not certified visibility or a calibrated probability.

The new scorer records stock CheatCode Cartesian targets as `gripper/tcp`
**delta poses**, whereas the first 29 native expert runs recorded `base_link`
absolute poses. The original source builder admitted zero new rows.
[`build_sfp_native_teacher_local_chunks.py`](../../scripts/build_sfp_native_teacher_local_chunks.py)
now accepts either frame and composes each body-frame delta with the causally
preceding measured TCP pose before relabeling it in the selected port frame.
On a sampled run, reconstructed targets matched the controller's subsequent
reference target at 0.0046 mm median / 0.081 mm p95 translation error; a few
transient maxima were larger. The repaired source contains
[1,914 synchronized near-port rows across ten successful episodes](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_teacher_local_first10_bodyconverted_summary.json).
The unchanged frozen RGB cascade accepted 1,478 for actor training after
rejecting 347 disagreeing-view rows, 74 position-label outliers, and 15
missing-view rows. Accepted *training* lateral pose error was 0.147 mm median,
0.298 mm p95. The [39-scene actor derivative](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_cascade_bc_39_bodyconverted/summary.json)
has 29 training, five checkpoint-selection, and five untouched development-test
episodes without scene overlap. One bounded actor continuation from the
29-scene checkpoint is running; no new insertion result is claimed yet.

The preceding warm SFP actor scored **0** on a near-port Gazebo development
handoff. A read-only [oracle-contact repeat](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_v4_oracle_contact_diagnostic/)
on the same reset also scored 0. The actor started at simulation time 28.25 s;
the plug tip first contacted the selected card around 31.6 s, reaching about
39.9 N maximum point normal force around 37 s while measured TCP axial progress
stalled. Cable link 6 had already contacted card 0 during the teacher approach,
with point-force maxima about 69.5 N before actor handoff, 41.0 N during early
actor motion, and 33.8 N before strong tip contact. This supports **both**
cable/card drag and a local tip/card block; tip contact is the clearest timed
event when forward motion stops. Oracle pairs and corrected normal impulses
are in [`contact_force_summary.json`](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_v4_oracle_contact_diagnostic/contact_force_summary.json),
with the [matching five-camera video](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_v4_oracle_contact_diagnostic/five_cameras_10fps.mp4).
The teacher prefix is privileged; the insertion actor itself uses RGB,
ordinary robot state, public task code, and its RGB-estimated port pose.
Reserved final scenes remain sealed.

The 39-scene warm actor finished its 1,200-update cap; checkpoint selection
chose update 800. Relative to the prior 29-scene actor on the same five native
selection episodes, first-target p95 improved 9.44→8.70 mm, near lateral
target p95 2.52→2.13 mm, and near orientation p95 5.69→4.89°. The separate
buffered-lateral-loss continuation from the old 29-scene data reached 2.23 mm
near lateral but worsened first-target error to 9.99 mm, so it was not
deployed. [39-scene summary](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_sfp_native39_bodyconverted_v1/summary.json),
[buffered-loss comparison](../../outputs/experiments/2026-10-02_two_family_perception/local_actor_sfp_buffered_lateral_v1/summary.json).

A same-reset scored Gazebo handoff then compared the old and new actors with
identical frozen perception and privileged teacher transport prefix. Both
scored tier 3 = 0. From approximately 65 mm before the selected port in TCP
coordinates, the old actor advanced at most 8.5 mm and stopped near -56 mm;
the 39-scene actor advanced 37.4 mm and reached -27.7 mm. The successful
teacher's final TCP was about -11.15 mm on this reset, so the new actor still
stopped roughly 16.5 mm short. At its closest axial approach it was about
2.46 mm lateral from the selected opening. Complete actor inference p95 was
233 ms across 294 proposals, below the 300 ms limit. These are posthoc TCP
coordinates, **not** plug-tip insertion depth. See the
[machine comparison](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_matched_comparison.json)
and [new actor video](../../outputs/experiments/2026-10-02_two_family_perception/live_sfp_actor39_matched_v1/five_cameras_10fps.mp4).
It is progress in approach control, not autonomous insertion.

One targeted data hypothesis is that nominal successful paths lack examples
of correcting near-port lateral and orientation errors. A new **privileged
training-data diagnostic**
[`CheatCodePerturbNearPort.py`](../../aic_example_policies/aic_example_policies/ros/CheatCodePerturbNearPort.py)
briefly holds the plug 3 mm sideways and 4° rotated around the selected port
at roughly 30 mm standoff, then releases the ordinary CheatCode to correct
it. Both positive and negative perturbations on the same card-2/port-1
training reset fully inserted (tier 3 = 75); the perturbation event is saved
with its release simulation time. The action-label builder now offers
`--post-perturb-release-only` so no deliberate error/hold command is copied
into the recovery expert targets. This proves one reset can yield causal
corrective demonstrations, not that the learned actor can recover. A bounded
two-edge-identity collection is running before scaling this data hypothesis.

A separate frozen-feature orientation ablation trained on the ten new starts
gave head-only held-out p95 7.94° on selection and 6.18° on test, versus
10.43°/6.37° for the prior runtime-matched head. In the complete cascade,
blending it with the old native head gave 8.02°/7.31°, a modest change from
the previously chosen blend's 8.95°/7.71°. It remains short of a dependable
orientation and is not deployed with the actor trained on the older blend.
[Head](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_orientation_39_v1/summary.json),
[full cascade replay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_cascade_benchmark_v39_ensemble_allframes.json).
An observation-only affine card-grid repair was also rejected: it could fit
only about 4.3% of held-out RGB views and did not reduce overall selected-port
p95. [Diagnostic](../../outputs/experiments/2026-10-02_two_family_perception/sfp_card_grid_fusion_diagnostic_v1.json).

### Corrective BC comparison and a three-card identity failure (October 2)

The bounded corrective collection covered all ten selected card/port identities
on separate five-card **training** resets. Each deliberately displaced the TCP
about 3 mm laterally and commanded a 4° yaw near the port, then resumed the
privileged successful teacher. All ten scored tier 3 = 75. The supervised
derivative includes only commands after perturbation release: 1,173 synchronized
four-command windows were available, of which the frozen RGB cascade accepted
935. The deliberate wrong move and hold were excluded. This is supervised
recovery data; the live actor never receives true port geometry. See the
[source selection](../../outputs/experiments/2026-10-02_two_family_perception/sfp_perturbed_recovery_selection_10/collection_manifest.json),
[accepted-row audit](../../outputs/experiments/2026-10-02_two_family_perception/sfp_perturbed_recovery_cascade_bc_10/summary.json),
and [axial coverage](../../outputs/experiments/2026-10-02_two_family_perception/sfp_recovery_bc_axial_coverage.json).

A same-initialization, same-update-budget continuation compared the old
39-scene dataset against that dataset plus 935 corrective rows. Both stopped
at update 800 on the prespecified native near-port checkpoint-selection metric;
both selected update 100. The recovery arm's native five-scene near-lateral
target p95 was 2.16 mm versus 2.10 mm for extra training alone, so it did
**not** improve this selection metric. On a separate, newly collected
three-card corrective development reset where RGB pose was accepted, the
recovery checkpoint reduced first-command p95 from 9.71 to 7.39 mm versus
the extra-budget control (old checkpoint: 9.09 mm). Only one of two held-out
episodes contributed accepted rows, and just 14 accepted rows were within
40 mm TCP axial standoff; this is weak evidence about near-contact recovery.
[Control](../../outputs/experiments/2026-10-02_two_family_perception/local_actor39_extra_budget_control_v1/summary.json),
[recovery](../../outputs/experiments/2026-10-02_two_family_perception/local_actor49_recovery_v1/summary.json),
[held-out action agreement](../../outputs/experiments/2026-10-02_two_family_perception/sfp_recovery_development_actor_offline_matched_v1.json).
Autonomous scored comparison is pending; no insertion improvement is claimed.

The other held-out three-card reset (`card1_port1_cards3_extra0_development`)
exposed a specific selected-card confusion. For 114 post-20-second frames,
the full-image locator's card-1/port-1 output lay near the **card-2/port-1**
opening in all three cameras. The broad crop corrected center and right,
while left remained on card 2. Its left final pixel median error was 256.5 px;
the resulting three-view reprojection median was 109.1 px, so the runtime
consistency gate rejected all 102 post-release source windows. The other
held-out reset had 0.61 px median reprojection and 0.14 mm median 3D position
error. These are posthoc simulator-label diagnostics, never actor inputs.
[View diagnosis](../../outputs/experiments/2026-10-02_two_family_perception/sfp_recovery_development_view_diagnosis_v2.json),
[accepted rows](../../outputs/experiments/2026-10-02_two_family_perception/sfp_perturbed_recovery_development_cascade_bc_2/summary.json).
The earlier 19 native locator training starts had one- and five-card scenes
but **zero three-card scenes**; three-card scenes were held out. A targeted
ten-reset three-card training collection is running with those held-out
resets and the reserved final split excluded. Its hypothesis and stopping rule
are explicit: train one matched locator continuation, then keep it only if
the wrong-card failure is repaired on a separate three-card development reset
without degrading the predeclared original selection/test and SC metrics.

### October 2 continuation: corrected handoff attribution and identity retry

The completed matched corrective-BC comparison did not improve autonomous
insertion. On the same five-card card-2/port-1 development reset, both the
extra-budget control and corrective-data actor scored tier 3 = 0. Posthoc
scoring TF showed only 2.01 and 2.17 mm of maximum TCP axial progress,
respectively. The tip was already close to the opening plane but several
millimeters sideways; this was an inadequate near-port starting state for the
comparison. The old 39-scene actor previously advanced the actual tip about
37.4 mm in the same reset but also failed to insert. See the
[matched summary](../../outputs/experiments/2026-10-02_two_family_perception/sfp_local_actor_recovery_match_v1_tip_summary.json).

Ten disjoint three-card training resets and four additional card-0-absent,
card-1-selected resets were collected to test the target-identity hypothesis.
All four focused teacher runs inserted. One ordinary locator continuation
reduced the original native held-out selection/test selected-point p95 from
217.9/113.2 px to 110.8/76.0 px, but the missing-card development scene still
had 69.6% of selected predictions closer to another card, and its cascade
accepted only 1/100 frames. The absent-identity loss reduced absent-slot false
positives but hurt present-target recall and SC. A further focused sampler
allocated 25% of updates to 1,895 missing-card training images; its
missing-card development wrong-target fraction improved to 38.6% (from 69.6%)
but selected-point p95 remained 225.9 px. Original native selection/test p95
regressed to 126.8/112.6 px and SC near p95 to 8.0 px. None of these
checkpoints is deployable. This is an unresolved card-indexing failure,
especially when card 0 is absent; more weight on the same four starts did not
solve it. [Focused checkpoint](../../outputs/experiments/2026-10-02_two_family_perception/joint_locator_missing0_focus25_v1/summary.json),
[held-out identity comparison](../../outputs/experiments/2026-10-02_two_family_perception/sfp_recovery_development_locator_focus25_v1.json).

One diagnostic mistake initially made a tier-3 score of 75 look like a local
actor success. The teacher handoff monitored the plug body link rather than
the actual SFP tip, so the teacher inserted the tip to 46.8 mm before the
actor's first decision; scored TF showed only 0.002 mm subsequent tip motion.
Switching to the tip link exposed a second issue: the Cartesian servo kept
executing the teacher's last target while the actor model loaded, and an
insertion event arrived before that actor started. The corrected handoff sends
an immediate zero-delta hold command and records the measured tip and
simulation time. In the held run, the teacher stopped with the tip already
23.0 mm inside the scored port opening, the startup moved it 2.6 mm, then the
frozen 39-scene actor advanced it another 21.1 mm and the official score was
75. Complete actor inference p95 was 223.3 ms. This proves some BC-controlled
insertion motion on this one training reset, but it does **not** establish
insertion from outside the port. The port-link frame used for the teacher
threshold differs from the scored opening frame by about 52 mm along the
axis. A subsequent frozen-actor test uses an earlier 85 mm port-link threshold
to put the tip outside the opening; it is pending. No final scenes were opened.
[Held-run tip attribution](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_physical_tip_handoff_held_35mm_v3_summary.json),
[teacher handoff audit](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor39_physical_tip_handoff_held_35mm_v3/physical_tip_handoff.json).

The first outside-opening test used an 85 mm port-link threshold with the
same frozen actor and hold. Posthoc scored TF put the plug tip at
`[+6.30,+9.32,-35.83]` mm (lateral 11.24 mm, axial 35.83 mm before the
opening) at handoff. It drifted 11.69 mm axially while the actor loaded, then
the actor advanced 25.80 mm but ended near the lip at
`[+7.27,-11.94,+1.52]` mm and scored tier 3 = 0. The actor did not solve a
roughly 10 mm lateral error before moving axially. The next controlled test
holds axial depth at the measured early handoff and lets the teacher finish
lateral alignment to 2 mm before starting the frozen actor. It tests a clean
outside-port insertion start; it is **not** an autonomous transport result.
[Outside-port summary](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_physical_tip_handoff_held_85mm_v4_summary.json),
[video](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor39_physical_tip_handoff_held_85mm_v4/five_cameras_10fps.mp4).

That axial-hold start-generator test did reach the intended outside-port state
at teacher handoff: the scored tip was 37.49 mm before the opening and only
1.23 mm lateral. It still failed (tier 3 = 0). The decisive flaw was a
17.2-second actor model-loading gap **after** handoff. During the gap, the
tip's lateral offset grew to 7.43 mm despite a zero-delta hold command. The
actor then advanced 49.53 mm axially, but its maximum insertion-depth point
was 16.91 mm sideways and it never entered correctly. The local actor now
preloads its perception and policy weights before teacher transport; a repeat
with the same frozen weights and axial/lateral handoff thresholds is pending.
This is a handoff control repair, not a new trained model.
[Axial-hold failed run](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_physical_tip_handoff_axialhold_85mm_lat2_v5_summary.json),
[preload implementation](../../aic_example_policies/aic_example_policies/ros/CheatCodeThenLocal.py).

The preloaded repeat removed the startup gap: the scored tip was
`[-1.54,+1.16,-34.76]` mm at handoff and
`[-1.19,+1.34,-34.73]` mm at the first actor decision, only 0.03 mm axial
motion apart. It still scored tier 3 = 0. The actor advanced 36.57 mm, but
the tip reached `[+5.03,-11.21,+1.84]` mm at its maximum axial point. This
is a genuine outside-port BC failure on one training reset, rather than a
handoff-timing artifact. The current complete inference p95 was 231.7 ms.
[Attribution and score](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_preloaded_axialhold_85mm_lat2_v6_summary_pose.json).

The posthoc geometry comparison separated translation from orientation in
the frozen RGB estimator at this clean first decision. The selected opening
position error was only 0.10 mm in 3D, but its frame orientation error was
3.17°; the median orientation error across the actor run was 3.06°. A 3°
rotation error over a roughly 90 mm TCP-to-port lever arm can create several
millimeters of wrong lateral control even with an accurate opening center.
This is a concrete perception hypothesis, not proof that the actor would work
with a correct frame. The official SFP asset and all 268 verified SFP episode
labels have a fixed mount roll/pitch and vary only in card yaw. A bounded,
observation-only ablation therefore refines **both** RGB-visible openings on
the selected card, triangulates each, computes yaw from their connecting line,
and uses the fixed mount roll/pitch. It rejects inconsistent pairs. No
scene-specific simulator geometry enters runtime; the known 23.2 mm port
spacing and mount rotation are static asset priors. Grouped development error,
acceptance, and complete live latency must pass before this representation
can replace the learned orientation head. Missing-card identity remains a
separate unresolved issue.

The first paired-opening implementation that reran the broad refiner for a
sibling snapped to the already selected opening in one view and rejected all
40 development frames. A direct 160px native sibling crop around the
projection from the selected refined point and the learned orientation seed
removed that ambiguity. Best-two-view triangulation handles a single
occluded or inaccurate sibling view, with a 23.2±3 mm spacing and <3 mm
height-consistency gate. On the original ten grouped native development
scenes, matched 12-frame-per-episode replay accepted 85/120 frames versus
93/120 for the old head. Among accepted frames, orientation p95 improved
from 8.77° to 4.17° overall and from 7.25° to 1.17° on the five untouched
development-test episodes. Some two-view outliers remain; this is not a
universal perception repair. The card-0-absent held-out scene still has a
wrong-card identity and is not rescued by orientation.
[Matched old head](../../outputs/experiments/2026-10-02_two_family_perception/sfp_pair_orientation_native29_baseline_hostpaths_v2.json),
[paired-opening head](../../outputs/experiments/2026-10-02_two_family_perception/sfp_pair_orientation_native29_robust2view_v3.json).

Replaying the **first 20 recorded images/states of the clean failed live
actor start**, the paired-opening mode accepted 20/20 and achieved 0.72°
orientation error on the first frame, then mostly below 0.5°. The old live
orientation error at that first frame was 3.17°. This replay uses scoring
geometry only to calculate error after prediction. A single same-scene live
comparison with the frozen actor, same privileged teacher start generator,
and pair mode enabled is running; it will test whether better orientation
actually changes insertion and whether complete inference p95 stays below
300 ms. [Replay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_pair_v6_live_first20_replay_v1.json).

That matched live pair-mode comparison **failed** tier 3 = 0. Its first RGB
port position error was 0.16 mm in 3D and orientation error 0.31°; after the
five-sample lock, orientation error was about 0.03°. Complete inference p95
was 239.8 ms, so latency passed. The plug tip began 34.5 mm outside the
opening at 1.6 mm lateral, advanced 60.8 mm axially, but ended 20.9 mm
laterally away. This rules out port-frame pose error as the sole explanation
of this actor's failure on this reset. The visible lock consumed five slow
initial estimates and the robot drifted from about 1.7 to 10.9 mm lateral
before the actor's first non-hold target. A one-observation-lock diagnostic
isolated that waiting delay from action prediction. It is not a promotion
candidate because it drops temporal consensus.
[Paired live summary](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_preloaded_axialhold_pair85_lat2_v7_summary.json).

The recorded action trace also shows that the actor later commanded TCP
targets well outside the successful teacher's usual lateral range. For
example, in the pair-mode run at 55–70 s the median commanded offset was
approximately `[+4.0,-3.7,-3.3]` mm from the measured TCP, while the TCP
and tip diverged laterally and the measured force stayed around 19–21 N.
The 39-scene native BC training source includes 219 training rows from 26
episodes with predicted TCP z in `[-100,-80)` mm relative to the opening,
so the outside-port depth is represented. However, held-out near-port target
p95 remains above 2 mm and the actor has not learned a reliable 0.5 mm
insertion corridor. Its local training labels used the previous learned
orientation representation; changing runtime pose to the paired representation
without reexpressing recorded targets for training is also a possible
distribution mismatch. No RL result is claimed.

### October 2: immediate-lock failure and matched actor-frame repair

The one-observation-lock repeat used the **same frozen 39-scene actor**, same
five-card training reset and same privileged teacher start generator. The
scored plug tip was `[+0.36,+1.45,-35.92]` mm in the true selected-port frame
at the first actor decision, so it began about 36 mm before the opening and
1.45 mm sideways. It moved only 0.01 mm axially during actor startup. The
actor advanced it 38.99 mm, but the point of maximum axial progress was
`[+2.69,+20.41,+3.07]` mm, far outside the 0.5 mm corridor. Tier 3 was 0.
The RGB port estimate differed from the scoring pose by about 0.11 mm laterally,
0.28 mm axially and 0.086° in orientation. Complete decision inference p95
was 234.7 ms; p99 was 247.2 ms. This single-reset diagnostic shows that
accurate perception and prompt handoff are insufficient for this BC actor.
It does **not** establish performance across resets or prove that the actor
training frame is the only remaining cause.
[Machine summary](../../outputs/experiments/2026-10-02_two_family_perception/sfp_actor39_pair85_lat2_lock1_v8_summary.json),
[five-view video](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor39_pair85_lat2_lock1_v8/five_cameras_10fps.mp4).

One concrete train/runtime mismatch remains: the actor's native labels were
expressed in the **old learned RGB orientation frame**, while this live run
supplied the new paired-opening RGB frame. A bounded matched repair now
reexpresses the **same recorded teacher target poses** through the paired
RGB frame. This is a coordinate conversion, not generated corrective actions:
for every one of the 9,424 matched four-target commands, reconstructing the
base-frame target from either dataset agrees within
`1.8e-13 mm / 2.3e-14°`. The paired estimator accepted all 2,220 native
training rows and 136/155 grouped validation rows; both arms train on the
same 2,356 accepted rows from 29 training and 5 validation episodes. The
reserved final split is excluded. Both arms resume the exact 39-scene actor,
use the same 600-update cap and hyperparameters, and train only on native SFP
rows. The old-frame arm is a matched control. This is a limited diagnostic of
coordinate consistency; SC and absent-card identity remain open.
The first relabel attempt (`sfp_native_pair_relabel_matched_v1`) incorrectly
parsed 9D actor poses as 7D quaternion poses. It was caught before training,
marked invalid, and superseded by v2; no model used those labels.
[Relabel script](../../scripts/relabel_sfp_native_pair_actor.py),
[run command](../../scripts/run_sfp_pair_actor_matched.sh),
[paired dataset](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_native_pair_relabel_matched_v2/summary.json).

Both matched native-only continuations completed their 600-update budgets.
On the same five grouped validation episodes and 136 accepted frames, the
old-frame control's near-port lateral target p95 was **2.20 mm**, versus
**2.07 mm** in the paired-frame arm, only about 6% better and far beyond the
0.5 mm insertion corridor. Buffered lateral p95 was 2.38 versus 2.13 mm;
first body-command p95 worsened from 9.05 to 9.81 mm. The paired checkpoint
was selected at update 100, so more updates in this setting did not reliably
help. This comparison does not support a live promotion. The first 20 RGB
observations of the frozen v8 rollout were then replayed through both new
checkpoints and the original actor. All three proposed a retreat from the
first pre-opening state: current RGB-relative TCP `[+1.27,+1.63,-93.68]` mm,
original first target `[-6.64,-6.60,-97.74]` mm, paired first target
`[-6.80,-5.62,-97.03]` mm. No true target pose entered that replay.
[Old-frame checkpoint](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor_oldframe_matched_v1/summary.json),
[paired checkpoint](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor_pair_matched_v1/summary.json),
[RGB replay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_v8_three_actor_first20_replay.json).

The remaining action error has a narrower, testable signature. In the
successful native teacher data, where RGB-relative TCP axial position is
`[-105,-80]` mm and lateral magnitude is below 3 mm, 69 training rows from
18 episodes move forward by more than 0.5 mm **93%** of the time. The five
validation episodes contribute only eight such tightly aligned rows; 75%
move forward. In v8, the actor instead repeatedly sent targets with more
negative axial coordinates while the TCP retreated to roughly -112 mm by
45 s. The measured robot initially moved toward those retreat targets;
later its lateral motion reversed while the robot had substantial contact
force. This supports a BC phase/action error at the handoff, but does not
exclude downstream cable or plug contact as a second failure.

One final bounded supervised diagnostic therefore oversamples **only these
recorded tightly aligned pre-opening rows**, adds a supervised first-target
axial loss, and selects on grouped validation action sign/axial error. It
starts from the paired-frame checkpoint, has an 800-update / 60-minute cap,
and uses no new episodes or invented targets. Continue to live testing only
if forward-sign accuracy reaches at least 75%, axial target p95 is below
3 mm and near-port lateral p95 is no worse than 2 mm. These values test whether
the specific retreat defect was repaired, not whether full insertion is
reliable. If the gate fails, stop this BC branch and reconsider phase-specific
control or genuinely on-policy successful corrections.

The aligned-approach run finished all 800 updates and selected update 600.
On the grouped validation subset, it reached **7/9** correct forward/hold
signs and **2.45 mm** p95 axial target error. However, the same grouped
validation had **2.17 mm** near-port lateral target p95, exceeding the preset
2 mm companion gate, so it was **not** promoted to live insertion. Replaying
the first 80 recorded RGB/state observations of the failed v8 rollout gives
a second warning: among nine tightly aligned pre-opening observations, the
original and paired-frame actors commanded forward in **0/9**; the targeted
actor commanded forward in only **1/9**. It changed the *first* action from a
4.1 mm retreat to a 1.3 mm advance, but it did not keep advancing after the
robot drifted away from its teacher distribution. No simulator geometry was
used in this replay. The result supports an on-policy state/phase mismatch
and a residual lateral-control problem; a further routine native-BC restart
is not justified by this evidence.
[Targeted checkpoint](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor_pair_aligned_approach_v1/summary.json),
[80-observation replay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_v8_aligned_approach_first80_replay.json).

A final same-reset **privileged feasibility control** restarted stock
CheatCode after the measured handoff. It must not be counted as autonomous
or BC performance. The first requested run did not forward its environment
flag into the rootless container and repeated the old actor (tier 3 = 0);
the second raised the handoff exception again during teacher restart and is
invalid. Both are retained with invalidity reasons in their manifests. The
corrected third run started the scored tip about 81.4 mm behind the port-link
origin and 1.96 mm lateral at handoff, then entered the opening and emitted an
insertion event. Official tier 3 was **50, partial insertion**, with no
excessive force or contact penalty. The original uninterrupted stock teacher
for this exact saved reset scored **75, full insertion**. Reentering the stock
teacher at the middle of its route repeats its interpolation and changes the
motion history, so this partial result neither validates nor rejects a
stage-preserving teacher takeover for on-policy corrective data. It does show
that the attempted simple restart is unsuitable as a reliable full-success
expert collector. [Restart score](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_preopening_teacher_restart_control_v3/results/scoring.yaml),
[handoff audit](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_preopening_teacher_restart_control_v3/physical_tip_handoff.json),
[original uninterrupted score](../../outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1/card2_port1_cards5_train/results/scoring.yaml).

**Decision and next bounded repair:** Stop unchanged native BC, pair-frame
relabeling, and near-phase oversampling; all three missed the action gate.
The next experiment should implement a **stage-preserving expert takeover**
that continues from the current pre-opening axial/lateral state without
restarting the five-second transport interpolation. Pilot it on this saved
development reset with 0, 5, and 20 visual-actor decisions before takeover,
at most two repeats each (six runs). Admit teacher continuations for DAgger
only if the official score is 75, the scored tip reaches full depth, force is
acceptable, and command/observation labels are causal. If 0-decision
takeover fails 2/2, repair the takeover controller first; do not collect
actor failures. If 0 succeeds but 5/20 do not, the actor leaves the teacher's
recoverable region, so shorten the handoff or train a dedicated recovery
controller rather than adding labels from failed continuations. Keep the
reserved final split sealed and do not start RL while autonomous insertion
remains 0.

The stage-preserving teacher takeover is now implemented as a **diagnostic
only** continuation of CheatCode's current z-offset and XY servo state; it
does not rerun its initial 5.5-second transport interpolation. In the first
zero-decision run, the axial hold began with the tip 11.6 mm sideways and the
start generator never reached the 2 mm handoff criterion, so that run never
tested takeover. The second zero-decision repeat reached handoff and scored
**75, full insertion**. In the first five-decision pilot, the frozen visual
actor issued 15 Cartesian commands from five proposals, then the privileged
teacher continued and scored **75, full insertion**. This establishes one
recoverable short actor prefix on one training reset; it is **not** an
autonomous insertion result or evidence of cross-scene reliability. The
planned 20-decision pilot tests how quickly the actor leaves that recoverable
region. [Zero-decision score](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_stage_preserving_teacher0_v2/results/scoring.yaml),
[five-decision score](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_stage_preserving_teacher_after5_v1/results/scoring.yaml),
[takeover code](../../aic_example_policies/aic_example_policies/ros/CheatCodeThenLocal.py).

The 20-decision prefix also recovered to tier 3 = **75** on the original
card-2/port-1 five-card training reset (56 actor Cartesian commands), then
on a changed card-0/port-0 five-card training reset (35 actor commands).
Both final successes are **privileged teacher-assisted**. They establish
that the late actor states in these two runs were recoverable by a controller
that preserves the near-port stage; they do not show that the actor can
recover or insert by itself. [Original-scene assisted score](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_stage_preserving_teacher_after20_v1/results/scoring.yaml),
[changed-scene assisted score](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_stage_preserving_teacher_after20_card0port0_v1/results/scoring.yaml).

A bounded single-GPU collection is now testing one new 20-decision takeover
per five-card training identity, at most 14 total attempts and two per
identity. It stops after five tested identities if fewer than three yield
full, event-timestamped expert continuations. Only tier-3=75 runs with a
retained takeover timestamp and actual actor commands enter the DAgger
derivative. The first admitted new run produced 134 **recorded four-command
teacher chunks strictly after takeover**; its paired RGB port relabel
accepted 134/134, with posthoc RGB lateral error 0.30 mm p95. The selected
port truth is used only for supervised label conversion and posthoc filtering,
never autonomous actor input. Corrective rows within two seconds of takeover
are marked so that a later matched BC run can weight the genuinely on-policy
region instead of simply adding many routine teacher frames.
[Collection driver](../../scripts/collect_sfp_stage_preserving_dagger.py),
[first causal chunk report](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_dagger_pilot1_teacher_chunks_summary.json),
[first RGB relabel report](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_dagger_pilot1_pair_bc/summary.json).

The first admitted actor prefix is a genuine off-trajectory correction
example, rather than merely routine teacher continuation. Posthoc scored tip
TF places the tip `[+0.39,+1.60,-35.24]` mm at the first actor decision.
During the 20-decision prefix it advanced 17.14 mm axially at best but ended
near `[+2.08,+5.60,-26.44]` mm before teacher takeover: still outside the
opening and 5.6 mm sideways. The stage-preserving teacher subsequently
recovered to full tier-3=75 insertion. These true-tip numbers are **analysis
only**; neither the actor nor its crop selection reads them. Complete actor
inference p95 on this prefix was 258 ms, below the 300 ms limit (p99 includes
initial setup and was 1,156 ms). [Prefix posthoc summary](../../outputs/experiments/2026-10-02_two_family_perception/sfp_dagger_pilot1_prefix_posthoc.json).

### Bounded on-policy correction comparison, declared before training

The five-card SFP collection completed at its ten-identity cap: 12 attempts,
10 full tier-3=75 **teacher-assisted** insertions, one per card/port identity.
The first repeats for card 3 ports 0 and 1 failed tier 3, and their traces and
videos remain archived. The accepted data contain only recorded commands
after a timestamped privileged teacher takeover, with the actor's preceding
20 visual decisions retained as the causal state history. The collection
success rate must not be reported as autonomous actor performance.
[Collection summary](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_onpolicy_takeover20_fivecard_v1/summary.json).

The next matched comparison starts both arms from the same paired-frame actor
checkpoint and uses the same 1,000-update budget. The control sees the prior
paired native expert data. The correction arm also sees accepted post-takeover
rows, with extra weight only on the first two seconds of actual corrective
teacher behavior. Grouped native validation episodes remain held out; the
reserved final scenes remain sealed.

**Promotion gate, set before inspecting these training results:** the
correction arm must reduce grouped validation near-port lateral target p95 by
at least 20% against the matched control and reach at most 1.5 mm; on the
frozen v8 RGB/state trace, it must command axial advance in at least 6 of
the nine tightly aligned pre-opening observations where previous actors
advanced in at most one. Complete live inference p95 must stay below 300 ms.
If this offline gate passes, compare both arms autonomously on identical
non-final development starts, with no post-handoff teacher. Full tier-3
insertion and measured lateral/axial error decide promotion. If it fails,
stop this BC continuation and investigate an explicit approach/recovery
phase policy or a bounded RL diagnostic; do not add routine expert data or
claim that DAgger fixed insertion.

**Observed result.** The causal extractor yielded 1,409 post-takeover
recorded four-command chunks; the RGB-pose consistency filter admitted 1,392
rows from ten complete training episodes, including 261 rows in the first two
seconds after takeover. Paired RGB pose lateral error on accepted rows was
0.35 mm p95. The grouped validation set had five other complete episodes.
Both BC arms completed 1,000 equal updates. The control selected update 800;
the correction arm selected 900. Held-out near-port lateral first-target p95
was **1.82 mm control versus 1.72 mm correction**, a 5.5% improvement, below
the declared 20% and 1.5 mm thresholds. Buffered lateral p95 was 1.95 versus
1.84 mm; first body-command p95 was 9.05 versus 9.05 mm. Among nine aligned
held-out teacher observations, forward-sign accuracy decreased from 5/9 to
3/9. On the same nine aligned, frozen v8 on-policy RGB observations, **both
models advanced in 0/9**. The correction arm's mean axial command was an
8.76 mm retreat versus a 10.21 mm retreat for control. This is insufficient
to start an autonomous live test, so the predeclared gate fails. It also does
not establish a live latency result for these checkpoints.

A bounded label-distribution diagnosis found a specific mismatch. The frozen
failed actor began near TCP port-frame z = −94 mm. The old expert dataset has
110 tightly aligned train rows around z = −90 mm, with 82.7% commanding axial
advance. The newly weighted corrective rows are mostly **later** states:
their median z is −58 mm and only one satisfies the same tight aligned-start
condition. Their first recorded target often advances about 41 mm, but it is
issued after 20 actor decisions, when the actor has already changed its state.
Thus the DAgger rows supervise recovery after drift; they do not cover the
specific first-decision phase in which the deployed actor chooses retreat.
The old start-phase labels are present, so this observation identifies a
coverage/conditioning hypothesis rather than proving that missing examples
alone caused the failure. No additional routine collection or restart is
justified from the current result.

**Next targeted experiment, if the BC branch is resumed:** use the already
recorded zero/five-decision teacher takeovers to test whether the same actor
can distinguish approach from recovery *at the point of handoff*. First
compare their causal RGB/state and recorded teacher command distributions to
the v8 start; only if there is overlap, add at most five new five-decision
development takeovers across distinct card/port identities, with a cap of
one GPU-hour of collection and 1,000 equal BC updates per matched arm on two
GPUs. Stop unless the same 20%/1.5 mm grouped lateral gate and 6/9 v8
forward gate both pass. If still missed, change the phase representation or
run a bounded online RL diagnostic on safe development starts before another
BC collection. RL, final scenes and any claim of autonomous insertion remain
gated. [Machine decision](../../outputs/experiments/2026-10-02_two_family_perception/sfp_dagger_matched_decision.json),
[matched commands](../../scripts/run_sfp_dagger_matched_bc.sh),
[frozen RGB replay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_v8_dagger_matched_first80_replay.json).

The read-only overlap check passed: the existing five-decision pilot's
predicted TCP port-frame z was −93.7 mm at first local observation and −93.9
mm at teacher takeover, compared with −93.7 mm at the frozen v8 initial
decision. Its teacher continuation fully inserted. A capped new phase probe
therefore uses five distinct training card/port identities, five actor
decisions, at most one attempt each on GPU 3. Only full-score causal
post-takeover commands may enter its derivative. The preceding gate and
one GPU-hour collection budget apply before any further matched BC training.

**Early five-decision phase-probe result:** Five of five distinct training
starts fully inserted only after stage-preserving teacher takeover. Their
recorded commands produced 709 accepted paired-RGB rows from five episodes,
139 within two seconds of takeover. Only **two** early rows were both within
the frozen failure's −105 to −80 mm TCP axial band and within 5 mm laterally;
the teacher rapidly moves out of that band. A 1,000-update continuation from
the same paired actor checkpoint reused the exact unchanged control arm.
Its five-episode held-out near-port lateral p95 was **1.75 mm**, versus 1.82
mm control; first body-command p95 worsened from 9.05 to 9.78 mm. On the
frozen v8 actor-generated RGB/state trace it commanded axial retreat in
**9/9** aligned observations, versus the control's 9/9. The predeclared
20%/1.5 mm and 6/9 forward gates fail again, so no live insertion rollout
was started with this checkpoint.

The decisive same-checkpoint diagnostic distinguishes fitting teacher data
from generalizing to its own state distribution. Among 110 aligned **training**
rows, the teacher advanced in 91, the control actor advanced in 86, and the
20-decision corrected actor advanced in 94; on nine actor-generated aligned
v8 states, both corrected actors advanced in **zero**. Task vector and 32D
state shapes match between live and recorded data, and the replay applies
the native RGB/JPEG resize path used by training. Same-scene train rows are
only around TCP z≈−81 mm while the live v8 first decision is z≈−94 mm.
This is an on-policy distribution/phase generalization failure. More ordinary
BC or repeating the same takeover collection is stopped. The next repair
must explicitly test a pose-conditioned approach phase or another mechanism
that preserves axial progress on actor-generated states, with a matched
unchanged-action baseline and no true geometry at runtime.
[Early collection](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_onpolicy_takeover5_fivecase_v1/summary.json),
[matched early checkpoint](/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_actor_dagger_corrected_early5_v1/summary.json),
[three-way frozen replay](../../outputs/experiments/2026-10-02_two_family_perception/sfp_v8_dagger_threeway_first80_replay.json),
[machine decision](../../outputs/experiments/2026-10-02_two_family_perception/sfp_dagger_early5_decision.json).

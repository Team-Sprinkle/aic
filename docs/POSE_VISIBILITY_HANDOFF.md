## October 18: gated close-up pose update and SC insertion result

V34 passes **4/4 matched SC development cases** (zero/five obstacle cards, both ports), followed by a passing SC round in the unchanged public sample: **15.510 mm depth, 0.141 mm lateral error, 0.469° orientation**, toolkit completion, no harm. Five fresh agreeing images support one bounded plug-pose update at 25 mm; the port frame stays fixed. Tighter lateral centering and stronger lateral tracking are also needed. All used-stage p95 timings pass; these are explicit RGB/public-state control results, not BC or RL.

The complete sample is **170.57/300, 2/3 strict** because SFP trial 2 has a 3.53° biased grasp estimate before contact. Its actual roll exceeds the original SFP training range; close-up predictions remain wrong. Repeats stop, and the next targeted repair is SFP orientation coverage rather than a stiffness-only change. No new training or collection is running. The SC model's original 1.026° qualification failure remains explicit; four successful development cases do not establish general reliability. Final scenes remain sealed.

[Detailed evidence, commands, images and next repair](experiments/2026-10-18-sc-explicit-controller-continuation.md).

> **October18 live milestone:** the discrete25mmRGBheld-plug pose update passes one failedSCdevelopment start and then **3/3strict insertions** in the unchanged public three-trial sample(257.36/300). A matched controller without the update failsSC. One local recovery occurs in the development success; this is **not evidence of hierarchical superiority or newRL**. Public-sample timing/guard repairs continue with failures preserved; model original subgroup1.026°stillfails1°. [Evidence and limits](experiments/2026-10-18-sc-explicit-controller-continuation.md#first-live25mm-update-full-sc-physical-insertion).

> **October18 SCexplicit continuation:** repeated original public full-sample scores **257.76/257.65 out of300**, bothSFPstrict passes each time; SCtoolkit75but~15.48mm depth still fails15.5mm. Fresh SCdevelopment1/4strict.12near-view perception holds are complete; bounded acquisition-view curation and matched pose fits continue. Discrete gated25mmgrasp re-estimation is the next policy experiment, not an unvalidated per-frame update. No newRL/world training. [Evidence, jobs and next gates](experiments/2026-10-18-sc-explicit-controller-continuation.md).

# Target-conditioned perception and two-actor handoff

**October 18 current result:** the unchanged three-trial public sample improves152.61→176.85/300 after an explicit-controller seating hold fix, with2/3combined successes(bothSFP). SCstillstopsnear2mm/2.17degrees; learnedBCremains0/8and is not the source of the SFP successes. An exact shared-JPEG preprocessing repair passes3/3liveSFP repeats with acquisitionp95<228ms. Current SC recovery work uses only RGB/public robot state; no newRL or final scenes. [Current cycle and gates](experiments/2026-10-18-three-trial-insertion-repair-cycle.md).

> **October 16 current scope:** follow [STATUS](STATUS.md) and [the active execution record](experiments/2026-10-15-rule-perception-hierarchical-execution.md). Rule transport now hands the full150mm approach to a shared SC/SFP actor. A single public requested-port crop per camera feeds the native five-reference pose refiner; no all-port precise prediction or learned confidence is required. Current coarse flags mean projected-in-frame, **not certified physical visibility**. Hidden-port/black-gripper examples remain an explicit label/observability limitation, and rigid geometry/identity/force/standoff gates still determine acquisition. Full refiner retraining and targeted SC/SFP coverage repairs are documented in the current record; earlier statuses below are historical.


**October 14 current execution:** Supported normal-board SC rule transport and RGB handoff passed small screens. Matched corrective-data, decoder and 6-second robot-history repairs did not pass autonomous full insertion. A separately frozen observation-only approach reached **29.12 mm before the opening / 0.148 mm lateral**, then the unchanged trained actor achieved **11.07 mm insertion depth / 0.413 mm lateral / 0.94° orientation error** on one reused start. Official score 75, but the unchanged ≥14 mm physical gate failed. Logs confirm contact-triggered completion and end-effector locking; no cable snag explains this run. This is meaningful approach progress, not a promoted insertion policy. All new jobs finished; no routine collection, RL, world-model work or final-scene access. SFP local insertion, rotated SC and port-0 rail minimum remain unapproved. [Execution record](experiments/2026-10-14-transport-handoff-execution.md) · [Easy explanation](answers/2026-10-14-autonomous-execution-outcome.md).

**October 7 SFP board context limit:** The actual rendered board screws are more meaningful labels than collision-envelope corners, but the two-screw early visibility gate failed 0/10 complete groups because cards obscure screw 0. A privileged one-screw true-position upper bound produced 22.62 mm median target error with a train yaw prior and 15.02 mm even with true yaw. Thus board origin alone cannot provide selected-port pose or a safe handoff; request-conditioned local port perception and explicit visual identity/visibility remain required. [Audit and images](experiments/2026-10-07-sfp-visible-board-origin-audit.md).

**October 7 label observability check:** Matched 20/40/70-group original-RGB fits had held-out endpoint p95 **34.58/34.67/32.17 px**; the unchanged causal board fit reached **1/10, 0/10, 3/10** accurate early development groups. A 30-panel single-reviewer packet found 23 CAD projections ambiguous/repeated/obscured and seven uncertain; none certified a uniquely visible numbered board origin. This is diagnostic rather than independently certified visibility. Stop scaling these labels and keep automatic handoff disabled. [Review panels and result](experiments/2026-10-07-sfp-rail-data-scale-and-visibility.md).

**October 7 temporal board-reference stop:** The fixed RGB endpoint model plus 3 s history yielded only 1/10 early accurate SFP board references; an original-resolution crop refiner reached 3/10 (required 9/10). The cropped prediction remained 9.92 px median / 32.17 px p95 from projected CAD endpoints. Projected-in-frame does not mean visually observable. Privileged 3 px-projection controls reached 10/10, but cannot choose runtime landmarks. Board-anchored association remains an unapproved research path pending an independently reviewed, genuinely visible fixed slot origin. This does not alter the transport, coarse check, precise pose or insertion deployment gates. [Record and example crops](experiments/2026-10-07-sfp-temporal-board-reference.md).

**October 7 target-cue outcome:** A local board-slot feature head failed three-camera consistency (1/10 early accepted groups), while a shared-3D head yielded wrong neighboring-card accepts and severely underpredicted error. An internally consistent 3D cue is **not** evidence of correct requested-port identity or calibrated visibility. The structural cue branch is stopped; no model output was connected to the transport or insertion actor. The observation-complete 30-panel packet still awaits independent review. [Bounded experiment](experiments/2026-10-07-sfp-slot-feature-port-cue.md).

**October 7 observation-completeness correction:** The first SFP blind review's 11 uncertain RGB panels omitted measured TCP pose, which a deployable coarse checker may use. A new 30-panel packet includes that public state, task request, camera history and fixed slot order but has no independent annotations. Do not treat the earlier abstentions as proof the full observation cannot identify the target. A fixed narrower 320 px crop gave 6/10 early clean two-camera development groups, below its 9/10 gate; no fit followed. Visibility still means evidence that the *requested* opening can be identified, not merely that its simulator projection lies inside a crop. Handoff remains off. [Review and crop audit](experiments/2026-10-07-sfp-observation-complete-review-and-tight-crop.md).

**October 6 active-view limit:** A true +10° guarded wrist-camera tilt from a reused nonfinal SFP scene still did not show a reliable absolute numbered card and chosen port side in two cameras. The controller held orientation but drifted 6.38 mm in position, failing the preset 5 mm gate. This is one view and one reviewer; it does not prove the visual task impossible. Do not convert geometric in-frame projections into visibility/confidence labels or enable the handoff. [Pilot, image and traces](experiments/2026-10-06-sfp-board-tilt-view.md).

**October 6 identifiability check:** A projected requested opening inside a camera image does not prove that its numbered card is visually identifiable. In the first 11 of 40 saved unmarked SFP triplets, the reviewer abstained on absolute requested-card/opening identity; a safe opposite-side view on one reused scene made the row less visible. This is partial evidence, not a calibrated visibility score or a 40-scene success rate. Keep visual accept/handoff disabled and seek an observable fixed slot reference before another identity fit. [Review and original RGB](experiments/2026-10-06-sfp-requested-card-identifiability.md).

**October 6 card-identity repair result:** A native crop model predicting rail anchor plus both ports rejected too many observations (5/119 guarded accepts, 2/10 early groups); a rail/TCP port prior missed by 111 px p95. Projected-in-crop flags remain geometric labels, not true visibility, and confident neighboring-card matches remain possible. Retain the single-point detector for offline comparison only. Before another handoff fit, review original RGB for whether the requested card and opening are actually identifiable or occluded; abstain on ambiguous views. [Experiment](experiments/2026-10-06-sfp-local-card-pair-repair.md).

**October 6 requested-card identity limit:** A confident crop prediction can be geometrically consistent in all three cameras and over several observations while pointing to the *wrong neighboring card*. On ten new nonfinal SFP scenes, top-five joint camera matching plus two-inspection stability still accepted eight wrong ports. Therefore a high in-crop flag, reprojection score or stable pose cannot replace explicit visual card/port identity or a reviewed visibility/safe-stop label. Keep automatic handoff disabled. [Failure overlays and comparison](experiments/2026-10-06-sfp-joint-camera-candidates.md).

**October 6 SFP pose/visibility clarification:** The latest crop model predicts a requested opening in each of three original-camera crops plus an in-crop flag. The crop center comes from RGB-predicted board rails and the requested card code; projected true poses are used after prediction only to score. Its 1.80 mm accepted 3D p95 is conditional on cross-view acceptance, achieved early in only 6/10 additional nonfinal groups. This flag is **not a calibrated visibility label or a safe-stop signal**: a confident view can point to a neighboring opening, and projected-in-frame does not prove unobstructed visibility. Handoff remains disabled pending joint identity matching, reviewed visibility and autonomous safe arrival. [Experiment](experiments/2026-10-06-sfp-slot-conditioned-crop-feasibility.md).

**October 5 SFP update:** The auxiliary five-rail occupied-slot head now uses local RGB features at its **own predicted** rail positions and gets 10/10 exact maps on grouped development plus 10/10 on additional disjoint nonfinal teacher scenes. This can help reject a request for an empty card rail, but it is **not** target-port identity, visual visibility, confidence, or a safe-stop signal. A strict causal multi-camera board reconstruction produced a valid reference in only 2/10 scenes and lost early port-candidate coverage. The transport → coarse check → precise selected-port pose → insertion contract below is unchanged; no automatic switch is enabled. [Matched experiment](experiments/2026-10-05-sfp-slot-local-occupancy-ablation.md) · [current handoff gates](SELECTED_PORT_HANDOFF_PLAN.md).

**October 4 active revision:** The requested-port coarse model now has a shared spatial/history backbone with two related outputs: current-image crop proposals and a proposal-conditioned **pause/accept for precise inspection**. The former separate late camera-check role moves into this coarse stage. History selection follows measured TCP travel, with 30 mm/up to twelve three-camera anchors as an offline candidate. This decision never starts insertion by itself; a precise selected-pose refiner and final handoff checks remain required. Current SFP identity and autonomous transport gates fail, and true visibility labels are incomplete. [Active execution plan](experiments/2026-10-04-coarse-shared-history-execution.md).

**October 3 gate:** The frozen selected-port identity check accepted a known SC wrong-posture stop. A one-port-aware orientation head improved independent reset errors but stayed above 3° p95. The automatic handoff remains disabled; [the fixed comparison](experiments/2026-10-03-frozen-handoff-gate-and-orientation-repair.md) and [current plan](SELECTED_PORT_HANDOFF_PLAN.md) supersede the earlier candidate-only proposal.

**October 3 evidence:** A task-conditioned SFP candidate matcher fixed four saved wrong-port proposals without losing 69 saved correct ones only after onscreen and offscreen examples were paired within the same reset groups. SC identity still accepts a known wrong-posture stop. This page's deployment handoff remains a proposed design; no automatic switch or insertion promotion followed. See [the bounded experiment](experiments/2026-10-03-selected-port-correspondence-continuation.md) and [current plan](SELECTED_PORT_HANDOFF_PLAN.md).

**Current decision (October 2):** The [active selected-port handoff plan](SELECTED_PORT_HANDOFF_PLAN.md)
supersedes this page's earlier proposal to make a learned confidence head part
of the first perception model. Precise near-target perception will predict
**one task-requested pose or invalid**, plus per-camera *identifiable*
visibility. Transport's arrival signal proposes a pause; fresh precise
perception verifies or rejects it. The current all-candidate locator and
projected-in-frame output remain comparison baselines. Historical results
and earlier design discussions below are retained for provenance.

## October 2 stopping verifier decision

The SC visual standoff is a useful *development* stop verifier on frozen
paths: it latched at a measured valid 10 mm / 3° TCP waiting pose in five
of seven trajectories, whereas the frozen transport-only head did so in
two. It still falsely latched on a wrong route with 6.71° orientation
error, so it is not a general safe handoff. The SFP card-0-absent scene
produced two accepted wrong-port estimates and no temporal lock among 100
sampled frames. Its visual stop remains disabled. A future switch must use
a stable correct-target pose, preloaded insertion actor, and measured-pose
hold. [Same-trace and live evidence](experiments/2026-10-02-transport-stop-comparison.md).

## October 2 corrective actor comparison

The selected SFP pose cascade was used to relabel 1,392 actual post-takeover
teacher action rows from ten full **teacher-assisted** five-card runs. This
tests whether a local actor can learn corrections after drifting away from a
good pre-opening pose. A matched 1,000-update continuation improved grouped
near-port lateral target p95 from 1.82 to 1.72 mm, but missed the preset
1.5 mm and 20% gates. On nine frozen aligned live observations, it still
commanded retreat in all nine. The corrective data mostly describe later
states after 20 actor decisions; the actor's initial handoff decision remains
unresolved. This result does not validate visibility, a general two-actor
switch, autonomous insertion, or RL. [Evidence](experiments/2026-10-02-two-family-perception-local-bc.md#bounded-on-policy-correction-comparison-declared-before-training).

## October 2 pre-opening diagnostic update

The RGB SFP cascade can identify the selected opening very accurately on a
tested five-card training reset when both ports are visible: the paired-port
estimate was about 0.11 mm sideways and 0.086° in orientation from scoring
geometry. That estimate is frozen at the near-port handoff and passed to a
separate BC insertion actor. A one-observation handoff placed the scored plug
tip 35.9 mm before the opening, with negligible startup motion. The actor
then missed laterally by about 20 mm and scored no full insertion. It often
commanded retreat when a tightly aligned successful teacher would advance.
Training the same recorded actions in the new RGB frame helped grouped offline
lateral error only slightly (2.20 to 2.07 mm p95); that actor was not promoted.
Missing-card target identity and actual occlusion visibility remain open, so
this accurate single-scene pose result cannot be treated as a general
handoff solution. See the [experiment record](experiments/2026-10-02-two-family-perception-local-bc.md#october-2-immediate-lock-failure-and-matched-actor-frame-repair).

## October 2 reality check

The implemented SFP locator predicts an opening for each card/port identity
and a **projected-in-camera** flag. It does not yet predict whether the
selected opening is physically visible behind the gripper or cable. The
currently useful reliability signal is repeated RGB agreement: five
three-camera estimates of a static opening must agree within 0.5 mm and
5 degrees before local control starts. This can reject a transient wrong
port, but it can still accept a consistently wrong one. A separately trained
RGB probability head was overconfident on a held-out wrong-port example and
was rejected. Ten additional successful five-card expert starts did not make
the full-image selected-port locator robust across old grouped holdouts.
See the [complete comparison and contact diagnosis](experiments/2026-10-02-two-family-perception-local-bc.md#october-2-follow-up-new-sfp-starts-body-frame-targets-and-live-contacts).

The insertion actor receives the RGB-estimated pose, images, measured robot
state, and the public card/port task code. True port geometry is used only to
build and score supervised labels. The successful stock teacher runs saved
TCP-frame delta commands in the new collection; the dataset builder converts
each through its causal measured TCP pose, then expresses the resulting
recorded target in the **estimated** port frame. This keeps training and
deployment action representations aligned.

**October 2 gate update:** A selected SFP target is now found with a learned full-view RGB locator followed by 480px and 160px original-resolution crops and three-view triangulation. A separate RGB-trained orientation head has a runtime-matched variant; a five-observation static visual lock holds action until the port estimate agrees across time. On ten development scenes this locked in every scene and gave 0.16 mm lateral p95 in a 722-frame replay, but one isolated wrong-feature three-view agreement proves reprojection consistency alone is not calibrated confidence. The projected-in-frame head still cannot certify target visibility behind the cable. A scored live insertion test remains open. The SC actor's latest corrected same-scene trial reached the port mouth but missed laterally by 1.44 mm and did not insert. [Full evidence](experiments/2026-10-02-two-family-perception-local-bc.md#october-2-continuation-complete-sfp-cascade-and-insertion-diagnosis).

**October 2 implementation update:** A first shared three-view selected-target locator now covers 268 verified SFP and 104 successful SC episodes. It predicts pixel locations for every physically present port and an **in-frame** score, with public task encoding selecting the required port. A native 160×160 SC crop, selected from the locator's own RGB prediction, improves near opening localization. A separate RGB feature head estimates orientation. The in-frame score is not visibility: a four-scene depth audit found many geometrically occluded openings accepted. A small SC visibility pilot is not ready for deployment and there are no SFP true-visibility labels yet. Calibrated pose confidence and the stable handoff window in this design remain unimplemented. Two predicted-port-frame insertion BC actors were trained offline from scratch and transport vision initialization, but neither has demonstrated autonomous full insertion. [Measured outcomes and limitations](experiments/2026-10-02-two-family-perception-local-bc.md).

**October 1 latest transport input:** A bounded 60-second-history actor reached 8 mm position error relative to the teacher's SFP pre-insertion TCP standoff on one non-final development scene, but orientation error was 6.9° and lateral error 8 mm, with no insertion or validated visibility/local-actor handoff. On a matched SC five-card start it failed the transport gate by 158 mm. The new route-head subgoal and handoff predictions also failed their offline gates. This does **not** promote the two-actor switch, perception model, or RL. [Controlled comparison and recordings](experiments/2026-10-01-transport-five-way-comparison.md).

**October 1 supervised sequence:** The complete system has three models: (1) a shared temporal transport actor that can route with the target offscreen, (2) a task-conditioned selected-port pose/visibility/confidence estimator, and (3) a separate port-relative insertion actor. There are two actors, transport and insertion. Their action heads use different frames and are trained separately. The selected-target perception model is trained only after transport passes its stable near-port gate; handoff then requires a calibrated stable observation window, hysteresis, and command continuity. Hierarchical backtracking is excluded from this baseline, and RL is gated on autonomous full-depth Gazebo insertion. [Current sequence and transport audit](experiments/2026-10-01-temporal-three-model-continuation.md).

**October 1 gate update:** The 104 verified SC Low Lateral Bypass successes now have an audited, episode-grouped cleaned derivative. Four supervised transport actors were compared on the same two non-final Gazebo layouts at 20 and 60 mm action caps: **0/16 full insertions and 0/16 stable 10 mm handoffs**. Offline teacher-frame fit did not translate into correct autonomous route progress. A bounded elapsed-time test and action-cap correction failed the gate. That temporal route-stage/subgoal candidate and its no-history control were trained; stage accuracy exceeded 90% in the primary arm, but full-held-out subgoal translation p95 was 41.51 mm against a 20 mm gate, so no temporal live rollout followed. [Temporal result](experiments/2026-10-01-temporal-three-model-continuation.md#completed-targeted-repair-and-full-held-out-check). Target visibility calibration, local pose-conditioned handoff, RL, and final scenes remain gated. [Commands, videos, metrics, and decision](experiments/2026-10-01-sc-llb-cleaned-bc.md#final-bounded-supervised-result-and-step-5-decision).

**September 30 latest:** teacher transport gate accepted. Two same-world Cartesian SC teachers fully inserted, providing v9 with 1,751 transport and 503 local chunks, plus the verified SFP source. The balanced-port BC retry finished 1,500 updates and failed both autonomous development starts: final planned-standoff errors 103 and 603 mm versus 76 and 74 mm for the prior masked actor; neither reached handoff or inserted. A bounded trace/video audit found early wrong-route commands without sustained cable snag. Stop routine feed-forward BC. A checkpoint-provenance audit found that this run reused the full action head from an older mixed-SC-route model and had only two new SC episodes; it was not a clean test of the new Low Lateral Bypass route. Audit clean route data and initialization before deciding on a route-subgoal model. No new training was started for this audit. True visibility, autonomous two-actor insertion and RL remain gated; reserved final scenes stay sealed. [Evidence and exact artifacts](experiments/2026-09-30-transport-to-two-actor-continuation.md).

**September 30 continuation:** transport gate accepted. Four changed-card-layout MoveIt replays reached handoff/standoff; one fully inserted, one failed local insertion and is excluded from expert BC, two were transport-only validation. Four full-success SC teacher episodes now provide 3,001 transport and 1,000 local training chunks, with 1,751 held-out transport chunks. True-visibility recorder checks caught invalid auxiliary depth during tilt; labels are being repaired before training. Autonomous two-actor insertion and RL remain pending. [Execution record](experiments/2026-09-30-transport-to-two-actor-continuation.md).

**September 30:** [Oracle contact diagnosis](experiments/2026-09-30-force-oracle.md) confirms fixed-scene teacher progress with the explicit tilt-before-return route (2/2 instrumented insertions). These trials use privileged teacher geometry and do not validate learned perception, visibility switching, or an autonomous actor. New-start generality and teacher admission remain open.

Status: proposed supervised architecture, September 29, 2026. It has not
passed a full-start Gazebo evaluation. The earlier local Isaac SFP-to-NIC
PoseInsert result does not establish this handoff or SC-to-SC performance.
The [September 29 label-readiness audit](experiments/2026-09-29-perception-label-readiness.json)
found 1,485 native SC frames in 21 scene groups with complete image paths and
separate train/validation scenes. This source labels only whether the selected
opening projects inside each camera. It cannot train identifiable visibility
or calibrated acceptance under occlusion; it also has no SFP rows or explicit
wrong-port negatives. The proposed two-family training gate remains closed.

## What each part does

The normal 10D task code identifies the task family and selected target port.
It contains two family bits, two selected-port bits, five SFP target-card
bits, and a card-valid bit; SC uses the selected-port bits and has no target
NIC card. The number and arrangement of obstacle cards must come from RGB.
This proposal has **two actors total**, each shared across SFP-to-NIC and
SC-to-SC through that task code; it is not one actor per task family.
The **transport actor** sees the three RGB cameras, robot state, and task code
and outputs complete TCP-body targets. It handles the initial and offscreen
route. It must learn SC transport from audited routes that keep the cable
beside the NIC cards, not merely from any teacher episode that scored an
insertion. Short observation/action history can help it remember which side
of the card row it approached from while the target is hidden. The **local
actor** sees the same task and state context plus a
visual estimate of plug-to-selected-port pose. It outputs complete
port-frame targets as in the local AIC-RPDP experiment. Neither actor receives
simulator TF at deployment. Command execution converts the chosen actor's
target using measured robot state and the estimated port frame.

The task-conditioned perception model estimates (a) the selected aperture's
per-camera visibility, (b) its image location and relative 3D pose when
observed, and (c) calibrated probability that its pose error is within a
specified tolerance. Visibility means that enough of the **selected**
aperture can actually be identified in that camera image. It is false when
the opening projects outside the image, is hidden by a card, cable, or robot,
or is too ambiguous to identify. Other visible ports are negative examples,
not substitutes for the selected one. Visibility and accuracy confidence
are different: a target can appear in frame yet still have an inaccurate
pose estimate.

Keep three concepts separate in the proposed handoff. **Visible fraction** is
the approximate fraction of the selected aperture's pixels that can be seen
in one camera; a half-occluded opening can have fraction 0.5. **Identifiable
visibility** is a per-camera yes/no decision: can that image support a
selected-port estimate? A half-occluded opening may still be identifiable.
**Pose confidence** is the calibrated probability that the fused pose error
falls within the specified translation and orientation tolerances. It can be
high for a partly visible opening when its remaining edges and the other
cameras constrain the pose well; it can be low for a fully visible but
ambiguous opening. For three cameras, retain three visibility values (and
optionally three fractions), plus the number of supporting views. An
"any-camera-visible" flag can be true when only one camera sees the port,
but must not be called 100% visibility across cameras. The fused confidence
must account for single-view ambiguity and disagreement between views.

The selected port still has a physical pose when it is offscreen. Simulator
geometry can compute that ground truth during label generation, but a
single offscreen image cannot determine a precise pose from visible evidence.
For a per-camera model, an offscreen/occluded frame gets visibility label
zero and its precise pose/keypoint loss is **masked**. It should not be given
an arbitrary zero-pose target. When at least one view sees the target, fuse
the available view estimates and supervise the 3D pose. All-view-absent
frames teach the transport actor and train the handoff to stay in transport;
they do not count as successful precise pose predictions. A separate coarse
offscreen bearing prediction could be studied, but must never satisfy the
local-actor gate without calibrated visual evidence.

For camera $c$, let $v_c=1$ only when enough of the selected aperture is
visible and identifiable. A simple supervised objective is

$$
\mathcal L = \sum_c \operatorname{BCE}(\hat v_c,v_c)
 + \mathbf 1\!\left[\sum_c v_c>0\right]\mathcal L_{\mathrm{pose}}
 + \mathcal L_{\mathrm{confidence}}.
$$

The mask prevents an invisible target's known simulator pose from being
treated as information that an image-only model should somehow recover.

## Label generation and current gap

Generate labels from synchronized Gazebo/Isaac RGB, camera intrinsics,
camera-to-TCP calibration, selected-port TF, plug/TCP TF, and scene geometry.
Use TF only to make training labels. Project the selected aperture into each
native image, then use depth, instance masks, or a rendered visibility test
to decide whether the aperture pixels are actually unoccluded. Keep a
separate label for projection inside the image, for true visibility, and for
pose. The current `build_sc_native_pose_labels.py` records projected
in-frame indicators and task one-hot bits; projection alone does **not**
detect occlusion. The current `train_sc_native_locator_v2.py` is a
task-conditioned keypoint regressor, not a calibrated visibility/uncertainty
model. Reuse its images and calibration, then build the missing labels.

Include both SC target ports in the same training batches and deliberately
show the wrong port as a hard negative. Also sample SFP-to-NIC target-card
identities, different card counts and rail poses, initial yaw-edge starts,
near-port views, cable occlusions, and robot self-occlusions. Failures can
contribute perception labels even if their actions are excluded from BC.
Additional randomized perception scenes do not require a successful expert
insertion; simulator labels and synchronized images are enough.
Split by complete episode and scene/reset configuration, never random
adjacent frames. Keep the reserved final scenes sealed.

Train a selected-target heatmap/visibility head on full images; use its
observation-only localization to crop the original high-resolution image
and refine port/plug keypoints and orientation. Fuse calibrated multi-view
estimates in the robot frame. Train an uncertainty head or small ensemble
to predict the chance that translation and orientation errors meet the
handoff tolerances. Calibrate that chance on held-out scene groups and
measure false acceptance, especially on frames with other visible ports.
Fit the error/confidence target from out-of-fold pose predictions so the
confidence head sees realistic mistakes rather than only its own training
fit. Report coverage at each confidence threshold together with the actual
fraction of accepted estimates inside the lateral corridor.
The camera projection label must not choose a crop during autonomous tests.

## Handoff rule to test

At each 200 ms decision, keep the last five estimates. A proposed starting
gate requires at least four of the five to identify the selected opening,
meet calibrated pose-confidence thresholds, agree spatially, and place the
plug in a validated near-port region. Require two supporting views when
possible; a single view needs a stricter calibrated threshold. Thresholds
come from held-out false-accept/error curves, not from arbitrary network
scores. Enter the local actor only after that gate passes. Leave it if
confidence falls for two or three decisions, measured pose jumps, or force
indicates unsafe contact. Add hysteresis and transform the first local
target so the commanded TCP trajectory is continuous at the switch.
The gripper may temporarily hide the opening during insertion. Carry the
last fused pose through a short occlusion only while measured TCP motion and
its growing uncertainty remain within validated bounds; do not instantly
switch actors on one hidden frame or extrapolate indefinitely.

This five-decision rule is a **proposal**, not a measured optimum. Report
false switches, missed handoffs, local pose error by phase, and full-start
insertion for alternative window lengths on development scenes. If the
selected opening remains offscreen, transport should seek a safe viewpoint;
the task code by itself tells the network which port is wanted, not where
an invisible port is.

## Initial data budget and pass condition

Start by relabeling existing synchronized SC/SFP episodes, including failed
ones for perception. A planning budget for a decisive SC test is roughly
**100–200 distinct scene/reset configurations across both selected SC
ports**, with about **5,000–15,000 temporally thinned timesteps** (three
camera images each). This is a starting estimate, not a claim that a fixed
image count guarantees 0.5 mm accuracy. Several hundred adjacent images
from one episode provide far less independent evidence than new card,
camera, yaw, and occlusion configurations. Audit existing coverage before
collecting any deficit. Add data only for held-out failure strata and
predeclare the stopping rule.

Before local-actor promotion, require low false-accept rate when the target
is offscreen or confused with another port, accurate held-out near-port
pose relative to the 0.5 mm lateral insertion corridor, stable handoff,
autonomous full-start insertion on new development scenes, acceptable
force/contact, and complete inference p95 below 300 ms. RL remains disabled
until this supervised gate passes.

Related primary references: [Multi-View Keypoints for Reliable 6D Object Pose
Estimation](https://arxiv.org/abs/2303.16833) uses multi-view keypoint
probabilities and alignment confidence to reject false poses;
[Ambiguity-Aware Multi-Object Pose Optimization](https://arxiv.org/abs/2211.00960)
addresses pose ambiguity under occlusion. They motivate confidence checks;
their reported task accuracy is not an AIC insertion result.


### September 30 continuation: deployment input audit

The transport gate is accepted; full insertion remains a separate gate. SC dataset v5 fixes canonical TCP quaternions and the raw ROS joint-prefix order: all 4,752 transport rows exactly match the live state assembler. Earlier v3 BC is retained as an invalid-input diagnostic. Corrected rootless GPU7 BC is bounded at 1,500 updates; auxiliary depth visibility remains rejected. See [the continuation record](experiments/2026-09-30-transport-to-two-actor-continuation.md). Autonomous handoff and RL have not passed.

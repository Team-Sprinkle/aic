# Target-conditioned perception and two-actor handoff

Status: proposed supervised architecture, September 26, 2026. It has not
passed a full-start Gazebo evaluation. The earlier local Isaac SFP-to-NIC
PoseInsert result does not establish this handoff or SC-to-SC performance.

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

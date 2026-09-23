# Current experiment status

## SC-to-SC mechanics and routing bring-up (September 23)

The approved multi-card SC continuation reached its simulator-mechanics gate.
The audit corrected the SC port mount, reversed cable endpoint topology and
fixed-joint transforms, reset ordering, multi-card scene support, and the SC
contact target. With normal collisions and no intervening cards, a scripted
14 mm retreat plus positive 4 mm lateral retry reached 0.099 mm final lateral
and 0.008 mm axial error. The matched negative retry entered the corridor only
briefly and diverged, confirming that recovery direction matters.

The apparent five-card cable snag was instead dominated by the gripper housing
striking a NIC card. Removing plug and cable collisions did not change the
failure; direct contact sensing measured about 128 N at the gripper/card
contact. The source reversed Gazebo grasp was not physically reachable in the
current Isaac base/board arrangement. Therefore no SC BC or RL was trained and
the failed rollout is not labeled as cable-snag data.

A follow-up source audit found that Gazebo intentionally removes two endpoint
collision groups near the palm and shortens/shifts the first rope collider. The
Isaac builder now reproduces that contract and uses the equivalent wrapped
wrist solution. This did not fix the full scene: a hold still produced a
36.98 kN peak. Removing the board, port, and self collision made the motion
low-force, but physical interpolation stopped 44.27 mm from the requested tip
pose. The remaining fault is narrower but unresolved: grasp transform, scene
placement, or articulation actuation. The SC learning gate remains closed.
The corrected asset is stable at its native initial posture, which localizes
the failure to near-port placement/reset rather than an always-unstable cable
topology. Gazebo welds to `ati/tool_link`; the prepared USD welds to the right
gripper finger. That frame contract and the source board/port placement are the
next items to reconcile.

A declared gripper-collision-disabled proxy tested routing while retaining
plug, cable, card, board, and port collisions. A direct high approach held the
final component gate in 1/3 seeds; a privileged 100 mm route around the card
edge held it in 3/3. This supports hierarchical routed transport as a future
candidate, but it is neither autonomous nor valid policy data. The active gate
is a reachable collision-faithful SC grasp/scene contract, followed by scripted
card-count 0--5 validation. See the
[SC mechanics record](experiments/2026-09-23-sc-mechanics-and-routing.md) and
[machine summary](../outputs/experiments/2026-09-23_sc_cable_bringup/summary.json).

## Probabilistic SERL recovery continuation (September 23)

The deterministic local RPDP controller was converted into a four-component
mixture over complete four-waypoint port-frame trajectories. It is a full
policy rather than a residual correction. Frozen perception plus the actor has
2,586,195 parameters, and complete live inference measured
20.43/22.67/23.78 ms p50/p95/p99. Measured-path backtracking, force plus stall
detection, lateral post-clearance exploration, replay ownership masks, balanced
prior/online replay, twin critics, and mixture SAC updates are implemented.

Frozen stochastic collection produced 3/7 local 8 mm seating successes. The
first backtracking implementation was invalid because force alone triggered on
normal insertion and abort persisted. After repair, no blockage trigger fired
in five episodes; its failures were lateral drift/timeouts, so backtracking had
no opportunity to help. A 12-episode prior contains seven successes and five
failures. A ten-episode critic warm-up ranked its one held-out success above its
one held-out failure, but this two-episode check is weak.

On eight new episode-grouped development starts, frozen BC scored **3/8**.
Initial correlated and backtracking SAC actors both scored **0/8** because the
actor collapsed deterministic probability onto unvalidated alternative modes.
Adding full-mixture behavior regularization stopped that collapse, but the
requested lower actor learning rate was silently overwritten when optimizer
state was restored and the result remained 0/8. Optimizer restoration now
retains momentum while reapplying explicit learning rates. The final tighter
trust-region run scored the same **3/8 on the same three episodes** as frozen
BC. It preserved the warm start but did not improve it.

A separate three-camera recording rerun on the same eight configurations scored
**1/8 BC, 1/8 tight-trust RL, and 3/8 tight-trust RL plus backtracking**. All 72
per-episode clips are on the [video review page](../outputs/experiments/2026-09-22_serl_recovery/videos/index.html).
Because BC and plain RL differed from their earlier 3/8 metrics-only outcomes,
the recording rerun is preserved as separate evidence and does not establish an
RL or backtracking improvement.

Do not promote this online checkpoint as better than BC and do not open the
reserved final split. Before another actor update, collect substantially more
episode-grouped recovery and blockage outcomes and validate critic ranking on
more than two held-out episodes. The selected checkpoint is preserved because
it verifies a mechanically safe continuation path. See the
[experiment record](experiments/2026-09-22-serl-mixture-recovery.md),
[actor explanation](SERL_PROBABILISTIC_ACTOR.md), and
[recovery strategy](SERL_RECOVERY_STRATEGY.md).

### Video diagnosis and current boundary

The three-camera rerun was reviewed configuration by configuration using the
recorded axial depth, lateral offset, orientation, and cable motion. The dominant
failure is lateral divergence: several actors move from a nearly centered reset
to 6--18 mm off-axis, sometimes crossing the entrance plane beside the port.
One configuration stalls at the port lip with roughly 0.04--0.05 rad angular
error. No episode provides conclusive evidence of a cable snag. Cable motion is
a possible contributor in configurations 12 and 14.

The recovery controller does more than retreat. It retraces measured TCP
positions in 0.25 mm commands until at least 1 mm of clearance and low force,
then constrains 12 control steps toward near-lateral motion. One millimeter is
twice the 0.5 mm local success corridor, so this can release a small lip contact.
The recording did not retain per-step recovery-mode traces, however, and the
visible backoff may also be a learned action. The two additional successes do
not prove that hard-coded recovery caused the improvement. See the
[committed failure analysis](experiments/2026-09-23-serl-video-failure-analysis.md).

### Approved SC continuation now in execution

The active work targets SC-to-SC with multiple intervening cards, where
the cable can require a larger retreat and a routed detour rather than local
realignment. Current selected pose/RPDP/SERL evidence is SFP-to-NIC only. The
ordered plan first audits Isaac SC support, defines cable snag causally, records
a bounded full-episode discovery set with compact force/motion/recovery
telemetry, and creates reproducible short incident resets. It then gates SC pose
estimation, port-relative BC, fixed multi-scale recovery, a learned recovery
option, offline critic validation, short-incident online SERL, and finally
full-episode evaluation. See [SC cable-snag recovery plan](SC_CABLE_SNAG_RECOVERY_PLAN.md).

## RPDP and DPPO continuation (September 22)

**Scope correction:** the RPDP evaluations below use
`scene.target.seated_depth_m: 0.008` and privileged near-opening resets. The
physical SFP entrance is about 45.8 mm from the seated port reference and the
cage is about 48.72 mm deep. Therefore these successes are **local 8 mm seating
events**, not official full-depth or full-episode insertions. The controller has
not yet been evaluated from the normal initial distribution or with the
official Gazebo insertion event. Earlier wording saying simply “strict
insertion” was too broad.

The selected supervised controller is now a PoseInsert-inspired, deterministic
AIC-RPDP policy. It predicts the connector pose after each of four recorded
50 ms teacher target commands, always relative to the fixed port opening. A
verified rigid transform converts those four waypoints back to four
TCP-body-frame commands. This is a complete target trajectory, not an additive
correction to ACT and not a base-frame absolute action.

The follow-up audit fixed four material BC problems: recovery data had used the
75% blended executed action instead of the recorded teacher target; four
different commands had been compressed into one 200 ms endpoint; a numerically
oversized checkpoint-improvement threshold kept undertrained update-200
models; and live insertion commands were sometimes clipped from 0.5 to 0.2 mm.
The original 322-row diagnostic also lacked insertion, but the current data
does not: the 823-row successful fit split contains 424 insertion decisions.
A DAgger extension keeps safe teacher queries from failed blended rollouts
while rejecting stale reset images and force at or above 5 N, producing 980 fit
decisions.

The repaired diffusion arm improved to 2/6. A matched deterministic trajectory
regression arm then achieved **6/6 autonomous local seating events** on the fixed
development starts and **8/8** on an additional set unused by RPDP fitting or
selection. Across all 14 episodes, final lateral error was 0.024--0.402 mm,
orientation error was 0.002--0.066 degrees, post-reset maximum force was
1.69--3.69 N, and complete p95 inference was 19.86 ms. Evaluation used no
guide, guard, exploration, privileged geometry, or RL.

A later frozen five-start video check achieved **4/5 local seating events** in the same Isaac
SFP-to-NIC scene and held-out cable-template family. The failure timed out at
5.987 mm depth despite 0.329 mm lateral and 0.032 degree orientation error;
the four successes reached 7.562--7.640 mm. Complete inference was 20.12 ms
p95. The cumulative post-repair development observation is therefore 18/19 and
shows that slow or stalled axial completion remains possible. Three per-episode
camera videos, a review page, and exact metrics are linked from the
[RPDP experiment record](experiments/2026-09-22-rpdp-dppo.md#post-selection-five-rollout-video-check).

The selected supervised arm has 2,004,329 trainable policy parameters plus 536,631 frozen
perception parameters, or 2,540,960 total. It passes the supervised development
gate. The older DPPO round remains a valid but unsuccessful 1/6 result and was
not continued. The four reserved final configurations remain sealed; no new
RL from the original RPDP/DPPO stage ran. The later probabilistic SERL
continuation is reported above. No world-model, SEER, or Gazebo-adaptation
training ran. See the
[RPDP/DPPO and BC repair record](experiments/2026-09-22-rpdp-dppo.md).

The staged
[perception, supervised control, and model-free RL experiment](experiments/2026-09-20-perception-supervised-rl.md)
has now completed its bounded supervised-controller diagnostic. Corrected
plug/opening supervision and oracle triangulation passed earlier, and the later
natural-cable observation-only estimator was frozen despite its remaining p95
tail. A matched action-only versus pose-conditioned GRU comparison then reached
**0/8 strict insertions in both arms** on new development starts. The policy
learned to make almost the same commands when pose inputs were zeroed or
shuffled. The predeclared stop rule therefore keeps RL, SEER adaptation, and
predictive world-model control parked.

The approved explicit-correction continuation is also complete. Force-safe
collection retained 322 fit rows from 25 episodes with all eight lateral
directions paired, plus 124 calibration rows from seven unseen-template
episodes on two paired axes. A 66,596-parameter correction was forced to pass
through the predicted plug-to-opening translation on top of the frozen
546,686-parameter nominal GRU. On calibration it reduced translation command
MAE from 0.04786 to 0.04638 mm, only 3.10%, and produced a 0.02909 mm median
correction; the fixed gates required 5% and 0.030 mm. A second checkpoint
selected directly on translation MAE still failed, shuffled pose slightly beat
correct pose on calibration, and zero pose beat correct pose on the untouched
offline development split. The six live starts were therefore not opened. No
autonomous rollout, final evaluation, or RL ran. See the
[explicit-correction record](experiments/2026-09-22-explicit-pose-correction.md).

The latest explicit-opening continuation used 20 train configurations, six new
development configurations, original-resolution 160x160 crops, six landmark
heatmaps, and per-observation three-camera calibration. Calibration rejected
the landmark and temporal variants and selected the coarse RGB locator. Across
240 near-port development decisions it reached **1.085/1.740 mm median/p95
lateral error**, 91.5% correction-direction accuracy, **1.204/2.928 mm
axial error**, and 0.211/0.363 degree orientation error. Complete inference was
1.406/1.982/58.099 ms p50/p95/p99. A world-coordinate temporal filter improved
lateral error to 0.862/1.505 mm only as a post hoc, promotion-ineligible
diagnostic. The accuracy gate remains closed.

A subsequent calibration-only ImageNet MobileNetV3-small landmark model was
close enough to earn six further fresh development starts. It is only 233,662
parameters and uses a 160x160 native crop. On 173 near-port decisions it reached
the current best observation-only result: **0.278/0.670 mm median/p95 lateral
error**, 94.1% direction accuracy, **0.664/1.363 mm axial error**, and
0.215/0.506 degree orientation error. Complete inference was 5.450/5.949/71.825
ms p50/p95/p99. It still fails the fixed 0.25/0.5 mm gate, so policy and RL
remain stopped.

A post hoc causal world-space opening filter reached 0.250/0.616 mm lateral
error and 97.8% direction accuracy, while axial error regressed to
2.085/4.407 mm. It was tested after that development split was opened and is
diagnostic only. Since its p95 still failed, no additional split was collected.
Heatmap-confidence camera rejection also failed to improve calibration, which
selected all three views; no further development collection was justified.

The approved temporal/multiview continuation is now complete. A replay audit
found strong cable occlusion in the sparse center-camera mask sample but clear
side views, and no independent cable-shape reset variable. It also fixed a
null episode-index bug so causal history resets at actual simulator resets.
Learned current-frame multiview and six-step temporal heads passed calibration,
then were evaluated once on six new reset positions. The current-frame head was
best on fresh data at **0.307/0.573 mm** near-port lateral median/p95, 99.4%
direction accuracy, and **0.304/0.796 mm** axial median/p95. The temporal head
reached 0.289/0.667 mm and did not reproduce its calibration p95. Full temporal
perception plus the existing trunk measured 6.82 ms p95. Accuracy still fails
the 0.25/0.5 mm gate, so no policy, final split, or RL was opened. The next data
collection must independently vary cable shape/occlusion and retain visibility
or depth supervision before another temporal attempt. The articulation has 40
cable joints in addition to the six arm joints, while the current reset event
selects only the arm. A safe follow-up needs seeded cable-shape templates,
physics settling, collision/force rejection, and a compensating arm solve that
restores the requested plug pose; unconstrained cable-joint noise would confound
the experiment.

The cable-shape/visibility continuation and its force-safe follow-up are
complete. Exact 40-joint resets initially caused a 35 N transient and plug
drift. A closed-loop collection controller now waits for settling, rejects
force above 5 N, restores the measured plug pose, and releases to the fixed
CheatCode teacher only after two safe observations. This is privileged data
generation and is never available to the autonomous actor.

The full natural-trajectory run saved 1,500 causal decisions across 50
episodes. Thirty-five episodes handed off successfully; filtering retained 460
safe near-port observations from all five included cable templates. The frozen
complete-template split used 205 fit, 79 calibration, and 175 development rows.
The calibration-selected current-frame model reached **0.322/0.626 mm**
near-port lateral median/p95 and 100% direction accuracy on 153 samples from ten
held-out episodes. A fixed temporal variant reached **0.247/0.617 mm** but still
failed the p95 requirement. Combining static and natural data was worse at
0.331/0.753 mm. Complete perception plus the existing trunk measured 7.18 ms
p95. The 0.25/0.5 mm gate still fails, so no policy, RL, or reserved-final
evaluation was opened. The subsequent frozen-model audit found that the right
camera was strongest and the center camera weakest, but every two-camera subset
had worse lateral p95 than the three-camera model. Exact labelled pixels gave a
sub-0.0011 mm two-camera triangulation ceiling, confirming consistent camera
geometry. A fixed per-camera bias correction worsened p95 to about 0.79 mm.
The tail was concentrated in one cable episode and small-motion rows; clear
views were worse than occluded bins, so occlusion is confounded with trajectory
rather than established as the cause. More independent safe cable layouts and
camera-specific landmark supervision are required before another frozen
representation test.
See the [execution record](experiments/2026-09-22-cable-visibility-perception.md).

A label-free qualitative audit now overlays only model predictions and their
three-camera reprojections; it never reads simulator plug/port pose labels from
the reviewed episodes. The predicted opening aligns visually with the SFP cage,
including cable-obstructed examples, and cross-view reprojection RMS is
0.807/1.309 pixels median/p95. The predicted stationary opening nevertheless
jitters by 0.369/1.316 mm and its normal by 2.98/7.97 degrees. This confirms the
model is attending to the intended region while independently motivating causal
temporal fusion; it is not a substitute for the labelled accuracy metrics.

The frozen-pose supervised control diagnostic is complete. Two matched
546,686-parameter policies used six causal decisions and emitted four 6D
TCP-body-frame delta commands. One received the frozen visual features, state,
measured state change, force, previous action, and visibility; the other also
received predicted relative translation, uncertainty, and observation-derived
phase. On offline held-out episodes, action-only was marginally better
(0.0367/0.0894 mm translation MAE/p95 versus 0.0369/0.0908 mm). On eight new
autonomous starts each, both scored **0/8 strict insertions**. Best lateral
p50/p95 was 1.479/1.678 mm for action-only and 1.417/1.871 mm for
pose-conditioned; terminal medians grew to 6.382 and 6.284 mm. A counterfactual
audit showed that zeroing pose changed development commands by just
0.0095/0.0264 mm p50/p95, so the generic BC head largely ignored the added
signal. Full inference was 7.135/7.388/8.249 ms p50/p95/p99. The final split
stayed sealed and no RL ran. See the
[execution record](experiments/2026-09-22-pose-conditioned-gru-policy.md).

The selected full-data world policy remains the strongest preserved Gazebo
controller: 2/20 full insertions, 9 partials, mean official score 45.14. Its
learned dynamics is worse than persistence, so predictive imagination remains
disabled.

The September 20 Isaac continuation resolved the earlier joint/reset diagnosis.
The expert joint convention is usable; the original supported YAML had placed
the port about 552 mm from the expert pose. A second silent issue then showed
that the first generated scene variations retained a joint-state reset and
ignored their requested Cartesian starts. The corrected reset uses 6D damped
IK and reproduces requested 4--10 mm axial and approximately 0.7 mm lateral
starts within the declared tolerances.

On those corrected starts, the privileged adaptive guide achieved 9/9 strict
insertions in its first pass and 7/8 in its second. The world-policy replay path
stores exact 24D four-command transitions, 384D frozen causal features, guide
labels, and commands actually executed. The trainable 333,024-parameter head
receives nonzero gradients; its frozen perception/world trunk has 22,236,320
parameters.

The autonomous gate still fails. The round-one aggregate, round-two aggregate,
and varied-only heads each achieved **0/4 strict insertions** on heldout IK
starts. The varied-only head was the most consistent: all four trajectories
reached within 0.10 mm before the opening plane, but lateral offset at closest
approach remained 3.25--3.89 mm against the 0.5 mm success threshold, and force
reached the configured 35 N clip. The four reserved final starts remain
untouched. No Gazebo transfer or actor-Q update was started because the
supervised anchor does not insert. The diagnostic checkpoints, videos, and
machine-readable summaries are under
[`outputs/experiments/2026-09-20_isaac_world_rl/`](../outputs/experiments/2026-09-20_isaac_world_rl/).
See the [execution record](experiments/2026-09-20-isaac-world-policy-rl.md).

The privileged guide is an **Isaac CheatCode policy**, not a new Gazebo
implementation. It uses the same ground-truth plug-to-port rigid-transform idea
as the existing Gazebo CheatCode, but is currently a local near-port feedback
servo rather than the Gazebo policy's full approach, settle, minimum-jerk
insertion, and completion-event sequence. The 552 mm finding is a spatial scene
placement mismatch, not a purely lateral port-frame error or proof that every
cross-simulator asset transform agrees.

DAgger fractions apply to every command rather than to a fraction of timesteps:
50% means `0.5 * model + 0.5 * guide`, while 25% means
`0.75 * model + 0.25 * guide`. Replay keeps the unblended Isaac CheatCode chunk
as the BC label and the blended command as the action that produced the next
state. The detailed record also specifies the Isaac shaping reward and makes
clear that it is not an official Gazebo score.

The causal replay audit is complete. Earlier replay joins were unsafe at every
terminal because Isaac had already reset before returning `next_obs`; new
collection retains a separate pre-reset terminal snapshot and all requested
causal decision fields. The frozen 384D pose probe was evaluated on four unseen
episode-grouped configurations. Near the port it reached 0.730 mm median and
2.533 mm p95 lateral vector error with 93.5% correction-direction accuracy.
This fails the predeclared 0.25/0.5 mm resolution gate, so no conditioned policy
was trained. The bounded native 576×512 crop ablation is also complete. Its
learned RGB locator uses simulator masks only as training labels and selects
evaluation crops from RGB. On the matched 225-decision heldout set, near-port
lateral error improved only from 0.962/2.630 mm median/p95 to 0.851/2.154 mm,
while correction-direction accuracy fell from 84.8% to 80.4%. It therefore
failed the material-improvement and absolute resolution gates. Complete live
inference measured 2.96/3.20/182.50 ms p50/p95/p99, satisfying the 300 ms p95
limit. The policy comparison was correctly skipped. The reserved final starts,
actor-Q, Gazebo transfer, reward-model training, and imagination remain closed.
See the
[full execution record](experiments/2026-09-20-isaac-world-policy-rl.md#causal-replay-audit-and-frozen-feature-probe).
The record's [plain-language interpretation](experiments/2026-09-20-isaac-world-policy-rl.md#plain-language-interpretation-of-the-crop-experiment)
explains the crop stages, median/p95 errors, label limitation, recent robotics
alternatives, and why pose prediction alone does not perform insertion.

The crop failure was subsequently traced more precisely. The old labels were the
SFP module centroid and nearest NIC-card mask pixel, not the plug tip and port
opening. An oracle-coordinate ceiling using those labels still failed at 1.132
mm near-port lateral median and 3.141 mm p95, so further fitting to them was
stopped. The Isaac camera projection path was repaired and visually checked on
all three views. The replacement 1,057-decision train and 225-decision held-out
collections are complete. Exact-keypoint triangulation reached 0.124 mm median
and 0.235 mm p95 near-port lateral error, proving the geometric ceiling is
sufficient. The deployable calibration-selected RGB locator reached only 2.380
mm median and 6.947 mm p95, with 82.6% correction-direction accuracy. It failed
the 0.25/0.5 mm and 90% gate. Complete inference p95 was 3.311 ms. Accordingly,
that stage did not train a pose-conditioned policy or RL. A later frozen-pose
diagnostic is reported above; the four final starts remain sealed. See the
[decision record](experiments/2026-09-20-perception-supervised-rl.md#executed-result).

## TCP-delta commands and tracking: current decision

**Train on the clean expert's commanded TCP-frame delta pose**, relative to the
TCP pose in the same observation. This delta is a desired *controller pose
reference*, not the displacement the TCP must achieve before the next image.
For corrective recordings, use `teacher_target_pose` for policy supervision and
`executed_target_pose` for action-conditioned dynamics; they can differ. Do not
replace the command label with the difference between two measured TCP poses.
The latter includes controller lag and contact and is useful for dynamics or
path evaluation, not as a drop-in `MotionUpdate` target. See
[`CollectCorrectiveCheatCode.py`](../aic_example_policies/aic_example_policies/ros/CollectCorrectiveCheatCode.py)
and [dataset eligibility](DATASETS.md).

An audit of **23,252 consecutive 50 ms observation pairs** in the 74 verified
one-NIC, no-SC aligned/corrective SFP episodes found these median translation
magnitudes: **9.83 mm** from the observed TCP to the *executed command target*,
**0.64 mm** actual TCP displacement during the next 50 ms, and **9.35 mm**
remaining from the next measured TCP pose to that same target. These are
separate medians, so they need not subtract exactly. In the last three recorded
seconds, the corresponding medians were 5.35 / 0.04 / 5.36 mm. The clean
teacher delta was also 9.83 mm median across all 32,183 native observations.
This is a target-tracking gap, **not** a 9.35 mm expert-label or model-prediction
error. The consecutive-pair audit used the saved `states.npy`,
`executed_physical.npy`, `teacher_physical.npy`, and each episode's native times
and command indices under the [Dreamer pilot data](../outputs/experiments/2026-09-18_dreamer60_pilot/artifacts/data), retaining only 50 ms intervals with
consecutive command indices. Position-delta magnitude is unchanged when a TCP
delta is composed into an absolute `base_link` target.

These episodes all have official full-insertion Tier 3 scores. They represent a
restricted aligned scene without SC distractors, so they do not establish
general performance in cluttered settings. The wider historical CheatCode SFP
collection has **140/140 verified full insertions**; its action labels are much
cleaner than the successful agent/VLM recordings with missing Cartesian labels.
The CheatCode policy itself issues a smoothly changing target about every
50 ms, including slow insertion and a settling phase. Its success despite the
gap above argues **against** waiting for every commanded pose to be reached or
shrinking the learned delta to the next observed displacement.

The existing [AIC controller](aic_controller.md) already runs at **500 Hz**,
interpolates position and orientation references, and applies Cartesian
impedance control. Its tracking-error reset is a coarse stuck-target safeguard
(`min_translation_error: 0.2 m`, timeout 2 s in
[`aic_ros2_controllers.yaml`](../aic_bringup/config/aic_ros2_controllers.yaml));
it is not a millimetre-level waypoint gate. The synchronized policy
[`Observation`](policy.md) arrives at up to **20 Hz**. In `insert_cable()`, use a
fresh observation as the inference event, publish the bounded Cartesian
`MODE_POSITION` target at the 20 Hz cadence, and let the controller run its own
fast loop. A chunked model may infer once per four commands, but every delta
must retain its trained observation-reference convention. For an
observation-relative delta, compose it with that observation's TCP pose and
publish the resulting absolute target in `base_link`; sending the delta in
`gripper/tcp` lets the controller apply it from the *later* TCP pose at receipt.
The runtime already has this observation-relative transport path in
[`RunACTTorchScript.py`](../aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py).

**Do not add general tracking-aware action-chunk pacing by default.** A second
fast loop would duplicate the controller's interpolation, and a reach-before-
advance rule would distort these successful expert commands, especially in
contact. Keep the nominal command cadence and log measured TCP, reference
pose, force, command, and simulated time. Consider a guarded
alignment-to-insertion transition only if the learned policy shows excess
off-path motion or harmful contact relative to experts, using observable
signals rather than privileged geometry. Compare the same frozen policy with
and without that guard on matched simulator scenes before adopting it. The
expert tracking-gap statistics alone do not demonstrate that extra pacing
will improve insertion.

## September 18 work

At 18:14 UTC the action contract was corrected: the verified expert label is
a **TCP-frame delta relative to the recorded observation**, rather than an
absolute base-link target for the model to predict. The completed all-data ACT
and fresh ACT60 results below used absolute predictions and remain historical
baselines. The new strict 60/14 ACT run completed 6,000 updates at 18:55 UTC
on the 32,183 native observation-command pairs, predicting one TCP delta per
observation. Its fixed held-out rule selected update 6,000 (2.96 mm first
command and 2.63 mm final-three-second translation error); four fresh paired
development scenes ran from 18:56 to 19:09 UTC: all four valid, **0/4 full
insertions**, mean official score **27.73**. Three ended 5 cm from the port;
one drifted to 33 cm. These imitation errors are not live insertion scores.
The corrected world dynamics completed 3,191 updates and stopped early after
its fixed held-out prediction gates failed; the selected update-1,000 model's
one-step TCP error was 10.33 mm versus 1.85 mm for persistence (14 episodes).
Here, one step means **200 ms and four actually executed 20 Hz commands**.
Starting from a recorded observation, the dynamics model predicts the TCP's
*measured position after those commands*. The 10.33 mm is the mean distance
between that predicted future position and the recorded future position in
`base_link`; it is not the error between a controller command target and the
robot. The 1.85 mm persistence baseline instead predicts that the future TCP
will remain at its initial measured position, then compares that unchanged
position with the same recorded future position. It works well over this short
interval because the TCP usually moves little, even when the commanded target
is farther away. See the [held-out dynamics evaluator](../outputs/experiments/2026-09-18_dreamer60_pilot/world_tcp_delta_final_runtime_archive/sources/dreamer_source/dreamer4/aic/evaluate_world.py).
A fresh supervised policy using corrected delta labels ran from 18:46 to
19:20 UTC on GPUs 2–3, stopping at update 4,509 after a documented held-out
plateau; fixed selection chose saved update 4,000 (2.16 mm combined
first-command error). The first live world startup found a camera-size
mismatch before any command. After matching the collector's resize to
288×256, trained raw-camera inference measured **32.76 ms p95** over 1,000
calls and the four-command callback loop **33.79 ms p95** in isolation.
The same-GPU live diagnostic measured **464.67 ms p95** across 168 published
commands, above the 300 ms requirement. Separating renderer and policy GPUs
alone did not fix this. A private inference worker preserved byte-identical
commands on verified native samples and passed a full live-scene diagnostic.
The fresh world development set was **4/4 valid, 1/4 full insertions**, mean
official score **52.14**, with pooled **81.00 ms p95** live decision latency.
The frozen paired final assessment completed all **20/20 eligible scenes per
policy**: ACT **0/20 full insertions**, one official partial, mean total
**22.69**; world **0/20 full insertions**, two official partials, mean total
**32.32**. The world policy's 8,908 live decisions measured **77.01 ms p95**,
under the 300 ms requirement. A failed ACT startup before trial 17's scored
rollout was preserved, then that same scene completed as an unchanged-model
retry. The previous absolute-action dynamics and BC runs are superseded.
The [follow-up](experiments/2026-09-18-world-followup.md) found the world
policy closer to the opening in more scenes but with persistent lateral error
and a large requested/measured TCP gap. Across all strict held-out 200 ms
dynamics windows, future measured TCP error was **12.06 mm** versus **2.29 mm**
for persistence; near the actual opening it was **13.94 mm** versus **0.57 mm**.
The 289-episode visual audit and matched tokenizer comparison improved SC
gross reconstruction with expanded data, but fine connector/port features
remained blurred. Reward and imagination training remain disabled. See the
[delta correction record](experiments/2026-09-18-tcp-delta-correction.md) and
[contract audit](../outputs/experiments/2026-09-18_dreamer60_pilot/act60_delta_contract_audit.json).

The [bounded supervised initialization comparison](experiments/2026-09-18-world-supervised-init-ablation.md)
trained the same six-view control architecture for 2,500 BC updates in each
arm and evaluated four fresh paired scenes disjoint from all 289 verified
episodes and the final set. Held-out first-command error was 2.285 mm with
selected world weights versus 2.578 mm from a fresh world trunk. Live mean
official score was 32.85 versus 26.24, with **zero full insertions in either
arm** and one partial for the selected-world arm. Both met the 300 ms limit:
pooled p95 command latency was 76.26/79.69 ms. The fresh arm's world weights
were a separate random draw rather than the exact pretraining ancestor, so
one seed cannot establish a causal pretraining advantage. The reward and
imagination gates remain closed.

The [September 19 full verified-data run](experiments/2026-09-19-full-world-training.md)
is complete. A fresh six-view tokenizer optimized all 250 canonical training
episodes, held out the 39 scene-disjoint validation episodes, and stopped at
78,000 updates under the declared validation plateau rule; update 75,000 was
selected. On 156 held-out frames it reduced whole-image/contact-field MSE by
91.25%/93.29% and contact edge L1 by 53.09% relative to the earlier bounded
reference. Corrected dynamics also converged by rule, but future TCP error was
11.46/19.17/20.12 mm at 200/400/600 ms versus 2.07/4.42/7.24 mm for persistence.
Every dynamics gate failed, so reward and imagination remain disabled.

The supervised full-data world policy selected update 1,000 and then completed
all **20/20 eligible sealed final scenes: 2 full insertions, 9 official partials,
mean total 45.14**. It beat the corrected60 ACT score on 13/20 paired scenes;
ACT remained 0 full / 1 partial with mean 22.69. Across 8,895 live decisions,
the new policy measured **80.77 ms p95**, 98.29 ms p99, and 146.74 ms maximum,
with no 300 ms misses. This is the strongest learned result in that paired set,
but 2/20 is not reliable and the complete-pipeline comparison does not isolate
which training change caused it. Next, run the full-data same-architecture
supervised comparison from the exact preserved random ancestor versus selected
world weights on new development scenes and multiple seeds. Keep the sealed
final scenes out of selection.

For size context, ACT has **16,354,566 inference parameters** (65,418,264
FP32 bytes), while the selected world policy has **22,569,344 acting
parameters** (90,277,376 FP32 bytes) and 27,130,942 parameters in the complete
training model. Thus the acting world policy is 1.38 times ACT's size. On their
matched corrected 60/14 data and frozen 20 scenes, ACT achieved 0 full / 1
partial insertion with a 22.69 mean score; the world policy achieved 0 full / 2
partial with a 32.32 mean. Neither met the reliability goal.

The user approved a new **6h50m ACT window, 11:51:28–18:41:28 UTC**, on physical
GPUs **0–1**. The canonical `expert_verified` collection now contains **289**
eligible episodes: 268 SFP across all five NIC counts and 21 newly collected
SC, with a grouped **250 train / 39 validation** split. The time-input
comparison used immutable v2 (260 episodes; 229/31). A matched visual-grid
comparison started on all of v3 at **15:18:10 UTC**, including its 29 additions.
The 253 successful agent recordings remain pending action-label repair;
212 non-insertions and 63 unresolved historical episodes are excluded.
Task conditioning and DDP passed GPU checks. The first SFP stage completed
6,000 updates in 53m33s at its requested cap. Mixed SFP/SC training started
13:16:39 UTC with batch 128 per GPU and balanced task/card-count sampling,
then stopped deliberately at 6,001 updates after 48m33s. This freed time for
a matched comparison of retaining versus masking elapsed time; it was not
stopped by the overall deadline or claimed to have fully converged.
SC collection accepted 21 of 36 attempts across two batches, all full insertions with
verified labels and terminal images, within a restricted scene distribution.
Historical SC command labels cannot be faithfully repaired. The initial
2,000-update ACT checkpoint completed five fresh development scenes with
**0/5 insertions**, mean score 31.71. A measured entrance stall was 7.91 mm
off axis. Step-4,000 and step-6,000 checks on two development scenes also had
zero insertions. The first mixed-task checkpoint completed **0/5 SFP and 0/4
SC insertions**; SFP NIC1 regressed while other SFP counts improved approach
scores. The time-input pair started at 14:06:35 UTC and completed 2,000 updates per
branch from the same mixed-task step-6,000 parent and immutable dataset.
Elapsed time is strongly associated with task identity in that dataset;
the first nine-scene evaluations gave **1/9 insertions for retained time**
(SFP 0/5, SC 1/4; mean 41.48) versus **0/9 for masked time** (mean 27.22).
The clean SC port-1 success scored 86.71. Those runs used a wall-time cap;
two masked SFP cases were still moving at the end. At 90 simulated seconds,
retained time completed **0/9**, mean **35.57**; its earlier SC success did not
repeat. Masked time had zero insertions and one shortened SC trial with contact
and force penalties. A separate full-duration retry also failed. Its supplementary
eight-valid-plus-retry comparison scored **23.22**, but does not erase the
original adverse outcome. Subsequent review found grossly displaced pre-command
robot starts in two masked SC trials, including the shortened one; the composite
is therefore not a clean policy comparison. The engine readiness source now
requires finite named arm joints near the configured home pose; it passed an
isolated rootless build/test but has not been installed into the pinned
evaluation image. Current trials restart the simulator for each scene and audit
the pre-command arm joints and first camera frames independently.
Retained time remains the parent for matched stride-32/stride-16 continuations.
Both completed 3,000 updates on v3, with held-out command errors **5.60 mm**
and **5.32 mm**; their fresh-simulator development trials completed with
physical start checks. No reliable learned insertion is established.
Their completed nine-scene fresh-simulator development results were
**0/9 insertions** each, mean score **33.27** for stride 32 and **24.44**
for stride 16. The earlier parent was also **0/9**, mean **23.03**.
The prespecified development rule selected stride 32; its frozen 32-scene
final assessment ran on GPUs 0 and 1 from 17:03 to 18:03 UTC. All **32/32**
trials passed the fresh-simulator initial-state and duration checks and were
officially scored. The policy achieved **0/32 full insertions** (SFP **0/20**,
SC **0/12**), mean official total **25.70** (SFP 21.04, SC 33.48). Two SC
trials were scored as partial insertions; nine trials incurred a prohibited
contact penalty. All 32 one-frame-per-second videos and start/end contact
sheets are archived with the selected model. The target was not met; no
offline or online SERL was trained in this window. See the
[ACT record](experiments/2026-09-18-act-all-verified.md) and
[frozen final report](../outputs/experiments/2026-09-18_act_all_verified_6h50/selected_act_final_single/final_single_results.md).

The parallel [Dreamer-v4 pilot](experiments/2026-09-18-dreamer-proposal.md)
was approved and started at 15:52:56 UTC, with a separate 22:42:56 UTC deadline.
Its strict first comparison uses the original 60 aligned SFP training episodes
and 14 held-out episodes. A fresh ImageNet ACT baseline completed 6,000 updates
on GPU 4; its four paired development scenes scored **0/4 insertions**.
Dreamer's first tokenizer stage stopped at 5,981 updates at its 50-minute cap;
two bounded detail refinements completed 2,500 updates each on GPUs 2–3, but
fine contact details remained blurred. A selected six-view tokenizer was frozen
for a supervised diagnostic. Its action-conditioned dynamics stage stopped
deliberately after 6,344 updates at 18:11:50 UTC; it beat persistence at
four/eight steps on average but failed the one-step and near-contact gates.
The held-out selected checkpoint was step 1,500. The superseded supervised BC
started at 18:12:54 UTC and stopped at 18:14:32 UTC when the action contract
was corrected. An untrained Dreamer inference path measured
28.49 ms p95 over 1,000 decisions including image
preprocessing and command conversion. Early tokenizer reconstructions blur
connector/port details. An audit of all 74 expert bags found the official
insertion event after the final saved observation in every episode, so the
strict pilot's insertion-state reward/imagination gate fails. The corrected
TCP-delta pilot's current training and live results are summarized at the top
of this page.

## Completed September 17–18 experiment

Updated: **2026-09-18**, branch `feat/hybrid-train`, base `7534090` plus
working-tree repairs and the direct visual actor. Evidence now includes CPU
regressions, the earlier 40-update BC smoke, **live Gazebo policy/control trials,
Isaac reset/terminal probes, a dataset audit, and new supervised ACT training**.
The earlier live-validation pass used at most two GPUs and did no training.
The eight-hour ACT experiment completed training and final evaluation using
physical GPUs 0–3, within its **2026-09-18 05:12:28 UTC** deadline.
[ACT experiment results](experiments/2026-09-17-act-verified-8h.md) ·
[Earlier results and videos](experiments/2026-09-17-live-validation.md) ·
[Initial audit](experiments/2026-09-17-reentry-audit.md) ·
[Accounting repairs](experiments/2026-09-17-evaluation-curriculum-fixes.md) ·
[Actor implementation](experiments/2026-09-17-direct-visual-policy.md).

## Where we stand

There is **no verified reliable general learned insertion policy in the evidence reviewed**.
The repository has usable training and evaluation machinery, ACT checkpoints,
many failed or partial RL experiments, and privileged local insertion diagnostics.
These have different evidential value; see the [experiment ledger](EXPERIMENTS.md).

The verified-data ACT experiment finished with **1/20 full insertions** on
fresh SFP NIC-1 / card-0 / port-1 / rail-0 scenes (mean total **39.16**).
All 20 evaluations completed normally with ground truth disabled. The **18/20
reliability target was not met**. The selected model had 2/10 development
insertions; its checkpoint, normalizer, runtime, lineage and exact scene set are
preserved in `outputs/experiments/2026-09-17_act_verified_8h/selected_act_final/`.
See the [report and videos](experiments/2026-09-17-act-verified-8h.md).

Historical RGB/BGR and wall/simulation-clock differences were identified; new
collections record aligned RGB observations and teacher commands. Quaternion,
command transport and warm-start normalization contracts were checked/repaired.
At the end of that earlier run, its newest cache held **87 recent episodes
(70 train / 17 validation)**;
the selected continuation used 60 train / 14 recent validation episodes from the
74-recording predecessor. There were 47 bounded ACT jobs, including small fitting
diagnostics and continuations. **No offline or online SERL updates ran.**

Training-command errors fell below 1 mm while recent held-out errors remained
around 4–5 mm. On one final failure, the tip was at the port entrance but
12.6 mm off axis, while the policy continued requesting descent. A separately
labeled privileged expert control on that same scene scored 94.68
(Tier 3 75); it is never counted as learned-policy success.

The latest located long run started June 14 and ended June 19. Its recovered
events show **200 evaluation cycles, zero successful evaluations, and zero
promotions**, ending at level 0. The recorded `episodes_used=2000` is a wrapper
budget counter: the old wrapper added 10 after every timed training segment, without
counting completed simulator episodes. Do not report this as 2,000 measured
episodes. The July handoff note remains useful historical context.

The ACT 175k CPU/CUDA exports, associated pretrained checkpoint/normalizer, and
historically named “clean” dataset are present locally. The new audit found
action-label and split-quality concerns in that dataset. Their presence establishes availability.
The inspected May 13 Gazebo evaluation has `policy_ready=false` and no score.
The May submission bundle's referenced model/evaluation directories are missing
from this checkout; its reported scores are historical, not reverified.

## Architecture direction

The current requested sequence is **ACT → offline SERL → online SERL**. The
completed experiment trained and evaluated ACT only, starting with verified
CheatCode demonstrations, then aligned expert and corrective collections.
Episode holdouts and a frozen 20-scene assessment were used. No offline or
online SERL updates have been run in this pass.

The [June 17 tutorial notes](https://github.com/yoonjung0705/tutorials/blob/master/python/libraries/tutorial_aic.md#617-2026-hybrid-train-branch-model-training-especially-for-isaac)
call for dropping the residual adapter, giving the actor direct visual features,
considering DINOv2, and reconsidering whether ACT warm starting is needed. They
also report an expert rollout withdrawing when initialized already inserted.
The two CheatCode controllers were found to request a 20 cm approach pose even
from such starts; they now preserve depth when an aligned final-descent gate is
met. The change has CPU coverage and still needs live verification.

The source still contains several distinct actors:

| Path | What the code does |
| --- | --- |
| **New `actor_mode=direct_visual`** | Shared image/state actor for offline learning, Isaac loading/training, and Gazebo inference/adaptation. Predicts the full command from visual features; no ACT action proposal or residual. Default for the offline CLI. |
| Direct actor backbones | Small conv, ResNet18, ImageNet ResNet18, and DINOv2 ViT-S/14. Optional ACT initialization copies visual weights only. Backbone parameters train by default. |
| Offline `ACTAdapterSERLActor` | Predicts a residual from encoded state plus ACT's proposed action chunk; the residual head does not receive image features directly. |
| Offline `actor_mode=act_direct` | Uses `ACTChunkActor`; this is a different implementation from Isaac's identically named mode. |
| Isaac `IsaacACTAdapterActor`, `act_direct` | Predicts the executed action without adding ACT's action, but still feeds state plus ACT's proposed actions to the trainable head. This retains the visual information bottleneck described in the notes. |
| Vision critics | The offline encoder supports small convolutional networks, ResNet18 variants, and ConvNeXt Tiny variants. DINOv2 is not implemented in this encoder. A visual critic does not give the actor direct visual features. |

The legacy actor modes remain for old checkpoints and recipes. Use the
[direct visual workflow](DIRECT_VISUAL_POLICY.md) for new architecture work.
The new actor completed 40 BC updates on saved demonstrations; a pretrained
DINOv2 forward/backward check also passed. These are implementation/learning
checks. The subsequent direct actor and ACT Gazebo trials both completed with
no insertion. Fix reset/control and dataset problems before comparing encoders.

## Repairs completed on September 17

| Area | Current behavior |
| --- | --- |
| Gazebo score interpretation | Correct insertion requires each trial's official Tier 3 score of 75. Tier 1 remains model validity. The parser exposes per-trial outcomes and counts; transfer validation requires complete scoring and all observed trials to succeed. |
| ACT runtime evaluation | Explicit command mode is required. Empty checkpoint selections and incomplete runs fail. Matching complete evaluations may be reused; failed, legacy, or changed-setting evaluations get fresh attempt directories. Completion requires readiness, clean engine exit, and all configured trials scored. |
| Episode accounting | Isaac writes one outcome per terminated/truncated environment episode, using pre-reset identity and terminal success flags. The wrapper counts these completed episodes. A checkpoint left by a failed process does not permit continuation. |
| Promotion and demotion | The wrapper enforces the configured success rate and consecutive-failure threshold. Evaluation must complete the fixed per-level episode set without updates, exploration, guide, or guard overrides. Empty, partial, or legacy sampled logs cannot authorize promotion. |
| Evaluation settings | SFP module consistency is required; defaults are 0.5 mm axial/lateral error, 0.03 rad orientation, and 1.0/1.5 mm module axial/lateral error. Modified collision geometry is an explicit diagnostic option. Settings are saved with the result. |
| Stops and schema checks | `--updates 0` no longer ends evaluation after its first step. The no-promotion limit stops the wrapper with exit 3. Actor state width mismatch raises an error instead of silently padding/truncating input. |
| Artifacts | Stateful/axial launchers default to mounted `outputs/experiments/`, preserve cycle directories, and reject accidental reuse. Missing checkpoints fail explicitly. Other historical scripts still require path inspection. |

The [repair record](experiments/2026-09-17-evaluation-curriculum-fixes.md) lists
tests and compatibility changes. These repairs do not reclassify historical
results or establish that a policy inserts successfully. Old metrics lack the
terminal evidence needed to reconstruct reliable episode success rates.

## Latest live findings

Subsequent [source inventory](DATASETS.md) found the broader 668-episode local
collection and recovered both CheatCode sources previously absent at their old
manifest paths. All 140 accepted CheatCode trials have historical official
Tier 3 success records. The subsequent audit verified raw state/action equality,
sampled final images, and checked the 130-episode cache's video timestamps.
Across all sources, 393 episodes have verified score/array lineage, 212 are
scored non-insertions, and 63 mappings remain unresolved. See [data eligibility](DATASETS.md#verified-insertion-episodes).

The ACT result above supersedes the early pilot status. The table below records
the preceding live-validation pass; it is not the final ACT assessment.

| Check | Result |
| --- | --- |
| ACT 175k / direct visual actor | One official SFP trial each, both complete, neither inserted. Total scores 36.58 / 23.28; these are unequal training budgets and execution settings, not an architecture ranking. Tier 2 sampling anomalies also limit aggregate-score comparisons. |
| Gazebo control | Final bridge probe completed 260 commands with three-camera recordings. Initial 20 zero commands drifted 7.13 mm over 2.35 simulation seconds. Signed translation effects remain confounded by drift. |
| Isaac resets | Requested −40/−2/+43 mm aligned starts drifted and never met strict insertion criteria in a five-second zero-action probe. It completed zero episodes. |
| Isaac terminal accounting | Separate one-second probe completed exactly three episodes, recorded three timeouts, and stopped at the requested count. Zero successes. |
| Demonstrations | 51.74% zero actions; 189,885 zero-action frames have recorded TCP speed above 1 mm/s. 6.11% of all labels exceed the direct actor's translation limits. Current holdout is entirely SFP card 0 / port 1. |
| Runtime | Rootless Unix-socket bridge, image-sized IPC messages, local model imports, visible policy exceptions, and one-second recording now work in live checks. Isaac cameras require the documented container-local workaround on this old driver. |

See the [full report and videos](experiments/2026-09-17-live-validation.md) for
settings, measured geometry, failure records, and interpretation limits.

## Still unresolved

- Calibrated Isaac placement across task families and scene variations. One SFP
  port-1/card-0 near-opening scene now has stable reset, correct action direction,
  guide insertion, and terminal validation; broader coverage is unproven.
- Demonstration command validity/time alignment, verified insertion labels,
  source-trial grouping, and balanced held-out scenes. Do not treat every zero
  command as corrupt or every historical scalar score as insertion success.
- Earlier policy trials had missing scorer world-frame connectivity. The new
  evaluator provides the fixed world transform while keeping object ground
  truth disabled for learned policies; old scores remain historical evidence.
- No verified reliable learned policy or controlled backbone performance comparison.
  Legacy Isaac `act_direct` still means the ACT-action-context head.
- Some historical model dependencies and container-local artifacts remain
  unbacked-up. See the [artifact map](../outputs_README.md).

## Next experiments, in order

The SFP pose-refitting branch remains parked with its failed dependence gate as
a fixed baseline. The active continuation is SC-to-SC and follows these gates:

1. **Repair the SC grasp/scene contract.** Reproduce a reachable grasp using
   the source-defined collision contract for the gripper, connector, cable,
   board, ports, and cards. Do not use the gripper-disabled diagnostic proxy
   for learning.
2. **Validate mechanics for card counts 0--5.** Run scripted complete insertion
   attempts with deterministic seeds and require valid reset identity, force,
   contact, plug motion, cable motion, and terminal observations in every
   stratum. Record failures rather than weakening the geometry gate.
3. **Collect the bounded discovery set.** Retain complete episodes and compact
   20 Hz telemetry. Manually validate that port-lip contact, simple
   misalignment, and cable/card snag labels are causally distinct. Require ten
   natural snag incidents across at least two layouts or conclude that this
   Isaac setup does not reproduce the intended failure.
4. **Train SC perception and port-relative BC.** Use episode-grouped splits,
   observation-only RGB/state/force inputs, full port-frame trajectory targets,
   and no privileged evaluation inputs. Require the frozen pose gate and
   autonomous insertion before adding RL.
5. **Compare recovery and then train model-free critics.** Start with fixed
   multi-scale measured-path retreat plus coherent lateral/routed exploration.
   Only after enough on-policy success, failure, block, retreat, and recovery
   outcomes, train critics offline and proceed to bounded online SERL with a
   strong supervised anchor.
6. **Measure Gazebo transfer after the Isaac gate.** Freeze the successful
   candidate, evaluate it on new Gazebo development scenes, and use the existing
   bridge for bounded adaptation only if accuracy, force, and latency hold.

World-model dynamics, SEER, reward-model training, imagination, and the reserved
final split remain parked. If explicit corrective control fails, return to the
perception restart checklist in the
[cable experiment](experiments/2026-09-22-cable-visibility-perception.md#parking-decision-and-restart-checklist).

Preserve actor-only evaluation for the June experiment lineage. Any privileged
guide or action override belongs to a separately labeled diagnostic experiment.
Report metrics for the same timestep/episode together: independent best depth,
lateral error, and orientation values cannot establish insertion.

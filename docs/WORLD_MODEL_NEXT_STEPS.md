# World-model follow-up: visual fidelity, dynamics, and control

## Parked after the bounded RPDP plus model-free RL evaluation (2026-09-23)

World-model and SEER work remains parked. The earlier PoseInsert-inspired
diffusion and bounded DPPO branch predicted full port-relative connector
trajectories and used a deterministic adapter to the TCP-body-frame command
contract. Its first valid DPPO round stayed at 1/6 and did not pass promotion.
A deterministic trajectory-regression version later became the selected
supervised controller. Details are in the
[RPDP/DPPO experiment](experiments/2026-09-22-rpdp-dppo.md). None of these
results authorizes imagination, reward-model, or predictive planning work.

The subsequent probabilistic SERL continuation implemented a mixture actor,
balanced prior/online replay, twin critics, and measured-path backtracking. Its
final trust-region actor retained the frozen BC result at 3/8 local successes
but did not improve it. Backtracking saw no qualifying force plus measured-stall
event after its false-trigger bug was repaired. This result does not reopen
predictive dynamics or imagination; see the
[SERL experiment](experiments/2026-09-22-serl-mixture-recovery.md).

A later three-camera rerun does not change that decision. It recorded 1/8 BC,
1/8 selected RL, and 3/8 selected RL plus recovery, but compact recovery-mode
telemetry was not retained and most failures were lateral divergence rather
than cable snagging. World-model, Seer, reward-model, actor-Q, and imagination
work remain parked. The next proposed experiment is the bounded
[SC-to-SC multi-card cable-snag plan](SC_CABLE_SNAG_RECOVERY_PLAN.md): establish
causal snag incidents, SC perception and BC, then fixed and learned recovery.
Reconsider temporal or predictive dynamics only if observation history remains
the measured bottleneck after those supervised and model-free stages.

This is the next-work handoff for the September 18 Dreamer-style pilot. Read
the [current status](STATUS.md), [pilot record](experiments/2026-09-18-dreamer-pilot.md),
[dataset guide](DATASETS.md), and [geometry/scoring terms](scoring.md) before
starting. The frozen paired final and the first visual/dynamics follow-up are
now documented in the [completed follow-up record](experiments/2026-09-18-world-followup.md).
Preserve the frozen pilot and final-scene records.

## Current direction (2026-09-20)

The direct world-policy Isaac path is implemented with the native four-command
TCP-delta contract. It stores one 24D, 200 ms transition plus the frozen 384D
causal observed feature and supports a differentiable 333,024-parameter policy
head. The legacy ACT residual adapter is not used.

The earlier contract failure was traced to a scene-placement mismatch: the
supported YAML put the port about 552 mm from the expert pose. On one calibrated
SFP port-1/card-0 near-opening scene, the privileged guide repeatedly inserts
and the action/reward/termination contracts pass. Two silent training bugs were
fixed: replay now stores all four actual guide commands, and actor-head forward
is no longer wrapped in `inference_mode()`.

Call this guide the **Isaac CheatCode** in future records. It mirrors the
Gazebo CheatCode's privileged plug-to-port rigid transform, but currently
implements a clipped near-port feedback servo rather than the full Gazebo
approach, timed settle, minimum-jerk insertion, and official completion-event
handling. The 552 mm value is a spatial placement mismatch between the original
Isaac opening and the expert plug pose. It is not a purely lateral port-frame
error or evidence that every Gazebo/Isaac asset transform matches.

During DAgger, the guide fraction interpolates every command; it does not choose
which fraction of timesteps belongs to each controller. Store the unblended
Isaac CheatCode chunk as the supervised target and the blended chunk as the
executed action for any dynamics or critic update. The current teacher covers
the calibrated one-card SFP port-1/card-0 task with near-port reset variation;
other task families and full approaches still need validation.

The initial variation generator retained a joint-state reset and silently
ignored requested Cartesian starts. After changing it to a tight 6D IK reset,
measured starts covered the requested 4--10 mm axial and approximately 0.7 mm
lateral offsets. Adaptive collection then achieved 9/9 and 7/8 strict guided
insertions over two passes, with the second pass intervening on 91.1% of steps.

Three autonomous heads still scored **0/4 strict insertions** on heldout IK
starts. The varied-only head consistently reached within 0.10 mm of the opening
plane, but stayed 3.25--3.89 mm laterally off center at closest approach and hit
the 35 N force clip. More of the same transition-level L1 fitting is not the
next experiment. Add an explicit observable alignment phase or geometry target,
split validation by episode, and require autonomous success on new development
starts before opening the reserved final split. Actor Q training and Gazebo
transfer remain gated. Predictive imagination remains disabled because the
learned dynamics loses to persistence at every measured horizon. See the
[Isaac execution record](experiments/2026-09-20-isaac-world-policy-rl.md) and
[contract summary](../outputs/experiments/2026-09-20_isaac_world_rl/contract_gate_summary.json).

### Approved supervised continuation

Before more action-head fitting, train a frozen-feature diagnostic probe to
predict the manipulated-object frame relative to the selected target frame:
3D translation, orientation, contact/blocked state, and uncertainty. This is a
general object-centric robotics target; axial and lateral insertion errors are
derived views rather than the permanent policy interface. Isaac geometry is a
label only and must not enter the actor at autonomous evaluation.

Use complete episodes/reset configurations for train and validation grouping.
Report heldout millimetre and degree errors, correction-direction accuracy,
contact classification, and phase/motion-size slices with state-only and
constant baselines. If the frozen 384D feature resolves the 0.5 mm corridor,
compare the current action head against a matched head conditioned on predicted
relative pose, phase, and uncertainty.

If it does not, add a high-resolution interaction crop taken from the original
render before global resizing and rerun the same probe. A crop made after the
288x256 resize cannot restore erased connector detail. At evaluation the crop
must be selected from observations, preferably by a learned plug/target locator
trained with simulator labels; ground-truth geometry may not choose the crop.
Keep the full camera feature for global task context and recheck complete live
latency against the 300 ms limit.

Use new development starts and keep the four reserved final IK configurations
sealed. Actor-Q, Gazebo adaptation, and imagination stay disabled until an
autonomous supervised policy inserts reliably with acceptable force/contact.
The detailed gates and stop rules are in the
[Isaac execution record](experiments/2026-09-20-isaac-world-policy-rl.md#approved-next-experiment-object-to-target-pose-supervision).

**Execution update:** the frozen-feature probe and original native-crop probe
failed, and the old crop labels were then found to describe a module centroid
and card-mask pixel. Exact plug-tip/opening projection was repaired and audited
visually. On a fresh episode-grouped 16-train/4-development collection, oracle
projected keypoints plus triangulation reached 0.124 mm median and 0.235 mm p95
near-port lateral error. The observation-only coarse-to-fine RGB locator
reached 2.380 mm median, 6.947 mm p95, and 82.6% direction accuracy. Its p95
inference time was 3.311 ms. The perception gate therefore failed; no
pose-conditioned actor or RL was started, and the reserved final split remains
sealed. Resume this branch only with a new, predeclared opening-boundary/corner
and temporal-occlusion representation, evaluated on new development starts.

That opening-boundary continuation is now complete. A six-heatmap native-crop
model was trained on 20 configurations and evaluated once on six new ones, but
calibration selected the coarse locator instead. Its near-port lateral error
was 1.085 mm median / 1.740 mm p95 with 91.5% direction accuracy; inference was
1.982 ms p95. A post hoc world-coordinate filter reached 0.862/1.505 mm but is
promotion-ineligible and still outside the gate. No control or RL branch was
opened.

A small ImageNet-pretrained spatial follow-up subsequently improved the fresh
held-out result to 0.278/0.670 mm median/p95 with 94.1% direction accuracy and
5.949 ms p95 complete inference. This is close but still fails both absolute
lateral thresholds. It does not change the world-model decision: current-frame
perception should clear the gate before predictive dynamics or imagination is
reintroduced.

A causal world-coordinate smoothing diagnostic reached 0.250/0.616 mm lateral
error but regressed axial error to 2.085/4.407 mm. Its p95 still failed and it
was ineligible for promotion because it was proposed after the development set
was opened.
Calibration also rejected simple heatmap-confidence camera dropping and kept
all three views. The next perception change therefore needs learned fusion or
more varied occlusion data, not a hand-selected view subset.

The learned-fusion continuation has now been executed. On four held-out
calibration configurations, a current-frame three-view residual head reached
0.141/0.421 mm near-port lateral median/p95 and a six-step causal head reached
0.133/0.300 mm. On six newly collected reset positions, those results regressed
to 0.307/0.573 mm and 0.289/0.667 mm respectively. The frozen spatial baseline
was 0.412/0.769 mm, so learned fusion helps but still misses the absolute gate;
the temporal calibration advantage did not generalize. End-to-end p95 latency
was only 6.82 ms with the existing frozen trunk.

The replay has no independent cable reset seed and sparse segmentation only;
therefore it cannot test the cable-occlusion hypothesis well. Collect varied
cable shapes and occlusion positions with depth-derived visibility labels
before another causal model. Keep predictive dynamics, conditioned policy
training, and RL disabled. See the
[executed plan](experiments/2026-09-21-temporal-multiview-perception-plan.md#executed-bounded-continuation).

The requested force-safe cable continuation is now complete. A privileged
collection-only controller settles each recorded cable state, rejects high
force, restores the measured plug pose, and then releases to the fixed
CheatCode teacher. The 50-episode run retained 460 safe near-port observations
from all five included cable templates. With complete-template splitting, the
calibration-selected current-frame model reached 0.322/0.626 mm near-port
lateral median/p95 and 100% direction accuracy on 153 held-out samples. A fixed
causal temporal variant reached 0.247/0.617 mm; a combined static plus natural
dataset was worse at 0.331/0.753 mm. Complete p95 latency was 7.18 ms. The p95
gate still fails. Do not run policy, RL, SEER, or another world-model branch on
this representation. The completed frozen audit found a weak center-camera
plug landmark, but no two-camera subset improved the three-camera p95. Oracle
pixels triangulated with sub-0.0011 mm lateral p95, while a fixed per-camera
bias correction regressed to about 0.79 mm p95. High errors clustered in one
cable episode and small-motion rows; ensemble spread was not calibrated.
Collect more independent safe cable shapes and improve camera-specific landmark
supervision before rerunning the frozen perception gate. Further pose fitting is
now parked. The frozen estimate may be tested as an input to a matched
supervised controller diagnostic, while retaining its failed gate in the
record. RL still requires autonomous insertion by that controller on new
development starts. See the
[cable/visibility record](experiments/2026-09-22-cable-visibility-perception.md).

The matched supervised controller diagnostic is also complete. Two identical
546,686-parameter six-step GRUs were trained on the same complete-template
split. One zeroed the ten pose/uncertainty/phase slots and one used them. The
action-only arm was marginally better offline. On eight new autonomous starts
per arm, both scored **0/8 strict insertions**. Best lateral p50/p95 was
1.479/1.678 mm without pose and 1.417/1.871 mm with pose; both drifted to about
6.3 mm median terminal lateral error. Zeroing the pose input changed the
conditioned arm's held-out translation command by only 0.0095/0.0264 mm
p50/p95, showing that ordinary BC largely ignored it. Complete p95 inference
was 7.388 ms. The reserved final split stayed sealed and no RL ran.

This result closes the approved generic pose-concatenation branch. A later
supervised attempt must use balanced corrective data and an explicitly
structured correction residual, with a counterfactual pose-to-command sign
test before live rollout. It must pass autonomous insertion before actor-Q.
World dynamics, SEER, reward learning, and imagination remain parked. See the
[controller execution record](experiments/2026-09-22-pose-conditioned-gru-policy.md).

The explicit continuation is now complete as well. Force-safe balanced
collection produced 322 fit rows from 25 episodes and 124 paired-direction
calibration rows from seven unseen-template episodes. The mandatory
odd-symmetric pose residual could not silently ignore pose, but correct pose
reduced heldout translation command MAE only 3.10% relative to physical zero,
below the fixed 5% gate. Its median correction was 0.02909 mm against a 0.030
mm floor. A bounded rerun selected directly on translation MAE still failed;
shuffled pose slightly beat correct pose on calibration, and zero pose was
better on offline development. The new live development recipe was therefore
not opened. This keeps actor-Q, world dynamics, SEER, and final evaluation
disabled. See the
[explicit-correction record](experiments/2026-09-22-explicit-pose-correction.md).

### Supervised RPDP repair update

The later RPDP audit found that the failed diffusion BC experiments were not a
clean test of the proposed port-relative target. Recovery rows supervised the
blended executed action instead of the recorded teacher target, four distinct
50 ms commands were compressed into one 200 ms endpoint, checkpoint selection
kept undertrained models, and some live runs clipped 0.5 mm insertion commands
to 0.2 mm. The corrected dataset composes each recorded TCP-body-frame teacher
command into a complete connector waypoint in the fixed port-opening frame.

A pose-gated diffusion arm improved to 2/6. A matched deterministic trajectory
regression arm achieved 6/6 strict insertions on the fixed RPDP development set
and 8/8 on an additional set unused for RPDP fitting or selection, with 19.86 ms
p95 inference and acceptable force. This passes the supervised development
gate without using a world model, guide, guard, exploration, or privileged
evaluation input. The reserved final split remains sealed. Any later RL should
start from this frozen BC anchor and use a matched comparison; the earlier 1/6
DPPO update is not evidence for continuing RL unchanged. See the
[RPDP repair record](experiments/2026-09-22-rpdp-dppo.md#supervised-bc-repair-continuation).

### Decision on any future use of dynamics

The world-policy controller scoring better than ACT does not show that its
learned dynamics is useful. The controller also changed representation,
architecture, cameras/crops, and supervised data. In the direct dynamics audit,
holding the current state fixed predicted future TCP position better than the
learned model at every tested horizon. Therefore future predicted latents,
poses, images, rewards, and imagined rollouts must remain outside control and RL
until a new dynamics model beats persistence on unseen episodes.

Current-frame pose estimation is a separate problem and does not require a
world model:

```text
camera images --- current pose estimator --- predicted plug-to-port pose
      |                                                   |
      |                                                   |
      +---------------- full visual features -------------+
                                                          |
                                                          |
robot state + force + previous commands ------------------+
                                                          |
                                                          v
                                                   BC / RL actor
                                                          |
                                                          v
                                                full TCP-frame command
                                                          |
                                                          v
                                                       robot
                                                          |
                                                          v
                                                next real observation
```

If future dynamics later becomes accurate, it should first act as a short-term
action evaluator rather than an open-loop controller:

```text
current real observation --- current pose/feature estimate
                                      |
                                      v
actor --- proposes several complete candidate commands
                                      |
                                      v
world dynamics --- predicts next pose, measured motion, force,
                   contact, blocked state, and uncertainty
                                      |
                                      v
selector / critic --- chooses one safe useful command
                                      |
                                      v
robot executes one short command --- next real observation
                                      |
                                      +--- repeat from real feedback
```

The world model never supplies true simulator geometry to the actor and never
runs long open-loop imagination near contact. If it continues to lose to
persistence, retain any separately validated encoder initialization but retire
the predictive dynamics, decoder, reward, and imagination components from the
control path.

More data helps only when it adds missing causal information. Repeating similar
successful approach trajectories will not identify how different actions alter
contact. A credible new dynamics dataset needs synchronized current and next
observations, actual executed commands, measured TCP and plug motion, exact
pre-reset terminal observations, and balanced examples of lateral corrections,
rotations, advances, retreats, free motion, edge contact, blocking, and
successful insertion. Near-port interventions from the same or closely matched
starts are especially valuable because they reveal action effects rather than
scene identity.

A future dynamics model should initially predict compact task state instead of
depending on photorealistic next-frame decoding:

```text
current predicted pose + robot state + force + executed command
                               |
                               v
       next pose change + measured motion + force/contact/blocked
```

Predicting the change from a persistence baseline is appropriate for dynamics;
it is not the removed residual-action adapter. Use short 50--200 ms horizons,
phase-aware losses, calibrated uncertainty, explicit action conditioning, and
native-resolution spatial features. The decoder remains an audit tool. Promote
the model only if it beats persistence and constant baselines by phase and
motion size, shuffled actions make it worse, near-port errors meet the control
need, contact/terminal recall is useful, and complete live latency remains below
300 ms.

### Cable dynamics, model-free RL, and SEER

The active non-world-model continuation, including the rationale and promotion
gates, is recorded in the
[perception and model-free RL plan](experiments/2026-09-20-perception-supervised-rl.md).

The repository's task names are `sfp_to_nic` and `sc_to_sc`. The latter is the
likely meaning of “SFP-to-SFP” when referring to the fiber cable obstructing
itself; verify this against the connector in the scene rather than renaming old
records.

Cable state is a sound reason to use temporal observations. A single image and
plug-to-port pose can miss slack, tension, a loop behind the gripper, or contact
with a neighboring card. The deployed actor should therefore see a short image
history together with measured TCP state, force, and previous executed
commands. A recurrent or temporal policy can learn an internal cable-state
estimate without first learning a pixel-generating world model.

Model-free RL can learn avoidance and recovery if its simulator exposes enough
variation and its observations reveal the obstruction. It should start from a
supervised temporal policy, use a privileged simulator-only critic if helpful,
and randomize cable slack, stiffness, damping, friction, card count, and contact
layout. Training must include blocked, retreat, reroute, and recovery cases;
roughly 1,000 mostly successful expert episodes do not cover those causal
outcomes. Simulator throughput can provide many more transitions, but simulator
contact fidelity remains a separate transfer risk.

[SEER (ICLR 2025)](https://arxiv.org/abs/2412.15109) is the relevant SEER here.
It is a predictive inverse-dynamics policy, not a Dreamer-style return-imagining
agent. It consumes image history, robot state, and a goal, predicts a future
visual representation, and predicts the intervening action chunk. Its released
real-world checkpoint was pretrained on DROID robot demonstrations. The paper
reports 316M total parameters for the standard model: 251M frozen visual/text
components and 65M trainable components. It used 76,000 successful DROID
trajectories, then collected 100 demonstrations per downstream real-world task.

This pretraining supplies broad robot manipulation, visual, temporal, and
action priors. It does not establish knowledge of thin cable deformation,
snagging among NIC cards, or self-obstruction: the paper reports no cable task,
and successful DROID demonstrations do not specifically teach recovery from
these failures. Fine-tuning is therefore plausible initialization rather than a
drop-in cable dynamics solution. Camera geometry, state and action conventions,
TCP-frame delta actions, and the 300 ms live limit must also be adapted and
measured locally.

A bounded SEER test should compare the same temporal policy and episode split
from (a) random initialization and (b) the DROID checkpoint. Begin with frozen
encoders and a new full TCP-frame action head, then unfreeze only the upper
temporal layers if the frozen version underfits. Use matched update counts and
evaluate both strict insertion and cable-specific events: card contact,
blocking, retreat, reroute, force, and recovery. This tests whether pretraining
improves sample efficiency without assuming that it already contains the
required cable physics. Retain the existing rule that predictive rollouts may
enter control only after they beat persistence on unseen causal transitions.

**Execution update:** the repaired causal collector produced 1,208 decisions
over 16 training/calibration configurations and 351 decisions over four unseen
development configurations, with true pre-reset observations for all 24
terminal transitions. The frozen feature probe failed the near-port resolution
gate: 0.730 mm median and 2.533 mm p95 lateral vector error versus required
0.25/0.5 mm, despite 93.5% correction-direction accuracy. The state-only and
constant baselines were worse. Policy conditioning is therefore still closed.
The single native 576×512 crop ablation is complete. Its crop comes from the
original render and is centered by a learned RGB locator trained from simulator
instance masks; evaluation receives neither geometry nor masks. On the matched
225-decision heldout collection, near-port lateral median/p95 changed from
0.962/2.630 mm for the frozen feature to 0.851/2.154 mm with crops, while
direction accuracy changed from 84.8% to 80.4%. This fails the absolute gate
and is not a material improvement. Complete trunk-plus-crop inference is
2.96/3.20/182.50 ms p50/p95/p99, below the 300 ms p95 limit. The conditioned
policy comparison was skipped by its stop rule. The four final configurations
remain sealed; actor-Q, Gazebo adaptation, reward learning, and imagination
remain disabled. See the
[probe result](experiments/2026-09-20-isaac-world-policy-rl.md#causal-replay-audit-and-frozen-feature-probe).

The approved implementation is now tracked in the
[September 19 full-training record](experiments/2026-09-19-full-world-training.md).
It completed a fresh tokenizer on the canonical 250/39 split under a held-out
validation plateau rule, then trained dynamics and supervised control using
native observed endpoints and actual executed TCP-delta commands. The tokenizer
selected update 75,000 and substantially improved held-out reconstruction.
Dynamics still lost to persistence at every tested horizon, while the supervised
policy achieved 2/20 full and 9/20 partial insertions on the sealed final set.
Reward and imagination remain disabled. See the linked record for metrics and
artifacts.

## What the pilot established

- The selected six-view tokenizer reconstructs broad scene geometry, but the
  connector and port remain blurred. The added crops came from images already
  resized to 288×256, so they cannot restore detail removed by that resize.
- The expanded corrected 200 ms world rollout, conditioned on four *actually
  executed* commands, misses the future measured TCP position by 12.06 mm on
  average; predicting no movement misses by 2.29 mm across all 924 strict
  held-out windows. The earlier three-window probe reported 10.33/1.85 mm. This is a future-state prediction
  test, not controller target-tracking error. The longer-horizon fidelity gates
  also failed. See [metric explanation](STATUS.md#september-18-work).
- All 74 pilot expert bags have an official correct-port insertion event, but
  none of their native saved images/states occurs at or after that event. These
  are successful command demonstrations; they supply no synchronized post-event
  RGB/state example. All 74 raw bags do retain short post-event controller-state
  tails, but none contains camera/video messages (audit below).
- The canonical `expert_verified` dataset has 289 BC-eligible episodes
  (268 SFP, 21 SC). BC eligibility does not imply that every source has
  synchronized images and the four executed commands needed for a valid
  200 ms dynamics transition. In particular, the historical CheatCode source
  retains a wall-clock sampling limitation. Keep failed rollouts separate from
  successful BC labels.

## Selected dynamics: future latent and pose audit

The selected dynamics checkpoint is **step 1,000**, SHA256 `839a1e9aaf77045cfe2c3db5aa6c5c2ed019f2cc9b8d271f4954aea23d515f87`. This is the world-only checkpoint selected before BC, not the subsequently updated BC transformer. No training or final-scene tuning was performed for this audit.

The expanded audit evaluates **every strict held-out native transition window**, with actual executed future commands provided only for this dynamics diagnostic. The three horizons contain 924/158/54 windows from 14/9/5 held-out episodes. Each episode receives equal weight after averaging its windows. This extends the earlier three-window probes; sampling and random noise differ, so the earlier report remains preserved rather than overwritten.

| Horizon | Episodes / windows | TCP error: world / persistence / shuffled | Rotation error: world / persistence / shuffled |
|---|---:|---:|---:|
| 200 ms | 14 / 924 | 12.06 / 2.29 / 13.43 mm | 2.28 / 0.237 / 2.45° |
| 400 ms | 9 / 158 | 14.92 / 4.81 / 19.84 mm | 3.24 / 0.390 / 3.94° |
| 600 ms | 5 / 54 | 21.92 / 9.85 / 24.23 mm | 4.97 / 0.942 / 6.28° |

RMS Euclidean position errors are **13.64 / 16.88 / 25.21 mm** for the world model, **2.98 / 5.94 / 10.73 mm** for persistence, and **15.07 / 22.05 / 26.92 mm** for shuffled actions at 200/400/600 ms. This RMS is the square root of episode-balanced mean squared Euclidean distance, not per-axis RMSE; `rmse_summary.json` also records angular RMS.

**The dynamics gate still fails.** The actual-command model is sensitive to actions, but holding the last observed pose is substantially better overall. Shuffled controls use all four verified commands from another validation episode at its nearest elapsed time; source episode, times and native indices are preserved. Their elapsed-time match is approximate, not an exact phase-matched intervention.

### Motion and phase matter

For windows with less than 1 mm true TCP displacement, world errors are **13.54 / 18.33 / 29.19 mm**, versus persistence **0.247 / 0.366 / 0.513 mm** at 200/400/600 ms. Coverage is 14/6/1 episodes and 308/21/3 windows. For the 5–20 mm motion bin, world errors are **9.01 / 13.99 / 23.79 mm**, versus **5.99 / 7.47 / 11.30 mm** for persistence, across 14/7/5 episodes. There are **no ≥20 mm held-out windows** at these horizons.

The final-three-seconds recording proxy gives world TCP errors **11.96 / 14.05 / 29.19 mm**, versus persistence **0.292 / 0.185 / 0.513 mm**, with **13/2/1 episodes and 73/5/3 windows**. This proxy is not an authoritative contact phase. Early/middle/late time bins and true-rotation bins are also recorded. A small high-rotation subset shows better translation than persistence at 400/600 ms, but only 3/2 episodes support it, rotational prediction remains worse, and the shuffled control is better at 600 ms. It does not establish usable action-conditioned dynamics.

### Distance to the physical opening: completed

Recorded scoring TF was matched to 1,690 native observations across all 14
validation episodes; **1,675 passed** the geometry/timing checks. Fifteen early
observations lacked a complete TF graph, excluding 25 transition windows
(75 actual/persistence/shuffled records). No missing transforms were invented.
Accepted controller stamps differ by at most 4 ms, moving cable-tip TF by at
most 2 ms, and native/controller TCP positions by at most 0.265 mm.

The task board is explicitly static and publishes its static poses at 1 Hz.
Its port/entrance transforms were exactly constant across the sampled times in
every episode, so older fixed-assembly header timestamps are treated separately
from moving-tip timestamps. The audit uses the **explicit recorded entrance
frame**, 45.8 mm from the port reference, rather than treating reference distance
as standoff from the opening.

For windows starting **within 5 mm of the entrance**, world TCP prediction error
is **13.94 mm versus 0.570 mm for persistence at 200 ms** (13 episodes / 163
windows), and **20.80 versus 0.360 mm at 400 ms** (5 episodes / 12 windows).
There are **no valid 600 ms windows starting within 15 mm of the opening**.
These geometric bins therefore confirm poor near-opening prediction while
exposing sparse longer-horizon coverage. They are not authoritative contact
labels. Prediction error still compares predicted future measured TCP with the
actual future measured TCP, not a command target.

The [opening-distance report](../outputs/experiments/2026-09-18_dreamer60_pilot/world_opening_geometry_audit/report.json)
contains start/end distance bins, all denominators and exclusion reasons.
Its archive includes compressed full records, exact TF snapshots, timing/source
hashes and scripts. The first broad reader's timeout is preserved; the completed
reader excludes the duplicate ground-truth `/tf` relay and retains authoritative
`/scoring/tf`, static transforms and observations near each requested timestamp.
It runs in a CPU-only rootless container; no policy or model was changed.

### Separate tokenizer loss from future prediction error

Two representative windows were selected using recorded motion and time, before examining prediction error:

- [Largest recorded 600 ms motion: all three cameras](../outputs/experiments/2026-09-18_dreamer60_pilot/world_future_latent_audit/largest_true_600ms_motion_full_cameras.png): episode 139, start 2.15 s, 18.42 mm displacement.
- [Latest valid 600 ms window in a different episode: all three cameras](../outputs/experiments/2026-09-18_dreamer60_pilot/world_future_latent_audit/latest_600ms_other_episode_full_cameras.png): episode 193, start 21.5 s, 0.511 mm displacement; its endpoint is still 2.85 s before the last native recording.

Each sheet shows recorded RGB, decoded true latent, decoded actual-action future, decoded persistence, and decoded shuffled-action future at 200/400/600 ms. Additional sheets show all three contact crops. Decoding shares the same true observed prefix; predicted decoders receive no true future latents. Both full-camera sheets were visually reviewed.

True-latent reconstruction already blurs connector/port detail. Predicted latents add drift as the horizon grows, particularly in the near-static example. At 600 ms, the moving example has RGB MSE **0.00425** for true-latent decoding, **0.01026** for world prediction, and **0.02021** for persistence. The near-static example gives **0.00219 / 0.00822 / 0.00229** respectively. These are selected examples across six views, not population-level image metrics. Gross image motion can improve over persistence while precise physical-state prediction remains poor.

Artifacts: `outputs/experiments/2026-09-18_dreamer60_pilot/world_future_latent_audit/`. `plan.json` pins the sampling/bins before inference; `report.json` includes all records, phase/motion denominators and source indices. The script ran read-only on physical GPU 3 in 18.94 seconds, without altering checkpoint or runtime sources.

### Implication for the next experiment

First require correct near-static pose behavior and meaningful motion prediction on fully recorded short trajectories. A state-change prediction objective with an explicit persistence baseline is a concrete candidate, but its benefit must be measured. Collect enough independent examples for contact and longer horizons before imagination RL. Better reconstruction alone would not repair the observed pose-prediction failure, and lower aggregate MSE alone would not demonstrate control readiness.

## Raw-bag post-event coverage: completed for the strict 74

All 74 original bag metadata hashes were checked against the prior event audit, including 11 archived bags. **None contains any camera/image/video topic.** Every bag has controller-state records after the first official correct-port insertion event, with **0.134–0.922 seconds** between the event's upper time bracket and the last controller stamp. Their controller, command and scoring-TF topics are retained. The native synchronized image/state dataset still has **zero post-event examples**.

This distinction corrects an overly broad reading of “no post-insertion states”: native RGB/state training examples are absent, while raw state-only tails exist. A timestamp after an event does not by itself establish sustained insertion. Matching scoring geometry and events could support a separately audited state-only reward or outcome model; it cannot recover missing RGB frames or justify assigning success to the last saved image. No reward labels, split or training run was changed.

The [all-verified metadata extension](../outputs/experiments/2026-09-18_world_followup/verified_bag_metadata_audit.json)
locates an unambiguous ROS bag for each of the **149 aligned** verified episodes,
including 23 archived bags. All 149 have insertion-event and controller-state
topics, and **zero have camera/image/video topics**. The other 140 historical
episodes link to LeRobot video datasets, but the canonical manifest does not
link them to ROS bags or event timestamps. Metadata alone does not determine
post-event state-tail duration in the additional 75 aligned bags, and it
does not establish historical video frames after insertion.

Evidence is in [the raw-bag coverage report](../outputs/experiments/2026-09-18_dreamer60_pilot/post_event_raw_bag_audit/report.json), with all 74 metadata files and the hashed controller/event reports. This conclusion is scoped to the strict 74, not automatically to all 289 expert episodes.

## Completion and remaining work

- **Completed:** both frozen final sets are archived (20 valid trials each, zero full insertions); all 40 scored rollouts have post-run plug-tip/port-opening geometry; the 289-episode visual provenance and leakage-safe split audit, matched 74-versus-expanded tokenizer comparison, contact-resolution ablation, strict held-out future-latent decoding at 200/400/600 ms, pose/rotation and persistence/shuffled controls, native near-opening TF bins, visual review, and strict-74 raw-bag topic coverage are recorded. See the [follow-up record](experiments/2026-09-18-world-followup.md).
- **Completed full-data continuation:** the fresh tokenizer reached the declared plateau at 78,000 updates and selected update 75,000. Dynamics and supervised policy also reached their declared plateau rules. The sealed final20 evaluation completed 20/20 valid scenes with 2 full insertions, 9 partials, mean 45.14, and 80.77 ms pooled live p95 latency. Dynamics remained worse than persistence, so the planning gate is closed. See the [full-training record](experiments/2026-09-19-full-world-training.md).
- **Remaining for dynamics diagnosis:** authoritative contact-phase labels and enough near-opening long-horizon transitions are unavailable: no valid 600 ms native window starts within 15 mm of the opening. If state-only outcomes become useful, audit event-relative tails in the additional 75 aligned bags; establish the historical videos' timing before treating them as post-event observations.
- **Separate supervised comparison completed:** both same-architecture arms finished 2,500 BC updates and 4/4 valid new development scenes; neither achieved a full insertion. The selected-world arm had one official partial and modestly lower held-out command error, but its fresh-world control used a different seeded random draw from the original world-pretraining ancestor. See the [comparison record](experiments/2026-09-18-world-supervised-init-ablation.md) and [paired live scores](../outputs/experiments/2026-09-18_world_followup/supervised_world_development_archive/paired_summary.json). A stronger repeat should use the exact preserved ancestral random initialization and multiple seeds. Improved near-static state-change prediction and complete near-contact/failure trajectories are still needed; these diagnostics do not authorize imagination RL.
- **Isaac supervised gate completed:** corrected varied resets and two adaptive intervention passes produced successful guided data, but three autonomous heads each scored 0/4 on heldout starts. The best controlled ablation reached the opening plane consistently while remaining 3.25--3.89 mm laterally off center. The reserved final starts, Gazebo transfer, actor-Q updates, and imagination updates were not opened. See the [Isaac execution record](experiments/2026-09-20-isaac-world-policy-rl.md).

## Work sequence

1. **Close the current comparison and localize failure.** Finish and archive
   the already frozen paired ACT/world evaluation before changing either
   model. For each complete rollout, measure plug-tip position in port
   coordinates: axial distance to the entrance, lateral offset, orientation,
   measured TCP motion, published target, and insertion/contact events. Use
   official scores and inspect the one-second videos. A reported 4–5 cm
   plug-to-port-reference distance alone does not mean the tip is 4–5 cm in
   front of the opening: the SFP entrance is 45.8 mm from that reference.
   Identify approach, alignment, and seating failures separately. Do not tune
   on the frozen final scenes; make a new development split for follow-ups.

2. **Audit the larger visual dataset without another simulator run.** Use all
   eligible expert images for a tokenizer experiment after checking source
   provenance, grouped scene splits, RGB/BGR convention, resizing, camera
   order, and near-port frame coverage. Keep the original 74-episode tokenizer
   and held-out scenes as a comparison. Compare 74 versus all eligible images
   with the same architecture, resolution, update budget, and selection rule.
   Keep source/task-balanced sampling visible so the 140 historical SFP
   episodes do not silently dominate the 21 SC episodes.

3. **Measure tokenizer reconstruction directly.** On held-out early, approach,
   near-opening, and last recorded frames, save side-by-side original/decoded
   images for each full camera and contact crop. Report whole-image and
   plug/port-region pixel MSE or RMSE, plus a local edge/detail measure.
   Identify crops where the target is absent. Inspect the actual images;
   whole-image MSE can improve while the small insertion feature stays blurred.
   Try a native-resolution contact crop or higher-resolution encoder input as
   a separate ablation if more-data-only training still blurs the port. Check
   the inference size and live latency after any resolution change.

4. **Separate decoder and dynamics errors.** First decode a *true* held-out
   next-frame latent to show the tokenizer's best available reconstruction.
   Then start from the same real observation, feed the recorded executed
   commands into the world model, decode its predicted next latent, and place
   both decodes beside the actual next image. Repeat at strict 200, 400, and
   600 ms horizons only where native observations and complete action windows
   exist; report episode counts per horizon. Compare predicted and observed
   TCP position with Euclidean error in mm and RMSE, orientation with angular
   error in degrees, and force/contact state where valid. Stratify by motion
   magnitude and distance to the opening. Include no-movement persistence and
   shuffled-action baselines. Pose error must compare predicted *future
   measured pose* with the actual future measured pose, never a command target
   with measured pose. A model useful for planning should outperform the
   simple baselines on held-out phases and respond correctly to actual actions.

5. **Test control value before scale-up.** Train the same policy architecture
   and action contract with and without world pretraining, using identical
   supervised data, split, runtime, and development scenes. Compare with the
   corrected TCP-delta ACT baseline. This isolates any benefit of pretraining
   from model size and policy-head differences. If dynamics still loses to
   persistence, keep imagination planning disabled even if supervised control
   improves. Consider predicting state change relative to the current measured
   state and explicitly modeling controller response/contact as a later
   dynamics ablation. Increase model size only after visual and data checks
   identify a capacity limit.

6. **Repair outcome coverage, then consider faster simulation.** Audit raw
   bags for post-event RGB/state before recollecting. If absent, collect a
   short timestamped observation tail after success and failure, with every
   applied command tied to its reference observation. Prefer targeted
   near-opening attempts over repeating long full Gazebo episodes. Isaac can
   later supply parallel expert or RL experience, but first demonstrate
   stable resets/zero-action behavior, matching TCP-delta action and camera
   semantics, and expert insertion on shared scenes. Its reward is not the
   official Gazebo score. Evaluate frozen Isaac-trained policies in Gazebo and
   use Gazebo experience to measure and close the simulator gap.

For each new run, record the exact source revision/diff, data manifest,
checkpoint, scene split, number of native transitions per horizon, reconstruction
sheets, pose-error report, simulator videos, official scores, and real ROS
latency. Keep the existing under-300 ms inference requirement. The pilot's
reward and imagination heads should remain disabled until post-event labels
and held-out dynamics fidelity support them.

# Isaac RL from the full-data world policy

Status: the joint, state, TCP-delta, reward, termination, and 200 ms macro
replay contracts are validated on calibrated SFP port-1/card-0 Isaac scenes.
After correcting a reset-variation bug, the privileged guide inserted from all
eight varied training starts in round one and from seven of eight in round two.
Autonomous supervised heads nevertheless achieved 0/4 strict insertions on
heldout IK starts. Actor Q updates and imagination remain disabled. The
diagnostic checkpoints, commands, and machine-readable summaries are preserved
under `outputs/experiments/2026-09-20_isaac_world_rl/`.

## Varied-scene supervised continuation

The first generated “varied” run was invalid as variation evidence. Its YAML
retained `reset_mode: robot_joint_state`, so the requested Cartesian starts
were silently ignored and every episode used the calibrated joint reset. The
generator now uses `reset_mode: body_start_position_world`, 20 damped IK
iterations, and a 0.25 mm reset tolerance. A 32-step audit measured axial
starts of 4.004--9.982 mm for requested 4--10 mm starts, lateral offsets of
0.614--0.792 mm for requested 0.7 mm offsets, and about 4.48 degrees of initial
orientation error. Earlier repeated-reset results are preserved but are not
reported as varied-scene evidence.

Adaptive collection applies the geometric guide separately to each environment
when lateral or orientation error, or actor/guide disagreement, crosses a
configured threshold. It records the guide chunk as the supervised label and
the commands actually sent as dynamics actions. Round one produced 9/9 strict
insertions, including all eight distinct training starts. Round two, initialized
from the round-one head, produced 7/8 strict insertions; one 300-step timeout is
retained as a failed corrective rollout. The teacher intervened on 91.1% of
round-two physical steps, so this is mostly guided corrective data rather than
evidence of autonomous competence.

The aggregate round-two head used 1,164 macro transitions. It stopped at update
3,900 under the declared validation-patience rule and restored update 2,900
(validation L1 0.00013246). A final controlled ablation removed 262 older
single-scene transitions and fit only the 902 genuine varied transitions. It
stopped at update 4,700 and restored update 3,700 (validation L1 0.00013223).
Both used the same 333,024-parameter head and frozen 22,236,320-parameter trunk.

| Autonomous head | Heldout strict insertions | Closest axial distance | Lateral error at closest approach | Peak reward range |
| --- | ---: | ---: | ---: | ---: |
| Round-one aggregate | 0/4 | -0.128 to +0.178 mm | 0.835--7.246 mm | 0.159--0.255 |
| Round-two aggregate | 0/4 | -0.592 to -0.089 mm | 2.060--10.166 mm | 0.112--0.197 |
| Varied-only round two | **0/4** | **-0.099 to -0.055 mm** | **3.246--3.894 mm** | **0.187--0.229** |

Negative axial values are before the opening plane. The varied-only policy
consistently reached the opening but remained several millimetres off center;
strict success requires lateral error at or below 0.5 mm. All four episodes
reached the configured 35 N force observation clip and then timed out. The
aggregate round-two head was worse than round one, while the varied-only head
improved consistency without completing alignment. Near-port 4x sampling and
a translation-heavy loss were also worse in prior bounded ablations. More
iterations of the same head and loss are therefore not justified by these
results.

`varied_supervised_eval_summary.json` contains per-episode axial, lateral,
orientation, command, measured tip-motion, force, reward, outcome, and timing
summaries. Full replay and simulator metrics are under
`/var/tmp/chmin_aic_20260920_isaac_world_rl/`; exact commands are under the
repository experiment directory. The exact eight training, four development,
and four still-sealed final YAMLs are archived under `configs/`. The
varied-only checkpoint is preserved at
`selected_varied_only/world_policy_online.pt` with SHA256
`3f408f2731d76c100bc54329c3e82ab939faff70ce4300abee1f00f10f4a48ac`.

The development gate remains closed. The four reserved final IK configurations
were not evaluated, no Gazebo transfer was run, and no critic, reward,
model-free RL, or imagination update was enabled. The next supervised design
should expose insertion geometry or an explicit alignment phase to the policy,
use episode-grouped validation, and first beat this 0/4 result on new development
starts. The current transition-level random validation split is useful for
optimization stopping but is correlated within trajectories and must not be
treated as a generalization metric.

## Approved next experiment: object-to-target pose supervision

The next step will test a general object-centric perception target rather than
hardcode an axial/lateral servo into the learned actor. From Isaac ground truth,
derive the plug frame relative to the selected port frame: relative 3D
translation, relative orientation, contact/blocked state, and optional
uncertainty. Axial and lateral errors are task-specific views of this relative
pose. The same representation applies to peg/hole, gripper/grasp, tool/workpiece,
and object/placement tasks.

First freeze the current 384D world representation and train a small diagnostic
probe on existing replay. Ground-truth geometry is a training label only; the
probe must infer it from deployable observations. Split by complete episode and
reset configuration, never by individual transitions. Report millimetre 3D,
axial, and lateral errors, orientation error in degrees, lateral correction-sign
accuracy, contact/blocked classification, and counts by approach/alignment/
contact phase and motion size. Include constant-mean and state-only baselines.
The four reserved final configurations remain sealed.

If heldout geometry prediction is accurate enough to resolve the 0.5 mm success
corridor, compare the existing action-only head with the same head conditioned
on **predicted** relative pose, phase, and uncertainty. Never feed true Isaac
geometry to the deployable actor. Use identical data, update budget, episode
split, commands, and autonomous development starts. Preserve model proposal,
Isaac CheatCode target, executed action, measured motion, contact, and terminal
observation for every transition.

If the frozen feature cannot resolve lateral position, run one visual ablation:
crop the plug/port interaction area from the original high-resolution render
**before** resizing the global image. Existing contact crops were taken after
the 288x256 resize and therefore only enlarge already lost detail. Encode the
new crop with a small shared encoder and rerun the same pose probe. For a
deployable and more general system, crop centers must come from a learned
plug/target locator or another observation-only rule; simulator geometry may
provide locator labels but may not select crops at evaluation. Retain the full
image for task and robot context.

Promotion requires autonomous insertion on new episode-grouped development
starts, acceptable force/contact behavior, and complete inference below 300 ms.
Do not open the reserved final split, start actor-Q training, transfer to
Gazebo, or enable imagination while the supervised autonomous gate remains at
zero. Stop the crop branch if it does not materially improve heldout near-port
pose error; stop the policy branch if geometry accuracy improves but autonomous
lateral alignment does not.

### Causal replay audit and frozen-feature probe

Execution of the approved continuation found that the two earlier 452-row
adaptive replay files cannot support a safe terminal join. Their ordinary
`next_obs` and post-step geometry already belong to the reset episode on all
9/9 and 8/8 terminal rows. They also omit the model proposal, direct causal
geometry, measured plug/TCP endpoint geometry, and the pre-reset terminal
observation. Geometry can be reconstructed from the previous row for 444/452
noninitial rows in each file, but that is insufficient for the requested
causal audit and is not used as new probe training evidence.

The collector now snapshots the causal episode, frozen 384D feature, plug and
TCP geometry, force, model proposal, Isaac CheatCode target, and executed 24D
chunk at every 200 ms decision. A pre-reset hook separately retains terminal
state, all three RGB observations, and plug/TCP geometry. New guide-only data
contains 1,208 decisions over 16 training/calibration reset configurations and
351 decisions over four unseen development configurations. All 1,559 rows have
the complete causal fields. All 24 terminal rows retain the true terminal
snapshot; the ordinary post-step record is visibly reset-contaminated on every
one and is never joined. The four reserved final IK configurations remain
sealed.

The split holds out whole reset configurations: 12 fit configurations, four
stratified calibration configurations, and four new development
configurations. Sampling is uniform over fit configuration. A five-member MLP
ensemble predicts target-frame translation, rotation vector, contact, blocked,
and phase. Ensemble spread plus calibration residual variance supplies the
reported regression uncertainty.

| Heldout development result | Frozen world feature | State only | Constant mean |
| --- | ---: | ---: | ---: |
| 3D translation, median / p95 | 1.073 / 3.060 mm | 12.644 / 20.446 mm | 5.108 / 9.077 mm |
| Axial, median / p95 | 0.564 / 2.303 mm | 9.461 / 16.336 mm | 2.444 / 7.558 mm |
| Lateral vector, median / p95 | 0.688 / 2.233 mm | 7.616 / 12.386 mm | 3.845 / 7.349 mm |
| Orientation, median / p95 | 0.214 / 0.505 degrees | 1.225 / 2.559 degrees | 1.723 / 2.609 degrees |
| Lateral correction direction | 95.4% | 69.0% | 90.2% |

Across 351 heldout decisions from four configurations, the frozen feature's
contact classifier has 1.000 precision and 0.889 recall on nine positive rows.
There are no blocked positives, so blocked precision and recall are
unsupported rather than evidence of a successful classifier. Phase accuracy is
88.6%. Regression uncertainty covers 88.4% and 97.3% of scalar targets at its
nominal 68% and 95% intervals, indicating conservative calibration.

Near the port (`signed_depth >= -3 mm`), lateral vector error is **0.730 mm
median and 2.533 mm p95**, with 93.5% correction-direction accuracy over 183
decisions. The predeclared gate requires at most 0.25 mm median, 0.5 mm p95,
and 90% direction accuracy. The frozen feature therefore fails the resolution
gate and cannot start policy conditioning.

### Bounded native-resolution crop ablation

The one permitted crop ablation is complete. A matched collection retained
native 576×512 RGB for 1,057 decisions across the same 16 training/calibration
configurations and 225 decisions across the same four development
configurations. All 20 terminal rows have a true pre-reset snapshot. A small
shared RGB locator was trained from simulator instance-mask labels, but at
evaluation it receives RGB only. Its predicted plug/target midpoint selects a
160-pixel crop from each original image before any global resize; a shared crop
encoder combines the three crops with the frozen 384D full-image feature.
Isaac geometry and masks never select an evaluation crop.

The locator's heldout median/p95 error is 1.44/7.66 pixels for the plug and
3.11/16.46 pixels for the target proxy over 675 camera views from four complete
development configurations. The matched frozen-feature probe and crop probe
produce the following heldout results:

| Development result, median / p95 | Matched frozen 384D | Frozen 384D + native crops |
| --- | ---: | ---: |
| 3D translation | 1.647 / 3.854 mm | 1.243 / 3.899 mm |
| Axial | 0.825 / 3.347 mm | 0.687 / 3.123 mm |
| Lateral vector | 1.074 / 2.555 mm | 0.816 / 2.222 mm |
| Orientation | 0.293 / 0.543 degrees | 0.234 / 0.585 degrees |
| Lateral correction direction | 92.3% | 90.4% |

Near the port, the matched frozen feature reaches 0.962/2.630 mm lateral
median/p95 with 84.8% direction accuracy. Crops reach **0.851/2.154 mm** with
80.4% direction accuracy. This is only an 11.6% median and 18.1% p95 reduction,
still well outside the 0.25/0.5 mm corridor, and direction accuracy regresses.
The crop ablation therefore fails both the material-improvement decision and
the absolute geometry gate.

| Crop probe slice | Rows | 3D median | Axial median | Lateral median | Orientation median | Direction accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Approach | 110 | 1.288 mm | 0.792 mm | 0.777 mm | 0.235 degrees | 100.0% |
| Alignment | 108 | 1.183 mm | 0.567 mm | 0.851 mm | 0.231 degrees | 80.4% |
| Contact | 7 | 1.294 mm | 0.965 mm | 0.623 mm | 0.165 degrees | 71.4% |
| Motion below 0.25 mm | 69 | 1.177 mm | 0.680 mm | 0.816 mm | 0.258 degrees | 98.6% |
| Motion 0.25--1 mm | 156 | 1.267 mm | 0.688 mm | 0.803 mm | 0.217 degrees | 86.4% |

Overall contact precision/recall is 1.000/0.857 on seven positives. No blocked
positive occurs in development, so blocked precision and recall remain
unsupported. The calibrated ensemble's nominal 68%/95% scalar coverage is
85.8%/96.2%; it is conservative. The state-only and constant-mean matched
baselines have overall lateral median/p95 errors of 5.909/11.272 mm and
5.104/9.369 mm respectively.

The first collection exposed a timing instrumentation ordering bug that wrote
`null` for every actor timing. After fixing the ordering, a minimal 100-decision
recollection on the same development trajectories aligned exactly with the
saved image rows (maximum relative-translation difference 0.0 mm). Complete
world-trunk plus RGB locator, native crop extraction, shared crop encoder, and
pose-head inference is **2.96 ms p50, 3.20 ms p95, and 182.50 ms p99**. Sensor
acquisition is excluded; the periodic world-trunk work causes the p99 tail.
The p95 passes the 300 ms latency constraint.

No conditioned policy was trained because neither representation passed the
predeclared perception gate. Consequently there is no new autonomous policy
rollout to promote. The four reserved final configurations remain unopened;
actor-Q, Gazebo adaptation, reward-model training, and imagination remain
disabled. Exact manifests, commands, checkpoints, predictions, plots, latency,
hashes, and machine decisions are under
`outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/`, starting with
`supervised_continuation_summary.json`, `artifact_map.json`, and
`replay_contract_audit.json`.

### Plain-language interpretation of the crop experiment

#### What the implementation actually does

The crop pipeline has a coarse stage and a detail stage, but it is **not** a
two-stage plug/port position estimator:

1. Isaac renders three 576×512 RGB images.
2. The RGB locator sees a 288×256 copy of each complete image and emits four
   continuous numbers: approximate 2D plug and target coordinates. Although
   these numbers can represent fractions of a pixel, their training labels
   come from integer-pixel instance masks.
3. The midpoint of the two predicted coordinates selects a 160×160 region
   directly from the original 576×512 image. That crop is resized to 128×128
   for the shared crop encoder.
4. The three crop features are concatenated with the frozen 384D whole-scene
   feature. The pose probe directly predicts relative 3D translation, relative
   orientation, contact, blocked state, and phase.

The detail stage does **not** predict refined 2D plug and port coordinates. It
does not use a heatmap, segmentation boundary, multiview triangulation, PnP, or
an explicit camera model. It must learn the final 3D relationship implicitly
from crop appearance plus the global feature.

There is another important label limitation. The locator's plug label is the
median pixel of the visible SFP-module instance mask. Its target label is the
nearest visible NIC-card pixel to that plug centroid. It is a useful
interaction-region proxy, but it is not the exact projected center, corners,
or axis of the port opening. The reported 1.44-pixel plug and 3.11-pixel target
median locator errors are against these proxies. They do not establish
subpixel physical localization of the insertion geometry.

A 0.5 mm physical corridor does not by itself prove that every camera needs
subpixel localization. The conversion depends on distance, focal length, view
angle, and crop resolution. Before choosing a network, render controlled
±0.25 mm and ±0.5 mm plug/port displacements in each lateral direction and
measure their pixel displacement in all three original images. This local
pixel-per-millimetre Jacobian will show whether integer-pixel heatmaps suffice,
whether soft subpixel coordinates are necessary, and which cameras provide
useful geometry near contact. Multiview fusion can also turn several modest 2D
measurements into a more accurate 3D estimate when calibration is sound.

#### Meaning of the millimetre errors

For every heldout camera/state decision, Isaac provides the true plug frame
relative to the selected port frame. The diagnostic predicts the same relative
pose from deployable observations. Translation error is their 3D difference.
Axial error is its component along the insertion axis. Lateral-vector error is
the length of the remaining error in the plane of the port opening.

A **0.851 mm median lateral error** means half of the 225 heldout decision
predictions have at most 0.851 mm sideways error and half have at least that
much error. It describes decisions, not episodes: the decisions come from four
complete heldout reset configurations and are temporally correlated within each
episode. A **2.154 mm p95** means 95% of those decisions have at most 2.154 mm
error and the worst 5% are larger. The configured insertion corridor allows
only 0.5 mm lateral error, so the typical prediction is already outside the
corridor and the tail is much farther outside it. The stricter diagnostic gate
of 0.25 mm median and 0.5 mm p95 asks for useful margin and reliability.

Correction-direction accuracy asks a simpler question: does the predicted
lateral vector at least point to the correct side? The crop probe reaches 80.4%
near the port, so roughly one decision in five recommends the wrong lateral
direction even before control and contact errors are considered.

#### Relevant recent robotics approaches

The following ideas are more suitable than asking a small crop encoder to hide
all geometry inside one feature vector:

1. **True coarse-to-fine keypoints or masks.** Use a whole image to find the
   interaction region. On the original-resolution crop, predict heatmaps or
   segmentation for multiple physical features such as plug-tip corners, port
   corners, and the opening axis. A soft coordinate expectation can return
   continuous coordinates; calibrated views can triangulate them and fit one
   relative SE(3) transform. [RVT-2 (RSS 2024)](https://www.roboticsproceedings.org/rss20/p055.pdf)
   uses 3D observations and keypoint-based action prediction and demonstrates
   plug and small-peg insertion. Its lesson here is explicit spatial prediction
   and 3D representation, rather than its complete language-conditioned policy.
2. **6D pose estimation and temporal tracking.** With usable depth and a CAD
   model or reference views, [FoundationPose (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html)
   estimates and tracks 6D object pose. Tracking across frames can reduce jitter
   and reject individual bad observations. Direct adoption depends on accurate
   runtime depth and complete latency under this project's limit.
3. **Explicit 3D scene representations.** [DP3 (RSS 2024)](https://www.roboticsproceedings.org/rss20/p067.pdf)
   and [3D Diffuser Actor (CoRL 2024)](https://3d-diffuser-actor.github.io/)
   condition policies on point-cloud or other 3D scene features. Calibrated
   multiview RGB could support a small task-specific 3D reconstruction when
   depth is unavailable. A metric 3D representation avoids inferring depth
   independently from every crop.
4. **Make geometry an action input or constraint.** [ReKep (CoRL 2024)](https://arxiv.org/abs/2409.01652)
   represents tasks as explicit relations between 3D keypoints and solves for
   robot poses in a closed perception-action loop. The applicable idea is to
   connect estimated plug-to-port geometry directly to the next target pose or
   policy input, with uncertainty, instead of merely encouraging an opaque
   visual feature to contain geometry.
5. **Use contact to remove residual uncertainty.** Exact visual pose is not the
   only route. [Robust Peg-in-Hole Assembly under Uncertainties (RSS 2025)](https://www.roboticsproceedings.org/rss21/p060.html)
   deliberately uses compliant contact to localize and funnel the peg when
   perception is uncertain. [Efficient Online Learning of Contact Force Models
   for Connector Insertion](https://arxiv.org/abs/2312.09190) learns a small
   online force model and uses optimization to reduce insertion resistance.
6. **Separate slow visual planning from fast contact correction.** [Reactive
   Diffusion Policy (RSS 2025)](https://www.roboticsproceedings.org/rss21/p052.pdf)
   uses a slow action-chunk policy with a faster tactile feedback path. This is
   directly relevant to the current 200 ms four-command policy: approach can
   remain chunked, while alignment/contact may need faster force-aware control.

These papers solve different tasks and use different sensors, so their reported
success rates are not directly comparable with this experiment. They support a
common design pattern: preserve explicit spatial structure, close the loop, and
use contact feedback for what vision cannot observe reliably.

#### Is accurate pose sufficient for insertion?

No. Accurate relative pose is valuable for reaching and pre-contact alignment,
but insertion also depends on camera/robot calibration, action-frame
correctness, controller tracking, latency, compliance, friction, cable motion,
contact state, force limits, and recovery after a blocked motion. A correct
visual estimate can still produce a failed insertion when any of these is
wrong. Conversely, compliant contact and search can sometimes insert despite
an imperfect visual estimate, so exact pose is not universally mandatory.

The current pose probe is even less connected to control than an auxiliary
actor loss: it is a separately trained diagnostic on a frozen feature. The
existing actor was not trained with this loss and never consumes the predicted
pose. If a later actor only shares a pose auxiliary loss, the representation may
become more geometry-aware, but there is no guarantee that the action head uses
that information correctly.

The next credible design should expose the observation-predicted relative pose,
phase, and uncertainty directly to a closed-loop controller or policy:

- use visual relative pose for approach, orientation, and lateral alignment;
- slow or stop when uncertainty is high instead of issuing a blind forward
  chunk;
- re-estimate after each small motion near the opening;
- use measured force, TCP/plug motion, and compliant search or recovery after
  contact;
- retain full-image features for context and ambiguity handling;
- train and evaluate with predicted geometry at the policy input, while keeping
  simulator geometry as supervision only.

This design still needs the perception gate first. With the present 0.851 mm
median, 2.154 mm p95, and 80.4% direction accuracy, explicitly feeding the
estimate to a controller would make its failures easier to diagnose, but would
not make insertion reliable.

### Corrected projected-label continuation

The earlier crop labels were later shown to be a module centroid and nearby
card-mask pixel. The rendered-camera projection was repaired, and projected
plug-tip/opening labels were visually inspected in all three views. Fresh
collection retained 1,057 decisions from 16 training configurations and 225
decisions from four held-out development configurations. The split remained by
complete reset configuration.

The bounded observation-only model used a downsampled full view to locate the
interaction region, a native 160x160 crop and shared heatmap refiner, then
three-view triangulation. Training calibration selected the coarse coordinates
because the refiner regressed. On 108 near-port held-out decisions it obtained
2.380 mm median / 6.947 mm p95 lateral vector error and 82.6% correction-sign
accuracy. Oracle projected points with the same triangulation obtained 0.124 mm
median / 0.235 mm p95 and 100%, isolating the remaining problem to visual
opening localization. Complete measured inference was 2.534/3.311/174.080 ms
p50/p95/p99.

This fails the unchanged 0.25 mm median, 0.5 mm p95, and 90% direction gate.
The approved stop rule therefore prevented pose-conditioned policy training,
actor-Q/RL, Gazebo adaptation, and final-split evaluation. Exact results are in
`pose_probe/perception_rl_continuation_summary.json`; the detailed rationale
and next representation requirement are in the
[perception-to-RL record](2026-09-20-perception-supervised-rl.md#executed-result).

### Opening-boundary and pretrained-feature continuation

The requested next representation was executed on a new 20-configuration
training grid and six fresh development starts. Explicit opening corners were
projected from the measured 14.0 x 8.9495 mm cage opening into all three native
camera images and visually audited. A scratch six-heatmap model failed;
calibration selected its coarse locator and held-out lateral error was
1.085/1.740 mm median/p95.

A calibration-only ImageNet MobileNetV3-small feature-pyramid ablation then
selected the coarse plug and learned opening corners at 0.289/0.531 mm. It
earned one evaluation on six further new starts. There it reached
0.278/0.670 mm lateral error, 94.1% direction accuracy, 0.664/1.363 mm axial
error, and 0.215/0.506 degree orientation error across 173 near-port decisions.
Complete inference was 5.949 ms p95. The median and p95 gates still failed, so
no conditioned policy, actor-Q update, Gazebo adaptation, or final evaluation
was started. The four reserved final configurations remain sealed. Full design,
failure analysis, commands, and artifacts are in the
[perception-to-RL record](2026-09-20-perception-supervised-rl.md#opening-landmark-continuation).

A post hoc causal world-space filter reached 0.250/0.616 mm lateral error and
97.8% direction accuracy, but axial error regressed to 2.085/4.407 mm. It was
not promotion eligible and its p95 still failed, so no further development
collection or control experiment followed.
Heatmap-confidence camera rejection was also calibration-rejected in favor of
all three views and did not open another split.

## Final continuation result

The early failure below was traced to scene placement rather than the expert
joint convention. The supported Isaac YAML placed the port about **552 mm**
from the pose represented by the expert joints. After converting the expert
joint state through the robot root frame, the resulting plug pose agreed with
the expected pose to about **5 mm and 1.24 degrees**. A calibrated near-opening
reset starts 24 mm before the opening and settles at roughly 9--12 N.

A privileged controller then established the physical gate on the calibrated
scene. With the corrected rotation sign it crossed the opening at step 206 and
terminated as a strict success at step 244. Immediately before the terminal
command, signed depth was 7.456 mm, lateral error 0.269 mm, and orientation
error 0.000372 rad (0.021 degrees). Two repeated guide-only episodes terminated
successfully at steps 244 and 516. Copying the Gazebo board transform directly
into Isaac was rejected: the different asset origin/collision frame created
cable contact even though the numeric board pose matched.

The approximately **552 mm** mismatch was the spatial separation between the
target opening in the original Isaac scene and the plug pose implied by the
expert joints. It was not measured as a purely lateral port-frame offset, and
it does not prove that every Gazebo/Isaac transform is correct. The narrower
evidence is that the robot root/joint mapping reproduced the expected plug pose
to about 5 mm and 1.24 degrees on this scene and that small TCP-frame commands
moved in the expected direction. Asset transforms still need individual
validation.

### Isaac CheatCode terminology and scope

The privileged guide is an **Isaac CheatCode policy**. Its implementation mode
is named `cheatcode_transform`, and its central calculation mirrors the rigid
transform in the existing Gazebo `CheatCode.py`: read exact simulator plug and
port poses, calculate the plug-frame correction, map it to the controlled TCP,
and emit an observation-relative TCP-frame delta.

It is a local variant rather than a line-for-line port. Gazebo CheatCode is a
complete trajectory policy with a distant approach, timed handoff and settle,
minimum-jerk insertion schedule, ROS TF and XY-error integration, and Gazebo
completion-event handling. Isaac `cheatcode_transform` is a clipped 20 Hz
near-port feedback servo. It recomputes lateral, axial, and orientation error,
gates forward movement on alignment, and compensates for plug-tip movement
caused by TCP rotation.

This work did not implement or rerun the Gazebo CheatCode. The initial Isaac
validation covered one calibrated SFP-to-NIC scene: one NIC card, target card
0, SFP port 1, and a near-opening reset. The later varied-reset continuation
expanded start-pose coverage for that task, but other ports, card counts, SC
tasks, and full-distance approaches remain unvalidated.

### Training path and fixes

The deployable world-policy path now exposes the frozen **384D causal observed
feature** and stores one transition for each four-command macro action. Replay
contains the actual 24D executed chunk, the four actual guide commands, the
next causal feature, and `gamma^microsteps`. Twin critics consume the full 24D
chunk. An online policy checkpoint can be restored on top of the immutable
full-data checkpoint.

Two silent failures were fixed before accepting training evidence:

- guide replay had repeated one 6D command four times instead of preserving the
  four commands computed during the macro step;
- `action_components()` was under `torch.inference_mode()`, so guide loss was
  nonzero while the actor-head gradient was exactly zero. The corrected smoke
  measured actor gradient norm 0.169368.

The frozen perception/world trunk has 22,236,320 parameters and the trainable
policy head has 333,024. The focused trainer fits only stored causal features
and 24D guide chunks, avoiding image decoding during every replay update.

Each decision has three distinct commands: the model proposal, the Isaac
CheatCode target, and the command actually sent to Isaac. Replay stores the
unblended CheatCode chunk as the supervised BC target and the executed chunk
as the action that caused the next state. The model proposal is logged and is
used to form the executed command during DAgger. Future dynamics and critics
must condition on the executed command; imitation should predict the teacher
target.

### Bounded supervised results

| Policy-head data | Macro transitions | Held-out chunk L1 | Autonomous result on the calibrated scene |
| --- | ---: | ---: | --- |
| Guide-only successful episodes | 262 | 0.000174 | 0 insertions; peak reward 0.038, final -0.064 |
| + 50% guide-blend DAgger | 388 | 0.000162 | 0 insertions; peak 0.098, final -0.043 |
| + 25% guide-blend DAgger | 513 | 0.000164 | **0 insertions**; peak **0.158**, final **0.091**, timeout at 300 steps |
| 25% blend data only | 125 | 0.000176 | 0 insertions; peak 0.062, final -0.054 |

The 50% blend collector achieved two strict insertions in 500 physical steps.
The 25% blend stayed near the opening but timed out. DAgger improved the best
autonomous reward trajectory, yet no zero-guide controller inserted. Small
held-out action error therefore did not predict closed-loop success.

These percentages are per-command interpolation, not alternating controller
steps. With guide fraction `alpha`, every executed command is
`(1 - alpha) * model + alpha * guide`. The successful 50% collection therefore
used half of each model command and half of each CheatCode command at every
step. The 25% collection used 75% model and 25% CheatCode at every step and
completed no insertion in its 500-step budget.

### Isaac reward definition

The reported reward is Isaac shaping and diagnostics, not the official Gazebo
challenge score. Every 50 ms it combines smooth and close-range distance to
the configured full-depth target, progress toward that target, orientation
reward gated by lateral alignment, lateral-error penalty, one-time success
bonus, force-change penalty, and small joint/action smoothness penalties. Force
change is unpenalized below about 10 N, rises gently to 20 N, becomes steeper
above 20 N, and saturates near 30 N.

Relative configured weights were 0.25 distance, 0.35 close distance, 0.25
progress, 0.10 orientation, 1.0 terminal bonus, -0.05 lateral error, and 0.30
force-change penalty. Isaac applies time-step scaling, so these are not direct
reward points per recorded row. Termination required axial error within 0.5 mm
of the approximately 8 mm full-depth target and lateral error within 0.5 mm.
Orientation was measured and 0.03 rad was used as the strict diagnostic
reference, but this run did not explicitly require it for termination; the
successful guided rows were nevertheless far better aligned. Supervised BC
optimized CheatCode imitation rather than reward. Reward supported diagnostics,
success detection, replay, and the later RL readiness decision.

The 513-transition aggregate checkpoint is preserved as the best diagnostic,
not as a promoted controller:

- `outputs/experiments/2026-09-20_isaac_world_rl/selected_dagger2/world_policy_online.pt`;
- guide-success and autonomous-evaluation videos under
  `outputs/experiments/2026-09-20_isaac_world_rl/videos/`;
- exact collection, training, and evaluation commands under
  `outputs/experiments/2026-09-20_isaac_world_rl/commands/`;
- full replay, metrics, commands, and additional videos under
  `/var/tmp/chmin_aic_20260920_isaac_world_rl/`.

### Final gate decision

Do not start actor Q optimization from this replay. It contains only a handful
of successful policy-distribution episodes, and no autonomous supervised
insertion establishes a safe anchor. Do not use world-model imagination: the
held-out dynamics remains worse than persistence. The next useful data step is
to collect more partially guided successful trajectories across calibrated
scene variations, retain terminal observations before auto-reset, and train a
phase-aware or geometry-supervised correction head. Re-evaluate autonomously
on held-out calibrated Isaac scenes before critics, then evaluate the frozen
candidate in Gazebo before any Gazebo adaptation.

Only physical GPU 0 was used. Warm deployment timing established earlier for
the complete policy path remains below the 300 ms requirement; per-call CUDA
metrics in these runs are asynchronous and are not used as a new p95 claim.

### Verification

`py_compile` passed for the trainer, wrapper, world-policy actor, focused head
trainer, and launcher test. `git diff --check` passed. The Isaac launcher test
file passed 15 tests with the one multi-GPU sharding assertion deselected. The
complete file reports the same 15 passes plus that one environment-dependent
failure because this container exposes one GPU and therefore creates one
five-episode shard instead of the test's expected two shards of three and two.

## 2026-09-20 execution record

The first fresh-container attempt failed because it omitted the repository's
documented container-local unsupported-driver workaround. This was an
integration oversight, not a host-driver change. The same host and image had
previously rendered cameras after that explicit workaround.

Implemented and checked:

- direct loading of the selected full-data world-policy checkpoint, without an
  ACT proposal or residual adapter;
- exact six-view RGB144 preprocessing through the Dreamer camera-view code;
- a 288x256 Isaac camera render contract matching the Gazebo input aspect. The
  legacy square 224x224 render would be rejected by the exact preprocessor;
  this correction is implemented but remains unverified in rendering because
  the driver failure occurs first;
- the base32-plus-task10 state contract and 24D four-command action contract;
- causal policy cache reset and previous-command history handling;
- fail-closed launcher checks requiring one environment, four executed
  commands, zero updates, 20 Hz, and no guide, guard, exploration or teacher
  changes;
- deterministic CPU checkpoint checks: output shape `(1, 24)`, finite output,
  bit-identical output after reset, and a changed output when command history
  is retained;
- a rootless Isaac 5.1 container on physical GPU 0. The checkpoint loaded, the
  environment was created, and a single environment reset completed;
- a successful eight-step frozen rollout after applying the existing
  `patch_isaac_rtx_driver_check.sh` inside the disposable container. It saved
  nine 288x256 frames per camera and three videos, issued two 24D chunks as
  eight physical commands, and performed zero gradient updates.

Host driver `535.104.05` still fails Isaac's normal RTX version check. As in the
September 17 validation, the retry explicitly set
`AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1` and ran
`patch_isaac_rtx_driver_check.sh`, which backs up and changes only the fresh
container's driver-requirements file. It does not reinstall or upgrade the
host driver. The current Isaac Sim 5.1
[requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
still list a substantially newer tested Linux driver, so every camera run on
this host remains an unsupported configuration and needs a camera smoke test.

The patched retry exposed and fixed a diagnostic-only state-normalizer bug:
the world policy has saved statistics for base32 and identity-normalizes the
task10 suffix, while the legacy diagnostic indexed 42 saved statistics. The
controller itself already used the correct base32-plus-task10 contract.

Preserved evidence is under
`/var/tmp/chmin_aic_20260920_isaac_world_rl/`:

- `execution_summary.json` records the checkpoint hash, passed gates, exact
  failure stage and artifact paths;
- `isaac_5_1_driver_failure.log` is the Isaac Kit log;
- `zero_shot_8step_v2/online_serl_plan.json` is the resolved launcher plan;
- `zero_shot_8step_v2/2026-09-20_02-21-21_world_policy_zero_shot_8step/train_config.json`
  is the simulator-side resolved configuration;
- `zero_shot_8step_v4/2026-09-20_02-48-29_world_policy_zero_shot_8step_patched/`
  contains the successful smoke metrics, audit log, all frames and three
  camera videos.

The smoke issued policy actions and performed no gradient update. It did not
authorize RL because the default scene lacked episode entrance/axis metadata
and the old replay path stored each 50 ms command separately. The follow-up
below resolved the replay schema and exposed simulator-contract failures.

## Contract-gate results

The preserved machine-readable summary is
[`contract_gate_summary.json`](../../outputs/experiments/2026-09-20_isaac_world_rl/contract_gate_summary.json).
Full metrics, frames, videos and replay files remain under
`/var/tmp/chmin_aic_20260920_isaac_world_rl/`.

### Macro replay passes

The world-policy path now stores one transition per four executed 20 Hz
commands. An eight-tick zero-command validation produced exactly two replay
items. Each item has a 24D action, four microsteps, discount
`0.99^4 = 0.96059601`, and the expected discounted reward sum. The launcher
also rejects an episode-configured world-policy run when near-gate reset
iterations are zero. Causal frozen policy features and complete guide chunks are now stored. The
subsequent supervised head updates are reported above; actor Q updates remain gated.

### Matched frozen rollouts fail

The first Stage C probe used port 0/card 0. That task is outside the supervised
agent manifest: the 148 dynamics/agent-eligible episodes comprise 127 SFP
port 1/card 0 episodes and 21 SC episodes. The wider verified dataset still
contains 289 episodes; this narrower count is a dynamics/action-history
eligibility constraint.

On the port 0 Stage C scene, a zero command drifted the plug tip 2.71 mm in one
second and reached the 35 N observation clip. The frozen policy moved the tip
123.04 mm, ending at 125.51 mm lateral error. Its mean translation command was
12.86 mm per 50 ms command, which is physically plausible for this dataset
but points in an unusable direction under the Isaac observation/rendering
contract.

A second probe used a near-gate port 1/card 0 configuration represented in the
agent training data. It also failed: after 20 ticks the frozen policy ended at
119.35 mm lateral error, 26.34 mm behind the opening, and the 35 N force clip.
The live warmed decision measurements were 201.7, 204.1, 229.6 and 205.2 ms;
all are below 300 ms, but four samples are insufficient for a live p95 claim.
The separate 1,000-call checkpoint benchmark remains 29.06 ms p95 and excludes
simulator capture and transport.

### Observation and reset contract fail

The supported port 1/card 0 reset reached its requested TCP pose to 1.10 mm,
but the policy observation was far outside its training distribution. Initial
normalized values included TCP y at -7.89 standard deviations, shoulder pan
at +8.80, wrist 1 at +3.10, wrist 2 at -2.40 and wrist 3 at +5.76. Seeding the
Isaac reset with expert terminal joint medians did not select an equivalent
joint convention or IK branch. This must be resolved with an explicit
Gazebo-to-Isaac base-frame and joint-convention audit; silently clipping or
adding an action residual would hide the error.

Small insertion-axis probes confirmed the translation sign and frame mapping:
four +0.25 mm commands advanced axial depth by about 1.00 mm, while the
opposite command reduced depth by about 0.42 mm. Both directions nevertheless
reached the 35 N force clip. The analytic corridor and phase-reward audits
passed their existing geometry checks, but a privileged scripted controller
has not yet demonstrated a valid insertion from these resets. The live reward,
contact and termination gate therefore remains failed.

### Initial decision after the first gates (superseded)

Do not start Isaac RL from these rollouts. First reproduce the Gazebo
`base_link` state and joint conventions in Isaac, use a collision-free reset
whose zero-command drift and force are acceptable, and validate scripted
insertions and terminal labels. Then expose and store the frozen observed
feature at each causal decision boundary so a fresh 24D critic and the policy
head can be trained without backpropagating through replayed temporal caches.
Repeat the frozen matched probe before enabling any update.

## Starting point

The selected controller is
`/var/tmp/chmin_aic_20260919_full_world/full_control.pt`, SHA256
`36e9cb8951473642d16a8ecae7430dee69352e9163bd1db4611053766788168d`.
The copy at
`/var/tmp/chmin_aic_20260918_dreamer60/full_data_control_20260919.pt` is
byte-identical and has the same source checkpoint metadata. Therefore the
completed frozen 20-scene result belongs to the current full-data controller:

- 2/20 official full insertions;
- 9 official partial insertions;
- mean official score 45.14;
- live observation-to-command latency 64.27 ms median, 80.77 ms p95, 98.29 ms
  p99 and 146.74 ms maximum, with no decision at or above 300 ms.

The corresponding ACT comparator had no full insertions in the matched
comparison. This makes the current controller the strongest learned starting
point in the preserved evidence, but does not establish that learned predictive
dynamics caused the gain. Six-view perception, contact crops, architecture and
the larger supervised dataset also changed.

## Decision

Use Isaac Lab for high-throughput **model-free** RL and Gazebo for short
high-fidelity adaptation and official-style evaluation. Transfer the tokenizer
encoder, observed-feature transformer and supervised full-action policy. Do not
use predicted latent rollouts, the decoder, the learned reward/value heads or
the failed dynamics model to create training targets.

The dynamics checkpoint failed every predefined fidelity gate and was much
worse than persistence at 200/400/600 ms. Reward and imagination training from
that model remain disabled.

## Learning architecture

Use an asymmetric off-policy actor-critic:

- **Actor inputs:** the same three RGB cameras plus three deterministic contact
  crops, deployable 32D robot state/wrench, canonical task vector, elapsed time
  and actual previously executed commands used by the Gazebo controller.
- **Actor output:** the complete four-by-six observation-relative TCP/body
  delta chunk. There is no ACT proposal and no action-residual adapter.
- **Critics:** fresh twin Q networks and target networks. During training they
  may also consume privileged plug/port geometry, contact and force signals.
  Privileged fields must never enter the deployable actor.
- **Actor retention:** initialize from the supervised policy, begin with the
  tokenizer and observed-feature transformer frozen, and retain verified
  Gazebo behavior with an interleaved supervised BC objective. Unfreeze only
  late representation blocks after a controlled ablation shows a need.

The existing Isaac SERL implementation is the preferred base because it has
replay, twin critics, cameras, strict insertion geometry, episode accounting,
diagnostics and checkpointing. PPO remains a backup baseline.

## Required action-time contract

The deployed policy makes one decision every 200 ms and emits four physical
20 Hz commands. Isaac must preserve that contract:

1. infer one 24D chunk;
2. execute its four 6D commands at 20 Hz;
3. accumulate discounted rewards across the four microsteps;
4. interrupt immediately on termination or unsafe contact;
5. store one replay transition from decision observation to the next decision
   observation, with the complete 24D chunk as the critic action.

The world-policy collection path now satisfies this storage contract and was
validated with two complete transitions. Replay now retains causal frozen features and supports differentiable 24D
policy-head training. The final result above keeps actor Q updates disabled
because the supervised head did not insert autonomously. A one-command replanning variant is a separate later ablation because
it changes deployment cadence and may violate the measured live latency budget.

## Execution gates

### Gate 1: checkpoint and preprocessing parity

- Load the selected control checkpoint without translating it into a legacy
  ACT/SERL checkpoint.
- Verify six-view RGB144 preprocessing against the Gazebo runtime on fixed
  tensors.
- Verify base-state normalization, positive-scalar quaternion convention,
  task vector, elapsed-time scaling and previous executed-command history.
- Verify identical deterministic physical action chunks within numerical
  tolerance.
- Add batched per-environment history and reset masks. Never share history
  across Isaac environments or across episode resets.

### Gate 2: Isaac/Gazebo simulator contract

On matched task YAMLs, record and compare:

- camera order, image statistics, crops and encoder latent statistics;
- requested and settled TCP/plug/module poses;
- zero-action drift;
- commanded TCP delta versus measured TCP/plug/module motion;
- port axis, lateral offset, orientation, contact and success signals;
- controller scale, sign, frame and 20 Hz timing.

Historical Isaac runs showed reset drift and controller/geometry discrepancies.
No RL run may start until zero-action, reset and action-direction probes pass.

### Gate 3: reward and termination

Use phase-gated geometry rewards: lateral and orientation progress before
insertion, axial progress only inside the alignment gate, module-consistent
terminal insertion, and penalties for off-axis forward motion, withdrawal,
force, forbidden contact, action rate and jerk. Tip-only depth is never a
success condition.

A privileged scripted controller must achieve physically valid insertions and
the reward/success signal must reject known bypass cases before policy updates.
Terminal observations and outcomes must be captured before automatic reset.

### Gate 4: bounded Isaac RL

Start nominal with a small number of environments and no representation
updates. Train fresh critics before enabling a small actor Q loss. Use low
exploration variance and a strong BC anchor, then decay the anchor only after
measured insertion improves.

Curriculum order:

1. aligned starts 2--5 mm before contact;
2. near-port starts with controlled lateral and angular error;
3. mid-range approach states;
4. full starts across task families and card-count settings.

Promote only on held-out Isaac episode configurations using strict insertion,
force and prohibited-contact metrics. Preserve a frozen supervised-policy
baseline and evaluate both with exploration disabled.

### Gate 5: calibrated sim-to-sim transfer

Measure paired Isaac/Gazebo differences, then randomize Isaac over the measured
range for lighting, exposure, color, camera pose/intrinsics, placement,
friction, contact stiffness, cable properties, controller response, latency
and sensor noise. Do not use unconstrained heavy randomization that removes
small connector details.

Evaluate the selected Isaac policy without updates on new Gazebo development
scenes. The completed 20-scene set is evidence only and cannot select future
checkpoints.

### Gate 6: short Gazebo adaptation

Keep the transferred actor, freeze its encoder initially, and initialize fresh
Gazebo critics because Isaac Q values are domain-specific. Collect recent
Gazebo replay with privileged geometry used only by rewards/critics. Train
critics first, then permit low-learning-rate actor updates with the verified
expert BC anchor. Stop on regression in full/partial insertion, official score,
force or prohibited contacts.

Compare on the same fresh Gazebo development scenes:

1. frozen supervised controller;
2. Isaac-trained controller before Gazebo adaptation;
3. the same controller after Gazebo adaptation.

Keep a same-architecture, no-world-initialization supervised comparison when
compute permits. That comparison is needed to attribute gains to pretraining.

## Implementation map

- Isaac launcher: `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- Isaac trainer: `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- Isaac environment/rewards:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/`
- Gazebo adaptation: `aic_utils/gazebo_rl/`
- Gazebo runtime: `aic_example_policies/aic_example_policies/ros/RunDreamerV4.py`
- World-policy source:
  `/data1/chmin/yj/ws_aic/src/dreamer-v4-aic-20260918/dreamer4/aic/`

The Isaac trainer now supports full-world-policy inference, exact 200 ms macro
replay, detached causal-feature replay, differentiable 24D policy-head updates,
and restoration of a trained online head. The focused offline head trainer is
`train_world_policy_head_offline.py`. Gazebo SERL loading still needs the same
actor type before sim-to-sim adaptation; it must not recreate the removed
residual-action design.

## Frozen-pose GRU continuation result

The later natural-cable estimator was deliberately frozen and used in the
approved matched supervised diagnostic. Identical 546,686-parameter six-step
GRUs were trained with pose/uncertainty/phase present or zeroed. The action-only
arm was marginally better on held-out action imitation, and both arms achieved
**0/8 strict insertions** on new autonomous starts. Best lateral p50/p95 was
1.479/1.678 mm without pose and 1.417/1.871 mm with pose. Both drifted to about
6.3 mm median terminal lateral error. A counterfactual audit showed that
zeroing pose changed held-out translation commands by only 0.0095/0.0264 mm
p50/p95: the generic BC head largely ignored the new signal. Complete p95
inference was 7.388 ms.

The branch therefore stopped under its declared rule. The four reserved final
starts remain sealed; actor-Q, Gazebo adaptation, reward learning, and
imagination were not started. See the complete
[execution record](2026-09-22-pose-conditioned-gru-policy.md).

## Later RPDP SERL continuation result

A later deterministic port-trajectory BC policy passed the local supervised
gate, so a separate bounded model-free continuation was authorized. It used a
four-component full-trajectory mixture, balanced prior/online replay, twin
critics, force plus measured-stall recovery detection, and measured-path
backtracking. The critic and actor consumed causal observations and executed
24D chunks; simulator geometry remained reward/diagnostic data and never actor
input.

Frozen BC scored 3/8 local 8 mm seating events on new starts. Initial SAC actors
scored 0/8 because deterministic policy mass moved onto unvalidated mixture
modes. Full-mixture behavior regularization and an optimizer-restore learning
rate fix produced a tight-trust actor that scored the same 3/8 on the same
episodes. It retained BC but did not improve it. Revised backtracking saw no
qualifying blockage in its five-episode sample. The reserved final split,
Gazebo adaptation, reward learning, and imagination remain closed. See the
[probabilistic SERL experiment](2026-09-22-serl-mixture-recovery.md).

## Resource and evidence rules

- Preserve all current checkpoints, reports and frozen rollouts.
- Do not tune on the completed final scenes.
- Save resolved commands, source/checkpoint hashes, environment configuration,
  reward components, replay schema, videos and strict outcome records.
- Keep deployable actor latency below 300 ms and report live p50/p95/p99.
- Use the rootless Docker socket at `/run/user/1008/docker.sock`; no sudo.
- Begin with contract and reward probes. GPU training starts only after their
  recorded gates pass.

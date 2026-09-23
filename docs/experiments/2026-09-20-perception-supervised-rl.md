# Perception, supervised control, and model-free RL

**Status:** perception continuation complete; gate failed. Supervised
pose-conditioned policy training and RL were therefore not started. World-model
planning and SEER adaptation remain parked. The four reserved final Isaac
configurations remain sealed.

**2026-09-22 decision:** further pose-estimator improvement is parked after the
natural cable experiment reached 0.322/0.626 mm lateral median/p95 and its error
audit isolated scene-dependent learned-landmark error. The checkpoint may be
frozen for a matched supervised controller diagnostic. This is a deliberate
diagnostic relaxation of the perception prerequisite, not a retrospective gate
pass. Model-free RL remains gated on autonomous insertion by the supervised
controller on new episode-grouped development starts.

## Why this direction

The insertion problem has two coupled requirements:

1. resolve the plug relative to the selected opening at submillimetre scale;
2. react to contact, cable slack, cable tension, neighboring cards, and blocked
   motion over time.

A current-frame pose estimate is necessary but cannot by itself establish a
safe insertion. The controller must close the loop using new images, measured
motion, force, previous commands, and short observation history.

### Why not pure RL from scratch

Pure visual RL would spend substantial simulator experience rediscovering
basic reaching and alignment already present in verified demonstrations. It is
also prone to exploiting errors in simulated cable/contact behavior. Roughly
1,000 mostly successful episodes would provide weak coverage of blocking,
retreat, rerouting, and recovery. The selected plan starts from supervised
behavior and introduces RL only after the observation-only actor can insert on
new development starts.

RL remains useful after that gate. Isaac can provide many intervention and
failure transitions, and a simulator-only critic may use exact geometry,
contact, and cable state to make learning more efficient. These privileged
values never enter the deployed actor.

### Why not use the current world model

The audited dynamics model predicted future measured TCP state worse than a
no-change baseline at 200, 400, and 600 ms. Its superior supervised controller
score does not isolate a dynamics benefit because the representation,
architecture, cameras, and training data also changed. Predicted latents,
decoded futures, reward learning, and imagined rollouts therefore remain out of
the control path.

The useful parts are retained only as fixed current-observation features where
they win a controlled comparison. No learned future is treated as evidence of
cable motion.

### Why not adapt SEER now

[SEER](https://arxiv.org/abs/2412.15109) is a supervised predictive
inverse-dynamics policy pretrained on DROID robot demonstrations. Its standard
model has 316M total parameters, including 251M frozen and 65M trainable. This
is broad visual, temporal, and robot-action pretraining, but the paper provides
no cable manipulation or obstruction result. SEER predicts a goal-conditioned
future visual representation and the intervening actions; it is not an
action-conditioned cable simulator that evaluates candidate actions.

SEER is a plausible later initialization comparison. Using it now would change
model size, representation, and action architecture before establishing whether
the present bottleneck is target localization, pose reconstruction, supervised
control, or cable interaction. It is parked with the generative world model so
the next result has a clear cause.

## Approved staged experiment

```text
original-resolution RGB
          |
          v
observation-only plug/opening locator and interaction crops
          |
          v
relative translation + orientation + contact/blocked uncertainty
          |
          v
temporal supervised actor using image features, pose, force, state,
previous executed commands, and task identity
          |
          v
complete four-by-six TCP-frame delta chunk
          |
          v
autonomous held-out development insertion gate
          |
          v
model-free actor-critic RL, only after the gate passes
```

### Stage 1: perception

- Use native images before the global 288x256 resize.
- Train the coarse locator with simulator labels, but select every evaluation
  crop from RGB predictions only.
- Refine plug and opening keypoints inside the native crop and expose their
  coordinates explicitly to the pose head.
- Use complete reset configurations as the split unit.
- Report translation, axial, lateral, and orientation error; correction sign;
  contact/blocked precision and recall; uncertainty; and live p50/p95/p99.
- Require near-port lateral median at most 0.25 mm, p95 at most 0.5 mm, correct
  lateral direction at least 90%, and p95 latency below 300 ms.

The first native-crop experiment failed this gate at 0.851 mm median and 2.154
mm p95. Its locator supervision used the SFP module centroid and the nearest
visible NIC-card pixel. That is a coarse region label rather than the semantic
plug tip and port opening. The collection code also contained a rendered-camera
projection helper, but referenced an obsolete Isaac Lab camera field and
silently returned no projections. The continuation first repairs and visually
audits this label provenance.

An explicit-coordinate ceiling test confirmed that this was a label problem,
not merely a weak image locator. Giving the pose probe the old mask-derived
coordinates directly produced **1.132 mm** near-port lateral median error and
**3.141 mm** p95; coordinates predicted from RGB produced **1.133 mm** and
**3.215 mm**. Their nearly identical failure means that more training on those
labels would not establish submillimetre plug-to-opening geometry. The exact
results and checkpoints are under
`outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/explicit_keypoints_v1/`.

The projection path now reads the rendered camera pose and ROS optical-frame
quaternion from Isaac's camera sensor data. All three cameras retain their
latest pose, intrinsic matrix, and projected plug-tip, port-entrance, and seated
target coordinates. A three-camera overlay was inspected before starting the
replacement collection: the plug-tip and opening markers land on the intended
features. The audit image is
`outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/projected_label_smoke.png`.
The failed projection diagnostics remain archived with the successful smoke
command so that the camera-convention repair is reproducible.

The replacement train collection ran from
`commands/run_pose_probe_projected_train_collect4200.sh`. Its held-out
development collection used
`commands/run_pose_probe_projected_development_collect900.sh`. The probe
located both points from RGB in each view and triangulated their relative 3D
position using rendered-camera calibration. Simulator plug/port geometry was
training supervision only; no true geometry selected a crop or entered an
autonomous actor.

### Stage 2: supervised closed-loop policy

- Initialize from verified supervised behavior.
- Predict the full TCP-frame delta action, never a residual adapter.
- Condition the policy on predicted pose and uncertainty, full image context,
  force/TCP state, recent observations, and previous executed commands.
- Use guide geometry only to label training data. Autonomous evaluation has no
  guide, guard, exploration, simulator geometry, or geometry-selected crop.
- Promote only after strict autonomous insertion on new development starts with
  lateral error at most 0.5 mm, explicit axial/orientation results, acceptable
  force/contact, and p95 latency below 300 ms.

### Stage 3: model-free RL

- Start with aligned 2--5 mm pre-contact states, then add lateral/angular error,
  mid-range approaches, more cards, and cable obstruction/recovery.
- Keep the actor observation-only. A fresh twin-Q critic may receive privileged
  simulator geometry, cable/contact state, and force during training.
- Train critics first. Enable a small actor Q objective only with a strong
  supervised anchor and low exploration.
- Randomize measured ranges of cable slack, stiffness, damping, friction,
  placement, card layout, cameras, controller response, and latency.
- Compare every candidate with its frozen supervised parent on identical unseen
  development starts. Stop if insertion, force, prohibited contact, or latency
  regresses.

### Stage 4: transfer

Evaluate the frozen Isaac candidate on new Gazebo development scenes. Only a
candidate that preserves the autonomous insertion gate may receive short Gazebo
critic training and low-rate actor updates. World-model imagination, learned
reward models, and the reserved final split remain disabled.

## Evidence and stop rules

Every run must retain its commands, source diff, scene manifest, causal replay,
checkpoint, metrics, latency, and rollout videos. Failures and matched baselines
are permanent evidence. A failed perception or supervised gate closes the RL
branch; it does not authorize relaxing the success corridor or adding privileged
actor inputs.

## Executed result

The corrected collection contains **1,057 training decisions from 16 complete
reset configurations** and **225 development decisions from four separate
configurations**. Twelve training episodes fit the models; four other training
episodes selected checkpoints, calibrated the fixed transform, and selected
coarse versus refined coordinates. The development episodes were evaluated
once. The four reserved final configurations were not opened.

The saved label montage confirms the projected plug-tip and port-opening points
land on the intended image features through changing cable occlusions. The
coarse full-view locator generalized well for the plug tip, at **0.569 px
median / 1.896 px p95**, but poorly for the small opening, at **4.627 px /
19.165 px**. The native 160 px heatmap refiner improved neither calibration nor
the final pose result, so the calibration-only selection rule chose the coarse
coordinates. This choice was fixed before reading development pose scores.

On the 108 near-port development decisions, the deployable observation-only
result was:

- 3D translation error: **4.551 mm median, 10.819 mm p95**;
- axial error: **3.650 mm median, 8.420 mm p95**;
- lateral vector error: **2.380 mm median, 6.947 mm p95**;
- orientation error: **0.246 degrees median, 0.572 degrees p95**;
- lateral correction direction: **82.6%**.

The oracle projected-keypoint ceiling passed the lateral perception gate on the
same rows: **0.124 mm median, 0.235 mm p95, 100% direction accuracy**. Its axial
error was 0.269 mm median / 0.568 mm p95. Thus the images and triangulation can
represent the required lateral geometry, but this bounded RGB locator does not
reliably identify the opening under held-out cable layouts and occlusion.

There were no positive contact or blocked examples in the 108 near-port rows,
so their reported all-negative accuracy is not evidence of useful event
detection. Complete measured perception plus frozen-trunk inference was
**2.534 / 3.311 / 174.080 ms p50/p95/p99**; p95 satisfies the 300 ms limit.

The predeclared perception gate required at most 0.25 mm median, 0.5 mm p95,
and 90% direction accuracy. It failed by a large margin. In accordance with the
approved stop rule, no pose-conditioned policy, actor-Q update, other RL,
Gazebo adaptation, reward model, imagination run, or final-scene evaluation was
started. The next experiment needs a materially better observation-only opening
representation, likely explicit opening-boundary or corner supervision and
temporal occlusion handling, before control training resumes.

Exact commands are in `outputs/experiments/2026-09-20_isaac_world_rl/commands/`.
Metrics, the 2.84 MB checkpoint, label audits, and the machine-readable decision
are under `pose_probe/triangulated_projected_v2_selection/` and
`pose_probe/perception_rl_continuation_summary.json`. Bulk PNGs and replays stay
under `/var/tmp/chmin_aic_20260920_isaac_world_rl/`.

## Opening-landmark continuation

**Status:** complete; the perception gate failed. The earlier four development
configurations remained historical evidence and were excluded from selection
and evaluation. The four reserved final configurations remain sealed.

The next bounded representation replaces an invisible center-only target with
explicit SFP-cage geometry. The existing offline collision audit measures the
opening as 14.0 x 8.9495 mm. Collection now projects its four physical corners,
the opening center, and plug tip into each original 576x512 image using the
rendered camera pose. A smoke overlay confirms that the projected cyan
quadrilateral follows the visible cage opening in all three cameras.

The model uses:

1. the downsampled full image for a coarse interaction-region estimate;
2. an original-resolution 160x160 crop selected by that RGB estimate;
3. six dense heatmaps for plug tip, opening center, and four opening corners;
4. a calibration-selected causal exponential tracker for temporary cable
   occlusion;
5. per-observation camera calibration for three-view triangulation.

Simulator geometry supplies training labels only. Development crops and
tracking use RGB predictions. Twenty new train configurations cover five axial
distances and four signed lateral offsets. Four complete train episodes are
reserved for checkpoint, representation, and smoothing selection. Six new
development starts are evaluated once after selection. The unchanged
0.25/0.5 mm lateral and 90% direction gate still controls whether supervised
policy work resumes.

The run retained 1,355 causal decisions from all 20 train configurations and
379 decisions from six new development configurations. Calibration selected
the original coarse plug and opening predictions with no temporal smoothing;
it rejected every explicit landmark alternative. On 240 near-port development
decisions, the selected model reached **1.085 mm median / 1.740 mm p95 lateral
error**, 91.5% correction-direction accuracy, **1.204 / 2.928 mm axial error**,
and 0.211 / 0.363 degree orientation error. Exact projected points with the
same per-observation calibration had effectively zero numerical triangulation
error, so labels and geometry were not the bottleneck.

The detailed heatmaps had median/p95 image errors of 3.15/16.74 pixels for the
plug, 3.41/17.40 for the opening center, and 2.17--7.32/9.00--21.41 across the
four corners. Occlusion tails made them less useful than the coarse locator.
A post hoc world-coordinate temporal-filter diagnostic improved lateral error
to 0.862/1.505 mm and direction accuracy to 94.0%, while worsening axial error
to 2.226/4.137 mm. Because that filter was proposed after this development set
was opened, it is diagnostic only and cannot promote a model.

Selected complete inference, including the measured frozen trunk, was
1.406/1.982/58.099 ms p50/p95/p99. The latency limit passed, but geometric
accuracy did not. No pose-conditioned policy, actor-Q training, Gazebo
adaptation, reward-model training, or imagination was started.

The first training invocation under `opening_landmarks_v1/result/` completed
optimization but hit an indexing bug while applying the target-point smoother,
before writing metrics or a checkpoint. The corrected implementation smooths
only the target coordinates; the identical rerun is preserved under
`result_v2/`. The incomplete montage remains as infrastructure-failure
evidence and is not presented as a scored run.

### Pretrained spatial-feature follow-up

A final bounded check replaced the scratch landmark network with an ImageNet
pretrained MobileNetV3-small feature pyramid. The crop stayed 160x160 and the
split, labels, heatmap objective, triangulation, and coarse locator stayed
fixed. The spatial model has 233,662 parameters because only MobileNet stages
through the 10x10 feature map and a small pyramid head are used. This adds about
1.1% to the 22.24M-parameter frozen world trunk. Initialization used torchvision
`MobileNet_V3_Small_Weights.IMAGENET1K_V1`; the cached official weight file has
SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

This check first used only the existing fit and four calibration episodes.
Calibration selected the coarse plug plus predicted opening corners and reached
0.289 mm median / 0.531 mm p95 lateral error with 90.8% direction accuracy.
That was sufficiently close to justify one evaluation on six newly generated,
complete reset configurations. No earlier development scene was used for
selection.

On 173 near-port decisions from those six new episodes, the fixed model reached
**0.278 mm median / 0.670 mm p95 lateral error**, **94.1%** correction-direction
accuracy, 0.664/1.363 mm axial error, and 0.215/0.506 degree orientation error.
The held-out overlay shows close corner estimates in clear views, with the
remaining tail concentrated in small or partially occluded openings. Complete
inference was **5.450/5.949/71.825 ms p50/p95/p99**, well below 300 ms.

This is the best observation-only lateral result in this experiment, but both
the 0.25 mm median and 0.5 mm p95 requirements still fail. The supervised
policy comparison and RL therefore remain unstarted. The next useful change is
better calibrated uncertainty and multi-frame/multi-view fusion focused on the
tail, evaluated through another predeclared calibration check rather than
loosening the insertion gate.

A final diagnostic selected a causal world-space opening EMA on calibration
episodes and applied it to the already opened pretrained development set. It
reached 0.250 mm median / 0.616 mm p95 lateral error and 97.8% direction
accuracy, but axial error worsened to 2.085/4.407 mm. This filter is
promotion-ineligible and still fails the lateral tail requirement, so it did
not justify another development collection.

Heatmap variance was also tested as an observation-only signal for dropping one
of the three camera views during triangulation. Calibration retained all three
views; fixed camera pairs and per-frame top-two selection were worse. The
development result therefore stayed at 0.276/0.671 mm in this diagnostic. This
rules out simple view rejection as the next fix; future work needs a trained
fusion/uncertainty model or more varied occlusion data.

### Prediction-only visual audit

To check what the trained model actually marks without drawing evaluation
labels, a separate renderer reads only RGB, camera calibration, frozen observed
features and trained weights. It overlays the predicted plug, four opening
corners, opening center, correction arrow, predicted opening axes and the
reprojection of triangulated predictions. Samples are selected from
prediction-only variance and reprojection statistics.

Manual inspection found the predicted opening polygon on the visible cage in
clear and cable-obstructed views. Across 301 observations, three-camera
reprojection RMS was 0.807 pixels median / 1.309 pixels p95. However, the
predicted stationary opening center jittered 0.369/1.316 mm and its normal
jittered 2.98/7.97 degrees relative to their per-episode medians. This supports
visual plausibility and cross-view consistency, while revealing the temporal
tail that remains. It cannot prove absolute accuracy without labels; the
separate held-out label comparison provides that measurement. See the
[review sheet and precise exclusions](../../outputs/experiments/2026-09-20_isaac_world_rl/pose_probe/opening_landmarks_v1/prediction_only_review/README.md).

The proposed causal fusion model, paper reading list, setting gaps, losses and
bounded ablations are documented in the
[temporal/multi-view design](2026-09-21-temporal-multiview-perception-plan.md).
That continuation has now run. Learned current-frame multiview fusion improved
the new-development near-port lateral result to 0.307/0.573 mm median/p95 with
99.4% direction accuracy, while the six-step causal model reached
0.289/0.667 mm. The temporal model's much stronger calibration result did not
generalize. Neither passed the 0.25/0.5 mm gate, so the policy and RL stages
remain closed. The replay also lacks independently randomized cable shapes;
targeted cable/visibility data is required before repeating the temporal model.

## Later supervised-controller result

After the force-safe cable/visibility continuation, the selected estimator was
frozen despite its remaining 0.322/0.626 mm near-port lateral median/p95. A
matched controller diagnostic then compared identical six-step GRUs with the
predicted translation, uncertainty, and phase either present or zeroed. Both
arms scored **0/8 strict insertions** on new autonomous starts. The conditioned
arm did not improve lateral p95, and zeroing its pose input changed held-out
translation commands by only 0.0095/0.0264 mm p50/p95. The branch stopped
before actor-Q, Gazebo adaptation, or final evaluation. Details and artifacts
are in the [controller execution record](2026-09-22-pose-conditioned-gru-policy.md).

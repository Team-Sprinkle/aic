# Cable-shape reset and visibility-aware perception continuation

## Decision

**Complete; perception gate still failed after the approved continuation.** A
force-safe closed-loop reset increased the usable static cable shapes from four
to five. A second collector then released control to the fixed CheatCode
teacher only after two stable observations within 0.25 mm, 1.75 degrees, and
5 N. This produced naturally evolving cable trajectories without training the
teacher, policy, or an RL agent.

The best calibration-selected natural-trajectory model reached **0.322 mm
median / 0.626 mm p95 near-port lateral error** on 153 samples from ten held-out
episodes of an unseen cable template. Direction accuracy was 100%. This is a
substantial p95 improvement over 0.909 mm, but it still misses the fixed
0.25/0.50 mm gate. A fixed causal temporal variant reached 0.247/0.617 mm; it
passed the median threshold and failed the p95 threshold. Combining restored
static frames with natural trajectories was worse at 0.331/0.753 mm. No policy,
reserved-final, or RL branch was opened.

## Force-safe and natural-trajectory continuation

The static force-safe run collected 960 causal transitions from eight recorded
cable templates crossed with ten starts. The filter retained 142 observations
from 34 episodes and five cable shapes. On a newly usable held-out cable shape,
the calibration-selected RGB model reached 0.338/0.746 mm lateral median/p95.
Complete perception plus the existing policy trunk measured 7.22 ms p95.

For natural trajectories, the reset controller first held the actor at zero,
waited for cable settling, restored the measured plug position in clipped
0.5 mm TCP-frame steps, and required two consecutive safe observations. It then
latched the reset complete and released control to the fixed CheatCode teacher.
This logic is privileged data-generation infrastructure. It is not available
to an autonomous actor.

The 50-episode collection saved 1,500 causal decisions. Thirty-five episodes
from all five included cable templates reached teacher handoff. The physical
and near-port filter retained 460 observations; the frozen split excluded the
single eligible row from template 4 and used:

| Split | Cable templates | Episodes | Rows |
| --- | --- | ---: | ---: |
| Fit | 0 and 2 | 18 | 205 |
| Calibration | 3 | 6 | 79 |
| Development | 5 | 10 | 175 |

The model starts from the original pretrained landmark checkpoint. This avoids
reusing the force-safe checkpoint, which had already used templates 3 and 5 for
fit or selection. Checkpoints and model variants were selected on calibration;
the development cable shape was then reported once.

| Model on natural trajectories | Lateral median / p95 | Axial median / p95 | Direction |
| --- | ---: | ---: | ---: |
| Equal-view triangulation | 0.328 / 0.822 mm | 0.296 / 1.926 mm | 100% |
| Predicted-visibility weighting | 0.284 / 0.788 mm | 0.348 / 1.219 mm | 100% |
| Oracle-visibility upper bound | 0.288 / 0.767 mm | 0.320 / 1.848 mm | 100% |
| **Visibility + current residual** | **0.322 / 0.626 mm** | **0.259 / 1.152 mm** | **100%** |
| Visibility + causal temporal residual | 0.247 / 0.617 mm | 0.270 / 1.117 mm | 100% |

The selected current-frame path plus the existing frozen policy trunk measured
6.89/7.18/7.32 ms p50/p95/p99 over 175 held-out decisions. Sensor acquisition
is excluded, as in the preceding latency tests. The saved prediction-only
montage was visually inspected: the cyan markers remain attached to the plug
and opening through large cable movement, but occasional pixel shifts remain
large enough to explain the p95 failure. No simulator label places these review
markers.

The matched combined-data ablation used 259/85/224 fit/calibration/development
observations with source-prefixed episode identities. Its selected result was
0.331/0.753 mm, showing that adding stationary restored frames did not
complement the natural trajectory distribution in this setup.

## Frozen-model error-tail audit

After the development cable shape had already been opened, the selected
natural-trajectory checkpoint was frozen and audited without changing its
models, data split, or scenes. This is a diagnostic analysis rather than a new
model-selection result.

The right camera localized the plug most accurately at 0.309/0.896 pixels
median/p95. The left camera reached 1.007/2.437 pixels, while the center camera
was weakest at 1.362/4.177 pixels and had a 1.33-pixel mean vertical bias. Port
errors were smaller: 0.977/1.912, 0.759/1.823, and 0.486/0.985 pixels for the
center, left, and right cameras. Exact labelled pixels triangulated through any
two cameras with at most 0.0011 mm lateral p95 error. The camera geometry is
therefore internally consistent; learned landmark error is the limiting term.

Removing a camera did not improve the all-camera model. Predicted two-camera
lateral median/p95 errors were 0.504/0.879 mm for center+left, 0.432/1.147 mm
for center+right, and 0.304/0.852 mm for left+right, compared with 0.322/0.626
mm for the selected three-camera current-frame result. A fixed per-camera
pixel-bias correction estimated from fit data also regressed to 0.283/0.791 mm.
The calibration-derived version was similar at 0.280/0.786 mm. These results
rule out camera removal and a constant offset as the next correction.

The worst errors were concentrated in one development episode: 13 of the
current-frame top 20 belonged to `cable_visibility_t05_p03`. The temporal
model's top 20 contained ten rows from that episode. The current-frame p95 was
0.680 mm for motions below 0.25 mm, versus 0.548 mm for 0.25--1 mm motion.
Clear opening views were worse than the partially or heavily obstructed bins,
so this split does not support cable occlusion as the direct cause. Occlusion
is confounded with cable template and trajectory. Predicted ensemble spread
was also poorly related to lateral error (correlation 0.18 current, -0.03
temporal), so it is not a calibrated rejection signal.

The bounded diagnostic is closed without another fit on this development
shape. A credible continuation requires new independent cable layouts and
camera-specific landmark supervision, especially for the center-camera plug.

## Question and controlled comparison

The preceding experiment had only one cable initialization family. This run
tested whether independent cable shape and explicit view reliability would
improve plug-to-opening pose prediction under cable occlusion.

The deployable path always uses RGB:

```text
three native RGB views
        |
        v
RGB locator -> 160 x 160 native crops
        |
        v
shared MobileNet landmark features
        |                       |
        |                       +-> RGB visibility probabilities per view
        v
plug/opening landmarks -> weighted three-view triangulation -> small residual
```

Isaac instance masks and geometry are used only to label and audit training
examples. They do not select a crop, set a camera weight, or enter the model at
evaluation time.

## Controlled reset audit

The new reset event restores all 40 cable joints from a recorded simulator
state, zeros their velocities, and then runs the existing tight six-dimensional
TCP IK reset. It records the template, seed, joint names, and reset report in
every causal transition. Historical episode YAML files without `cable_reset`
remain no-ops.

The short repeated test looked accurate at the first decision: all 40 cable
joints were restored, TCP reset error was below tolerance, and the requested
plug position had **0.053 mm median / 0.112 mm p95** error. That result alone was
misleading. Every reset began at the configured 35 N force clip. In the longer
zero-action test, observations 800 ms or more after reset had **3.96 N median /
18.36 N p95** force and the plug had moved **2.07/3.41 mm median/p95** from its
requested position. Templates 0, 3, and 5 commonly settled below 2 N; several
others remained at 8--22 N or were inconsistent across repeats.

Consequences:

- The all-template immediate-reset dataset was rejected.
- Training uses measured causal geometry after settling. It never assumes that
  the requested pose remained fixed.
- Samples less than 400 ms after reset or above 5 N were removed.
- Arbitrary cable-joint noise remains prohibited.

## Collection and split

The bounded collection crossed eight recorded cable templates with ten
near-port starts: two axial distances and five two-dimensional lateral offsets.
It ran zero policy actions and zero learning updates on one isolated A6000.
Each causal decision retained 576×512 RGB and an instance mask for all three
cameras.

Of 480 transitions:

- 160 were removed as reset-settling frames;
- 197 were removed for force above 5 N;
- 123 remained physically accepted;
- 96.2% of accepted plug views were visibly labelled;
- 84.8% of opening views were clear under the declared cable-mask criterion;
- opening rope coverage had 0 / 0.379 / 1.0 p50/p95/max.

Only four cable templates produced accepted observations. Complete cable shapes
were separated:

| Split | Cable templates | Episodes | Rows |
| --- | --- | ---: | ---: |
| Fit | 0 and 2 | 19 | 53 |
| Calibration | 3 | 9 | 33 |
| Development | 5 | 10 | 37 |

Templates 1, 4, 6, and 7 contributed no samples after the physical filter. The
four reserved final IK configurations were not opened.

## Models

The frozen baseline reuses the 233,662-parameter ImageNet
MobileNetV3-small FPN landmark model. The matched continuation fine-tunes the
same architecture on the cable-varied fit split and selects its checkpoint on
the held-out cable-shape calibration split. It stopped by patience after 1,800
updates and selected update 1,300.

The visibility head consumes only the shared RGB landmark representation and
heatmap confidence statistics. It predicts separate plug-visible and
opening-clear probabilities for each camera. On the unseen development cable
shape, opening-clear precision/recall were 0.977/0.977. The development set had
no plug-occluded positives, so its 0.919 plug-visible recall does not measure
occluded-plug recall.

## Held-out cable-shape results

All entries below use the 23 near-port rows from eight complete development
configurations. The selected model was chosen on calibration before reading
these development metrics.

| Model | Lateral median / p95 | Axial median / p95 | Direction |
| --- | ---: | ---: | ---: |
| Equal-view triangulation | 0.604 / 1.092 mm | 1.295 / 2.201 mm | 100% |
| Predicted-visibility weighting | 0.501 / 1.107 mm | 1.186 / 2.197 mm | 95.5% |
| Oracle-visibility upper bound | 0.537 / 1.117 mm | 1.206 / 2.204 mm | 95.5% |
| **Visibility + current residual** | **0.346 / 0.909 mm** | **0.781 / 1.336 mm** | **100%** |
| Visibility + six-frame temporal residual | 0.482 / 1.102 mm | 1.218 / 2.186 mm | 95.5% |

The oracle visibility weights did not improve the p95 tail. This indicates that
binary cable visibility is not the limiting error in this small split. Landmark
domain shift, triangulation calibration across cable shapes, and the limited
number of physically usable cable states remain larger problems. Fine-tuning
did improve the selected current-frame result over the frozen-landmark version,
which reached 0.480/1.177 mm. Temporal history again failed to generalize.

The saved prediction montage was visually inspected. Predicted landmarks remain
on the plug/opening region across all three cameras, including frames in which
the orange cable crosses the center view. The overlay contains only model
predictions and predicted visibility scores; simulator masks are not used to
place its markers.

## Latency

On 37 held-out decisions, the in-memory three-view perception path measured:

| Scope | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Locator, crops, shared landmark/visibility encoder, weighted geometry, residual ensemble | 6.62 ms | 7.02 ms | 7.22 ms |
| Same path plus existing frozen policy trunk | 6.72 ms | 7.12 ms | 7.32 ms |

Sensor acquisition is excluded, matching the prior perception benchmarks. The
measured p95 is safely below the 300 ms constraint.

## Commands

The durable launchers are:

```bash
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_reset_smoke.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_reset_settle_smoke.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_visibility_collect.sh
```

Dataset preparation:

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  outputs/experiments/2026-09-22_cable_visibility_perception/prepare_visibility_dataset.py \
  --replay /var/tmp/chmin_aic_20260920_isaac_world_rl/cable_visibility_20260922/visibility_collect/replay.pt \
  --config-manifest outputs/experiments/2026-09-22_cable_visibility_perception/configs/visibility_manifest.json \
  --output-dir outputs/experiments/2026-09-22_cable_visibility_perception/dataset
```

The exact selected training and latency commands are retained in
`run_visibility_training.sh` and `run_visibility_latency.sh` in the experiment
artifact directory.

The approved continuation adds these durable launchers:

```bash
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_force_safe_restore_smoke.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_force_safe_restore_collect.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_force_safe_training.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_force_safe_latency.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_natural_guide_smoke.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_natural_guide_collect.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_natural_guide_prepare.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_natural_guide_training.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_natural_guide_latency.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_combined_training.sh
bash outputs/experiments/2026-09-22_cable_visibility_perception/run_natural_pose_tail_audit.sh
```

## Artifacts

- [Machine summary](../../outputs/experiments/2026-09-22_cable_visibility_perception/summary.json)
- [Artifact map](../../outputs/experiments/2026-09-22_cable_visibility_perception/artifact_map.json)
- [Dataset manifest](../../outputs/experiments/2026-09-22_cable_visibility_perception/dataset/dataset_manifest.json)
- [Selected metrics](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_finetuned_landmarks/metrics.json)
- [Selected checkpoint](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_finetuned_landmarks/visibility_ablation_checkpoint.pt)
- [Prediction montage](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_finetuned_landmarks/development_predicted_visibility.jpg)
- [Live latency](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_finetuned_landmarks/live_latency.json)
- [Artifact hashes](../../outputs/experiments/2026-09-22_cable_visibility_perception/artifact_sha256.txt)
- [Force-safe metrics](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_force_safe/metrics.json)
- [Natural-trajectory handoff audit](../../outputs/experiments/2026-09-22_cable_visibility_perception/natural_guide_audit/audit.json)
- [Natural-trajectory metrics](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_natural_guide/metrics.json)
- [Natural-trajectory prediction montage](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_natural_guide/development_predicted_visibility.jpg)
- [Natural-trajectory latency](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_natural_guide/live_latency.json)
- [Combined-data metrics](../../outputs/experiments/2026-09-22_cable_visibility_perception/training_combined/metrics.json)
- [Frozen-model error-tail audit](../../outputs/experiments/2026-09-22_cable_visibility_perception/natural_pose_tail_audit/audit.json)

Bulk RGB, masks, the 2.58 GB force-safe replay, the 4.09 GB natural-trajectory
replay, and derived filtered datasets remain under
`/var/tmp/chmin_aic_20260920_isaac_world_rl/cable_visibility_20260922/`.

## Parking decision and restart checklist

This perception branch is parked on 2026-09-22. Preserve the selected
current-frame and temporal checkpoints, complete-template manifests, prediction
montage, raw audit, and failed bias correction. The achieved 0.322/0.626 mm
lateral median/p95 is useful as an observation for a closed-loop controller,
but it did not pass the original 0.25/0.50 mm promotion gate. Do not describe it
as resolving the success corridor.

The evidence available when revisiting this branch is:

1. exact labelled pixels give sub-0.0011 mm two-camera lateral p95, so camera
   geometry and triangulation are not the limiting term;
2. learned landmarks are limiting, especially the center-camera plug estimate;
3. all two-camera subsets are worse than the selected three-camera model;
4. constant per-camera correction regresses p95 to approximately 0.79 mm;
5. high errors cluster in `cable_visibility_t05_p03` and sub-0.25 mm motion;
6. occlusion is confounded with cable template and trajectory, and the current
   uncertainty estimate does not identify the failures.

If pose estimation is revisited, first collect independent cable layouts with
complete-template splits, strengthen camera-specific plug supervision, and fit
uncertainty without reading the new development split. Then rerun the unchanged
gate once.

The next control experiment may freeze this pose estimator and use it in a
clearly labelled **supervised diagnostic**. Compare the existing action-only
head with the same phase-aware head conditioned on predicted pose and history;
the actor must never receive true Isaac geometry. This diagnostic does not
authorize RL. Require autonomous insertion on new development starts, with
acceptable force, orientation, and latency, before training a critic.

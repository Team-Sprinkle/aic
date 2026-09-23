# Explicit pose-correction supervised continuation

Status: completed offline; gate failed, 2026-09-21 UTC under the existing
September 22 artifact convention. No autonomous rollout or RL followed.

## Reason for this experiment

The preceding matched GRU comparison achieved 0/8 insertion in both arms. Its
pose-conditioned arm made almost the same commands when pose was zeroed or
shuffled. Concatenating pose into a generic imitation network therefore did not
make pose causal for control.

This continuation tests a smaller and more constrained design. It retains the
frozen action-only GRU as a nominal approach policy and adds a bounded
correction that must pass through the current predicted plug-to-opening
translation. Context may change correction magnitude, but cannot create an
independent correction direction.

## Data plan

The first iteration isolates lateral correction and holds initial relative
orientation aligned. This avoids changing lateral coverage, orientation
coverage, and controller architecture simultaneously.

- Fit cable templates: `observed_cable_00` and `observed_cable_02`.
- Calibration cable template: `observed_cable_03`.
- Offline development evidence: the already frozen `observed_cable_05` natural
  trajectories.
- Fresh live development recipe: six new template-05 starts, unopened until
  the offline gate passes.
- Reserved final IK configurations: sealed.

Fit starts use two axial depths, 0.75 and 1.5 mm lateral radii, and eight
directions spaced by 45 degrees. Every direction has a 180-degree counterpart
with the same radius and depth. Calibration uses directions shifted by 22.5
degrees and different radii/depths. Collection uses the force-safe restore and
the fixed Isaac CheatCode at 100% blend. Replay retains the unblended teacher
chunk, executed chunk, measured state change, force, images, geometry for
offline labels, and pre-reset terminal observation where applicable.

Simulator plug/port geometry and masks may filter and label collection. They do
not enter the deployed actor or choose its crop.

## Architecture

```text
frozen action-only six-step GRU ---------------------- nominal 24D chunk

predicted plug-to-opening translation (current) --> no-bias pose map --+
                                                                    multiply
uncertainty + phase + measured state change + force                 |
previous executed chunk + visibility --> temporal GRU --> gain -----+
                                                                    |
                                             bounded 24D correction residual
                                                                    |
nominal chunk ---------------------------------------------------- add
                                                                    |
                                             four 6D TCP-frame deltas
```

The correction network has 66,596 trainable parameters. The pose map has no
bias and context controls magnitude only. For fixed context it has exact odd
symmetry:

\[
c(-p, h) = -c(p, h).
\]

Thus zero pose produces zero correction, and reversing pose reverses the
correction. The network can still learn a world-to-TCP coordinate mapping and
phase-dependent magnitude. Translation residuals are bounded to 0.5 mm per
command and rotational residuals to 0.002 rad before the learned gain of
0.25--1.75.

The first version has no separately predicted relative-orientation input. That
remains a later isolated extension after lateral correction passes.

## Predeclared offline gate

Calibration uses a complete unseen cable template. Before collecting the six
fresh live starts, all conditions must pass:

1. True pose improves translation MAE by at least 5% relative to replacing pose
   with zero.
2. The median correction-vector magnitude is at least 0.03 mm, showing a
   material rather than numerical dependency.
3. The correction has a positive dot product with the required
   teacher-minus-nominal translation residual on at least 65% of eligible
   chunks.
4. The implementation check confirms exact odd symmetry within floating-point
   tolerance.

Zero pose here means a physical predicted plug-to-opening vector of
`[0, 0, 0]` mm. It is different from the preceding matched model's normalized
zero slots: the explicit correction architecture guarantees that physical zero
pose contributes exactly zero correction.

If the gate fails, preserve the result and stop before live rollout. It may be
reasonable to revise data coverage or the supervised correction target using
the calibration split, but the live recipe cannot select those revisions.

## Live gate

If offline gates pass, run six fresh autonomous starts without guide,
exploration, action guard, or privileged actor input. Compare against the
preserved nominal controller and report same-timestep axial, lateral,
orientation, force, terminal behavior, and complete p50/p95/p99 latency.

Promotion requires repeated strict insertion, lateral error at or below 0.5
mm, acceptable orientation/force, and p95 latency below 300 ms. RL, world-model
dynamics, SEER, reward learning, imagination, Gazebo adaptation, and final
evaluation remain disabled until that gate passes.

## Implementation and artifacts

- Config generator and manifest:
  `outputs/experiments/2026-09-22_explicit_pose_correction/make_corrective_configs.py`
- Collection launcher: `run_corrective_collect.sh`
- Dataset preparation: `prepare_corrective_dataset.sh`
- Trainer:
  `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train_explicit_pose_correction.py`
- Runtime:
  `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/explicit_pose_correction_actor.py`

Bulk RGB/replay stays under
`/var/tmp/chmin_aic_20260920_isaac_world_rl/cable_visibility_20260922/`.

## Execution result

### Collection and coverage

The fit collection saved 960 causal decisions. The unchanged force and restore
filters retained 322 rows from 25 complete episodes: 253 rows on cable template
00 and 69 on template 02. All eight 45-degree directions had an accepted
180-degree partner.

The first calibration collection was too unsafe and sparse to satisfy the
predeclared coverage rule. Three retries reduced the offset from 1--1.75 mm to
0.5 mm and then 0.3 mm. Every rejected transition remained rejected; the force
limit was never relaxed. Nine calibration episodes were accepted in total.
Two lay on directions without an accepted opposite and were archived but
excluded before looking at model metrics. The fixed coverage rule retained 124
rows from seven complete template-03 episodes at 22.5, 157.5, 202.5, and 337.5
degrees. This supplies two direction axes with exact 180-degree partners. The
[coverage audit](../../outputs/experiments/2026-09-22_explicit_pose_correction/dataset_coverage_audit.json)
passes all four dataset checks. The pair selector records that it used coverage
only and did not inspect model results.

The merged replay also contains the frozen 175-row, ten-episode template-05
offline development split. It was used only as a diagnostic after checkpoint
selection. The separate six-start live development recipe was not opened.

### Training and fixed gate

The deployed stack would contain 1,149,913 parameters: 536,631 in the frozen
pose estimator, 546,686 in the frozen nominal GRU, and 66,596 trainable
parameters in the explicit correction. It still emits four 6D TCP-frame delta
commands.

The primary run selected update 400 using calibration normalized action loss.
On calibration, its true-pose translation MAE was 0.04638 mm versus 0.04786 mm
with physical pose set to zero, a **3.10% improvement**. Median correction
magnitude was 0.02909 mm, correction-direction accuracy was 67.09%, and exact
odd-symmetry error was zero. It passed direction and symmetry, but missed the
5% improvement and 0.030 mm magnitude requirements.

The first selector mixed translation and rotation through normalized action
loss. A bounded rerun kept the same data, architecture, optimizer budget, and
seed, but selected directly on calibration translation MAE. It also selected
update 400 and did not change the decision:

| Split / ablation | Translation MAE | Translation p95 | Correction p50 | Direction accuracy |
| --- | ---: | ---: | ---: | ---: |
| Calibration, predicted pose | 0.04663 mm | 0.11621 mm | 0.02942 mm | 69.00% |
| Calibration, pose set to zero | 0.04786 mm | 0.12226 mm | 0 | n/a |
| Calibration, shuffled pose | 0.04618 mm | 0.11626 mm | 0.02942 mm | n/a |
| Offline development, predicted pose | 0.03730 mm | 0.09402 mm | 0.03448 mm | 63.44% |
| Offline development, pose set to zero | 0.03671 mm | 0.08943 mm | 0 | n/a |

The translation-selected model improves calibration by only **2.58%**. More
concerning, shuffled pose is slightly better than the correct pose on
calibration, and zero pose is better on the untouched offline development
split. The architecture demonstrably changes commands with pose and usually
points toward the teacher residual, but the correct sample-specific pose does
not provide reliable control information across cable templates.

The post-training slice audit adds two useful limits. Every retained row in all
three splits was classified as **approach**; this dataset did not actually add
contact, blocked, retreat, or recovery supervision. On calibration, pose helped
the 48 rows whose nominal residual was at least 0.10 mm by 6.38%, but hurt the
65 medium-residual rows by 0.55% and the 11 smallest-residual rows by 15.36%.
On offline development it hurt the medium and small groups and improved the 20
largest-residual rows by only 2.45%. A correction enabled only for a pose-size
bin would also be unsafe: the sole helpful calibration bin was harmful on
development. The full episode, phase, motion-size, and predicted-lateral slices
are in `correction_slice_analysis.json`.

### Decision

The offline gate is closed. Running the six autonomous scenes would turn a
failed offline candidate into an unnecessary evaluation and could encourage
tuning on those scenes. No live latency claim is made for this combined actor,
no autonomous insertion result was collected, and RL remains disabled. The
reserved final configurations remain sealed.

The useful finding is narrower than a controller success: balanced corrective
data plus a mandatory pose path prevents the earlier silent “ignore pose”
failure, but it does not establish that this frozen pose estimate maps
consistently to the required TCP-frame corrective action. A later restart
should first improve cross-template pose/control calibration or supervise a
short-horizon visual servo target with direct progress labels. It also needs a
separate safe collection that actually reaches contact, blocked, retreat, and
recovery phases. Repeating BC or adding a critic to this failed controller is
not justified.

## Reproduction records

- Machine summary: `outputs/experiments/2026-09-22_explicit_pose_correction/summary.json`
- Artifact map: `outputs/experiments/2026-09-22_explicit_pose_correction/artifact_map.json`
- Dataset manifests and pair selection: `dataset_manifests/`
- Primary failed checkpoint and metrics: `training/`
- Translation-selected failed checkpoint and metrics:
  `training_translation_selection/`
- Checksums: `artifact_sha256.txt`

The raw replays and simulator logs are large and remain at the bulk root stated
above. The durable experiment directory contains the exact generated YAMLs,
coverage decisions, small checkpoints, metrics, and hashes.

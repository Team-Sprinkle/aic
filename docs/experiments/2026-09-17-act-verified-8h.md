# Verified-demonstration ACT experiment: eight-hour window

Started 2026-09-17 21:12:28 UTC. Hard deadline: **2026-09-18 05:12:28 UTC**.
User authorized dataset verification, ACT training, bounded parameter sweeps,
and iterative simulator evaluation. At most four GPUs concurrently;
physical GPUs 0–3 were used. GPU 0 initially handled evaluation; GPUs 1 and 2
later moved to isolated simulator evaluations as training finished.
Rootless Docker only. No offline or online SERL training in this experiment.

Artifacts: `outputs/experiments/2026-09-17_act_verified_8h/`.
Start/deadline/resource limits are recorded in `budget.json`.
Training/evaluation finished before 04:46 UTC; all three owned containers
were stopped at 04:45:40 UTC and GPUs 0–3 released. `resource_cleanup.json`
records the checks. The wall budget was not extended.

## Final result

**Reliability target not met: 1/20 full insertions**, mean official total
**39.16**. All 20 trials completed with the frozen ACT model, ground truth
disabled, and unchanged runtime. The target was 18/20. Scope is **SFP, NIC count
1, rail 0, card 0, port 1**, with fresh randomized board/rail/yaw/grasp settings;
this experiment does not establish performance for other NIC counts or SC.

- Audited 668 historical episodes: **393 verified official insertions with raw
  array lineage**, 212 scored non-insertions, 63 unresolved. Inspected sampled
  terminal three-camera sequences across every source family. A success score
  alone did not make an action-label stream suitable for training.
- Started with a 27-episode subset of the verified 130-episode CheatCode source,
  expanded to all five NIC-count groups, then collected **111 additional verified
  expert/corrective episodes**. The most recent aligned pool has 87 episodes
  (70 training / 17 recent validation). Expert collection success is not learned
  policy success.
- Completed **47 bounded ACT training jobs**, including continuations and fitting
  diagnostics, plus two derived weight averages. Swept chunk/execution horizon,
  state inputs, action representation, learning rates, dropout and sampling.
  **No offline or online SERL training.**
- Fixed or exposed image-channel, clock, quaternion, target-transport and
  normalization issues. Recent training errors fell below 1 mm while held-out
  command errors stayed around 4–5 mm; more updates did not establish live
  reliability.
- Selected `act_aligned74_noerror_c8/002500`: **2/10 development insertions**.
  Froze model, normalizer, runtime and scene hashes before the final sample.
  [Selection and training lineage](#frozen-selection-0427-utc) explains the five
  training stages; “2,500” is only the last stage's checkpoint count.

Read the [20 official results](../../outputs/experiments/2026-09-17_act_verified_8h/selected_act_final/final_results.md),
[model bundle](../../outputs/experiments/2026-09-17_act_verified_8h/selected_act_final/README.md),
[video review](../../outputs/experiments/2026-09-17_act_verified_8h/review/index.html), and
[all run summaries](../../outputs/experiments/2026-09-17_act_verified_8h/results_latest.md).
[Local reproduction](../LOCAL_WORKFLOW.md#test-the-frozen-september-act-model)
includes an exact rootless evaluation command. Models/data/videos are local
ignored artifacts, not an external backup.

## Plan and stop rules

| Elapsed budget | Work | Decision |
| --- | --- | --- |
| 0–45 min | Resolve episode-to-official-score lineage; inspect final camera frames; audit action labels per source. | Initial training uses verified insertions, with explicit source IDs and no fabricated success labels. |
| 45–120 min | Small ACT training run on the strongest SFP setting; evaluate saved checkpoints in Gazebo. | If imitation improves without rollout progress, inspect normalization, action/control timing, and deployment compatibility. |
| 2–5 h | Small learning-rate/chunk/execution-horizon sweep; stop weak candidates. | Select by complete official runtime outcomes, with held-out imitation checks as supporting evidence. |
| 5–7 h | Scale the best configuration to more verified data/settings if the pilot supports it. | Do not scale a failing recipe simply to spend the budget. |
| 7–8 h | Fresh repeated trials, videos, checkpoint/config packaging, report. | Stop all training/evaluation by the deadline; report failures or an unmet target honestly. |

Reliability target: **at least 18 official Tier 3 = 75 insertions in 20 fresh
trials for each claimed setting**. Record total scores and force/efficiency
categories separately. A result applies only to the evaluated scene/reset
distribution. No privileged teacher, pose override, or success gate is added to
learned-policy evaluation. This finite sample is an empirical target, not a
statistical guarantee of 90% population success.

Split by source episode/trial; keep pilot validation and final evaluation
separate from fitting. Preserve failed candidates and runtime attempts. Derived
datasets and any repaired labels must retain links to their original episodes.

## Chronological work record

The dated entries below preserve decisions as the experiment progressed.
The final result above supersedes their provisional status statements.

### Verification and initial runs

- 668 accepted episodes audited: 393 verified official insertions with matching
  raw state/action arrays, 212 scored non-insertions, 63 unresolved mappings.
  See [dataset eligibility](../DATASETS.md#verified-insertion-episodes).
- Inspected sampled three-camera final frames and preceding seconds for every
  source family. Representative raw/accepted CheatCode end frames match within
  compression differences. The decoded 130-episode cache checks frame PTS at
  the expected 20 Hz timestamps, without changing action labels.
- Pilot setting: SFP card 0 / port 1, NIC count 1. Of its 34 demonstrations,
  27 episodes / 14,837 frames train and 7 episodes / 3,674 frames validate.
  Splits are deterministic and stored in `cache_cheat130/cache.json`.
- Three fresh ACT pilots: chunk 16 / LR 1e-4, chunk 8 / LR 1e-4, and chunk 16 /
  LR 3e-5. ResNet18 ImageNet initialization, trainable backbone LR 1e-5,
  transformer width 256, 3 encoder layers, batch 16, L1 + KL weight 10.
  All use the installed LeRobot ACT implementation via
  `scripts/train_verified_act.py`; no offline or online SERL updates.
- State/action normalization uses training episodes only. Image normalization
  uses ImageNet statistics. Small state standard deviations have documented
  floors in the trainer, avoiding amplification of nearly constant channels.
- Runtime explicitly disables translation/rotation deadbands, preserves the
  recorded TCP-relative command convention, and records camera snapshots about
  once per simulation second. Initial execution horizon is 4 actions at 20 Hz.
- First export check: nine validation frames, maximum physical translation
  component difference 1.05 micrometres between PyTorch and TorchScript. Saved
  normalizer tensors match the trainer's arrays.
- Controller-error/velocity channels strongly correlate with expert commands.
  This is a possible imitation shortcut; falling validation loss alone is not
  treated as evidence of live insertion skill.

Run directories contain exact arguments, source splits, normalization arrays,
training/validation JSONL, standard LeRobot checkpoints, and optimizer state
for the latest checkpoint. Runtime attempts retain logs and official scores.

### Early deployment findings (not final results)

- Baseline chunk 16, 500 updates: total 36.99, Tier 3 24.79, no insertion,
  approximately 5 cm final distance on held-out episode 11's scene. The
  5,500-update model also stopped approximately 5 cm away (Tier 3 24.68).
- Chunk 8, 6,000 updates, one action per inference, simulation-time cadence:
  no insertion, approximately 6 cm final distance. Lower imitation loss has
  not yet translated into insertion success.
- On 256 held-out frames, the 6,000-update full-state model's translation error
  was 1.237 mm normally, 1.441 mm with shuffled images, and 6.102 mm with
  normalized controller error/velocity zeroed. The 2,000-update model trained
  without those channels was 3.824 mm normally and 7.177 mm with shuffled
  images. This supports testing a state shortcut; it does not prove images are
  unnecessary or establish a successful alternative.
- State ablations exclude either controller error (indices 13–18) or both
  velocity and controller error (7–18). Both ACT state-projection layers store
  exact zero columns for excluded inputs; no inference-only masking flag is
  needed. Each checkpoint save verifies those columns remain zero.
- An absolute-target-pose variant converts the same recorded command through
  the observed TCP pose. All 71,670 source frames passed a numerical roundtrip
  check. Rotation vectors use target-quaternion X >= 0 to avoid a 180-degree
  sign discontinuity in this collection. Runtime bounds the displacement from
  current proprioception and sends the bounded absolute target in `base_link`.
  Export metadata identifies the action representation and rejects a mismatch.
- Simulation-clock execution gates actions on new center-camera timestamps;
  wall-clock execution remains available for legacy comparisons. Maximum
  runtime is still a wall-clock bound. Snapshot logs now include proprioception
  and wall/simulation timestamps for timing and distribution checks.
- Fixed score parsing for valid trial names not beginning with `trial_`. The
  first attempt's original summary is preserved alongside a revalidated
  summary; its official score is unchanged. Runtime evaluation now guards
  against simultaneous use of the same container and bounds Docker restarts.

### First-hour contract checks

- Replaying original relative commands on episode 11's original scene drifted
  and failed (Tier 3 = 0). Replaying their reconstructed absolute targets tracked
  free-space motion much better but reached only partial insertion (Tier 3
  38.01). Open-loop replay is a diagnostic, not learned-policy performance.
- A fresh privileged CheatCode run on that same scene **inserted**, total
  **93.72**, Tier 3 **75**. Its final TCP position differed from the recorded
  demonstration by approximately 0.81 / 0.42 / 0.014 mm on X/Y/Z. This supports
  the historical data's physical relevance; the learned models still need
  sufficient precision and feedback. This diagnostic did not train ACT or count
  toward its success rate. CheatCode does not call `get_observation`, so the
  policy-boundary camera recorder produces no expert frames; endpoint evidence
  was extracted from the official controller-state bag instead.
- Found the cause of the zero trajectory-path reports: the normal launch omits
  the fixed `world` → `aic_world` identity when ground truth is disabled. The
  evaluator now publishes only that fixed transform, leaving object-pose relay
  disabled. A subsequent ACT trial reports a measured 0.25 m path. Earlier
  Tier 2 efficiency scores are not directly comparable; insertion events remain
  the primary criterion.
- Absolute ACT targets are now sent directly in `base_link`, after bounding
  their displacement. Converting them to a relative message using an older
  camera observation would otherwise add intervening robot motion. The runtime
  performs two dummy forward passes before trial start to avoid first-inference
  delay and uses the model callback's real Cartesian-mode service rather than
  assigning its cached mode flag.
- At 22:00 UTC, new candidates train on either the same 27-episode subset or
  all five NIC-count groups from the verified 130-episode source, with the same
  episode holdouts. They append an elapsed-time feature to the state, since the
  expert includes timed approach/settle/descent phases. This is an experimental
  ACT input extension, stored in checkpoint/export metadata, capped at 40 s;
  use `RunACTTorchScript` for these checkpoints. It uses no target geometry.

### Quaternion and recording-clock findings

- Canonicalizing input quaternions to W >= 0 causes an input discontinuity at
  the initial half-turn about X. The W-sign absolute/time model stalled near
  that boundary, then descended late; its 5,000-update three-scene pilot had
  **0/3 insertions**. A new X >= 0 input convention removes that visible initial
  stall, but the 10,000-update one-scene pilot still failed (total 8.89,
  Tier 3 21.53, contact penalty). This is a representation repair, not a
  demonstrated insertion solution. Its sampled rollout video is saved.
- The original recorder's timer uses **wall time**. LeRobot timestamps and
  video PTS are frame indices divided by FPS, not preserved ROS simulation
  timestamps. Across these 130 episodes, official simulation duration divided
  by dataset duration ranges 0.59–0.74 (median 0.686). A separate position/
  velocity fit also confirms unequal clocks. Exact per-frame simulation times
  cannot be recovered from these parquet columns. See
  `verification/collection_clock_audit.json`.
- Consequently the early time-conditioned candidates feed dataset elapsed
  time during fitting but camera simulation elapsed time during deployment.
  Their timing assumption is approximate and is a known limitation. Matching
  video PTS proves image/frame-index alignment, not simulation timing.
- Replaying reconstructed absolute targets directly in `base_link` still
  reached only partial insertion (total 57.44, Tier 3 38.02), similar to the
  earlier replay. The privileged online teacher's successful feedback is not
  reproduced by open-loop command replay.
- Further controlled candidates test ACT's optional VAE disabled and the
  X-sign convention with relative commands. A training-only proprioception
  noise option is also available for absolute targets; it changes observations
  without changing target poses and is explicitly recorded in run arguments.

### Historical camera convention confirmed

The 130-episode source predates the collector's RGB fix. At historical commit
`d1efda5`, `PolicyRecorder._image_to_rgb` actually returned **BGR** arrays;
LeRobot saved them as RGB images. Raw and accepted videos therefore both have
red/blue reversed relative to current ROS RGB observations. The matching-scene
first-frame comparison shows this in both cable and board-marker colors;
lower-image color differences drop from roughly 87–120 to 5–20 pixel levels
after swapping red/blue. Evidence and historical source are preserved under
`verification/`.

Cache pixels are preserved. These checkpoints must receive BGR channel order
at runtime, now represented by `image_channel_order` metadata or the explicit
evaluation override `--image-channel-order bgr`. New training requires an
audited cache convention. Newer datasets must be audited independently before
mixing them with this source. The current recorder already converts to RGB
correctly. Nineteen focused camera/runtime tests pass.

The matched 10,000-update absolute/time/X-sign checkpoint still failed with
the compatibility fix (total 30.29, Tier 3 18.91). Correct channel order is
necessary, but this test did not establish a learned insertion. Earlier runs
without the BGR override used mismatched camera channels.

### First learned insertion (22:53 UTC)

`act_all130_absolute_time_qx_c8`, checkpoint **020000**, BGR runtime input,
four executed actions per inference at 20 Hz simulation cadence:

| Held-out source scene | Total score | Tier 3 | Inserted |
| --- | ---: | ---: | --- |
| Episode 11 / trial 13 | 8.83 | 21.83 | No |
| Episode 18 / trial 20 | 29.81 | 19.37 | No |
| Episode 27 / trial 29 | **86.39** | **75** | **Yes** |

This is **1/3 held-out pilot insertions**, not reliable performance or a fresh
20-trial evaluation. The successful run took 58.32 simulation seconds and had
no contact penalty. No object ground truth was exposed to the ACT policy.
Official scores, once-per-second three-camera frames, and
`success_ep27_020000_bgr.mp4` are saved under that run. Its source episode was
excluded from fitting. Exact run/evaluation configurations remain in artifacts.

`scripts/summarize_act_experiment.py <artifact-root>` regenerates
`results_latest.json` and `results_latest.md` from saved training records and
official attempt summaries; diagnostics remain separately labeled.

### Corrective-data pilot and phase-two plan

The supervised goal variant continues the 20k ACT checkpoint with a fresh
optimizer and an additional head trained to predict the same demonstration's
final achieved TCP pose. Its extra encoder pass uses zero VAE latent and no
action labels, avoiding access to future action targets through the VAE. The
head is discarded at deployment; ACT still predicts and executes its full
action chunk. These are additional ACT updates, not offline or online SERL.

The first corrective-data pilot completed insertion (score 88.90). The next
five new scenes produced four insertions (scores 92.05, 92.51, 93.44, 92.94)
and one partial insertion, which was rejected. These are privileged expert
collection outcomes and **do not count as learned-policy successes**.

- `CollectCorrectiveCheatCode` records the clean teacher target while executing
  intermittent bounded XY/rotation perturbations. Images therefore include
  physical deviations and labels describe the expert correction, rather than
  duplicating the deliberately perturbed command. Only training collection
  enables object ground truth; normal ACT evaluation continues to forbid it.
- Recorded state/action pairs reconstruct the clean target within 1e-6 metres/
  radians. Collection also requires complete official scores and all camera
  frames. The merger accepts Tier 3 = 75 and total >= 80. Source ordinal 5 in
  each five-trial block is held out; this assignment remains stable when more
  batches are added. Scene seeds are distinct from the untouched final test.
- The synchronous image-writing pilot sampled approximately every 0.1 sim s.
  Later collection writes images asynchronously with bounded queues and one
  OpenCV thread; measured median camera sampling interval is 0.05 sim s. Actual
  stamps are retained. The cache uses a causal hold on a 20 Hz simulation grid.
- `cache_cheat130_corrective5` contains the original 130 episodes plus five
  verified corrective episodes (four train, one validation). Sharded image
  access shares the original 47 GB cache and reconciles old BGR/new RGB pixels.
  Three focused tests cover channel/order preservation, score eligibility,
  timestamp resampling, and teacher-label reconstruction.
- Paired small fine tunes start from the same 25k total-update goal-supervised
  checkpoint: one retains elapsed time; one keeps the input dimension but zeros
  its projection weights. Both sample 50% legacy and 50% corrective training
  frames. This tests whether correction coverage and removal of the mixed
  clock input improve live control before collecting/scaling further.

The final reliability set remains unused. Physical GPUs 0–3 remain the only
GPUs used, and the original hard deadline is unchanged.

### Follow-up diagnostics around 23:40 UTC

- The absolute-goal auxiliary continuation at +5k updates failed all three
  original pilot scenes. The best reference checkpoint also failed all three
  when replanning every action (`n_action_steps=1`), so that execution change
  is not an improvement over its earlier 1/3 result with four actions.
- An offline diagnostic evaluated the auxiliary head on the failed live camera
  recordings. Its final ten predictions averaged 22.0, 12.1, and 34.1 mm from
  the corresponding demonstration's achieved final TCP position. This is much
  worse than its roughly 2–3 mm held-out demonstration error. Reference poses
  were used only for this retrospective measurement, never for control.
  Evidence: `act_all130_absolute_time_qx_goal_continue/goal_live_diagnostic_005000.json`.
- The five-episode corrective fine tune with elapsed time reached a **partial**
  insertion on pilot episode 11: total 50.21, Tier 3 38.04. It is not counted
  as a completed insertion. The paired time-masked run is being evaluated.
- The next auxiliary variant predicts the final pose relative to the current
  TCP, with proprioception zeroed only in its extra observation-only pass.
  This requires camera features to carry the alignment information. Its head
  remains training-only; ACT's normal state inputs and full action output are
  retained. A matching fine tune without auxiliary supervision is the control.
- Optional temporal averaging of overlapping absolute action chunks is now
  available for an execution ablation. It is disabled by default, requires
  replanning every action, and explicitly favors newer predictions with
  `exp(-coefficient * age)` weights. Focused runtime tests: 22 passed.

### Translation-limit audit

The corrective-data partial insertion's ACT continued requesting downward
motion while the measured TCP stayed near Z=0.237 m. Its nearby logged target
was Z=0.172 m. The runtime limited each TCP-relative translation component
to 20 mm. With the gripper tilted, this changed the requested world delta
from approximately `[+3.3, -1.8, -65.3]` mm to `[+5.0, +10.5, -26.0]` mm:
about **24.5 degrees of direction error**, including reversed lateral motion.
The values use the last saved observation and a nearby logged action, not an
exactly synchronized pair. Evidence is saved in
`verification/absolute_translation_clamp_diagnostic.json`.

An optional norm limit now preserves translation direction (after any explicitly
configured deadband). The existing component limit remains the default for
compatibility. Planned controlled evaluations compare 100 mm and 20 mm norm
limits, with zero deadbands. No performance improvement is assumed until those
official evaluations complete. The norm-limit/runtime tests pass (23 tests).
The expert command audit found 4.34% of source frames exceeded a 20 mm component
limit; these full pose targets are not measured per-frame displacements.

### Expanded corrective data and longer ACT training

The 20-scene training collection with seed 202609173 finished with **19 full
insertions** (91.92–94.66 total) and one partial insertion (50.82, rejected).
Six accepted episodes and the rejected episode were visually inspected at
end-minus-two seconds, end-minus-one second, and the final frame in all three
cameras. The images agree with the official full/partial distinction. Audit
records and four contact sheets are in `corrective_data_batch2/verification/`.

`cache_cheat130_corrective24` now contains **154 verified episodes / 85,374
frames**, sharing the original image cache. The 24 corrective episodes have
19 training and five validation assignments. Both partial collection episodes
remain excluded. The new data uses actual camera simulation timestamps.

The longer plain ACT run `act_mixed24_time_velocity_b64` initializes from the
original 20k checkpoint with a fresh optimizer. It uses batch 64, a 50/50
legacy/corrective frame sampler, mild color jitter, and cosine learning-rate
decay. Physical TCP velocity is available; controller target-error channels
remain excluded. It has no auxiliary goal loss. Separate corrective-holdout
metrics were added because the pooled validation sample mostly covers legacy
episodes. Training is bounded by 30k additional updates / 135 minutes and the
original experiment deadline.

The first larger-limit pilot did not improve insertion. The remaining limit
comparisons are still running; the measured direction distortion is not being
treated as proof that it explains all learned-policy failures. A fresh 10-scene
development set (seed 202609179) is prepared, with training forbidden. The
separate 20-scene final reliability set remains untouched.

The runtime/data-gate test suite passes all 26 focused tests when run with
Pixi's native JPEG and PNG libraries preloaded. Without that process-local
setting, the combined suite exposed native-library symbol conflicts
(`jpeg12_write_raw_data`, then `png_set_cICP`) on this host. No packages or
global environment settings were modified. Evidence:
`expanded_data_runtime_tests_preload_codecs.txt`.

```bash
LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
  .pixi/envs/default/bin/python -m pytest -q \
  aic_example_policies/test/test_act_runtime_time.py \
  aic_utils/gazebo_rl/test/test_runtime_evaluator.py \
  aic_utils/lerobot_robot_aic/test/test_verified_act_cache.py
```

`act_diagnostic_overfit_corrective131` deliberately fits one **training** episode
from the new collection, with training fitting errors reported separately.
Its matching-scene runtime evaluation is a control-contract diagnostic and
must not count toward generalization. The trainer now accepts an explicit
absolute deadline instead of embedding this experiment's date as a permanent
default; expired explicit deadlines fail before creating a run.

### Dropout fitting diagnostic

The one-episode ACT retained about 7 mm inference fitting error after 1,500
updates. A saved diagnostic compared float32, bfloat16, posterior-latent and
zero-latent passes with/without dropout on identical training observations.
Float32/bfloat16 and posterior/zero latent differences were small relative to
the training/evaluation behavior difference. The diagnostic script and JSON
are retained under the artifact root and the one-episode run.

A continuation with **dropout set to zero** reduced training-frame inference
error to 0.97 mm after 250 additional updates (0.50 mm over the final three
seconds). This is a fitting diagnostic, not a generalization result, and still
needs a matching-scene simulator check. The larger mixed-data run was saved
after 3,642 updates and continues as `act_mixed24_time_velocity_b64_dropout0`
with the same data, state/action conventions and a fresh optimizer. It saves
every 1,000 updates to allow earlier simulator checks.

### Clean collection and inference fitting checks (00:50 UTC)

The dropout-zero single-episode model reached 0.57 mm mean training-frame
translation error, but its matching training-scene rollout still only partially
inserted (49.86 total, Tier 3 38.01). This diagnostic is excluded from learned
held-out success counts. Fitting error alone remains an inadequate selector.

A new **unperturbed** expert collection uses the same absolute `base_link`
command helper as ACT, stores actual simulation timestamps, and checks that
the recorded teacher target equals the executed target. Its first episode
fully inserted: **94.84 total / Tier 3 75**. Terminal images from all three
cameras at end-minus-two, end-minus-one, and the end were inspected. This
confirms the absolute-action command path can complete the task. It does not
establish ACT performance. The 409-frame episode is cached as episode 130 in
`cache_cheat130_nominal1`; a matching-scene ACT fitting diagnostic is running.
A separate 20-scene nominal collection (seed 202609174) is in progress; each
new episode must pass official score and image/label gates before training.

The mixed 24-episode continuation at 4,000 updates failed all three fresh
development scenes (mean total 25.28). These are the first three scenes of the
10-scene development set, now used for tuning. The independent 20-scene final
set remains untouched. New candidates compare canonical RGB, elapsed-time
input, dropout zero, and a dilated ResNet50 backbone with full TCP-relative
commands. These are plain ACT policies, without a residual adapter.

The trainer now computes the ACT VAE KL term in float32 using
`mu^2 + expm1(log_variance) - log_variance`; this avoids small negative values
from bfloat16 cancellation near the prior. Two numeric/gradient tests cover
this calculation. The running BGR continuation predates this change; the RGB,
ResNet50 and new nominal diagnostic runs include it. Adding time to an older
32-state checkpoint explicitly zero-pads the new input-projection column.

All 29 focused runtime, data-gate and numerical tests passed with the
process-local JPEG/PNG preload described above. Evidence:
`nominal_collection_runtime_tests.txt`. The new nominal acceptance gate checks
both collector metadata and evaluator runtime settings; perturbations must be
zero and executed/teacher targets must match.

### Observation-relative target transport and contact weighting

The new collector derives TCP-relative labels from the recorded observation.
An optional `--delta-pose-reference observation` runtime path composes that
label with the observed TCP pose and sends the resulting absolute target in
`base_link`. The existing controller-relative path remains the default for
legacy checkpoints. This makes the reference choice explicit, because applying
an observed-pose delta from a later controller pose shifts its target by the
intervening motion. A numerical target-reconstruction test and evaluator
identity/compatibility test cover the option; the focused suite passes 31 tests
(`observation_delta_runtime_tests_rerun.txt`). Its rollout benefit is unproven.

The clean expert's `verification/target_and_tcp.png` shows the TCP pausing in
contact while the commanded target continues approximately 4 cm deeper,
followed by insertion. These labels are full compliant-controller targets,
not measured frame-to-frame displacement. Optional terminal-frame sampling
has been added for a bounded contact-phase weighting comparison; it preserves
training-source and holdout boundaries. Existing running jobs are unaffected.

The images/time-only fitting diagnostic exposes a reporting-only baseline bug:
its old hold-position baseline used the masked state position. Future runs use
the preserved physical TCP pose for this baseline. Training targets, gradients,
and measured prediction errors were unaffected; ignore that diagnostic's old
`hold_pose_translation_error_m` field.

### Midpoint: clean data, longer chunks and student corrections

The unperturbed 20-scene collection (seed 202609174) completed with **18 full
insertions**, total 93.14–94.70, and two rejected failures (36.71 and -11.00).
Six accepted episodes and both failures were visually inspected in all three
cameras at the end and preceding two seconds. Together with the earlier clean
pilot, `cache_cheat130_nominal19` has 19 new episodes / 9,834 frames: **15 train
/ 7,748 frames and four validation / 2,086 frames**. The old 130-episode image
cache is shared on disk, but `act_nominal19_absolute_time_c8` and `_c64` use
**only the 15 new training episodes for both updates and normalization**.
Their initialization weights come from the mixed RGB run at 7,907 updates;
optimizers are fresh. Both use dropout zero, RGB, elapsed time clipped at 30 s,
controller-error masking, batch 64, and 50% sampling from the final five seconds.
The comparison changes action chunks from 8 to 64, with intended runtime
execution horizons 4 and 16. The longer chunk explicitly resamples decoder
query embeddings and regenerates fixed VAE positions; other weights load
strictly. Runs are bounded at 20k updates / 105 minutes and the original deadline.

The single clean-episode ACT diagnostics still failed in their matching training
scene: **49.85 and 49.80**, both partial insertions. The second masks all robot
state inputs and uses images/time. These results are excluded from held-out
performance. On the first model's recorded observations, axis mean errors are
0.080 / 0.053 / 0.694 mm in base-link X/Y/Z; its final-three-second error is
mostly Z. PyTorch and TorchScript predictions matched on the checked batch.
The plots and JSON are under `act_diagnostic_nominal130_dropout0/`.

`CollectCorrectiveCheatCode` now supports explicitly bounded ACT-influenced
collection. During selected segments the student proposes an absolute target;
the collector limits its disagreement with the expert to 30 mm / 0.08 rad,
executes that target, and keeps the clean expert label for the actual observed
state. Student checkpoint SHA256, active frames, proposed/executed targets,
probability and timing are recorded. It is a privileged data collector, never
a learned-policy score. The acceptance gate checks lineage, executed-target
reconstruction, intervention bounds, official insertion and images.
The first matching-scene pilot scored **92.55 / Tier 3 75**, with 223 recorded
student-active frames. Its terminal images were inspected. A two-episode ACT
fits the clean pilot plus this correction episode as a separate diagnostic.
The focused suite passes **33 tests**, including the new correction-data gates
and the terminal sampler (`student_correction_runtime_tests.txt`).

Collection runs in a second rootless container, `aic_collect_validation_20260918`,
with bridge networking, ROS domain 117 and Gazebo partition
`aic_collect_20260918`. It shares **physical GPU 1** with training; evaluation
continues on GPU 0, and training also uses 2–3. No fifth physical GPU is used.
Container details are saved in `secondary_container.json`. New independent
training scenes use seeds 202609175 (student correction) and 202609176 (nominal),
in five-scene batches with an explicit reset between batches.

The batch-size choice also addresses a recorder problem: in the previous
20-scene collection, trial 1 stores 437 distinct pose commands once, trial 5
stores 433 commands five times, and trial 20 stores 423 commands twenty times.
The serialized payloads include identical command timestamps. Insertion-event
copies likewise grow from 1 to 5 to 20. Evidence:
`verification/recorder_duplicate_audit.json`. The middleware/scorer root cause
is not yet established. Do not count event-message copies as independent
successes; acceptance uses each trial's official Tier 3 plus visual evidence.
Shorter engine batches bound this recording overhead. The independent final
reliability set still has no training or tuning use.

### First learned recovery after student correction data (01:43 UTC)

`act_diagnostic_nominal_student2/checkpoints/002000` completed **one full
insertion, total 86.49**, on the deliberately reused training scene. It was
trained on the clean pilot plus the score-verified student-correction episode;
its clean-only predecessors scored 49.85/49.80 with partial insertion on that
scene. This is evidence for learning the correction in a fitting diagnostic,
not a held-out success rate. The runtime used the ACT alone, with object ground
truth disabled, absolute commands, RGB, a four-action execution horizon,
simulation cadence, zero deadbands and a 100 mm norm bound. It had no teacher
or runtime success gate. The saved evaluation kind is `training_scene_diagnostic`.

The broader clean-only c8/c64 candidates both failed all three development
scenes at 2,500 updates. The 64-action model approached consistently (totals
36.46, 35.61, 36.55) but did not insert. Both R50 delta-reference comparisons
also failed all three development scenes. These results motivate scaling the
student correction data, rather than claiming that low fitting error or longer
chunks alone solved the task.

The first five-scene student batch contributes three student-correction
insertions, one nominal insertion with no randomly selected student segment,
and one rejected partial insertion. The unused-student case is accepted only
when recorded intervention count is zero and every executed target exactly
matches the expert, with explicit metadata/runtime checks. It has a separate
`nominal_expert_no_student_intervention` label. All terminal contact sheets were
viewed. The focused suite now passes **34 tests** (`student_unused_gate_tests.txt`).
The first additional nominal batch completed 5/5 insertions with all terminal
frames inspected. Remaining independent five-scene batches continue collecting.

A full-state ACT comparison restores controller-error inputs only on the new
causal recordings (observations are captured before issuing the next command).
This tests whether previous control state helps reproduce accumulated expert
corrections. Earlier masking was motivated by the historical action/observation
contract; the new comparison does not retroactively validate that old contract.


### Incremental clean and student data (01:56 UTC)

The clean recorded-command replay reached only partial insertion (57.25 total,
Tier 3 38.01), even with causal timestamps and five seconds holding the final
command. It is a recorded-trajectory diagnostic, never a learned-policy result.
The full-state two-episode fitting comparison also remained partial (49.80,
Tier 3 38.02). These outcomes do not establish an advantage for controller-error
inputs; the broader development comparison is still running.

`cache_cheat130_aligned42` adds 23 newly verified episodes to the 19 clean
recordings. It includes student corrections and explicitly identified nominal
trajectories, excluding all failed outcomes. All terminal contact sheets have
been visually inspected. Its 130 legacy episodes are retained as shared cache
references but excluded from the new model's updates and normalization. The
incremental merger now reuses previous immutable image shards and rejects
repeated collections; seven focused cache/acceptance tests pass. Shard roots are
absolute paths: retain the parent caches and referenced image directories, and
update those roots in `cache.json` if relocating the artifacts.

`act_aligned42_fullstate_c1` tests a one-action prediction/execution horizon on
this data, with the full ACT backbone and decoder. The hypothesis is that current
expert labels on student-visited states are more directly supervised than a
future chunk along a trajectory partially driven by a different controller.
This is an experimental comparison, not a demonstrated improvement.

The 20 final scenes are split into separate engine configs in
`evaluation_configs/final_fresh_nic1_seed202609180/single_trials/`. They remain
unused for training and tuning. The selected checkpoint and runtime must be
frozen before running them.


### Broader collection and continued training (02:17 UTC)

The independent five-scene collection batches produced 20/20 nominal insertions
and 14/20 accepted student-collection episodes. The latter contain twelve actual
student-intervention trajectories and two nominal episodes where the random
schedule selected no intervention. All accepted terminal frames were inspected;
six failed student trials remain excluded. Combined with the earlier clean19
and student pilot, `cache_cheat130_aligned54` contains **44 train / 10 validation
new episodes**, with 25,005 / 5,518 frames. Historical examples are excluded from
these continuations' updates and normalization, although the warm-start weights
inherit earlier training. Further collection is bounded to 20 clean plus 20
student scenes; the latter uses the frozen aligned42 one-action checkpoint.

The 19-episode c8 model at 7,500 updates failed all three fresh development
scenes (35.76, 35.94, 36.38). Restoring full state also failed all three (36.09,
1.00, 36.20). Those small-data training runs stopped at 9,127 and 5,130 updates.
The original all130 model at 20,000 updates, previously successful on one pilot
scene, failed the same fresh scenes too (9.13, 30.71, 28.55). Its earlier 1/3
pilot outcome therefore does not establish generalization to fresh scenes.

Three continuations now use the aligned54 cache: full-state one-action ACT,
controller-error-masked c8 ACT, and full-state c8 ACT with backbone LR 3e-5
instead of 1e-5. All use head LR 3e-5, batch64, RGB, no dropout, absolute targets,
actual clipped simulation time and 50% terminal-window sampling. Exact commands
and distinct warm-start parents are in `act_aligned54_*_command.json` and
`launch_aligned54.py`. These are candidate comparisons with different histories,
not isolated causal estimates of one hyperparameter's effect.

A 512-frame fresh-holdout input probe of aligned42/c1 at 2,500 updates found
4.55 mm mean target error normally, 8.37 mm with another scene's images at a
similar elapsed time, and 4.78 mm with controller-error inputs at their training
mean. This supports visual sensitivity; these perturbed inputs are off
trajectory and do not establish rollout performance. The historical and new
NIC-1 scene-coordinate/grasp ranges were also audited and substantially overlap
(`verification/scene_distribution_audit.json`). The focused suite passes 35 tests.


The aligned42 one-action checkpoint at 2,500 updates completed all three fresh
development trials without insertion (36.39, 36.60, 34.39; Tier 3 approximately
25). A shorter prediction/execution horizon alone has not solved the problem.
Its fitting run stopped for data expansion, and the aligned54 continuation
retains that checkpoint as its initialization. Comparable offline curves for the
three aligned54 variants are rendered by `plot_aligned_learning.py`; the fresh
holdout and training metrics are shown separately.


### Normalization-preserving data expansion (02:33 UTC)

Changing the dataset previously fitted new normalization statistics while loading
unchanged ACT weights. A direct 64-frame check of the same aligned42 checkpoint
under aligned54 statistics changed physical predictions by **2.799 mm mean
translation**, up to **6.978 mm per component** and **0.04752 rad per rotation-vector
component**. This is a continuation discontinuity, not evidence that it explains
all the insertion failures.

The new opt-in `--rebase-normalization` adjusts both state input projections,
the VAE action projection, and the action output head for the changed affine
normalizers. It rejects changed input conventions/masks and unsupported auxiliary
goal heads. With unchanged input semantics, the actual-checkpoint test preserved
physical translations within **0.0000298 mm** and rotation-vector components
within **2.38e-7 rad**. See `verification/normalization_rebase_actual_policy.json`.
The focused suite passes **36 tests**. All three next continuations enable this
option and retain fresh optimizers/schedules.

All twenty additional clean trials succeeded and their terminal contact sheets
were viewed. `cache_cheat130_aligned74` now contains **60 train / 14 validation
new episodes**. `act_aligned74_*` continues each corresponding aligned54 model's
final saved weights using the corrected normalization handling. Exact parent
steps and commands are recorded by `launch_expanded_act.py`.

The later R50 delta checkpoint (7,500 updates) failed the three development
scenes (36.56, 1.00, 1.00). The aligned54 masked-error c8 model at 2,500 updates
also failed all three (36.91, 36.20, 36.64). No reliable general insertion result
has been established. The next student-data queue initially failed its explicit
preflight: the default four-action horizon exceeded the student's one-action
chunk. No simulation ran in that failed attempt. The retry explicitly uses
`--n-action-steps 1`; its original failure log remains intact.


### Development progress and remaining budget (02:47 UTC)

At 2,500 updates on aligned54, the one-action full-state model scored 36.73,
35.90 and 49.93 (0/3 insertions). Its third-trial video and first/middle/end sheet
are saved as `act_aligned54_fullstate_c1/dev3_trial3_partial_002500.mp4/.jpg`;
visual inspection confirms the plug remains protruding. The full-state c8 model
with faster backbone learning reached 61.06, 36.10 and 49.95, with best Tier 3
49.60. This is the strongest fresh development progress so far, still **0/3**.

The aligned74 continuations use final aligned54 parents at steps 4,633
(masked-error c8), 4,519 (full-state c1), and 4,228 (faster-backbone c8). New
2,500-update exports are ready, with the faster-backbone configuration evaluated
first. Separate checks test temporal averaging of overlapping action chunks and
a longer wall-clock cap on the strongest partial-insertion scene. Neither
changes the official full-insertion criterion.

`prepare_final_bundle.py` is prepared but has **not been run**: after selection,
it copies the chosen model, normalizers, relevant runtime sources and the twenty
single-trial configs into `selected_act_final`, hashes the files, and creates the
frozen final evaluation queue. No final scene has yet been evaluated.

### Final data expansion and controlled backbone comparison (02:58 UTC)

The last 20 bounded student-intervention collections produced **13 verified
insertions**, seven score-based exclusions. All 13 accepted episodes had their
three-camera end-minus-2/1/0-second contact sheets inspected; the visual reviews
are recorded separately in each collection's `verification/audit.json`. These
are privileged expert-labelled demonstrations, never learned evaluation wins.

`cache_cheat130_aligned87` contains **87 recent verified recordings**, split into
**70 training episodes / 41,052 frames** and **17 validation episodes / 9,805
frames**. It references the original 130-episode image shards without copying
them. The final continuations update and normalize on the 70 recent training
episodes only. Earlier checkpoints in their ancestry also saw legacy data.

The 74-recording fast-backbone candidate at 2,500 updates scored 36.45 / 36.33 /
36.19 on the three fresh development scenes, **0/3 insertions**. Normalization
rebasing fixes a real warm-start discontinuity; this result does not establish
an improvement in insertion success. The earlier 54-recording temporal-ensemble
variant also failed all three scenes. Extending the 54-recording fast candidate
to 180 wall-clock seconds failed (36.59, Tier 3 24.93); extra time is not a
supported fix.

The remaining training comparison uses a shared full-state chunk-8 parent and
identical data/normalization, with backbone learning rates **3e-5 versus 1e-4**.
Head LR remains 3e-5. The third candidate retains chunk size 1. Each is bounded
at 12,000 updates / 65 minutes; all have the experiment deadline guard. Final
model selection will use development outcomes, then freeze model and runtime
before the untouched 20-scene assessment.

### First insertions on fresh development scenes (03:12 UTC)

Two 74-recording continuations now complete one insertion each on the three
fresh development scenes (ordinary ACT rollout, ground truth disabled):

| Candidate / checkpoint | Trial 1 total | Trial 2 total | Trial 3 total | Full insertions |
| --- | ---: | ---: | ---: | ---: |
| `act_aligned74_noerror_c8` / `002500` | **86.36** | 36.24 | 49.90 | **1/3** |
| `act_aligned74_fullstate_c8_fastbackbone` / `005000` | 36.05 | **86.77** | 49.77 | **1/3** |

Both full successes have official Tier 3 = 75. Their three-camera videos and
first/middle/end sheets were rendered and inspected, under the respective run
folders as `dev3_trial1_insertion_002500` and
`dev3_trial2_insertion_005000`. Videos hold the roughly one-second snapshots
until the next simulation timestamp; they do not reconstruct intermediate motion.
The one-action candidate at 5,000 updates reached only partial insertion.
These repeatedly used development scenes support candidate selection, **not**
a final reliability claim. Both successful candidates are queued on the seven
remaining development scenes.

An exact scene/task-config audit across all 45 ACT runs found **zero overlaps**
with the 20 predeclared final scenes and no ACT warm start outside this
experiment. The audit covers 191 distinct training scene/task configurations,
including ancestry and small diagnostics; it does not imply those scenes are
independent samples or measure how similar two layouts are.
`verification/final_scene_split_audit.json` retains the references.

### Storage preservation

At 19 GB free, ten inactive `corrective_data_batch2` diagnostic bags (trials
11–20) were losslessly archived as adjacent `.tar.zst` files. Every archived
member was decompressed and SHA256-checked against its original before removing
the expanded directory. This saved **16.05 GB**. Scores, camera frames, training
caches and checkpoints remain directly accessible. The archive checksums,
member checksums, original paths and exact restore arguments are in
`verification/archived_diagnostic_bags.json`. Restore a bag before rerunning a
bag-reading diagnostic; the original duplicate-audit result remains available.

A second verified archival pass compressed 20 more bags from completed attempts,
saving another **27.48 GB**. Across both passes, 30 bags saved **43.53 GB**;
`verification/archived_completed_bags.json` records the second set with the same
member-by-member verification and restore arguments. No model, image cache,
score file or rollout JPEG was removed.

Before the final evaluation, a third pass archived 12 further completed bags
and saved **10.55 GB**, restoring 22 GiB of free space. Its index is
`verification/archived_pre_final_bags.json`. Total across the three passes:
**42 verified archives / 54.08 GB saved**. The same decompression/hash checks
were required before removing expanded copies.

At 03:17 UTC, the 87-recording one-action continuation also inserted on
first development trial at checkpoint `002500`: totals **86.57 / 36.16 /
35.52**, **1/3 full insertions**. It remains in the comparison. This is not an
independent three-trial reliability estimate: candidate runs reuse the same
three development scenes. A local video review index is available at
`review/index.html` under the artifact root.

### Broader development checks and final candidate variants (03:38 UTC)

- The 54-recording fast-backbone candidate failed all seven additional scenes:
  **0/10** development insertions in total. Its isolated partial score was not
  a useful selection criterion.
- `act_aligned74_fullstate_c8_fastbackbone/005000` failed the additional seven:
  **1/10** total. `act_aligned74_noerror_c8/002500` inserted on additional trial
  8 (87.74), yielding **2/10** total. Neither is reliable.
- `act_aligned87_fullstate_c1/005000` inserted on first-set trial 3 (87.80),
  while its 2,500-step version inserted on trial 1. Both are **1/3**; the
  5,000-step checkpoint is now undergoing the additional seven trials.
- The backbone-LR 1e-4 branch failed all three first-set scenes at both 2,500
  and 5,000 updates. Training stopped cleanly at **5,667** updates. GPU 2 now
  trains `act_aligned87_fullstate_c8_dropout01`, from the same 74-recording
  parent as the 87-recording baseline, with the same data/LRs and dropout 0.1.
  Its 12,000-step LR schedule is unchanged; the wall limit is 38 minutes.
- `act_average_aligned87_c1` is a **derived model, not another optimization
  run**: an equal float64-accumulated average of the 2,500/5,000/7,500-step
  weights from the same one-action run. Configurations and normalizers must
  match; nonfloating buffers must be identical. The result is one ordinary
  ACT network with checkpoint label `000000`, not an adapter or action
  residual. `model_average.json` records component hashes. Four arithmetic/
  compatibility regressions pass. Evaluation is pending.

The third isolated rootless simulator, `aic_compare_validation_20260918`, shares
physical GPU 2 with its assigned trainer. It uses ROS domain 118 and a separate
Gazebo partition. Its first scored ACT evaluation completed normally with
ground truth disabled. **Only physical GPUs 0–3 are used.** All three containers
use the same recorded image ID. Final trials will still use one fixed model
and runtime, with each scene run in a fresh engine process.

`development_ranking.json` combines the explicit first3/remaining7 configs and
rejects duplicate trial mappings for the same checkpoint/runtime. It keeps
3-trial screens separate from complete 10-trial development results.
`aligned87_learning_curves.png` plots training and held-out command errors;
these errors are supporting diagnostics, not insertion scores.

### Last fresh fit and revised final reserve (03:44 UTC)

The corrected recent data had so far primarily trained absolute-target
continuations. One last bounded test trains **fresh ACT** on their original,
verified full TCP-relative command labels: `act_aligned87_relative_fresh_c8`.
It uses ImageNet ResNet18 initialization and newly initialized ACT layers,
**no prior ACT checkpoint and no residual/adapter**. Configuration: the same
70 recent training episodes / 17 held-out episodes, chunk 8 with one executed
action, controller-error inputs masked, dropout 0.1, head LR 1e-4, backbone LR 3e-5,
8,000 requested updates, maximum 38 minutes, and an absolute training deadline of
04:25 UTC. Runtime uses the already tested observation-relative transport
helper: reconstruct the absolute target from the observed TCP before sending
it in `base_link`. Earlier relative candidates mixed historical data, so this
is a fresh test of that representation with aligned clocks and RGB.

The one-action absolute run stopped after saving its 10,000-step checkpoint;
its earlier and later exports remain available for evaluation. The new fit
reuses physical GPU 3. Exact launch and stop counts are recorded in
`fresh_relative_launch.json`.

With three verified, isolated simulators, the 20 final independent trials can
be divided into 7/7/6 jobs. Measured single-trial overhead is about 144 seconds,
so the largest partition is about 17 minutes at the fixed 90 second wall bound.
The model-freeze target is revised to **04:30 UTC**, preserving **40 minutes**
for final evaluation and reporting. This revision is recorded in `budget.json`;
the **05:12:28 UTC eight-hour hard stop is unchanged**.

A score of 1 in a completed development trial was checked against the logs:
the policy ran and sent commands, but the plug remained 0.20 m from the target.
This was a physical failure, not an inferred missing policy invocation.
The engine's “Successful” trial-execution count is not an insertion count;
only the official Tier 3 result is used for learned-policy success here.

### Last candidate checks and completed training (04:22 UTC)

All ACT training is now stopped. The fresh relative-command run completed its
requested **8,000 updates in 2,274.5 seconds**. Its 2,000-, 4,000- and 6,000-step
checkpoints each failed the first three development scenes; changing the
4,000-step execution horizon from one to four actions also failed all three.
At 6,000 updates the totals were 1.00 / 49.60 / 49.79, with two partial insertions
and no Tier 3 = 75. The final 8,000-step checkpoint is saved but has not been
selected or evaluated in simulation. Training completion alone is not a result.

Other late checks did not establish an improvement: the higher backbone LR,
0.1-dropout continuation and 10,000-step full-state chunk-8 model each failed
all three initial development scenes. The averaged one-action model finished
**1/10**, as did its 5,000-step component. Averaging the two no-controller-error
checkpoints failed all three initial scenes. A 180-second wall allowance for
one partial trial of the leading model still failed to insert; final trials
retain the 90-second allowance.

The last comparisons extend the earlier one-action checkpoint and the leading
checkpoint with uniform temporal action averaging to all ten development
scenes. Selection uses completed ten-scene results, ordered by full insertions,
then mean official total score. Three-scene screens are retained separately.
No additional architecture or training branch is being started.

Recent-data validation must be read separately from legacy validation. For the
74-recording continuation there are 60 recent training / 14 recent validation
episodes, plus seven legacy NIC-1 episodes retained in the overall validation
list. The 87-recording runs use 70 recent training / 17 recent validation, again
with those seven legacy validation episodes reported separately in the logs.
Use `corrective_holdout` for the recent-only validation metric. At the end of
the 74-recording no-controller-error run, sampled training translation error
was **0.734 mm**, versus **4.270 mm** on recent held-out frames; these are command
prediction errors, not measured plug alignment. This gap and the live failures
argue against treating lower training loss as sufficient progress.

## Frozen selection (04:27 UTC)

The selected model is `act_aligned74_noerror_c8/checkpoints/002500/pretrained_model`,
copied with its matching normalizer and CUDA TorchScript to `selected_act_final/`.
Selection was frozen at **04:27:48 UTC**, before any final trial. Of 38 screened
checkpoint/runtime combinations on the fixed development splits, seven completed
all ten scenes. Selection used full insertions first, then mean official total.

| Candidate | Step | Execution | Insertions / 10 | Mean total |
| --- | ---: | --- | ---: | ---: |
| `act_aligned74_noerror_c8` | 002500 | 4 actions | 2 | 47.88 |
| `act_aligned74_noerror_c8` | 002500 | 1 actions + uniform temporal averaging | 1 | 42.67 |
| `act_aligned74_fullstate_c8_fastbackbone` | 005000 | 4 actions | 1 | 40.85 |
| `act_aligned87_fullstate_c1` | 002500 | 1 actions | 1 | 40.20 |
| `act_aligned87_fullstate_c1` | 005000 | 1 actions | 1 | 39.81 |
| `act_average_aligned87_c1` | 000000 | 1 actions | 1 | 38.91 |
| `act_aligned54_fullstate_c8_fastbackbone` | 002500 | 4 actions | 0 | 38.07 |

The selected checkpoint’s last stage used **60 recent training episodes** from
`cache_cheat130_aligned74`, with 14 recent validation episodes. Its weights
inherit earlier ACT training; it is not a fresh 2,500-update model. Exact lineage:

| Stage, earliest first | Saved updates in stage | Training episodes sampled |
| --- | ---: | ---: |
| `act_mixed5_rgb_relative_visual_goal` | 26198 | 108 |
| `act_mixed24_rgb_time_velocity_dropout0` | 7907 | 123 |
| `act_nominal19_absolute_time_c8` | 7500 | 15 |
| `act_aligned54_noerror_c8` | 4633 | 44 |
| `act_aligned74_noerror_c8` | 2500 | 60 |

Each continuation resets optimizer/schedule. The earliest stage used an
auxiliary visual-goal loss; later stages removed it. The selected stage rebases
normalization to preserve physical predictions across the data expansion.
All ACT ancestors were trained in this experiment, with ImageNet ResNet18
initialization at the root. No historical task ACT checkpoint was imported.
The root name `relative_visual_goal` refers to its auxiliary goal head; its
main ACT action representation was already absolute pose.

Selected recipe: ResNet18, ACT width 256, three encoder / one decoder layers,
eight heads, chunk 8, batch 64, dropout 0, L1 + VAE KL weight 10; head LR 3e-5
and backbone LR 1e-5 at this continuation’s start. Half of sampling targets the
last eight seconds. Inputs are three RGB images and 33D state including elapsed
simulation time; controller-error projections are zeroed. It predicts the full
absolute TCP pose command, with no action residual or adapter. Runtime: 20 Hz
simulation-clock cadence, four actions per chunk, 100 mm translation-norm and
0.2 rad rotation bounds, zero deadbands, and a 90-second wall limit.

The final sample has 20 fresh SFP NIC-1 / card-0 / port-1 / rail-0 scenes.
Each uses a fresh engine process; three isolated containers divide the fixed
trial list 7/7/6. Exact scene audit: zero overlaps with the 191 training scene
configurations across 47 runs. `frozen_selection.json` hashes the model,
normalizer, runtime sources and scene files. `selection_decision.json` retains
the development table; `training_lineage.json` records every ACT ancestor.
The completed final result is summarized at the top of this report.

## Final failure inspection and expert control

The frozen assessment completed **1/20 insertions**, mean total 39.16.
`selected_act_final/final_results.json` checks readiness, clean engine exit,
all expected trial IDs, disabled privileged inputs, model/runtime hashes, and
unchanged settings. Each scene ran in a fresh engine process. Missing or
incomplete trials could not be silently dropped from the denominator.

The first final failure's official plug-port distance was about 0.05 m.
Post-hoc transforms from its official bag place the tip essentially at the
entrance (**−0.004 mm depth**) with **12.598 mm lateral error** and **0.943 degrees
orientation error**. Port entrance-to-goal depth is 45.800 mm. The last ACT
command still requests a target roughly 41 mm beyond the measured TCP; controller
tracking error is 41.265 mm while terminal motion is nearly stopped. Assuming a
rigid measured TCP-to-tip relation, executing that target fully would still
leave **11.617 mm lateral error**. This is evidence of a misaligned target and
blocked descent, not just a policy that stopped requesting insertion.

These are post-hoc diagnostics, not policy inputs. Transform timestamps differ
slightly; the graph TCP and controller TCP agree numerically. The rigid-grasp
calculation does not simulate contact or cable deflection. Source hashes and
assumptions are in `selected_act_final/posthoc/trial_000001_geometry.json`;
`posthoc/final_motion.json` summarizes recorded motion for all final scenes.

A separate privileged CheatCode control, run only after GPU 2 finished its
frozen ACT trials, scored **94.68, Tier 3 = 75** on the
same first scene. It inserted, confirming that this scene is reachable by the expert.
It is excluded from all learned-policy counts, and no model or runtime was
retuned. Evidence: `recorded_replay_diagnostic/eval_posthoc_final_scene1_expert/`.
The replay checkpoint path in that generic evaluator is a compatibility artifact;
CheatCode does not load or execute it.

All 20 terminal three-camera views and representative videos are under
`selected_act_final/review/`. Videos hold snapshots captured approximately once
per simulation second; they do not reconstruct unrecorded motion.

All four terminal sheets (20 trials × three cameras) and the first/middle/end
sheets for trials 12, 18 and 17 were visually inspected. Trial 12 seats; trials
18 and 19 receive partial-insertion scores; trial 17 ends away from the port.
The remaining trials do not fully insert. The recordings contain **1,328**
three-camera snapshots across the final sample. Visual inspection is recorded
in `selected_act_final/review/manifest.json`; full videos were generated, not
claimed to have been watched frame by frame.

## Reproduction and code verification

The frozen bundle includes the normalizer, CUDA export/sidecar, checkpoint,
runtime source copies, selected development summary, training lineage, image
identity and dependency versions. `environment/` at the experiment root stores
345 installed Python package versions and copies of `pixi.toml` / `pixi.lock`.
`source_snapshot/` records the base commit and complete dirty source/docs snapshot,
including changes from earlier authorized work. It is not a commit or upload.

Focused CPU regressions passed in separate runs: **36** runtime/data/numerical
checks, **6** frozen-result accounting checks, and **4** checkpoint-averaging
checks. The first two logs are `focused_tests_rebase.txt` and
`final_report_tests.txt`; the four averaging cases are in
`aic_utils/lerobot_robot_aic/test/test_average_act_checkpoints.py`. Live final
rollouts additionally verified the packaged model and evaluation path. These
checks establish contracts and accounting, not reliable insertion performance.

## What the next experiment should test

1. **Collect corrections at the actual failed states near insertion.** Reuse
   successful nominal approaches, then query the expert on the learned policy's
   stalled/misaligned contact states and record observation timestamp, teacher
   target and executed command together. Keep failed rollouts with explicit
   labels. The existing bounded student-correction pilot helped fitting but did
   not establish broad recovery. Test whether targeted contact coverage helps
   before collecting another broad nominal batch.
2. **Test visual precision with a controlled comparison.** The recent runs
   reach submillimetre training-command errors while held-out errors stay around
   4–5 mm. First compare a fresh absolute-pose ACT against a warm start on the
   same 87 recent episodes: the late fresh run used relative commands, so it
   does not isolate initialization. Then compare higher-resolution connector
   crops or multiscale features and one pretrained backbone change, with the
   same full-action head, data,
   action convention and live trial budget. A backbone change remains a
   hypothesis; this run did not compare DINOv2 insertion performance. The
   [recent learning curves](../../outputs/experiments/2026-09-17_act_verified_8h/aligned87_learning_curves.png)
   show why additional updates alone are not a convincing next step.
3. **Separate approach fitting from contact recovery measurements.** Report
   arrival at the insertion region, alignment, and official full insertion
   separately. Evaluate a short force/proprioception history as one change,
   without privileged inference or an ACT residual adapter. Inspect timing and
   command traces on failures; a low target-prediction loss does not validate
   feedback behavior under contact.
4. **Then resume ACT → offline SERL → online SERL.** Preserve the corrected
   observation/action contract and use the direct visual full-command actor.
   Include recorded failures for critic learning with trustworthy terminal
   labels. Start with a small fixed-scene evaluation gate before spending a
   large online budget or transferring to Isaac. The Isaac reset/control
   concerns from the earlier report still need their own live gate.

The ten development scenes were reused heavily, so their results are selection
measurements. The frozen final scenes must not be used to tune this model.
Future work should create a new held-out scene set and report per-setting
results before extending claims to other NIC counts or SC connectors.

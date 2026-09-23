# Frozen-pose-conditioned GRU policy diagnostic

Date: 2026-09-21 UTC (stored under the existing September 22 experiment naming convention)

## Question and stop rule

This experiment asked whether the parked observation-only pose estimator makes
a supervised controller better than an otherwise identical action-only
controller. Both controllers had to use the same replay, episode split,
initialization, update budget, architecture, and fresh development starts. True
Isaac plug or port geometry was forbidden from actor inputs. Actor-Q, reward
learning, imagination, and the four reserved final configurations stayed off.

The branch had to stop before RL if pose conditioning did not materially improve
autonomous lateral alignment. Promotion required repeated strict insertion,
lateral error at or below 0.5 mm, acceptable orientation and force, and full
p95 inference below 300 ms.

## Architecture

Each trainable policy head has 546,686 parameters. The frozen locator,
landmark, visibility, and three residual pose networks contain 536,631
parameters, so one deployed arm has **1,083,317 parameters total**. The saved
4.2 MiB training bundle contains both alternative heads; deployment loads one.

```text
three native 576x512 RGB views
        |
        +--> frozen RGB locator --> three 160x160 native-resolution crops
        |                              |
        |                              +--> frozen MobileNet landmarks/visibility
        |                                             |
robot state + state change + force + previous command + visual features
        |
        +--> 297 base values
        |
pose arm only: predicted relative translation (3)
             + ensemble uncertainty (3)
             + observation-derived phase (4)
        |
        +--> 307 total values at each of six causal steps
        |
LayerNorm --> Linear(256) --> GELU --> one-layer GRU(256)
        |
LayerNorm --> Linear(256) --> GELU --> Linear(24)
        |
four consecutive 6D TCP-body-frame delta-pose commands
```

The action-only arm keeps the same 307-wide input and zeros the ten
conditioning slots. This preserves the exact parameter count and initialization.
Both arms use the existing four-command macro transport. They do not load a
world model or use predicted future images.

This first diagnostic includes explicit predicted relative **translation**,
uncertainty, and phase. It does not contain a separately validated predicted
relative-orientation channel. Orientation is still present indirectly in the
RGB features and robot state, and the output includes rotational deltas. The
frozen perception work had validated translation more strongly than opening
normal and plug orientation; adding a noisy explicit orientation estimate would
have changed two variables at once. This limitation is material when interpreting
the result.

The 42-value state already contains force-related values, and the actor also
receives the causal three-axis force explicitly. This duplication is shared by
both arms and therefore does not bias the matched comparison.

## Data and split

The source is the force-safe natural-cable guide replay. Complete cable templates
define the split:

| Split | Cable template(s) | Rows | Complete episodes |
| --- | --- | ---: | ---: |
| Fit | 0, 2 | 205 | 18 |
| Calibration / early stopping | 3 | 79 | 6 |
| Offline development | 5 | 175 | 10 |

The one-row template-4 fragment is excluded. Normalization uses fit rows only.
Rows are sampled with inverse episode-frequency weighting. A six-decision causal
history never crosses an episode boundary. The target is the Isaac CheatCode's
24D four-command TCP-frame delta chunk.

For live evaluation, eight new starts use held-out cable template 5, axial
distances 4.5 or 7.5 mm, lateral radii 0.5, 0.9, 1.3, or 1.7 mm, and angles not
used by the preceding development set. Privileged geometry establishes and
checks the initial condition only. After handoff, the actor runs without the
guide, exploration, action guard, privileged crop selection, or object/target
geometry. The four reserved final IK configurations remain sealed.

## Commands

Training used one physical GPU through rootless Docker:

```bash
CUDA_VISIBLE_DEVICES=1 docker run --rm --gpus device=1 \
  # same image and bind mounts as the preserved Isaac container
  aic-isaac-cable-visibility:20260922-installed \
  /workspace/isaaclab/aic/outputs/experiments/2026-09-22_pose_conditioned_gru_policy/run_offline_matched_training.sh
```

The complete arguments are preserved in
[`run_offline_matched_training.sh`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/run_offline_matched_training.sh).
Live commands and every safety/reset option are in
[`run_live_variant.sh`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/run_live_variant.sh).
The two invocations were:

```bash
./run_live_variant.sh action_only
./run_live_variant.sh pose_conditioned
```

The pose-conditioned run's sixth start failed the privileged force-safe reset
before actor release. Only that untouched scene was recollected with the same
checkpoint and settings. The merged result substitutes this retry; it does not
hide or count the infrastructure failure as a policy episode.

## Offline result

Both checkpoints selected update 100 and stopped at update 1,100 after the
fixed 1,000-update calibration patience elapsed. Lower is better:

| Arm | Development translation MAE / p95 | Development rotation MAE / p95 |
| --- | ---: | ---: |
| Action-only | **0.0367 / 0.0894 mm** | **0.00594 / 0.01421 deg** |
| Pose-conditioned | 0.0369 / 0.0908 mm | 0.00597 / 0.01427 deg |

The differences are tiny, but pose conditioning did not improve held-out
command imitation.

## Autonomous result

All values below use the same timestep for axial, lateral, and orientation
measurements. “Best” selects the timestep with the smallest three-dimensional
plug-to-opening distance within each episode; terminal is the pre-reset final
observation.

| Arm | Strict insertions | Best lateral p50 / p95 | Best absolute axial p50 / p95 | Best orientation p50 / p95 | Terminal lateral p50 / p95 | Max force p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Action-only | 0/8 | 1.479 / 1.678 mm | 0.998 / 1.302 mm | 0.220 / 0.637 deg | 6.382 / 7.473 mm | 1.884 / 2.963 N |
| Pose-conditioned | 0/8 | 1.417 / 1.871 mm | 0.842 / 1.585 mm | 0.219 / 0.252 deg | 6.284 / 9.445 mm | 1.707 / 3.543 N |

The conditioned median best lateral distance is 0.062 mm lower, while its p95
is 0.193 mm worse. Both policies move closer axially and then drift sideways;
terminal lateral error is roughly 6.3 mm at the median. Neither approaches the
0.5 mm insertion corridor reliably. Endpoint sheets for center, left, and right
cameras and the complete MP4s are under
[`visual_review/`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/visual_review/).
Manual review agrees with the geometry metrics: cable pose changes across
starts, the plug approaches the panel, and lateral/rotational drift develops
after handoff.

## Did the GRU use the pose?

A post-training counterfactual audit reran the fixed pose-conditioned checkpoint
on held-out offline sequences with its ten conditioning channels intact, zeroed,
or shuffled. It did not retrain or select a checkpoint.

On the 175-row development split, zeroing pose conditioning changed a predicted
translation command by only 0.0095 mm at the median and 0.0264 mm at p95. The
expert command magnitude was 0.192/0.322 mm p50/p95. Shuffling conditioning had
similarly small effects, 0.0094/0.0231 mm. Command error was essentially
unchanged: 0.03688 mm MAE with the true input, 0.03709 mm with zeros, and
0.03698 mm when shuffled.

The network therefore learned an almost pose-independent imitation solution.
The training trajectories correlate visual state, pose, phase, and expert
action too strongly for ordinary BC to force causal use of the new channels.
This explains why good offline action error did not become closed-loop lateral
correction. It does not prove that the frozen pose estimate could support a
better explicitly structured controller.

## Latency and resources

The complete in-memory inference path includes RGB locator, native crops,
landmark and visibility networks, triangulation/residual ensemble, six-step GRU,
and 24D head. Sensor acquisition is excluded. Across 454 decisions after five
warmups it measured **7.135/7.388/8.249 ms p50/p95/p99**, with 10.828 ms max.
The runtime's approximately 0.1 ms `model_inference_s` is only cached dispatch
and must not be reported as full inference. The 300 ms requirement passes.

Training and evaluation used one GPU at a time. Existing containers and GPU 7's
unrelated allocation were not interrupted. No RL training ran.

## Decision and next work

The predeclared stop rule fired. Do not start actor-Q from either checkpoint,
do not open the reserved final split, and do not interpret the small median
difference as improvement. Further pose-estimator fitting remains parked.

The next useful supervised test should make correction explicit rather than
hope a generic BC head discovers it:

1. Generate balanced, force-safe corrective demonstrations from many lateral
   directions and magnitudes, including retreat and blocked/contact recovery.
2. Predict a bounded correction residual from the observation-only pose,
   uncertainty, recent measured motion, and force, while retaining a learned
   nominal approach command. Train losses that directly supervise correction
   direction and progress, not only per-command L1 error.
3. First test the correction mapping offline by counterfactual pose changes and
   then autonomously on new development starts. Require evidence that changing
   the pose input changes the lateral command in the correct direction.
4. Consider DAgger-style collection only as supervised corrective data
   collection, with unblended teacher labels and executed commands both saved.
   Start model-free RL only after this supervised controller inserts reliably.

If this explicit controller still cannot align, the remaining perception tail
and control/data coverage must be revisited before RL. World-model, SEER,
reward-model, imagination, and Gazebo-adaptation branches remain parked.

## Artifacts

- [`summary.json`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/summary.json): compact decision and main numbers.
- [`artifact_map.json`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/artifact_map.json): durable and bulk artifact locations.
- [`artifact_sha256.txt`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/artifact_sha256.txt): hashes for preserved experiment files.
- [`offline_matched/metrics.json`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/offline_matched/metrics.json): training curves and all split metrics.
- [`matched_live_summary.json`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/matched_live_summary.json): episode-level live metrics and merged retry.
- [`pose_reliance_audit.json`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/pose_reliance_audit.json): true, zero, and shuffled pose counterfactuals.
- [`live_latency.json`](../../outputs/experiments/2026-09-22_pose_conditioned_gru_policy/live_latency.json): complete inference benchmark.

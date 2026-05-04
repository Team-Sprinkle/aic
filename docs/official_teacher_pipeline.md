# Official Teacher Trajectory Pipeline

## Why move away from Gazebo-gym stepping

The experimental `aic_gym_gz` path is useful research code, but its custom
`step()` loop is not the execution stack used by the challenge container. It
adds another timing, observation, and controller abstraction exactly where the
teacher data needs to match evaluation. The new teacher path should therefore
use the official ROS/Gazebo bringup, the `aic_model.Policy` interface, and the
existing LeRobot recorder path. The custom gym code remains reference material
for planning ideas, reward diagnostics, and trace formats, but not the runtime
used to collect final demonstrations.

## Two-run process

The pipeline is split into two runs.

First, a slow oracle planning run may call GPT/VLM backends, inspect images,
pause for multi-step reasoning, and run trajectory optimization. This run
produces a piecewise trajectory JSON. Because the robot should not be blocked
inside the official evaluation control loop while a VLM thinks, this run is not
the final recorded rollout.

Second, a replay-policy run loads a smooth trajectory JSON and executes it in
the official ROS/Gazebo environment with no VLM calls. The LeRobot recorder
records this deterministic replay through the same path used for CheatCode and
teleop data.

## Current working pipeline

As of the May 4, 2026 reliability pass, the working expert-data pipeline is:

1. Launch a live planner recording pass with
   `aic_teacher_official.OfficialExpertGeneratorPlanner`.
2. Capture live task context from the official ROS/Gazebo stack: TFs, current
   TCP pose, plug/port transforms, wrist force baseline, camera observations,
   and task config.
3. Ask GPT-5-mini only for high-level scene strategy and cable-risk reasoning.
   The VLM does not output executable Cartesian waypoints, joint targets, or
   final insertion motions.
4. Generate deterministic candidate staging/pre-insertion targets from the live
   task context.
5. Use MoveItPy for required free-space transport planning to the staging pose.
   If MoveIt is unavailable or cannot plan, the candidate is rejected; there is
   no geometric fallback for free-space transport.
6. Write the planner result as `piecewise_trajectory.json`, then postprocess it
   to `smooth_trajectory.json`.
7. Replay with `aic_teacher_official.OfficialTeacherReplay` in
   `joint_position_then_cheatcode` mode: MoveIt transport is replayed in joint
   space, then the final approach/insertion switches to online
   CheatCode-style Cartesian geometry.
8. Before descent, run local preinsert alignment, precontact port alignment,
   and tracking gates using live TF, TCP speed, and force deltas.
9. During guarded insertion, use F/T trend and threshold checks to decide
   whether to continue, hold/re-align, back off, retry, or reject.
10. Record the official replay with the existing LeRobot recorder and parse the
    official scoring YAML. GPT-5 failure analysis is run only after the episode,
    using sampled images and runtime traces.

The current stable command shape is:

```bash
env AIC_OFFICIAL_TEACHER_ENABLE_LIVE_Z_REPAIR=false \
    AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_MODE=tcp_away_from_port \
    pixi run python scripts/generate_expert_trajectories.py \
      --nominalrecovery \
      --target-accepted-trajectories 3 \
      --max-total-attempts 6 \
      --candidates-per-scene 2 \
      --score-threshold 96 \
      --ft-threshold 2.0 \
      --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
      --output-dir outputs/expert_debug/<run_name> \
      --strategy-model gpt-5-mini \
      --analysis-model gpt-5 \
      --debug \
      --moveit-required true \
      --backup-distance-m 0.02 \
      --backoff-increment-m 0.006 \
      --backoff-stage-sec 0.5 \
      --min-backoff-distance-m 0.002 \
      --max-retries 2 \
      --recovery-release-force-threshold 2.0 \
      --force-confirm-sec 0.10 \
      --gazebo-gui false \
      --launch-rviz false \
      --startup-delay-sec 12 \
      --recorder-drain-sec 120 \
      --planner-recorder-drain-sec 45 \
      --per-trial-timeout-sec 0 \
      --launch-moveit true
```

The latest verified reliability run was:

```text
outputs/expert_debug/vlm_backoff_reliability_cycle10
```

It produced three successful official replays:

```text
attempt_000001_candidate_00: 97.035335987039559
attempt_000002_candidate_01: 97.038856798717887
attempt_000003_candidate_00: 97.056801265524768
```

All three reached insertion, had no official force penalty, and did not need a
runtime backoff. The center-camera videos are:

```text
outputs/expert_debug/vlm_backoff_reliability_cycle10/replay_attempts/attempt_000001_candidate_00/dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
outputs/expert_debug/vlm_backoff_reliability_cycle10/replay_attempts/attempt_000002_candidate_01/dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
outputs/expert_debug/vlm_backoff_reliability_cycle10/replay_attempts/attempt_000003_candidate_00/dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
```

GPT-5 post-run feedback for the best run is saved at:

```text
outputs/expert_debug/vlm_backoff_reliability_cycle10/replay_attempts/attempt_000003_candidate_00/gpt5_replay_analysis_center/analysis.md
```

## What worked and what did not

What worked:

- Keeping the VLM at the strategy/critique level and leaving executable motion
  to MoveIt, deterministic geometry, and guarded replay.
- Replaying MoveIt transport in joint space and switching to online
  CheatCode-style Cartesian insertion near the port.
- Disabling live-Z repair by default for the stable runs. Live-Z repair can be
  useful only if strongly gated by lateral error and force context.
- A 5-sample force median window with centers separated by 5 control samples
  (`250 ms`) for trend checks. This reduces false positives from single noisy
  force samples.
- Preinsert servo compensation with a small cap (`1.5 mm`) and a slightly
  relaxed lateral tracking gate (`2.2 mm`) before final insertion.
- Treating precontact force aborts as recovery/retry opportunities instead of
  immediate terminal failures in nominalrecovery.
- Requiring measured physical backoff, not just commanded backoff. The current
  minimum measured retreat is `2 mm`; this catches cases where the controller is
  pinned against the face and the commanded target is not actually reached.
- Stopping above-contact descent when force is already meaningful but below the
  hard contact threshold. The default soft realign threshold is now `2.0 N`,
  preventing long pushes at `2-5 N` before a harder collision.

What did not work:

- Letting the VLM or planner own fine insertion motion. The successful path uses
  deterministic port/plug geometry for final insertion.
- Assuming command traces prove recovery. Several failed runs logged commanded
  backoff but measured less than `1 mm` of TCP retreat.
- A strict `4 mm` required measured-backoff threshold. One run measured about
  `2.38 mm` of actual retreat and released force; rejecting that was too strict
  for the current controller and task geometry.
- Ignoring above-contact force until the early-contact threshold (`5 N`). Runs
  that sat at `2-4.8 N` above the port eventually pinned hard enough that
  recovery could not retreat.
- Treating precontact port-align force abort as terminal. It should trigger
  the same backoff/realign/retry machinery in nominalrecovery.
- Threshold tuning alone. Raising F/T thresholds may hide early contact but does
  not fix bad lateral alignment or pinned recovery.
- Running with stale `aic_model`/recorder processes. Orphaned policy processes
  caused repeated model-validation failures unrelated to the replay policy.

## Piecewise vs continuous trajectory

Legacy oracle/debug paths can emit coarse or piecewise-continuous waypoints:

- symbolic VLM strategy metadata for scene understanding and cable-risk notes.
- Optimizer-derived alignment waypoints where local constraints matter.
- CheatCode-derived final insertion waypoints once port and plug geometry are
  known.

The postprocessor owns the conversion from that piecewise plan to a single
smooth, timestamped trajectory. The current implementation computes a global
C1 cubic Hermite trajectory across all piecewise waypoints and keeps the
minimum-jerk helper available for future time scaling. This avoids the
stop-and-go behavior of independent local segments while still giving strictly
increasing timestamps, continuous position, continuous velocity at piece
boundaries, explicit phase labels, and TODOs for a future constrained
spline/trajectory optimizer. The current high-scoring generator does not use
VLM-emitted executable waypoints; it uses MoveIt joint-space transport plus
online geometric alignment, guarded insertion, and recovery.

## Hybrid VLM and CheatCode insertion

The intended split is:

- Approach, alignment, and obstacle avoidance may use VLM perception plus a
  numerical optimizer.
- Final insertion should use CheatCode-style geometric logic rather than asking
  the VLM to guess fine contact motion.

In the JSON model, final insertion samples are explicitly phase-labelled
`final_insertion` and marked `source: cheatcode` after postprocessing. This
allows later filtering, scoring, and critique to distinguish semantic planning
from geometric insertion.

The official replay trial on `trial9_2026_0425_205620` showed that the final
descent must be slow, not just geometrically correct. A 2 second final segment
reached the port but only scored partial insertion; stretching the same
CheatCode-derived final waypoint to a 12 second segment produced a successful
insertion and a score of 97.02 on that trial. The generator therefore defaults
`insertion_duration` to 12 seconds.

## Postprocessor responsibility

The postprocessor takes `PiecewiseTrajectory` JSON and emits
`SmoothTrajectory` JSON. It must validate timestamp monotonicity, preserve or
normalize phase labels, interpolate TCP pose, attach diagnostics, and make the
final insertion segment cheatcode-derived. Future versions should add collision
clearance, controller limits, contact-force constraints, and better orientation
and velocity continuity.

## Replay policy responsibility

The replay policy loads a `SmoothTrajectory`, maps elapsed execution time to a
target TCP pose, and converts that target to the official action interface. It
must not import or call VLM/planner backends. The default action mode is now
`relative_delta_gripper_tcp`: each absolute replay target is converted to a
current TCP-relative delta using TF, then sent through
`Policy.set_delta_pose_target()`. This mirrors CheatCode's controller style
more closely and avoids asking the VLM to reason about joints or absolute robot
state during execution.

`absolute_cartesian_pose_base_link` remains available through
`AIC_OFFICIAL_TEACHER_ACTION_MODE` or `--teacher-action-mode`; it matches the
documented `Policy.set_pose_target()` path and the `WaveArm` example.

For nominalrecovery/recovery backoff, the command-frame rule is more specific:
the retreat direction is computed relative to the TCP/tool and port geometry,
but success is judged in measured `base_link` TCP motion. The current
nominalrecovery default commands staged `gripper/tcp` deltas along
`tcp_away_from_port`, tracks actual TCP displacement, and requires at least
`2 mm` measured retreat before retry. Commanded backoff distance is not enough:
if measured retreat is too small, the replay policy tries a measured-backoff
fallback lift and rejects if the TCP remains pinned. The LeRobot recorder should
still store delta-pose labels for learned-policy compatibility.

The May 3 isolated debug run `outputs/debug_cheatcode_modified/run20` remains a
useful frame-direction reference. It used a `1.7 N` force-vector-drop trigger
and measured about `+3.3 mm` actual base-z retreat. Later reliability work
showed the more general lesson: video, recorded TCP state, and force release
must agree before a recovery is considered correct.

## Commands

Generate a first piecewise oracle artifact using explicit geometry:

```bash
pixi run python scripts/official_teacher_generate_piecewise.py \
  --output artifacts/piecewise_trajectory.json \
  --start-position=-0.35,0.35,0.32 \
  --port-position=-0.10,0.45,0.12 \
  --orientation-xyzw=1,0,0,0
```

Legacy standalone piecewise generation can still be exercised for debugging.
Do not use this as the current expert-generator contract: the current
`generate_expert_trajectories.py` path uses GPT-5-mini for symbolic strategy
only and lets deterministic candidate generation plus MoveIt produce executable
motion.

```bash
pixi run python scripts/official_teacher_generate_piecewise.py \
  --output artifacts/piecewise_trajectory.json \
  --use-vlm \
  --max-vlm-calls 20
```

The slow planner can also capture current TCP and target port from TF in a
running official sim:

```bash
pixi run python scripts/official_teacher_generate_piecewise.py \
  --output artifacts/piecewise_trajectory.json \
  --auto-context \
  --target-module-name <module> \
  --port-name <port> \
  --use-vlm
```

In the legacy standalone artifact format, the pre-insertion staging pose is
optimizer-labelled; the final insertion waypoint remains deterministic
geometry, `source: cheatcode`, and has diagnostics referencing
`aic_example_policies/aic_example_policies/ros/CheatCode.py`.

Postprocess the piecewise artifact:

```bash
pixi run python scripts/official_teacher_postprocess.py \
  --input artifacts/piecewise_trajectory.json \
  --output artifacts/smooth_trajectory.json \
  --sample-dt 0.05
```

Print the replay command:

```bash
pixi run python scripts/official_teacher_replay.py \
  --trajectory artifacts/smooth_trajectory.json
```

Run the build-and-replay dry run:

```bash
pixi run python scripts/official_teacher_build_and_replay.py \
  --piecewise-output artifacts/piecewise_trajectory.json \
  --smooth-output artifacts/smooth_trajectory.json \
  --dry-run
```

Use the dataset layout requested for VLM trajectory attempts:

```bash
pixi run python scripts/official_teacher_build_and_replay.py \
  --use-dataset-layout \
  --timestamp 2026_0425_205620 \
  --use-vlm \
  --sample-dt 0.05 \
  --dry-run
```

This writes the first artifact under:

```text
outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620
```

and the postprocessed replay artifact under:

```text
outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620
```

Build a future GPT-5 critique manifest:

```bash
pixi run python scripts/official_teacher_collect_review_context.py \
  --trajectory artifacts/smooth_trajectory.json \
  --output artifacts/review_bundle.json \
  --samples 8
```

Run the single GPT-5 VLM failure-analysis call after a recorded rollout has
images/actions/observations available:

```bash
pixi run python scripts/official_teacher_collect_review_context.py \
  --trajectory artifacts/smooth_trajectory.json \
  --output artifacts/review_bundle.json \
  --wrist-image-dir <sampled_wrist_images> \
  --gazebo-image-dir <sampled_gazebo_images> \
  --samples 8 \
  --use-gpt5-review
```

## LeRobot recording path

Recording should use the existing official-compatible tooling:

```bash
export AIC_OFFICIAL_TEACHER_TRAJECTORY=artifacts/smooth_trajectory.json
bash ./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh \
  --engine-config ./outputs/configs/random_trials.yaml \
  --policy-class aic_teacher_official.OfficialTeacherReplay \
  --teacher-trajectory artifacts/smooth_trajectory.json \
  --teacher-action-mode relative_delta_gripper_tcp \
  --dataset-repo-id ${HF_USER}/official_teacher_dataset \
  --dataset-root ./outputs/lerobot_datasets \
  --gazebo-gui false \
  --launch-rviz false \
  --startup-delay-sec 8 \
  --per-trial-timeout-sec 0 \
  --recorder-drain-sec 120 \
  --require-recorder-save-log true
```

Keep `--startup-delay-sec` near the `exp/data` default of 8 seconds. The engine
expects the model node within its discovery window; a longer delay such as 25
seconds can make model validation fail before the policy starts.

`launch_policy_recording_per_trial.sh` and
`launch_policy_recording_tmux.sh` now both accept `--teacher-trajectory` and
`--teacher-action-mode`, which set `AIC_OFFICIAL_TEACHER_TRAJECTORY` and
`AIC_OFFICIAL_TEACHER_ACTION_MODE` for the policy process. The simulation and
recorder do not need these variables.

## Future GPT-5 critique loop

After a replay is recorded, a critique job should sample equidistant timesteps
from the trajectory, export synchronized camera frames plus metadata, and ask a
GPT-5 VLM critic to identify likely failure causes or unsafe geometry. Critique
belongs after replay/recording so it can compare the planned trajectory,
executed TCP state, phase labels, action metadata, and images without adding VLM
latency to execution.

The current review manifest builder handles missing image directories
gracefully and records `missing_wrist_images` / `missing_gazebo_images` flags.
Passing `--use-gpt5-review` makes exactly one GPT-5 review call.

## Iterative improvement loop

The autonomous loop repeats the same two-run process up to four times on the
same scene spawn settings:

1. Generate or reuse a `PiecewiseTrajectory`.
2. Postprocess it into a smooth replay trajectory.
3. Replay and record through the official LeRobot path.
4. Build a review bundle from equidistant recorded frames, observations,
   actions, phase labels, and score metadata.
5. Optionally make one GPT-5 VLM failure-analysis call and feed that feedback
   into the next GPT-5 mini planner call.

Dry-run the loop and print the exact official replay commands:

```bash
pixi run python scripts/official_teacher_iterate.py \
  --root-dir outputs/trajectory_datasets \
  --base-run-name trial9_2026_0425_205620 \
  --engine-config outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/oracle_engine_config.yaml \
  --max-loops 4 \
  --dry-run
```

Run the loop in the official eval container path:

```bash
pixi run python scripts/official_teacher_iterate.py \
  --root-dir outputs/trajectory_datasets \
  --base-run-name trial9_2026_0425_205620 \
  --engine-config outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/oracle_engine_config.yaml \
  --context-json outputs/trajectory_datasets/sfp_to_nic/vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620/piecewise_trajectory_from_official_v2.context.json \
  --use-vlm \
  --use-gpt5-review \
  --score-threshold 80 \
  --max-loops 4 \
  --record
```

Loop 1 uses the base run name plus an explicit loop suffix:

```text
outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_1
```

Later loops keep the same trial timestamp and increment the loop index:

```text
outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_2
```

Each loop writes `loop_manifest.json` with the planner root, postprocessed
root, smooth trajectory path, LeRobot dataset root, score path, parsed score
if available, and the exact replay command. The loop stops once the parsed
score reaches `--score-threshold` unless `--force-all-loops` is set. New base
trials should use a new `--base-run-name` under the same `n1` attempt folder;
the script creates new directories and does not delete existing trials.

For loop feedback, the runner samples 10 equidistant timesteps by default. If
more than one prior loop exists, the GPT-5 review context includes both the
immediate previous loop and the best-scoring prior loop. For example, if loop 1
scored 97 and loop 2 dropped, the loop-3 review bundle includes both loop 1 and
loop 2, with planned TCP pose, recorded observation values, recorded action
values, wrist images for each sampled frame, phase/source labels, the parsed
official score breakdown, raw `scoring.yaml` scorer messages, `score_summary.csv`
text when available, and scoring-related excerpts from the container-generated
simulation/policy/recorder logs. GPT-5 review receives all extracted sample
images, not a truncated image subset.

Build that comparison bundle manually:

```bash
pixi run python scripts/official_teacher_collect_review_context.py \
  --output artifacts/loop1_loop2_comparison_review.json \
  --samples 10 \
  --comparison-run 'loop_1|outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_1/smooth_trajectory.json|outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_1/raw_dataset|outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_1/scores/trial_1_trial_000001/scoring.yaml' \
  --comparison-run 'loop_2|outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_2/smooth_trajectory.json|outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_2/raw_dataset|outputs/trajectory_datasets/sfp_to_nic/vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620_loop_2/scores/trial_1_trial_000001/scoring.yaml'
```

## Known limitations

- Automatic context extraction currently reads TF in a running official sim; it
  does not yet subscribe to full observations or task messages.
- The current expert generator does not let the VLM emit executable waypoints.
  Older standalone piecewise scripts still exist for debugging historical
  artifacts and should not be treated as the production contract.
- Final insertion is CheatCode-style geometry with guarded timing. The current
  best VLM/MoveIt nominalrecovery path uses online CheatCode-style final
  alignment/insertion after joint-space MoveIt transport.
- Replay defaults to relative TCP deltas and falls back to absolute pose for a
  tick if TF lookup fails. Velocity-mode replay is not implemented yet.
- Gazebo scene images and LeRobot action/force summaries are only referenced in
  the review manifest until recorded dataset extraction is implemented.

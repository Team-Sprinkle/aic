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

## Piecewise vs continuous trajectory

The oracle can emit coarse or piecewise-continuous waypoints:

- VLM-derived approach waypoints for scene understanding and obstacle avoidance.
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
spline/trajectory optimizer.

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

## Commands

Generate a first piecewise oracle artifact using explicit geometry:

```bash
pixi run python scripts/official_teacher_generate_piecewise.py \
  --output artifacts/piecewise_trajectory.json \
  --start-position=-0.35,0.35,0.32 \
  --port-position=-0.10,0.45,0.12 \
  --orientation-xyzw=1,0,0,0
```

Generate with GPT-5 mini VLM planning. The prompt asks for Cartesian delta
waypoints in `base_link`, not joints and not final insertion. Up to 20 planner
calls are allowed by the CLI budget, but the current implementation uses one
call per generated trajectory:

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

Without `--use-vlm`, the approach and alignment phases are `placeholder_vlm`
and `placeholder_optimizer`. With `--use-vlm`, approach/alignment waypoints are
`source: vlm`; the pre-insertion staging pose is optimizer-labelled; the final
insertion waypoint remains deterministic geometry, `source: cheatcode`, and has
diagnostics referencing `aic_example_policies/aic_example_policies/ros/CheatCode.py`.

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
- The VLM planner emits Cartesian delta waypoints, but the optimizer is still a
  conservative pre-insertion staging adapter rather than a full constrained
  trajectory optimizer.
- Final insertion is CheatCode-style geometry with slow timing. The replay
  policy also has an opt-in live final insertion mode via
  `AIC_OFFICIAL_TEACHER_ONLINE_CHEATCODE_INSERTION=true`, but the best observed
  official-teacher score so far came from the smooth slow-final replay path.
- Replay defaults to relative TCP deltas and falls back to absolute pose for a
  tick if TF lookup fails. Velocity-mode replay is not implemented yet.
- Gazebo scene images and LeRobot action/force summaries are only referenced in
  the review manifest until recorded dataset extraction is implemented.

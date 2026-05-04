# Agent Teleop Failure Analysis

This debug path is opt-in. It does not change normal `aic_model` policy execution or the default official teacher replay path.

## What It Records

`scripts/run_agent_teleop_failure_debug.py` runs the current GPT-5-mini official-teacher planner path several times through the official recording launcher and writes a bundle under `outputs/failure_analysis/<run_id>/`.

Each attempt includes:

- `planner_prompt.json`, `planner_response.json`, and `planner_response.txt` when the VLM call succeeds.
- `piecewise_trajectory.json` with parsed local planner waypoints and CheatCode-derived insertion waypoints.
- `smooth_trajectory.json` with the postprocessed replay trajectory.
- `segment_vlm_trajectory.json` from the existing build script.
- `trace.json` with 0.5 second samples by default.
- `image_manifest.json` with image validation and optional image descriptions.
- command stdout, stderr, exact argv, exit code, seed, run mode, git commit, branch, and dirty status.

Unavailable live fields are recorded as `null` plus a reason. Dry runs do not have actual robot pose, joint state, wrench, score, or controller feedback.

## Three-Attempt Debug Run

The runner does not accept offline scene-image paths for planning. It starts the official recording stack with `OfficialTeacherOraclePlanner`, captures a short live image window from `/observations`, validates those images, and sends only valid nonblank live images to GPT-5-mini.

```bash
pixi run python scripts/run_agent_teleop_failure_debug.py
```

Useful options:

```bash
pixi run python scripts/run_agent_teleop_failure_debug.py \
  --num-attempts 3 \
  --seed 0 \
  --config ./outputs/configs/random_trials_10.yaml \
  --output-dir outputs/failure_analysis \
  --sample-period 0.5 \
  --validate-images \
  --describe-images \
  --max-images 16
```

For each attempt, the script runs an official live planner recording pass, postprocesses the resulting `piecewise_trajectory.json`, then runs an official replay recording pass. It does not expose a planner/replay dry-run mode for failure capture. ROS, Gazebo, the engine, and the LeRobot recording launcher must be available.

## Image Validation

For every image found under an attempt directory, the runner checks:

- file exists and has nonzero bytes,
- image decoder can open it,
- width, height, channels, min, max, mean, and standard deviation,
- all-black/all-zero, all-white, and near-constant images.

Invalid or blank images are marked clearly and are not sent to the image-description model.

## Image Descriptions

When `--describe-images` is enabled, valid nonblank images are sent to the configured vision-capable model, default `gpt-5-mini`, for a short scene description. If `OPENAI_API_KEY` is unavailable, the manifest records that description was unavailable instead of failing the run.

Use `--dry-run-image-descriptions` only to avoid image-description model calls while still running the official recording path.

## Analysis

Build the prompt and compact payload without calling GPT-5:

```bash
pixi run python scripts/analyze_agent_teleop_failure.py \
  --run-dir outputs/failure_analysis/<run_id> \
  --dry-run
```

Call GPT-5:

```bash
pixi run python scripts/analyze_agent_teleop_failure.py \
  --run-dir outputs/failure_analysis/<run_id>
```

The runner also accepts `--analyze-with-gpt5` or `--dry-run-analysis`. `--dry-run-analysis` only skips the final GPT-5 failure-analysis call; it does not make the planner or recording path a dry run.

## Outputs

At the run root:

- `summary.json`: compact attempt status, image counts, trace sample counts, and scores if present.
- `prompt.md`: exact GPT-5 analysis prompt.
- `bundle_manifest.json`: all files in the bundle.
- `bundle.zip`: optional, if `--zip` is passed.
- `failure_analysis.md` and `failure_analysis.json`: written by the analysis script.

## What Is Sent To GPT-5

The analysis payload is selected and compacted to control cost. It includes all three attempts, planner prompts/responses, parsed VLM plans, piecewise waypoints, smooth trajectory metadata, sampled trace rows, image validity/descriptions, command results, and score/recording metadata when available.

It does not send every smooth waypoint by default. Full local traces remain on disk.

## Privacy And Cost

Planner calls, image descriptions, and failure analysis can send prompts, selected images, and compact trace data to OpenAI. Review `prompt.md` and `image_manifest.json` before using `--analyze-with-gpt5` or enabling image descriptions on sensitive data.

## Interpreting Results

Use the three attempts to separate planner variance from deterministic pipeline bugs:

- If all attempts lack images or ROS context, the local planner is under-specified.
- If VLM plans differ substantially but smoothing/replay behaves consistently, planner capability or prompt context is suspect.
- If piecewise plans look plausible but smooth samples bend through obstacles or arrive at bad handoff poses, focus on smoothing and frame semantics.
- If smooth trajectories are plausible but actual reached poses diverge, focus on action mode, timing, velocity limits, controller behavior, and replay rate.
- If CheatCode works alone but fails after handoff, verify the pre-insertion pose, plug-tip/gripper offset, orientation frame, and online CheatCode TF state at the handoff.

For contact/recovery failures, do not rely on command traces alone. Sample the narrow interval around the first large F/T change at higher temporal resolution, include center-camera frames, and compare:

- median-filtered force now versus about `150 ms` earlier;
- commanded TCP-frame recovery delta;
- transformed absolute `base_link` target;
- measured TCP/base z during the next `0.5-0.8 s`;
- whether force release coincides with a measured physical retreat.

The May 3 `CheatCodeModified` run20 is the reference behavior for this check:

```text
outputs/debug_cheatcode_modified/run20
center video:
/home/ubuntu/ws_aic/src/aic/outputs/debug_cheatcode_modified/run20/dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
```

Run20 showed that the earlier apparent non-backoff was a frame-direction bug, not mainly a stiffness/damping issue. The policy detected a force-vector drop, computed a TCP-frame delta, converted it to a fixed absolute `base_link` target with positive base z, resent that target for the latch duration, and the recorded TCP state moved upward about `3.3 mm` with default gains. If future GPT-5 failure analysis sees a backoff command but no measured z retreat, first check whether the transformed base-frame axial component points deeper into insertion.

GPT-5-mini should remain in the control loop only if it has enough scene context, produces stable and physically executable subgoals, and deterministic robotics tools can verify or repair its output. If it cannot produce metric waypoints reliably, prefer using it for scene interpretation, high-level strategy, subgoal selection, or review while IK, MoveIt, cuRobo, FCL/collision checking, visual servoing, and trajectory optimization generate executable motion.

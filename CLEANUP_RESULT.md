# Cleanup Result

## Branches

- Base branch: `main` at `origin/main`
- Cleanup branch: `cleanup/main-vlm-data-gazebo-rl`
- Merged branch: `origin/feat/gazebo-rl`
- Fetched reference branch: `origin/exp/data`

## Removed Bad Files

- Removed tracked `install/` artifacts.
- Removed tracked `pixi.lock`.
- Added ignores for `install/`, `log/`, and `pixi.lock`.

Validation note: `git diff --name-only origin/main...HEAD | grep -E '(^install/|^pixi.lock$)'` reports these paths because they are intentionally deleted relative to `origin/main`.

## Affected Remote Feature Branches

Branches containing `install/` or `pixi.lock`:

- `origin/feat/agent-teacher-teleop`: `install/`, `pixi.lock`
- `origin/feat/data-serial`: `pixi.lock`
- `origin/feat/gazebo-rl`: `install/`
- `origin/feat/hybrid_teleop`: `pixi.lock`

Remote history was not rewritten.

## Removed Tool References

- Removed the obsolete official-teacher follow-up prompt document.
- Raw repository text search for the removed tool name returns no matches after generated local environments/artifacts are removed.

## Pixi Decision

- Kept the root `pixi.toml` change that adds `ros-kilted-gazebo-rl = { path = "aic_utils/gazebo_rl" }`.
- Did not keep `pixi.lock`.
- Docs indicate Pixi is required for local install/runtime workflows, while challenge evaluation uses participant containers. Keeping the minimal Gazebo RL package dependency is necessary for local package installation and validation.
- `pixi install` passed after fixing local `.pixi` ownership left by previous root-owned files.

## Policy And CheatCode Decision

- Preserved `aic_model/aic_model/policy.py` delta-pose support.
- Preserved `aic_example_policies/aic_example_policies/ros/CheatCode.py` delta-pose behavior.
- Confirmed absolute Cartesian pose remains available through `set_pose_target()`.
- Confirmed joint commands are still supported through `move_robot(joint_motion_update=...)`.
- Delta-pose is additive, not globally forced.

## LeRobot Decision

- `aic_robot_aic_controller.py` supports Cartesian delta-pose and joint velocity modes through `teleop_target_mode`.
- `aic_teleop.py` provides both end-effector delta controls and joint teleop controls.
- `types.py` keeps separate Cartesian delta and joint action schemas.
- Dataset pipeline expects Cartesian delta-pose actions, but joint/velocity control remains available.

## Test Results

- `pixi install`: passed.
- `python -m compileall ...`: bare `python` is unavailable on this machine; `pixi run python -m compileall aic_teacher_official aic_example_policies aic_utils aic_model` passed.
- `pixi run python -m pytest aic_teacher_official/test/test_official_teacher_pipeline.py -q`: 23 passed.
- `pixi run python -m pytest aic_model/test/test_policy_delta_pose.py -q`: 2 passed.
- `pixi run python -m pytest aic_utils/lerobot_robot_aic/test/test_generate_trajectory_dataset.py -q`: 11 passed.

## CheatCode Dataset Generation

Request YAMLs found:

- `aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_minimal.yaml`
- `aic_utils/lerobot_robot_aic/config/data_generation_templates/sc_to_sc_minimal.yaml`
- `outputs/trajectory_requests/sfp_to_nic_minimal_n3_test.yaml`

Command run with `outputs/trajectory_requests/sfp_to_nic_minimal_n3_test.yaml`.

Result after reverting the default sim distrobox from `aic_eval_0415` to `aic_eval` and creating the local `aic_eval` distrobox from `ghcr.io/intrinsic-dev/aic/aic_eval:latest`:

- Episode saved with `success=True`.
- Score: `96.247497515004227`.
- Summary: `outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/scores/score_summary.csv`.

## Agent Teacher Smoke

- `OPENAI_API_KEY` was present.
- Literal requested loop failed because current `scripts/official_teacher_build_and_replay.py` does not support `--output-dir`.
- Adapted current CLI dry run with `--root-dir`, `--use-dataset-layout`, `--use-vlm`, and `--dry-run` passed and generated piecewise/smooth trajectories.
- Adapted record attempt first failed because the default `./outputs/configs/random_trials_10.yaml` engine config does not exist.
- Adapted record attempts with explicit generated engine config ran after the local `aic_eval` container was created.
- Attempts before fixes scored `0` or `1`; scorer reported no insertion and final plug-port distance around `0.40m`.
- Fixes added in this cleanup:
  - Segment manifest is written at `metadata/segment_vlm_trajectory.json` with all piecewise VLM/optimizer/cheatcode segments, insertion boundary, postprocessing metadata, run status, and score.
  - `smooth_trajectory.json` recording metadata is updated after recording.
  - Dataset-layout runs now pass `--results-root` and `--tmp-dir`, so score summaries and logs stay under the attempt output.
  - Global smoothing is excluded from the final insertion segment; insertion samples are linear between cheatcode geometry waypoints and marked `insertion_smoothing_protected`.
  - Official replay enables online cheatcode insertion by default and copies the CheatCode XY integrator for the final phase.
- Post-fix scores:
  - Attempt 7: `51.405467971421693`; partial insertion detected, final distance `0.05m`.
  - Attempt 8: `38.48677683183066`; no insertion event, final distance `0.05m`.
- No score over 90 was reached in the 8 attempts run so far. The remaining issue appears to be that the offline generated approach uses static CLI default start/port context instead of live task TF context; final online cheatcode insertion can partially recover but not enough for high score.
- Agent-teacher smoke artifacts were generated as local ignored outputs during validation and are not included in the cleanup commit.

## Gazebo RL Results

- Literal `pixi run python -m pytest aic_gym_gz/tests -q || true`: path does not exist.
- Actual merged package tests after the stale Zenoh cleanup fix: `pixi run python -m pytest aic_utils/gazebo_rl/test -q`: 18 passed.
- Literal `pixi run python scripts/train.py --episodes 1 --steps 10 --headless || true`: `scripts/train.py` does not exist.
- Actual training entry point was run with:
  `pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_train_short.py --workspace-dir . --engine-config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml --sim-distrobox aic_eval --max-iterations 2 --max-steps 3 --max-minutes 3 --per-trial-timeout-sec 120 --output-dir outputs/gazebo_rl_smoke`
- Initial result: failed because a stale Zenoh router kept port `7447` busy, the engine could not find a valid clock, and the trainer timed out waiting for the Gazebo RL bridge connection.
- Fix added: `GazeboRLRunner` now clears stale host and distrobox `rmw_zenohd` processes before launching each rollout, matching the cleanup already used by the LeRobot per-trial launcher.
- Final result: completed end to end with `--max-iterations 2 --max-steps 3`. Both iterations connected to the bridge policy, ran 3 real steps, wrote `outputs/gazebo_rl_smoke/checkpoints/gazebo_rl_short.pt`, and wrote `outputs/gazebo_rl_smoke/run_summary.json`.
- Final smoke metrics: 2 requested iterations, 2 completed iterations, about 85.6 seconds elapsed, rewards `-0.02` and `-0.02`, losses `7.752313422315638e-07` and `4.337087773365056e-07`.
- Gazebo RL smoke artifacts were generated during validation, then removed from the worktree because they are ignored local outputs and should not be committed.
- Minimal Gazebo RL import smoke passed for action clipping, env class import, and reward helper import.

## Risks

- Live dataset generation, official teacher scoring, and Gazebo RL smoke validation require the local `aic_eval` distrobox; it was created locally for validation but is not part of the repository.
- The requested final bad-file grep is inconsistent with removing files that already exist on `origin/main`; the PR necessarily shows deletions for `install/` and `pixi.lock`.
- Agentic official-teacher scoring remains below target until the first-pass VLM approach is generated from live task context or the two-pass `OfficialTeacherOraclePlanner` flow is wired into `official_teacher_build_and_replay.py`.

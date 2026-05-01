# Expert Trajectory Generator

This branch now separates high-level strategy from executable motion generation for AIC cable insertion.

## Architecture

The expert generator has two explicit modes:

- `nominal`: smooth, high-scoring, minimal-contact demonstrations.
- `recovery`: guarded recovery demonstrations for later HIL-SERL or online RL guidance.

The VLM is not allowed to generate Cartesian waypoints, joint targets, velocities, or executable trajectories. GPT-5-mini is used only for scene understanding, cable sweep/collision risk assessment, and symbolic approach strategy. GPT-5 is reserved for offline critique/debug analysis.

Free-space approach planning requires MoveIt. There is no geometric fallback. If MoveIt is unavailable or planning fails, the candidate is rejected and the generator records a structured failure.

Final alignment and insertion use CheatCode-style ground-truth geometry. MoveIt is used only for obstacle-aware free-space motion to staging/pre-insertion poses.

MoveIt approach plans are replayed in joint space. The MoveItPy backend extracts the planned joint trajectory, computes TCP poses for metadata/sampling, stores joint names/positions/velocities in the trajectory JSON, and the replay policy sends `JointMotionUpdate` commands until the final-insertion phase. Final insertion then switches to the existing online CheatCode-style Cartesian insertion.

## Package Layout

The implementation lives under:

`aic_teacher_official/aic_teacher_official/expert_generator/`

Key modules:

- `scene_snapshot.py`: serializable robot, target, camera, TF, F/T, config, and collision-object context.
- `vlm_strategy.py`: strict JSON schema and validation for symbolic strategy output.
- `vlm_strategy_client.py`: GPT-5-mini client for strategy-only prompts.
- `candidate_generation.py`: deterministic candidate staging poses.
- `moveit_planner.py`: MoveIt-required planner wrapper with no fallback.
- `moveit_py_backend.py`: MoveItPy sequential planning and joint-trajectory extraction.
- `nominal_expert.py`: nominal trajectory assembly, no F/T correction.
- `recovery_expert.py`: recovery trajectory assembly and `recover_from_state(...)` API.
- `ft_guard.py`: guarded insertion/backoff/retry state machine.
- `trajectory_validator.py`: replay acceptance filters.
- `dataset_writer.py`: LeRobot-compatible sidecar metadata writer.
- `generation_loop.py`: target-accepted generation loop.

## VLM Strategy

Nominal VLM output must be strict JSON:

```json
{
  "mode": "nominal",
  "approach_side": "above_left",
  "cable_risk": "medium",
  "reason": "The cable appears to trail near the front edge of the board.",
  "mitigation": "Use a high-clearance approach and avoid sweeping laterally across the NIC face.",
  "preferred_clearance_m": 0.12,
  "avoid_regions": ["front_of_nic"],
  "insertion_strategy": "straight_slow_descent",
  "recovery_allowed": false
}
```

Recovery output may additionally include:

```json
{
  "probe_pattern": "small_cross",
  "backup_distance_m": 0.006,
  "retry_count": 3
}
```

The parser rejects malformed JSON, validates required fields, and clamps numeric values to safe ranges. Raw prompts/responses and parsed strategy are saved in debug metadata.

## Nominal Mode

Nominal generation:

1. Capture a `SceneSnapshot`.
2. Ask GPT-5-mini for cable-risk/strategy JSON.
3. Generate candidate staging poses.
4. Use MoveIt to plan free-space approach.
5. Extract the MoveIt joint trajectory for replay.
6. Append CheatCode-style geometric insertion.
7. Replay in Gazebo.
8. Validate score, insertion event, F/T/contact limits, and tracking metrics.
9. Accept only passing trajectories into the dataset.

Nominal mode does not use F/T correction, intentional probing, backoff, or retry logic.

## Recovery Mode

Recovery generation starts like nominal mode through free-space approach. Once contact matters, it uses F/T data:

1. Guarded insertion.
2. If soft F/T threshold is exceeded, stop descent.
3. Back off by `backup_distance_m`.
4. Realign using CheatCode-style geometry.
5. Retry insertion up to `max_retries`.
6. If hard threshold or off-limit contact occurs, reject unless a later retry succeeds cleanly.

The recovery expert exposes:

```python
recovery_expert.recover_from_state(current_observation, current_scene_snapshot)
```

This keeps the recovery logic reusable for future HIL-SERL or online RL guidance.

## Dataset Metadata

The standard LeRobot dataset structure should remain unchanged. Extra expert labels are written as sidecars under `meta/`:

- `meta/expert_trajectory_metadata.jsonl`
- `meta/phase_labels.jsonl`
- `meta/validation_results.jsonl`
- `meta/vlm_strategy.jsonl`

Each episode metadata record includes mode, scene ID, candidate index, validation metrics, VLM cable risk, MoveIt result, phase labels, and source candidate metadata.

## Live CLI Flow

Without `--dry-run-config`, `scripts/generate_expert_trajectories.py` now runs an accepted-trajectory loop:

1. Launch a live planner-recording pass with `aic_teacher_official.OfficialExpertGeneratorPlanner`.
2. Capture live TF, observations, wrist/camera images, F/T baseline, task config, and rigid scene metadata.
3. Ask GPT-5-mini for strategy/cable-risk JSON only.
4. Try the requested candidate index with MoveIt-required free-space planning.
5. Write `piecewise_trajectory.json` with MoveIt joint positions/velocities when planning succeeds.
6. Postprocess to `smooth_trajectory.json`.
7. Launch `aic_teacher_official.OfficialTeacherReplay` with `joint_position_then_cheatcode`.
8. Parse official scoring YAML and validate thresholds.
9. Count only accepted trajectories toward `--target-accepted-trajectories`.

Planner attempts are stored under `planner_attempts/`; replay attempts are stored under `replay_attempts/`; accepted sidecar metadata is stored under `accepted_metadata/meta/`.

The current MoveIt integration is deliberately strict: if the required MoveIt Python planning backend is unavailable, generation fails before silently producing geometric trajectories. The ROS message package `moveit_msgs` alone is not treated as a planner. The workspace now depends on `ros-kilted-moveit-py`, and the live policy uses `MoveItPyPlanningBackend` to call MoveItPy for sequential free-space planning and joint-trajectory extraction.

## CLI

Nominal example:

```bash
pixi run python scripts/generate_expert_trajectories.py \
  --expert-mode nominal \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 100 \
  --max-total-attempts 300 \
  --candidates-per-scene 5 \
  --score-threshold 95 \
  --max-offlimit-contacts 0 \
  --require-insertion-event true \
  --rerandomize-scene true \
  --strategy-model gpt-5-mini \
  --output-dir outputs/expert_datasets/nominal_sfp2nic
```

Recovery example:

```bash
pixi run python scripts/generate_expert_trajectories.py \
  --expert-mode recovery \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 100 \
  --max-total-attempts 500 \
  --candidates-per-scene 5 \
  --score-threshold 90 \
  --ft-soft-threshold 1.0 \
  --ft-hard-threshold 3.0 \
  --backup-distance-m 0.006 \
  --max-retries 3 \
  --probe-pattern small_cross \
  --require-insertion-event true \
  --rerandomize-scene true \
  --strategy-model gpt-5-mini \
  --output-dir outputs/expert_datasets/recovery_sfp2nic
```

## Validation

Every candidate must be replayed in Gazebo and filtered before acceptance. Validation metadata includes:

- score
- insertion event reached
- max F/T magnitude
- F/T impulse, when available
- max tracking error
- off-limit contact count
- trajectory duration
- replans/retries
- mode
- VLM cable risk
- MoveIt success/failure
- candidate index
- scene seed/config
- phase labels

The generation loop targets accepted trajectories, not raw attempts. `--max-total-attempts` bounds runtime.

## Rerandomization

The CLI exposes:

- `--rerandomize-scene true|false`
- `--respawn-assets true|false`
- `--scene-randomization-config PATH`

Current repo utilities distinguish generated engine configs from per-run robot reset. The CLI exposes rerandomization flags in metadata; full per-attempt respawn is still governed by the engine config passed to the per-trial launcher. Full asset respawn should be treated as expensive.

## Limitations

- Cable dynamics are not explicitly modeled.
- GPT-5-mini estimates cable collision/sweep risk from live images and context only.
- MoveIt handles rigid obstacles and inflated keep-out regions, not deformable cable physics.
- Gazebo replay/score/contact/F/T validation remains the final filter.
- The live planner captures snapshots and uses MoveItPy for free-space planning. Replay uses the MoveIt joint trajectory for approach, then switches to online CheatCode-style final insertion.

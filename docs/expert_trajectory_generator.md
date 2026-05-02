# Expert Trajectory Generator

This branch now separates high-level strategy from executable motion generation for AIC cable insertion.

## Architecture

The expert generator has three explicit modes:

- `nominal`: smooth, high-scoring, minimal-contact demonstrations for ACT training. Recovery is disabled; F/T threshold or validation failures reject the trajectory.
- `nominalrecovery`: nominal-first demonstrations that may include labeled backoff, realign, and retry segments after slight contact or near-failure.
- `recovery`: guarded recovery demonstrations for later HIL-SERL or online RL guidance.

The VLM is not allowed to generate Cartesian waypoints, joint targets, velocities, or executable trajectories. GPT-5-mini is used only for scene understanding, cable sweep/collision risk assessment, and symbolic approach strategy. GPT-5 is reserved for offline critique/debug analysis.

Free-space approach planning requires MoveIt. There is no geometric fallback. If MoveIt is unavailable or planning fails, the candidate is rejected and the generator records a structured failure.

Final alignment and insertion use CheatCode-style ground-truth geometry. MoveIt is used only for obstacle-aware free-space motion to staging/pre-insertion poses. The handoff now has an explicit `local_preinsert_align` phase before settle and guarded insertion.

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
- `nominal_expert.py`: nominal trajectory assembly, no post-contact F/T correction.
- `trajectory_repair.py`: pre-contact nominal approach retiming that rewrites executable sampled targets and pins the pre-insertion pose.
- `recovery_expert.py`: recovery trajectory assembly and `recover_from_state(...)` API.
- `ft_guard.py`: guarded insertion/backoff/retry state machine.
- `debug_artifacts.py`: 0.5 second debug sidecars, transition metrics, F/T window aggregation, and GPT-5 failure-analysis payloads.
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
  "backup_distance_m": 0.002,
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
6. Append `local_preinsert_align`, pre-insertion settle, and CheatCode-style guarded insertion.
7. Replay in Gazebo.
8. Before descent, enforce the pre-insertion tracking gate. The default gate requires low pose error, low TCP speed, and no F/T threshold change. If it fails in nominal mode, reject without descending.
9. Validate score, insertion event, F/T/contact limits, phase speed metrics, and tracking metrics.
10. If the pre-insertion portion was contact-free but too rough, repair only the approach-to-pre-insertion portion with minimum-jerk retiming, rewrite executable action targets, preserve the exact pre-insertion pose/orientation, and rerun live validation.
11. Accept only passing trajectories into the dataset.

Nominal mode does not use post-contact F/T correction, intentional probing, backoff, or retry logic. If contact/F/T violation happens before or during insertion, nominal rejects the attempt rather than recovering.

## Recovery Mode

Recovery generation starts like nominal mode through free-space approach. If the pre-insertion tracking gate fails, nominalrecovery/recovery performs one local realign and re-checks the gate; if the gate still fails, it rejects before descent. Once contact matters, it uses F/T data:

1. Guarded insertion.
2. If the configured F/T threshold is exceeded, stop descent.
3. Back off by `backup_distance_m`.
4. Wait for force release before lateral realignment.
5. Realign using CheatCode-style geometry.
6. Retry insertion up to `max_retries`.
7. If off-limit contact or validation failure remains, reject unless a later retry succeeds cleanly.

The recovery expert exposes:

```python
recovery_expert.recover_from_state(current_observation, current_scene_snapshot)
```

This keeps the recovery logic reusable for future HIL-SERL or online RL guidance. The default future takeover behavior is `finish_episode`, with the code shaped so later callers can support `assist_and_return`, `finish_episode`, and `terminate_and_reset`.

## HIL-SERL Recovery Design

In a later HIL-SERL setup, the learned policy should roll out normally while a failure detector monitors F/T threshold, off-limit contacts, insertion stuck state, timeout near the port, and controller tracking error. When failure risk triggers, pause the learned policy, call `RecoveryExpert.recover_from_state(obs, scene_snapshot)`, execute the recovery expert, and by default finish the episode with the recovery expert.

Each rollout should log the policy-controlled segment, recovery segment, takeover reason, final outcome, F/T summary, tracking-error summary, and score. Store recovery transition data from near-failure states so SERL/HIL-SERL can learn which states require backoff, realignment, retry, or reset.

## Dataset Metadata

The standard LeRobot dataset structure should remain unchanged. Extra expert labels are written as sidecars under `meta/`:

- `meta/expert_trajectory_metadata.jsonl`
- `meta/phase_labels.jsonl`
- `meta/validation_results.jsonl`
- `meta/vlm_strategy.jsonl`

Each episode metadata record includes mode, scene ID, candidate index, validation metrics, VLM cable risk, MoveIt result, phase labels, and source candidate metadata.

For split clarity, generated outputs should be separated by mode, for example `accepted_dataset_nominal`, `accepted_dataset_nominalrecovery`, and `accepted_dataset_recovery`, or carry equivalent metadata recording mode, phase labels, recovery segments, validation metrics, F/T threshold, VLM strategy, MoveIt plan summary, and whether GPT-5 debug analysis was run. These sidecars do not alter the native LeRobot data files.

## Debug Artifacts

Pass `--debug` to write expert debug sidecars under the mode-specific dataset folder:

```text
accepted_dataset_<mode>/
  debug/
    observations_sampled.jsonl
    actions_sampled.jsonl
    ft_windows.jsonl
    tracking_error_sampled.jsonl
    sampled_images/{center,left,right}/
    image_manifest.jsonl
    trajectory_segments.json
    moveit_plan_summary.json
    replay_command_trace.jsonl
    transition_metrics.json
    phase_speed_metrics.json
    runtime_trace.jsonl
    gpt5_failure_payload.json
    gpt5_failure_prompt.md
    gpt5_failure_analysis.md
```

Observations, actions, images, and F/T windows are sampled at 0.5 seconds for local storage. F/T streams are represented as per-window min/max/median for `fx`, `fy`, `fz`, `tx`, `ty`, `tz`, force norm, and torque norm rather than raw high-frequency samples. The debug records also include phase label, command source, action representation, frame, phase-specific actual TCP speed summaries, tracking-gate state, repair metadata, and recovery runtime events. If live controller feedback is unavailable, the field is marked unavailable instead of inventing values.

Run GPT-5 failure analysis after collecting a complete debug folder:

```bash
pixi run python scripts/analyze_expert_trajectory_failure.py \
  --debug-dir outputs/expert_datasets/nominal_sfp2nic/accepted_dataset_nominal/debug
```

The analyzer first builds a compact 0.5 second payload. If that payload is too large, it automatically retries with 1.0 second sampling. If it is still too large, or if observations, actions, F/T windows, transition metrics, expected images, or GPT-5 output are missing, it fails fast with a clear error.

## Live CLI Flow

Without `--dry-run-config`, `scripts/generate_expert_trajectories.py` now runs an accepted-trajectory loop:

1. Launch a live planner-recording pass with `aic_teacher_official.OfficialExpertGeneratorPlanner`.
2. Capture live TF, observations, wrist/camera images, F/T baseline, task config, and rigid scene metadata.
3. Ask GPT-5-mini for strategy/cable-risk JSON only.
4. Try the requested candidate index with MoveIt-required free-space planning.
5. Write `piecewise_trajectory.json` with MoveIt joint positions/velocities when planning succeeds.
6. Postprocess to `smooth_trajectory.json`.
7. Launch `aic_teacher_official.OfficialTeacherReplay` with `joint_position_then_cheatcode`.
8. In nominal mode, optionally repair contact-free rough pre-insertion motion and rerun the repaired executable trajectory.
9. Parse official scoring YAML, debug phase speed metrics, and validation thresholds.
10. Count only accepted trajectories toward `--target-accepted-trajectories`.

Planner attempts are stored under `planner_attempts/`; replay attempts are stored under `replay_attempts/`; accepted sidecar metadata is stored under `accepted_metadata/meta/`.

The current MoveIt integration is deliberately strict: if the required MoveIt Python planning backend is unavailable, generation fails before silently producing geometric trajectories. The ROS message package `moveit_msgs` alone is not treated as a planner. The workspace now depends on `ros-kilted-moveit-py`, and the live policy uses `MoveItPyPlanningBackend` to call MoveItPy for sequential free-space planning and joint-trajectory extraction.

## CLI

Nominal example:

```bash
pixi run python scripts/generate_expert_trajectories.py \
  --nominal \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 100 \
  --max-total-attempts 300 \
  --candidates-per-scene 5 \
  --score-threshold 95 \
  --max-offlimit-contacts 0 \
  --require-insertion-event true \
  --rerandomize-scene true \
  --strategy-model gpt-5-mini \
  --ft-threshold 1.0 \
  --debug \
  --output-dir outputs/expert_datasets/nominal_sfp2nic
```

Nominal-with-recovery example:

```bash
pixi run python scripts/generate_expert_trajectories.py \
  --nominalrecovery \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 100 \
  --max-total-attempts 500 \
  --candidates-per-scene 5 \
  --score-threshold 90 \
  --ft-threshold 1.0 \
  --backup-distance-m 0.002 \
  --max-retries 3 \
  --probe-pattern small_cross \
  --require-insertion-event true \
  --rerandomize-scene true \
  --strategy-model gpt-5-mini \
  --output-dir outputs/expert_datasets/nominalrecovery_sfp2nic
```

Recovery example:

```bash
pixi run python scripts/generate_expert_trajectories.py \
  --recovery \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 100 \
  --max-total-attempts 500 \
  --candidates-per-scene 5 \
  --score-threshold 90 \
  --ft-threshold 1.0 \
  --backup-distance-m 0.002 \
  --max-retries 3 \
  --probe-pattern small_cross \
  --require-insertion-event true \
  --rerandomize-scene true \
  --strategy-model gpt-5-mini \
  --output-dir outputs/expert_datasets/recovery_sfp2nic
```

`--expert-mode nominal|nominalrecovery|recovery` is kept only for backward compatibility. Prefer the mutually exclusive flags `--nominal`, `--nominalrecovery`, and `--recovery`.

Final insertion remains CheatCode-style geometry, not MoveIt. The default online replay now keeps guarded insertion in explicit base-link targets: `AIC_OFFICIAL_TEACHER_CHEATCODE_Z_MODE=cheatcode_offsets` and `AIC_OFFICIAL_TEACHER_INSERTION_COMMAND_MODE=exact_position`. The port and plug TFs are read in `base_link`, CheatCode geometry computes each exact TCP target, and descent streams absolute base-link positions on a minimum-jerk depth profile. This avoids switching from a base-link local pre-insert align into gripper/tcp relative deltas, which live debug showed could produce actual guarded TCP speeds far above the commanded value. The default commanded insertion speed is 1.3 mm/s, down from 2.0 mm/s, and exact-position targets are bounded by the configured step size. A fully pinned XY/orientation insertion experiment is available behind `AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET=true`, but it is not the default because live testing reduced speed while missing insertion. Relative gripper/tcp insertion remains available behind `AIC_OFFICIAL_TEACHER_INSERTION_COMMAND_MODE=relative_delta` for comparison only. The local pre-insertion alignment uses minimum-jerk interpolation and is distance-rate-limited by `AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SPEED_MPS` (default 80 mm/s), so large near-port moves are not compressed into a fixed duration. A conservative force-gated pre-contact port alignment loop is available behind `AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC`; it is disabled by default because live tests showed sub-threshold force buildup before descent and inconsistent contact timing. A TF-based relative TCP-frame preinsert micro-align experiment exists behind `AIC_OFFICIAL_TEACHER_PREINSERT_MICRO_ALIGN_SEC`, preserves z, and fails closed if F/T rises near the threshold; it is disabled by default because live tests showed both absolute and relative x/y correction worsened contact timing. The gate uses controller TCP error when available; its TF fallback threshold defaults to 15 mm, also requiring low TCP speed and force delta below `AIC_OFFICIAL_TEACHER_TRACKING_GATE_FORCE_FRACTION` times the F/T threshold (default 1.0). Recovery uses smooth staged absolute TCP backoff: 5 mm increments up to `AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M` (default 30 mm), with 0.45 s minimum-jerk stages and 0.10 s force-release checks so nominalrecovery/recovery backs off promptly after contact instead of waiting for a new VLM decision. After force release, recovery returns smoothly to the original pre-insertion height captured before the first insertion attempt, then holds that exact z target until measured TCP z is within `AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_THRESHOLD_M` (default 4 mm). The same measured z gate is repeated after recovery realignment, and retry tracking gates pin the original pre-insertion z only for TF-depth experiments; default CheatCode-offset retries use the same relative z-offset semantics as plain CheatCode. A bounded TF-derived retry x/y bias is available for experiments behind `AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS=true`, but it is disabled by default because the first live smoke with that bias contacted earlier and then failed a tracking gate. The online replay path exposes environment overrides for experiments, including relative delta clamps, but the CLI keeps only one F/T threshold flag and one debug flag.

## Validation

Every candidate must be replayed in Gazebo and filtered before acceptance. Validation metadata includes:

- score
- official total score
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

Score parsing detail: `score` is the official scorer's top-level total, so partial insertion/proximity points and Tier 2 motion-quality points are preserved. The parser also records `task_score_excluding_tier_1` so debug reports can distinguish real task progress from `total: 1` runs that only received Tier 1 model-validation credit. Expert acceptance must not rely on score alone; it also checks insertion, F/T, contact, tracking, and mode-specific validation gates.

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

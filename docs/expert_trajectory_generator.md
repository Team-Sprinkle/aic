# Expert Trajectory Generator

This branch now separates high-level strategy from executable motion generation for AIC cable insertion.

## Architecture

The expert generator has three explicit modes:

- `nominal`: smooth, high-scoring, minimal-contact demonstrations for ACT training. Recovery is disabled; F/T threshold or validation failures reject the trajectory.
- `nominalrecovery`: nominal-first demonstrations that may include labeled backoff, realign, and retry segments after slight contact or near-failure.
- `recovery`: guarded recovery demonstrations for later HIL-SERL or online RL guidance.

The VLM is not allowed to generate Cartesian waypoints, joint targets, velocities, or executable trajectories. GPT-5-mini is used only for scene understanding, cable sweep/collision risk assessment, and symbolic approach strategy. GPT-5 is reserved for offline critique/debug analysis after a full episode has been recorded.

Free-space approach planning requires MoveIt. There is no geometric fallback. If MoveIt is unavailable or planning fails, the candidate is rejected and the generator records a structured failure.

Final alignment and insertion use CheatCode-style ground-truth geometry. MoveIt is used only for obstacle-aware free-space motion to staging/pre-insertion poses. The handoff now has explicit `local_preinsert_align`, precontact port-align, tracking-gate, and guarded-insertion phases before the final insertion event is expected.

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
8. Before descent, enforce the pre-insertion tracking gate. The default gate requires low pose error, low TCP speed, and no F/T threshold change.
9. Validate score, insertion event, F/T/contact limits, phase speed metrics, and tracking metrics.
10. If the pre-insertion portion was contact-free but too rough, repair only the approach-to-pre-insertion portion with minimum-jerk retiming, rewrite executable action targets, preserve the exact pre-insertion pose/orientation, and rerun live validation.
11. Accept only passing trajectories into the dataset.

The current reliable nominal path is Idea 3 from the May 4 reliability pass:
use the same clean transport, local alignment, precontact port alignment, and
guarded insertion phases as nominalrecovery, but remove post-contact recovery.
Nominal can do pre-contact realignment and gate rechecks, but it must not
backoff, retry, or recover after confirmed contact.

Current nominal behavior:

1. If the initial preinsert tracking gate misses with low force, run one slow
   nominal pre-contact realign and recheck.
2. If the recheck is still a small low-force lateral miss, allow it to continue
   into the port-frame aligner instead of rejecting immediately.
3. Run precontact port-frame lateral alignment with slow speed, force aborts,
   and a capped correction.
4. Smooth the CheatCode handoff to insertion start with a minimum-jerk profile
   so the direction change into descent is not a sharp corner.
5. Run a final slow port-frame lateral alignment immediately before descent.
6. Descend with guarded exact-position CheatCode insertion and speed-gate holds.
7. Reject on confirmed contact or force abort. There is no backoff, retry,
   probing, or recovery segment in nominal.

The latest stable nominal run is:

```text
outputs/expert_debug/nominal_clean_align_v4_20260504T192000Z
attempt_000001_candidate_00: 96.83110630045765
attempt_000002_candidate_01: 96.82091717491022
attempt_000003_candidate_00: 96.80927941955909
```

All three accepted attempts reached insertion, had no official contact, had no
excessive force, and had no runtime backoff/recovery events. The representative
GPT-5 review is:

```text
outputs/expert_debug/nominal_clean_align_v4_20260504T192000Z/replay_attempts/attempt_000002_candidate_01/gpt5_replay_analysis_center/analysis.md
```

Nominal mode does not use post-contact F/T correction, intentional probing,
backoff, or retry logic. If contact/F/T violation happens before or during
insertion, nominal rejects the attempt rather than recovering.

What worked for nominal:

- Idea 3 was the best design: share nominalrecovery's near-port alignment and
  guarded insertion stack, but keep recovery disabled.
- Pre-contact realign/recheck fixed borderline tracking-gate failures without
  introducing recovery labels.
- Small low-force gate-miss allowance prevented over-rejection when lateral
  residuals were around `2-3 mm` and the fine port-frame aligner could still
  correct safely.
- Minimum-jerk handoff smoothed the sharp transition into descent.
- Raising nominal `ft-threshold` and above-contact soft threshold to `2.5 N`
  avoided rejecting harmless `~2.1 N` transients before true insertion, while
  official scoring still reported no force penalty in successful runs.
- Runtime-trace guarded speed is the right validation source. Sampled phase
  labels can overestimate guarded speed because the online replay inserts
  dynamic phases that are not fully represented by the static trajectory labels.

What did not work for nominal:

- Pure nominal with only a strict preinsert gate was not reliable; slight
  lateral/attitude mismatches still missed insertion.
- Threshold tuning alone did not solve lateral misalignment.
- Treating every tracking-gate miss as terminal was too brittle.
- Letting live-Z repair run freely in nominal was risky; GPT-5 feedback and
  trace inspection both point to live-Z repair as a source of immediate contact
  when lateral alignment is not already good.
- A `0.25 mm` precontact port-align correction cap was too small for the
  observed `2-3 mm` residuals. The reliable v4 nominal run used `0.5 mm` cap
  with slow speed and force gating.

## Recovery Mode

Recovery generation starts like nominal mode through free-space approach. If the pre-insertion tracking gate fails, nominalrecovery/recovery performs one local realign and re-checks the gate; if the gate still fails, it rejects before descent. Once contact matters, it uses F/T data:

1. Guarded insertion.
2. If the 5-sample force median windows show a meaningful rise over a 250 ms
   center gap, or the force threshold is exceeded in the contact zone, stop
   descent immediately.
3. If force is meaningful but still above the nominal contact zone, stop and
   realign instead of continuing to push. The default soft realign threshold is
   `2.0 N`; this prevents long `2-5 N` pushes that can pin the plug before
   recovery starts.
4. Command backoff in `gripper/tcp` using the `tcp_away_from_port` direction by
   default, while measuring the actual TCP displacement in `base_link`.
5. Require at least `2 mm` of measured physical retreat before accepting the
   backoff. Commanded backoff alone is not sufficient.
6. Require stable force release before lateral realignment. The strict release
   check defaults to `1.0 N` for `0.25 s`, even if the CLI release threshold is
   higher.
7. If the first backoff command does not produce enough measured retreat, try a
   measured-backoff fallback lift before giving up.
8. Realign using CheatCode-style geometry and recovery gains.
9. Retry insertion up to `max_retries`.
10. If off-limit contact or validation failure remains, reject unless a later
    retry succeeds cleanly.

The May 3 `CheatCodeModified` debug runs clarified the required frame semantics. The force-derived correction should be interpreted relative to the tool/port, but recovery must be verified in the world frame by measured TCP motion. Trace events such as `backoff_distance_achieved_m` are command progress, not proof of physical retreat. A recovery is correct only when the recorded TCP state and force history show actual unloading.

The successful isolated run is:

```text
outputs/debug_cheatcode_modified/run20
/home/ubuntu/ws_aic/src/aic/outputs/debug_cheatcode_modified/run20/dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
```

Run20 used the original/default backoff gains, not the diagnostic high-gain setting:

```text
backoff stiffness: [90, 90, 90, 50, 50, 50]
backoff damping:   [50, 50, 50, 20, 20, 20]
force L2 drop trigger: 1.7 N
```

It triggered on `delta_force=[-0.328, -1.408, 1.560]`, produced `delta_base=[-0.00078, -0.00103, 0.01135]`, and measured about `+3.3 mm` actual base-z retreat over the backoff window. A stronger-gain diagnostic run reached a larger retreat, but the original gains were sufficient once the direction was corrected. This means nominalrecovery should first port the trigger/frame/latch behavior before changing controller gains.

The earlier implementations failed for several reasons:

- It overemphasized absolute force values even though the wrist force baseline is nonzero while holding the cable.
- It allowed the transformed force-derived delta to point in negative `base_link.z`, which can command deeper insertion for this TCP orientation.
- It did not always guarantee a latched fixed target; one-shot deltas are easy to erase or overwhelm with the surrounding insertion stream.
- Trace events alone were misleading. The video and recorded TCP state must both show physical retreat before calling recovery correct.
- It ignored above-contact force until too high a threshold. Runs that held
  `2-4.8 N` above the port often became pinned before backoff could unload.
- It treated precontact port-align force aborts and low-force lateral gate
  misses as terminal failures. In nominalrecovery these now feed recovery
  backoff/re-align/recheck logic.
- It used a `4 mm` measured-backoff requirement during one reliability pass.
  That rejected a valid case with about `2.38 mm` measured retreat and force
  release; the current default is `2 mm`.

Latest stable nominalrecovery verification:

```text
outputs/expert_debug/vlm_backoff_reliability_cycle10
attempt_000001_candidate_00: 97.035335987039559
attempt_000002_candidate_01: 97.038856798717887
attempt_000003_candidate_00: 97.056801265524768
```

All three accepted runs reached insertion without official force penalty. The
best run's GPT-5 critique is at:

```text
outputs/expert_debug/vlm_backoff_reliability_cycle10/replay_attempts/attempt_000003_candidate_00/gpt5_replay_analysis_center/analysis.md
```

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

Final insertion remains CheatCode-style geometry, not MoveIt or VLM waypoint execution. The current reliable online replay path uses MoveIt for free-space joint-space transport, then switches to local preinsert alignment, precontact port alignment, tracking gates, and guarded geometric insertion near the port. The port and plug TFs are read in `base_link`; local corrections and recovery directions are reasoned about relative to TCP/tool and port geometry, then verified against measured TCP motion in `base_link`.

The current defaults intentionally keep live-Z repair disabled unless explicitly enabled. Earlier live-Z repair runs could begin guarded insertion after a failed handoff gate and a large pose-tracking error, producing high speed or early contact. The stable path instead preserves measured TCP Z during preinsert alignment/gates, lets guarded insertion own axial descent, and uses low-force tracking gates to stop before a small contact becomes a pinned insertion.

The local pre-insertion alignment uses minimum-jerk interpolation and is distance-rate-limited by `AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SPEED_MPS` (default 80 mm/s), so large near-port moves are not compressed into a fixed duration. Preinsert servo compensation is capped tightly enough to improve alignment without creating a large lateral sweep near the port. The precontact port-align phase is no longer treated as a terminal failure when force appears; it routes into the same recovery/realign/retry machinery as guarded insertion contact.

Recovery uses measured TCP retreat rather than trusting commanded backoff traces. The default recovery direction is `tcp_away_from_port`, commanded as `gripper/tcp` deltas, while success is judged by measured TCP displacement in `base_link`. The minimum accepted measured retreat is `AIC_OFFICIAL_TEACHER_MIN_BACKOFF_DISTANCE_M=0.002`; a larger commanded retreat is still allowed when the policy elects to return to a higher preinsert pose before realigning. If measured retreat is too small, the replay falls back to a small upward lift and retries only while retry budget remains. Force-release checks use the stricter stable release threshold by default so retries do not start while the plug is still loaded.

A historical exact-position guarded insertion path remains available behind environment overrides (`AIC_OFFICIAL_TEACHER_INSERTION_COMMAND_MODE=exact_position`, speed gates, pinned insertion targets, and TF-derived retry biases), but it is not the current recommended path. The best stable May 4 run set is `outputs/expert_debug/vlm_backoff_reliability_cycle10`: three accepted nominalrecovery trials scored `97.0353`, `97.0389`, and `97.0568`.

The generator now forwards recovery controls into online replay instead of only recording them in metadata: `--backup-distance-m` maps to `AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M`, `--max-retries` maps to `AIC_OFFICIAL_TEACHER_RECOVERY_MAX_RETRIES`, and `--recovery-release-force-threshold` maps to `AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N`. The release threshold is intentionally separate from `--ft-threshold` so strict contact-trigger stress runs can still declare force released after small residual noise. Recovery realign preserves current TCP Z by default (`AIC_OFFICIAL_TEACHER_RECOVERY_REALIGN_PRESERVE_CURRENT_Z=true`) and uses its own slower speed cap (`AIC_OFFICIAL_TEACHER_RECOVERY_REALIGN_SPEED_MPS=0.02`). Live forced-backoff testing showed 5 mm backoff plus Z-preserving realign can pass release, return, and retry gates with guarded insertion speed below 5 mm/s, but repeated strict-threshold contacts still prevent accepted recovery trajectories. GPT-5 analysis of the best failed recovery run identified live-Z repair after a failed handoff gate as the remaining risky path: speed is acceptable, but guarded insertion can begin after a large pose-tracking error. The next recovery experiment should make live-Z repair conditional on XY/yaw being in spec, use body/port-frame micro-align when the handoff gate fails laterally, and then use adaptive staged backoff, increasing the second retreat to 10-15 mm when the retry contacts before meaningful insertion depth.

Before changing recovery sequencing, tune command-level stiffness/damping in nominalrecovery. `aic_controller` logic should remain fixed; only the gains sent in replay commands should vary. The replay path accepts `--cartesian-stiffness`, `--cartesian-damping`, `--recovery-cartesian-stiffness`, `--recovery-cartesian-damping`, `--joint-stiffness`, and `--joint-damping`, which map to `AIC_OFFICIAL_TEACHER_*` gain environment variables. The first tuning target is recovery/backoff compliance: try lower recovery Cartesian translational stiffness and higher damping, then inspect runtime trace events plus sampled center-camera images and run GPT-5 failure analysis. Initial candidate: `--recovery-cartesian-stiffness 45,45,55,35,35,35 --recovery-cartesian-damping 70,70,80,30,30,30`. If backoff still fails to release contact, try `30,30,45,30,30,30` translational/rotational stiffness before changing the state machine.

The first live gain-tuning run, `outputs/expert_debug/nominalrecovery_gain_tune_v1_20260503`, confirmed that those recovery gains allow prompt staged backoff and force release. The run was still rejected because every retry contacted immediately near the guarded insertion start and exhausted retries. GPT-5 failure analysis on the sampled center-camera images, observations, actions, F/T windows, and runtime trace identified lateral/yaw misalignment and the absolute base-link insertion handoff as the remaining failure mode. Keep the recovery gain profile as the baseline and move the next experiment to port-frame preinsert gating and micro-align before descent.

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

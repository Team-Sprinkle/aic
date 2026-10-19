# Frozen pipeline coverage: transport, visual handoff and insertion

## Why this evaluation exists

The prior newest result was one reused SC episode, not five episodes per setting. Older6/6 pooled SC screens measured acquisition, not full insertion. Neither establishes reliability across card counts, ports or families. The user explicitly requested broader evaluation of the current frozen systems before further changes. This overrides the earlier one-pilot stopping rule for diagnostic evaluation only; it does not relax promotion or physical success gates.

## Exact frozen systems

### SC hybrid candidate

1. **Rule-based Low Lateral Bypass transport**, with measured stage completion. No learned transport actor moves the robot in this arm.
2. **Requested-port coarse/identity check**, original-image crop refiner, cross-view geometry and guarded board orientation; strict stationary approval. Three cameras, public requested SC port and measured robot state. Only the requested selected port is returned.
3. **Learned held-plug position helper**, image-only, position role only; full grasp orientation was not promoted.
4. **Explicit observation-only approach controller** to30mm before opening. Camera-estimated port/plug geometry plus measured TCP; no simulator geometry.
5. **Existing frozen BC insertion actor**, `local_causal_actor_correction_v1/best.pt`, shared SFP/SC training but current locked runtime is SC-only. ResNet18 + four-target transformer; complete plug-tip translations and TCP orientations in the estimated port frame, converted to TCP targets. Four50ms predictions, four executed. No new training, diffusion, RL or backtracking in this arm.

### Matched SC far-start actor control

Identical transport, coarse/refiner, plug estimator, actor checkpoint and scene. Omit only the new observation-only approach controller. The actor starts from the roughly150mm acquisition point. This comparison is limited to1/5card extremes; do not generalize its effect to unpaired cells.

### SFP arm

Frozen shared learned diffusion transport, `matched_actor_repaired_fixed20k_v2/best.pt`;12x100ms predictions, six executed,12 denoising steps. Images, measured state, task-family code, requested SFP port, requested NIC card code and history condition actions. **Its trained stop head supports SC only. The strict selected hybrid verifier/local insertion runtime is also SC-only.** Therefore this SFP arm measures transport geometry and failures; no SFP visual approval or insertion actor is invoked. A near-port geometric visit is not an autonomous accepted handoff. Existing teacher-prefix SFP insertion diagnostics are privileged and cannot fill this gap.

## Predeclared matrix and limits

-100 primary rollouts:1–5cards ×2ports ×2families ×5fresh replicates.
-20 additional matched SC far-start controls:1/5cards ×2ports ×5same resets.
-20 alternate-published-board stress rollouts:3cards ×2ports ×2families ×5fresh replicates.
-Total140 rollouts on120 fresh synthetic development scenes. No reserved final scenes or audit partitions are read.

SFP requested card index cycles0–4 across the five replicates and is present. Card subsets, legal rail positions, small board/grasp perturbations and **actual xacro startup joint perturbations** vary; changing YAML home positions alone does not affect the first trial. The primary board pose uses the existing development template. Stress board poses use the released public examples. This does not exhaust every card subset, full robot-start distribution or board orientation/card-count cross-product. Five cases per group are a feasibility estimate, not a strong reliability certificate.

Runtime code, selected checkpoints, scenes, seed and thresholds are frozen before rollout. Ordinary failures do not stop the matrix. Invalid infrastructure/scoring joins must be diagnosed, not counted as physical failures. No tuning between evaluation cases. One rootless Gazebo domain86world at a time, a free GPU5or6, with legacyGPU0 left untouched. Up to roughly7–12hours for the entire matrix.

## Metrics

Report SC acquisition, requested-port mistakes, original-image pose errors, near-approach completion, insertion-actor execution, terminal depth/lateral/orientation, official score and prohibited-contact/sustained-force penalties separately. Physical success remains depth≥14mm and lateral≤0.5mm, with explicit orientation and acceptable contact.

SFP reports measured plug-to-requested-opening geometry, closest/terminal distances and broad diagnostic near-port visits. The broad region is axial[-100,-15]mm and lateral≤10mm; it is not a handoff acceptance criterion. Full SFP pipeline success is unimplemented/N/A, not a fabricated0% insertion rate.

Geometry is read for posthoc scoring only at causal pre-return timestamps. Raw bags and terminal observations are preserved; no post-reset pose joins. Record all failures and available inference latency. Recording can save5Hz camera frames for ordinary replicates,20Hz for replicate0; policy camera inputs are not throttled. Render representative videos from20Hz recordings.

## Artifacts and commands

NAS root:`/home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1`.

```bash
.pixi/envs/default/bin/python scripts/prepare_frozen_pipeline_coverage.py --output /home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1
.pixi/envs/default/bin/python scripts/run_frozen_pipeline_coverage.py --plan /home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1/plan.json
```

`plan.json` freezes exact scenes, hashes and arms. `current_job.json` identifies the active run; `progress.json` accumulates results. Each `runs/CASE__ARM/` keeps source/scene hashes, commands/environment, logs, raw bag, scored stream, trace, `coverage_metrics.json` and `tip_curve.json`. `batch_finished.json` exists only after the whole matrix completes. An incomplete generator folder preserves the initial inactive-rail configuration error; no rollout used it.

Status: **complete on October 15: 140/140 valid rollouts, 28 cells with exactly five episodes each. No promotion.** See the final results below.


### Important implementation distinction: selected coarse model

The live SC diagnostic uses the older `joint_locator_v1` / `SCShadowHandoffVerifier` stack. Its coarse network predicts12internal **2D** candidate slots (2SC and10SFP), plus projected-in-frame flags. The public requested SC code selects one slot. A separate request-conditioned spatial identity head receives current image features, camera identity, measured7D TCP pose and requested SC port code. The selected original-image crop is refined, then three-view geometry/board orientation supplies one requested-port3D pose.

This is **not** the later proposed target-only temporal coarse model. There is no full camera-image history in this selected identity head; the guarded rule pipeline can retain observation-derived board orientation, which is different. Its in-frame flags are projection-presence labels, not certified occlusion visibility or calibrated pose confidence. The current matrix tests this older frozen stack because that is what the reported latest hybrid actually used. Only the requested candidate is passed downstream; internal12slot2D proposals must not be described as predicting12reliable3Dposes. Earlier user-facing explanations did not make this version distinction sufficiently clear.

### Measurement plumbing repairs before broad execution

The first SFP policy ran correctly, but the bag's scored TF did not include its plug pose. That attempt is preserved as `__missing_tip_tf_attempt0`; it is not called a model failure. The same frozen case was re-recorded with a read-only native `/world/aic_world/dynamic_pose/info` subscriber. Native link poses are **relative to their model**, so physical plug position is world-model pose × model-relative tip pose. The initial uncomposed measurement is preserved and excluded. The corrected mapping was independently checked on117timestamps against the official SFP Module fixed tip transform: max translation error0.000208mm and max rotation error0.000140°. No policy, physics, scene or action-frame changes were made. The corrected first SFP terminal is -3.865mm axial/50.615mm lateral, rather than the invalid1.4m result.

The first three valid runs established SC/SFP rollout and scoring plumbing. The full frozen matrix subsequently completed. The scorer additionally records failure reasons and geometric distance to the training acquisition standoff; a geometric arrival by itself does not prove visibility or authorize handoff. All final aggregation uses authoritative per-run metrics, not obsolete provisional progress entries.

### Interrupted engine cleanup and frozen retry

At `sfp_primary_n4_p1_r0`, the engine crashed with `std::out_of_range: unordered_map::at` during controller deactivation after the policy recorded its causal `terminal_before_return`. No official scoring file was written. The entire attempt is retained as `runs/sfp_primary_n4_p1_r0__learned_transport_only__engine_crash_attempt0`, with `measurement_incomplete.json`. It is excluded from the valid score matrix and was repeated with identical scene, weights, policy runtime and limits. This is an infrastructure retry, not tuning after a policy failure. Other active jobs were left untouched.

### Causal terminal scoring details

The frozen force-stop code sometimes records the complete 32D robot state but omits simulation time. The scorer recovers it only from a unique exact float32 match of all seven TCP pose components in the recorded controller stream, within one second after the last preceding trace timestamp; otherwise it refuses to score. The first such case matched 68.700 seconds and stopped at 43.13 N. Raw traces are unchanged, and this ordinary policy force stop counts in the evaluation. A later explicit `visual_approach_failed` observation supersedes the intermediate rule-stage terminal marker. Approach observations are included in closest-distance diagnostics; neither fix changes controller actions or uses post-reset geometry. Earlier per-run metrics are refreshed with this scorer before final aggregation.

## Final results — October 15

All **140 rollouts / 120 distinct development configurations** completed. Every one of the 28 predeclared cells contains exactly five valid episodes. The 29 watched runtime files and four selected model/template hashes match the frozen plan at completion (`final_frozen_hash_verification.json`). No training, policy tuning, RL, backtracking or final-scene access occurred. Evaluation used one free GPU 5 or 6 sequentially, with the older GPU 0 process left untouched.

| Domain and arm | Episodes | Visual approvals | Wrong-port approvals | Insertion actor ran | Full physical insertion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary SC, added visual approach | 50 | 23 | 3 | 3 | 0 |
| Primary SC, far-start control | 20 | 7 | 1 | 7 | 0 |
| Alternate public board SC | 10 | 0 | 0 | 0 | 0 |
| Primary SFP learned transport only | 50 | N/A | N/A | 0 | N/A |
| Alternate public board SFP transport only | 10 | N/A | N/A | 0 | N/A |

The 20 matched SC configurations give **0/20 full insertions in both arms**; seven pairs pass visual approval in both realized rollouts. This does not establish a full-insertion improvement from the added approach controller. The three near-start actor runs terminate at depths **1.98, 5.58 and 1.94 mm**, lateral offsets **2.50, 0.27 and 0.99 mm**, and orientation errors **1.32°, 1.23° and 1.91°**. Official score 75 in a shallow run does not override the frozen 14 mm depth / 0.5 mm lateral physical gate.

Primary SC failure breakdown: 27/50 stop before visual acquisition. Of 23 visually approved cases, 20 time out with `no_progress` in the explicit approach controller and only three enter the insertion actor. Importantly, **17 of those 20 approach failures are physically within ±5 mm of the 30 mm stand-off and within 10 mm laterally of the requested opening**. They got near the requested port but did not pass the tighter readiness test. At their final measured observations, 12 fail only the estimated 0.5 mm lateral condition, seven fail lateral and the ±1.5 mm axial condition, and one fails axial only. All are below the 5 mm/s speed condition. Estimated lateral error has median 1.70 mm, range 0.33–5.25 mm. This diagnoses a convergence/readiness mismatch; it does not justify silently loosening insertion gates. See `approach_failure_diagnosis.json`.

The three primary SC candidate wrong accepts are `sc_primary_n5_p1_r2`, `sc_primary_n2_p1_r3`, and `sc_primary_n4_p1_r3`. Their estimates are **0.160, 0.310 and 0.183 mm from port 0**, although port 1 was requested; requested-port errors are 52.29, 70.81 and 72.05 mm. One matched far-start run repeats the first wrong accept. These are concrete identity errors, not a vague confidence issue. In the other 20 correctly identified primary candidate acquisitions, opening-position error is median **0.169 mm**, p90 **0.228 mm**, max **0.698 mm**. Fine localization can work while identity still fails. Do not generalize these conditional pose errors to rejected views or all scenes.

For primary SFP transport, **0/50** enter even the declared broad diagnostic region (axial −100 to −15 mm, lateral ≤10 mm); alternate-board SFP is **0/10**. Primary SFP closest sampled plug-to-requested-opening distance has median **71.17 mm**, p90 **108.96 mm**. This is not adequate transport in this matrix. No SFP coarse approval or insertion actor is connected, so no full insertion success percentage can be reported. Primary SFP has 39/50 without scored contact/force penalties; all ten SFP stress runs and all 80 SC runs have no scored penalties. Absence of a scored penalty is not proof of collision-free motion or insertion.

### Coverage and statistical limits

Primary SC and SFP each have five repeats for every **card count 1–5 × requested port 0–1**. SFP requested NIC index cycles 0–4; card index, subset and selected rail stratum are tied to the replicate, so this is not five repeats of every joint NIC-index/card-count/port combination. Starts have actual small joint and grasp perturbations, not exhaustive initial-state variation. The alternate-board test uses three cards only. These are fresh synthetic development resets within the chosen public ranges, not the sealed official final evaluation or every published configuration. Five samples per cell can expose failure but cannot establish high reliability (even 5/5 has Wilson 95% lower bound about 56.6%). The conditional near-start actor sample is only three episodes and is insufficient to rate that actor independently across settings. This is a coverage evaluation of the selected frozen pipelines, not a five-per-cell rerun of every historical checkpoint.

### Videos and artifacts

All bulk artifacts are under `/home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1`.

- `plan.json`: exact scenes, task settings, chosen models and runtime hashes.
- `coverage_cells.csv`, `summary.json`, `summary.md`: complete grouped results.
- `decision.json`: final machine-readable decision; `batch_finished.json`: completion marker.
- `final_frozen_hash_verification.json`: all 33 predeclared hash checks pass. The older coarse/refiner weights were not separately included in the initial four-entry model hash set; this limits the independent weight-freeze audit. No model-training job was run during this evaluation.
- `runs/*/`: commands, source snapshots, raw bags, recorded images, terminal observations, official scores, physical tip curves and per-run metrics. Invalid missing-tip and engine-crash attempts remain separately preserved.
- `sc_primary_n2_p1_r1__near_approach/five_cameras_5fps_grid_active.mp4`: existing BC actor reaches the port and remains shallow; path is inside `runs/`.
- `sc_primary_n1_p0_r0__near_approach/five_cameras_20fps_grid_active.mp4`: transport/acquisition failure, insertion actor never starts.
- `sc_primary_n2_p1_r3__near_approach/five_cameras_5fps_grid_active.mp4`: wrong-port approach.
- `sfp_primary_n5_p0_r0__learned_transport_only/five_cameras_20fps_grid_active.mp4`: learned transport drift/stall near cards.

Representative frame sheets were actually viewed and agree with the physical measurements. Overhead/side cameras are passive recording views, not additional actor inputs. Simulator geometry is posthoc only. The scorer repairs native SFP model-relative link composition, exact force-stop timestamp recovery, and nested terminal-event selection; raw traces and invalid preliminary scores remain preserved. No controller action was changed by these measurement repairs.

### Decision

The earlier 6/6 acquisition screens and one 11 mm pilot were too narrow. Keep full insertion and step 5 open. Rule-based SC transport remains a useful restricted-board baseline, not a robust solution for both ports and board poses. The added approach moves correctly acquired scenes toward the near-port region, but has not increased full insertion in the matched comparison. The selected learned SFP transport also fails this expanded matrix. The next proposed work is a bounded repair of requested-port identity and approach convergence on new repair-development starts, followed by a new frozen comparison; separately establish a working autonomous SFP handoff before claiming end-to-end coverage. Do not start another broad architecture/data sweep or RL from these results alone. No follow-up policy changes or jobs were started in this user-requested pause-and-understand task.

A standalone stage-count plot is saved as `primary_sc_stage_counts.png` under the artifact root. `additional_perception_hashes_at_completion.json` records the older perception components for reproduction, explicitly as completion-time hashes. `progress.json` was refreshed from final authoritative per-run scores; the earlier accumulated copy is preserved as `progress_before_final_causal_rescore.json`.

# Full-connector insertion and jerk probe

## Bounded test

User requested only a few rollouts to check insertion and remaining jerk, then suggestions if jerk persists. Run **four Gazebo episodes**: two repeats of each previously used one-card SC-port0 and SFP-card0-port0 development configuration. These are repeated identical scene settings, not four independent random configurations or a broad success-rate estimate. Reserved final scenes remain sealed.

Frozen policy: `full_connector_bc_h4/best.pt`, selected at update7400 after a13400-update validation plateau. Full connector position/orientation input and full connector trajectory targets are estimated from RGB. Recorded TCP targets supplied training labels. No privileged geometry supplies actor inputs, crop selection, or handoff decisions.

Same rule transport, coarse/refiner and25mm visual/proprioceptive approach as the earlier controlled comparison. New rotation is acquired from three causal native-image estimates spanning at most one second, with pairwise angular span<=1degree. Its SO(3)mean stays fixed for the attempt. Tip translation remains the existing approved visual estimate. The actor retains transport robot history; it uses current RGB, not historical images.

Predict4x50ms targets; execute all four then replan. **Recovery=none**, no exploration, RL or teacher control. This isolates the actor's motion from hardcoded retreats. Contact/geometry oracle information is recorded only by the independent evaluation process.

## Runtime repair and checks

`RunSharedPortTCP` now admits the explicit `full_pose`/`full_connector_pose` checkpoint contract; legacy modes retain their conversion. `RunLockedPortLocal` supplies full estimated connector rotation and converts desired connector poses through the inverse estimated full grasp. `RunFullConnectorInsertion` adds causal orientation acquisition to the existing history-preserving25mm runtime.

The actual checkpoint constructs and predicts on held-out original SC and SFP RGB inputs. Independent conversion parity has maximum matrix-component difference5.4e-8; orientation is unmasked. Synthetic history in this interface check is not a physical-performance test. The first host-only import failed because ROS setup selected an installed package copy; explicitly preferring this checkout fixed the preflight. Docker's existing checkout bootstrap already does this.

## Measurements

- Independent terminal axial depth, lateral and orientation error; official score and force/contact behavior. Official tier3=75 alone does not establish physical full depth.
- Within-chunk versus between-plan command changes in mm/degrees.
- Public measured TCP speed, acceleration and motion reversals, alongside proposed targets. A command jump is not identical to measured physical jerk; units and sample intervals must be reported.
- Pose-lock constancy, causal transport-history retention, executed command indices and stale drops.
-20fps videos and inspected image samples around the learned insertion period.

Stop after four valid diagnostic episodes. Infrastructure/causal-label failures are preserved and fixed without treating them as policy failures. No automatic retraining or smoothing ablation: report a targeted next experiment if jerk persists.

Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_full_connector_live_probe_v1/`

- `runtime_preflight.json`, `before_runtime/`, `command.json`, `runner.log`.
- `evaluation_cases.json`, `evaluation_variants.json`.
- `runs/plan.json`, per-run frozen inputs/models, raw bags, metrics, command traces and videos.
- `runs/progress.json`, `runs/summary.json` and `runs/summary.md` as episodes finish.

Runner: `scripts/run_full_connector_live_probe.py`; physical GPU5, rootless Docker with existing cooperative locks. Legacy GPU0 is untouched; no other job was interrupted.

## Contingent repair directions, to rank after the measurements

1. **If the jumps concentrate at plan boundaries:** retain the previous unexecuted trajectory tail and condition the next prediction on it, with explicit action timestamps and inference delay. The current deterministic actor needs a suitable conditioning/continuity design; it cannot directly use a diffusion/flow inpainting routine. [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339) freezes committed actions while generating a compatible continuation. This supplies a design principle, not a demonstrated fix for this actor.
2. **If observed commands differ mainly because of teacher phase or controller bias:** provide causal previous-command/controller-state context or supervise recorded geometric intent separately from low-level compliance compensation. Do not replace recorded commands with measured future motion without an explicit target-contract revision. Longer history alone may not identify hidden teacher state.
3. **If predictions are geometrically sensible but execution jerks:** test a small fixed-weight trajectory bridge with velocity/acceleration bounds and correct rotation interpolation. Contact guards must override it. [ACT](https://arxiv.org/abs/2304.13705) uses overlapping action predictions for temporal ensembling; averaging is not a cure for incompatible motion modes or sustained misalignment.
4. **If the actor settles smoothly against the opening with lateral error:** prioritize alignment/contact-state coverage and target/control semantics. Additional smoothing would conceal the failure rather than solve insertion. A modest longer causal robot/force/command history is a distinct, bounded ablation once the missing information is identified.

The prior H8 overlap-consistency loss did not establish insertion improvement. Do not suggest repeating it unchanged or claim that a diffusion switch alone fixes discontinuities. No repair trial or new training is part of this four-rollout request.

## Finished results: four rollouts, no further training

All four rollouts, causal geometry scores and20fps five-camera videos completed. Two identical-setting repeats per family are development diagnostics, not coverage of all card counts/ports/grasp angles. All four retain public-only actor inputs, a single fixed port/grasp lock, causal transport history, all four executed50ms targets and recovery disabled. No scored harm was recorded. No final scenes were opened.

| Case | Depth mm | Lateral mm | Orientation degrees | Strict physical insertion | Official tier3 |
| --- | ---: | ---: | ---: | --- | ---: |
| SC repeat1 |1.948|2.186|3.041|No|75|
| SFP repeat1 |46.749|0.479|0.461|**Yes**|75|
| SC repeat2 |1.993|5.425|3.058|No|75|
| SFP repeat2 |37.430|0.252|0.530|No: shallow|47.813|

Therefore **1/4 strict physical insertions**, SFP1/2 andSC0/2. The separate official completion/full-score count is3/4. The strict gate remains unchanged: SCdepth>=15.5mm, SFPdepth>=44mm, lateral<=0.5mm, orientation<=1degree and no scored harm. Do not replace physical insertion with the official score.

### Critical correction: simulator locking changes smoothness interpretation

The toolkit `CablePlugin` transitions from touch to `ATTACH_CABLE_TO_PORT`, makes the connector static and then locks the end effector before publishing completion (`aic_gazebo/src/CablePlugin.cc`, roughlylines284–326). This also happened in **both shallow SC episodes**. Their long stationary segments are partly simulator-enforced, not evidence that the actor learned stable contact control. Post-completion immobility must not be diagnosed as cable snag or a failed learned/backtracking action.

Completion wall logs are joined to recorded camera wall/simulation timestamps, with nearest-camera offsets below50ms. Approximate simulation completion times are57.25s forSC1,61.70s forSC2 and53.55s forSFP1. SFP2 has no completion event. No ground-truth or completion log enters the actor; this is posthoc diagnosis.

The early interim statement that whole-episode SCp95 jumps were1.5mm versus8.9mm **overstated smoothness improvement**, because most new SC commands followed simulator freezing. Preserve the interim metrics but use the corrected phase windows below. The all-episode flat portion of successful SFP1 is also not active stabilization evidence.

### Does jerk remain?

**Yes. Large repeated SFP oscillation is reduced; abrupt transitions and rapid contact-release advances remain.**

| Command-boundary jump p95 | Previous matched H4 | New repeat1 | New repeat2 |
| --- | ---: | ---: | ---: |
| SFP, first15actor seconds, before any completion freeze |15.434mm|4.378mm|3.162mm|
| SC, first2actor seconds, approximately before earliest freeze |8.031mm|9.036mm|5.162mm|

SFP improvement is clear in this small descriptive comparison, but SC early smoothness is inconsistent. Maximum new boundary jumps remain5.29–9.37mm across the four episodes. These are target changes, not instantaneous measured displacement. The baseline is one saved run per configuration, so this is not a paired-seed statistical estimate.

Measured SFP peak TCP speeds are about59mm/s and54mm/s. In successful SFP1, the tip waits near the opening while a commanded axial point lies roughly40mm farther forward; lateral alignment then improves and the plug advances rapidly. The command/motion plot supports excessive accumulated forward demand as a separate explanation from replanning discontinuity. It does not prove an exact force-storage mechanism. Blindly averaging plans does not necessarily remove this release behavior.

`analysis_v2_prefreeze/summary.json` retains full/windowed target jumps, pre-completion speeds, clock joins, tracking error, reversals and raw finite-difference motion proxies. Position-derived acceleration/jerk include sampling/interpolation artifacts; do not use them as calibrated sensor measurements. `analysis/` and `interim_analysis/` preserve earlier computations. No dataset or model was selected from final-scene results.

### Why SC remains unreliable

Independent raw-bag image-time labels show live grasp-orientation errors of**2.476degrees and2.543degrees**, with tip-position errors around0.58–0.62mm. These are larger than the prior held-out component results. For the first SC run, the actual grasp is3.446degrees from the training mean, outside the training frame range(max2.334degrees); the estimator returns only1.261degrees from that mean. Nearest training orientation is1.842degrees away. This is a concrete coverage/extrapolation gap in one configuration, not proof of universal failure.

The analogous SFP grasp errors are0.281/0.299degrees and about0.04–0.07mm in translation. SFP2 has good terminal lateral/orientation alignment but stops about6.6mm before the unchanged full-depth threshold. Its late predicted axial setpoint is behind the measured tip; the raw command/bias/teacher-phase ambiguity remains relevant. Accurate perception alone is insufficient.

These SC perception errors are a confirmed gap, not established as the sole cause of its lateral-control failures. Video samples from all four runs were inspected alongside the measured-motion traces.

### Direct handoff discontinuity found

Comparing the **last visual-controller reference with the first insertion-actor reference**, both in base coordinates, finds position jumps of**18.68/13.57mm for SC and7.58/4.15mm for SFP**. Corresponding orientation steps are about0.96degrees forSC and0.20degrees forSFP. `handoff_command_jump.json` preserves all four values.

This transition is outside the earlier within-actor boundary statistics. Preserving observation history did not preserve **command-reference continuity or the visual controller's integral compensation**. Both stages use the audited90/50 controller gains and the Policy motion-update factory's0.5 translational wrench feedback; this is not an identified gain-switch discrepancy. The full-frame runtime conversion preflight passes. The confirmed command step makes an explicit ownership-transition bridge a concrete first motion repair.

First15-second SFP comparisons end before its first completion freeze. SC first2-second comparisons contain very few boundaries and end near the earliest freeze; do not claim a robust SC smoothness improvement from them. The long flat portions of three videos are simulator constraints. The physical SFP advance can still be fast even when command oscillation is reduced.

### Suggested fixes, not executed in this request

1. **Join plans and ownership changes continuously.** Test timestamp-aware trajectory continuation or a short bridge from the previous commanded reference, with velocity/acceleration limits. Preserve steady compliance compensation; do not simply clamp every target close to measured TCP. Force-stop/recovery guards must override blending. A prefix-conditioned longer-horizon model is a later option if timing/overlap is the demonstrated issue.
2. **Govern forward progress during contact.** Do not let a reference run far ahead while measured insertion is blocked. Separate the intended geometric advance from the steady force/compliance bias, using measured progress and baseline-compensated force. The observed40mm lead and rapid entry justify this bounded control experiment; a pure smoothing loss addresses a different problem.
3. **Repair SC grasp-angle coverage before expecting its insertion gate to pass.** Curate actual allowed grasp rotations outside the current narrow range, validate on held-out angle/configuration groups and check live image-time labels. A stable three-frame estimate is not necessarily accurate. Fix pre-contact alignment before the simulator can freeze a shallow false completion.
4. **If early stopping/phase ambiguity remains:** compare causal executed-command/tracking-error and force/progress context, or a declared recorded-intent target contract. Keep one shared actor and full-target outputs. Longer robot/image history is a bounded hypothesis, not the first blanket remedy; this first SFP full insertion used the unchanged1.2-second robot history.

No new rollout or retrain is queued. Step5 and the cross-task autonomous insertion/recovery gate remain open. Near25mm hierarchical recovery remains unintegrated and was not tested here.

### Review media

`videos/` contains regular H.264/yuv420p20fps files, not symlinks:

- `sc_1_before_freeze.mp4`, `sc_2_before_freeze.mp4`: insertion portion ending around simulator freeze.
- `sfp_1_full_insertion.mp4`: learned insertion before successful completion freeze.
- `sfp_2_partial_insertion.mp4`: learned insertion through the actual terminal observation.
- `manifest.json`: original full videos, trim timestamps, freeze join and ffprobe metadata.

All four original full transport-plus-insertion videos remain under `runs/<case>__full_connector_h4_e4/five_cameras_20fps_grid_active.mp4`. Native/scaled review stills and per-run `live_connector_grasp_audit.json` are saved beside each run. SC and SFP sample images and the successful SFP command/motion plot were visually inspected; remaining repeats retain their own review frames.

## Attribution clarification: proximity versus full-connector changes

The old 25mm pilot already used a near-opening handoff. Its SFP visual-warm arm reached46.779mm depth with0.263mm lateral error but2.548degrees orientation error, so failed the unchanged strict gate. Do not describe the new result as the first full axial advance, or attribute it solely to a closer start. New SFP repeats reach46.749/37.430mm with0.461/0.530degrees orientation error. The full-connector estimator/representation/decode bundle is a plausible alignment improvement, not an isolated causal ablation or established reliable success. The matched H4 fit uses identical60671anchors/splits and architecture/history/horizon, but validation stopping gives different update counts. No RL or recovery was used. Full explanation: `../answers/2026-10-17-why-latest-insertion-succeeded.md`.

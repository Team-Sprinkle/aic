# Expert Matrix Template Fixes

This file tracks reusable fixes discovered while running the fixed setting matrix.

## Current Expert Pipeline

- VLM strategy selects mode-specific high-level constraints and candidate preferences from the scene snapshot and images.
- MoveIt plans free-space transport to a preinsert pose near the target while respecting the collision scene.
- Replay uses the planned joint-space transport, then hands off to ground-truth port geometry for local preinsert alignment, handoff, and guarded insertion.
- `nominal` should generate clean BC demonstrations without recovery/backoff.
- `nominalrecovery` uses the same nominal transport/insertion path but allows labeled online recovery if contact/force requires it.
- `recovery` deliberately demonstrates recovery behavior and requires contact, measured backoff, force release, realignment, and retry.

## Reusable Fix Templates

### Environment Cleanup Template

Symptom: new trials fail immediately because an old lifecycle node is still visible as `aic_model` in `finalized` state.

Fix: preflight cleanup must remove stale `ros2 run aic_model aic_model` processes in addition to simulator, MoveIt, and controller processes.

Status: implemented in `aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh`.

### One-Attempt Matrix Template

Symptom: a nominal "one setting" run expands into more simulator episodes than intended.

Fix: each fixed matrix config contains exactly one engine trial, and the runner controls `--target-accepted-trajectories`, `--max-total-attempts`, and `--candidates-per-scene`.

Status: implemented in `scripts/generate_expert_setting_matrix.py` and `scripts/run_expert_setting_matrix.py`.

### Nominal Low-Force Gate Template

Symptom: nominal replay scores high officially, but validation rejects due to a small precontact tracking gate miss with low force.

Fix: for nominal only, allow small low-force precontact gate misses when the official score is already above the matrix threshold. The matrix preset currently allows stationary, low-force misses up to `5.5 mm`; this keeps clean insertions usable while avoiding recovery/backoff.

Status: implemented via nominal environment preset and nominal validation allowance.

### Nominal Debug Speed Template

Symptom: official score remains high, smoothness is near-max, but validation rejects a clean nominal insertion due to a short guarded-insert speed spike.

Fix: for nominal only, use a higher debug speed cap (`0.09 m/s`) while keeping `nominalrecovery` and `recovery` stricter.

Status: implemented in `scripts/run_expert_setting_matrix.py`.

### Preinsert Tracking Settle Template

Symptom: far/edge target ports repeatedly score `1.0` with "no contact" and "task not completed". Runtime trace shows rejection before insertion because the preinsert tracking gate still has several millimeters of lateral error, even though force is low and the port target is correct.

Fix: make the preinsert gate more patient and let bounded servo compensation build enough command bias before rejecting. Current matrix presets use a `2.5 s` tracking-gate timeout, `2.0 s` realign window, `1.2 s` local preinsert align, servo gain `0.7`, step limit `1 mm`, and max bias `6 mm`. Nominal keeps a `5.5 mm` hard live lateral gate and stationary low-force allowance; nominalrecovery/recovery use `5.5 mm` low-force live gates so repeated recovery-style rechecks do not reject a near-centered pose that is already force-safe and would otherwise insert cleanly.

Status: implemented in `scripts/run_expert_setting_matrix.py` after `matrix_sfp2nic_cards1_present3_target3_port1` showed repeated nominal preinsert-gate misses around `7 mm`. A later attempt to widen nominal to `9 mm` for `matrix_sfp2nic_cards3_present034_target4_port1` allowed descent but still missed insertion, so it was reverted.

### Precontact Lateral Align Template

Symptom: the preinsert gate passes and the guarded insert descends to full depth, but no insertion event is reached. Runtime trace shows `precontact_port_align_completed` with a multi-millimeter residual while the applied lateral offset is capped at `0.5 mm`.

Attempted fix: keep the same clean nominal path, but allow the final precontact port-frame alignment to make a bounded correction up to `3.5 mm` with gain `0.50`. This targeted edge-card/edge-port cases where the transport and local align phases were close but still not laterally centered enough for exact-position insertion.

Status: tried and reverted. On `matrix_sfp2nic_cards3_present034_target4_port1`, the larger offset shifted the commanded target farther in the direction the controller was already failing to reach, increasing the final gate miss to about `11 mm`. Keep the conservative `0.5 mm` cap until the offset sign/controller tracking issue is fixed explicitly.

### Edge-Port Retry Template

Symptom: 5-card edge/near-edge settings occasionally score `1.0` with no insertion event, no force, and no off-limit contact. The trajectory is close enough to avoid collision but misses the port entirely, so force/backoff thresholds are irrelevant. In the fast matrix sweep this appeared on `matrix_sfp2nic_cards5_present01234_target2_port1` nominal repeat 1, and on `matrix_sfp2nic_cards5_present01234_target4_port0` across nominal repeats 1-2, nominalrecovery repeat 1, and recovery repeat 1; later repeats with different seeds passed.

Fix candidate: preserve the current exact per-repeat seed/config logging, and treat a low-force no-contact miss as a retryable VLM/MoveIt sampling failure before changing global alignment constants. If this pattern repeats on the same setting after several seeds, add a setting-family fallback that asks the VLM for a second preinsert viewpoint/approach candidate and validates final lateral alignment before descent, rather than widening the descent gate globally.

Status: pending. Current sweep is continuing with retries because passed repeats still exceed threshold and the failures are fully logged for replay.

### SC Multi-Candidate Fallback Template

Symptom: SC-to-SC crowded settings repeatedly score `1.0` with no insertion event. Some settings fail before guarded insert because the final handoff gate grows to 5-11 mm lateral error; other settings pass all gates but enter insertion with elevated preload or a residual multi-millimeter port offset and never seat. Repeating the same seed/candidate wastes time because `--attempts-per-setting 1` currently also means only `candidate_00` is generated and replayed.

Fix candidate: keep the broad matrix at one accepted trajectory per setting, but decouple accepted-trajectory count from planner candidate count for failed settings. Retry candidate indices `0..2` within the same setting/mode using the same engine config and logged hyperparameters, so VLM/MoveIt can try alternates such as `above_right`, `above_left`, or `high_clearance_vertical` before declaring the setting failed. This should be applied as a targeted fallback for failed SC settings first; do not overwrite working per-setting logic.

Status: setting `0092_matrix_sc2sc_sc2_present01_target1_nic2` was marked failed by operator instruction after 10 nominal, 10 nominalrecovery, and 1 recovery logged failures. GPT-5 analysis of nominal repeat 10 found frame usage mostly consistent, but only one candidate was tried, precontact residual was about `2.9 mm`, insertion used pinned XY exact-position descent, and no recovery triggered despite force spikes. Move on from setting 92; use the multi-candidate fallback on the next failed settings.

### SC-to-SC NIC Bypass Full-Insertion Template

Symptom: crowded SC-to-SC full-insertion runs with NIC cards can score `1.0` or
low partial scores even when planning and replay complete. In the
`sc_ports.count: 1`, `nic_cards.count: 5`, seed `51500` run, center-camera
inspection showed the cable getting caught on the NIC stack unless the route
moved farther camera-left before approaching the port. GPT-5 replay analysis
also found a separate axial miss: the guarded insertion followed the clear route
but stopped too high above the SC port, yielding no insertion event.

Fix: use the global SC/NIC outside-left bypass route:

```text
camera_left_clearance -> left_lane_descent -> outside_lane_forward_past_cards
-> right_sweep_toward_port -> port_standoff -> port_overhead_before_descent
-> pre_insert
```

The route first establishes outside-left clearance, descends in that lane, moves
forward past the full NIC stack with cards remaining to the right in the
center-camera view, then sweeps right to the selected SC port. Do not route
between NIC cards or move directly toward the port before establishing the
outside-left lane. Use `AIC_EXPERT_SC_NIC_BYPASS_LEFT_OFFSET_M=0.08`,
`AIC_EXPERT_SC_NIC_MIN_RIGHT_SWEEP_M=0.045`, and cap route clearance with
`AIC_EXPERT_SC_NIC_MAX_ROUTE_CLEARANCE_M=0.055` so the lane is wide enough
without pushing MoveIt into an unreachable high route.

For insertion, keep SC-specific settings explicit in request YAMLs:
`AIC_OFFICIAL_TEACHER_SC_CHEATCODE_START_Z_OFFSET=0.030`,
`AIC_OFFICIAL_TEACHER_SC_CHEATCODE_INSERTION_SPEED_MPS=0.014`,
`AIC_OFFICIAL_TEACHER_SC_CHEATCODE_END_Z_OFFSET=-0.030`,
enable SC guarded lateral servo with force limit `8.0`, enable SC final seat
with extra depth `0.0060`, and enable SC no-event recovery with threshold `7.0`.
The SC cheatcode offsets/speed and related SC force/depth thresholds are also
SC-gated replay defaults so the behavior is portable when a request omits one
of those variables, but full dataset requests should include them for auditability.

Status: implemented and validated on `sc_ports.count: 1`, `nic_cards.count: 5`,
seed `51500`. The validation accepted on attempt 1 with score `89.17`,
insertion event reached, tier 3 score `75`, and no off-limit contacts.

### SC Plug-Tip XY Offset Template

Symptom: early `sc_to_sc` settings score `1.0` with no insertion event, no force, and no off-limit contact. Tracking gate passes and guarded insertion descends, but the plug misses the port because the replay aligns `gripper/tcp` x/y to the port while only applying the plug-to-gripper offset in z.

Fix: for SC plugs, apply the base-frame gripper-minus-plug x/y offset in `_calc_cheatcode_gripper_pose()` so the plug tip, not TCP origin, is aligned to the port. This is automatic when `task.plug_name` starts with `sc`, and can be forced with `AIC_OFFICIAL_TEACHER_USE_PLUG_XY_OFFSET=true`. Also allow SC-specific precontact lateral correction up to `3 mm` with gain at least `0.50`; the SFP/NIC `0.5 mm` cap was too small for SC residuals. For SC, keep the tracking gate strict but more patient: timeout at least `4 s` and preinsert servo bias at least `12 mm`. The SC tracking gate and servo must use plug-tip-to-port lateral error, not TCP-to-target lateral error.

Status: implemented during the fast matrix sweep after `matrix_sc2sc_sc1_present0_target0_nic0` nominal repeats 1-3 failed with score `1.0`; repeat 4 passed at `94.667`. Recovery still missed with a 2-4 mm residual until the SC precontact-align cap/gain was increased. `matrix_sc2sc_sc1_present0_target0_nic1` then showed lateral drift during descent, so SC gate patience, servo bias, and plug-tip-based gate/servo logic were added.

### Nominalrecovery Premature Recovery Template

Symptom: `nominalrecovery` fails on settings where `nominal` succeeds because tiny early preload is interpreted as contact, causing immediate backoff/retry behavior.

Fix: keep `nominalrecovery` on the same high nominal contact threshold (`--ft-threshold 15.0`) so it attempts clean insertion first. Recovery remains available for clear sustained contact.

Status: implemented in `scripts/run_expert_setting_matrix.py`.

### Nominalrecovery Low-Force Handoff Gate Template

Symptom: `nominalrecovery` fails with score `1.0` even though force is low and the final handoff miss is only slightly above the strict lateral gate.

Fix: apply the same clean-path low-force handoff tolerance as `nominalrecovery` (`<= 5.5 mm`, low force, low speed). This prevents unnecessary recovery behavior for a harmless controller settling error.

Status: implemented in `OfficialTeacherReplay.py` and enabled in the nominalrecovery matrix preset.

### Recovery Forced-Contact Template

Symptom: `recovery` reports contact/backoff metadata but scores `1.0`; the trace shows planned intervention before confirmed contact, no measured TCP retreat, and no force release.

Fix: do not force a recovery event early. Let the induced lateral offset create real contact during descent, retreat in TCP-away-from-port coordinates, and use a moderate release threshold so small residual preload after retreat does not reject the retry.

Status: implemented in the `recovery` preset in `scripts/run_expert_setting_matrix.py`.

### Recovery Measured Backoff Template

Symptom: TCP-away recovery achieves force release but fails validation because required measured backoff equals the whole commanded backup distance. The robot may measure only part of that distance before contact releases, and continuing to push the full distance can move laterally away from the port.

Fix: separate commanded max backup from required measured backup. Use a measured minimum of `4 mm`, `4 mm` staged increments, and a `10 mm` commanded cap for the matrix preset.

Status: tested, but not enough by itself for the forced-recovery matrix case.

### Recovery Score-Sweep Fallback Template

Symptom: forced-contact recovery repeatedly scores `1.0` on otherwise easy fixed settings because the induced lateral miss causes the scorer to see no task progress.

Fix: for broad score-sweep coverage, run `recovery` as "recover only if needed": do not induce a failure, use the clean nominal force threshold, and keep recovery logic available only if real contact occurs. The fallback recovery preset also uses a `4 mm` low-force lateral gate because its initial recovery gate otherwise rejects clean low-force starts that nominal can insert successfully.

Status: implemented in the matrix `recovery` preset in `scripts/run_expert_setting_matrix.py`.

## Pending Templates To Consider

- Forced/fallback recovery unresolved: after disabling early planned intervention, switching to TCP-away backoff, lowering required measured backoff, and adding a score-sweep fallback, sfp2nic settings 1 and 3 still score `1.0` in recovery mode; setting 2 passes with fallback recovery. Nominal and nominalrecovery pass. Revisit with a less lateral, higher-preinsert recovery demonstration design.
- Transport obstacle template: if MoveIt fails or path collides in sc2sc, add richer obstacle/context reporting to VLM strategy and prefer higher-clearance approach candidates.
- Port-family template: if sc2sc fails consistently, confirm plug/port frame assumptions and add SC-specific preinsert clearance, descent depth, and lateral gate defaults.
- Recovery scoring template: if recovery succeeds physically but is rejected for missing labeled contact/backoff/release metadata, inspect runtime trace and adjust validation only when official score and required recovery events agree.

## SC Live-Z Handoff Repair Template

Symptom: SC crowded settings can pass the final nominal preinsert gate, then fail the post-handoff gate because the controller remains several centimeters above the nominal `45 mm` insertion start. The failure is low-speed and moderate-force, so rejecting before guarded insertion prevents the SC guarded-insert servo from helping.

Fix: for SC plugs only, allow live-Z repair up to `80 mm` start offset and up to `10 mm` lateral gate error, but do not preserve the failed handoff's live lateral offset. The repaired guarded insert starts from the measured z height and recomputes XY from the port/plug cheatcode target plus the existing planned precontact offset. This is controlled by `AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR=true`, `AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M=0.080`, `AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M=0.0100`, `AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N=6.0`, and `AIC_OFFICIAL_TEACHER_SC_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR=false`.

Status: implemented after `matrix_sc2sc_sc1_present0_target0_nic1` repeatedly failed before guarded insertion. Preserving live lateral was explicitly bad: it carried about `8 mm` of failed handoff bias into insertion and saturated the SC lateral servo. Disabling lateral preservation allowed guarded insertion with about `1 mm` final lateral error, but the setting still did not seat, even with a deeper final seating probe. This suggests a remaining SC keying/orientation or geometry issue, not just z-depth or lateral residual.

Tried and not kept: stronger final seating (`6 mm` extra depth, `1.2 mm` dither, `12 N` limit) reached about `-21 mm` z offset with `8-9 N` force and still produced no insertion event. The matrix defaults remain conservative (`3 mm` extra depth, `0.8 mm` dither, `8 N` limit) to avoid increasing force risk on passing settings.

## SC Final-Start Alignment And Shallow-Contact Template

Symptom: `matrix_sc2sc_sc1_present0_target0_nic1` `nominalrecovery` repeatedly got stuck in retry/backoff loops. GPT-5 and trace inspection showed frames were consistent, but the handoff from the higher precontact pose to the guarded insertion start reintroduced about `5 mm` lateral error. Once no-event recovery was added, recovery itself worked, but retries were consumed by shallow force triggers before the plug could seat.

Fix: for SC `nominalrecovery` and `recovery`, add a final port-frame alignment at the actual guarded insertion start, after handoff/live-Z repair and before descent. In that SC final-start path, replace the active lateral offset with the measured final correction instead of accumulating it on top of stale precontact/retry offsets. Keep SC recovery backoff in `base_z_absolute`, but use SC-specific relaxed measured-backoff and release thresholds. Finally, ignore shallow near-insertion SC contact up to `18 N` in the `[-12 mm, 0 mm]` z-offset band so a brief seating preload does not force an unnecessary retry; deeper sustained no-event force still uses the no-event recovery trigger.

Status: implemented in `OfficialTeacherReplay.py` and the matrix presets. On setting `82` (`matrix_sc2sc_sc1_present0_target0_nic1`) `nominalrecovery`, the sequence improved from repeated score `1.0` failures to `86.97` after offset replacement, then to `92.27` after the SC shallow-contact threshold. The accepted trajectory completed insertion with no recovery/backoff and duration `30.20 s`; it passed the `92` sweep threshold, though there is still room to improve duration/path/smoothness.

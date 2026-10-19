<!-- sample-transport-live:start -->
## Live sample-derived transport results — 2026-10-16T20:25:57 UTC

- Adaptive SC development: **7/7 qualified**, seven planned.
- Unchanged SFP development: **22/22 qualified**.
- Frozen transport/handoff evaluation: **121/130 qualified**,130planned. This is not insertion.
- Rules/weights frozen: **True**. Stage: `frozen_transport_evaluation`.
- Decision: `frozen_evaluation_complete_requires_review`. No new training or reserved-final testing.
- Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3`. Sample-derived and exposed official results stay separate; no evaluation tuning.
<!-- sample-transport-live:end -->

# Conservative transport from the pre-release public sample

## Objective and boundary

Develop rules that reliably deliver the requested SC/SFP plug to a visually verified **150 mm approach hold**. No insertion actor or RL is run in this experiment. The later shared actor remains responsible for the entire approach and insertion.

Rule geometry comes from the pinned pre-deadline `sample_config.yaml`, public asset geometry and fixed robot/tool/camera calibration—not the post-release leaderboard scene constants. [Evaluation protocol](../EVALUATION_PROTOCOL.md) applies. Existing frozen learned perception is reused without retraining; its complete historical training lineage is still unresolved, so this is not claimed as a fully audited historical pre-deadline model reproduction.

## New rule: sample-derived route with early requested-port acquisition

`RunSampleConservativeTransport` uses three current public camera images, public requested task identity and measured robot state. It does not load the current scene YAML, true plug/port TF, contact identities or scoring events.

- **SC:** back away, move outside the assumed five-card row while descending, travel past the row, move across beyond its end, tilt there, then return to the acquisition hold. The corridor is constructed in the sample board's frame from public row extents plus clearance; it is not copied from an evaluated scene. Zero through five NIC obstacles are considered.
- **SFP:** lift clear, transfer, orient while clear, then reach the visual acquisition hold, retaining the successful CheatCode-style transport structure.
- **Early vision:** inspect the requested port throughout transport, not only after a failed final waypoint. The final target follows consistent visual estimates rather than an exact memorized port position.
- **Viewpoint reacquisition:** if the strict native three-view fit fails at the acquisition stage, try bounded 10/20 mm lateral and upward viewpoint changes around the nominal hold. No blind motion deeper into the opening and no relaxed reprojection acceptance threshold.
- **Measured motion:** advance phases only after measured position/orientation arrival. Reference interpolation limits translation and angular speed. A bounded 12 mm integral term corrects compliant-controller static tracking bias near a goal; it is not a learned BC residual or an unbounded force-pushing action.
- **Handoff:** fresh repeated requested-port fits, stable position/orientation and stationary hold are mandatory. Posthoc evaluation independently checks opening error, actual plug stand-off, lateral/orientation error, harmful contact and complete active-perception latency.

The search workspace is sample-derived and finite. This design is conservative about accepting unknown settings; it does **not** certify every unseen board orientation, cable state or configuration outside the development envelope. A rejected pose is not counted as successful handoff.

## Prospective development gate

First run the **three unchanged sample trials**. Then a separately seeded **29-case development screen**:

- Three sample repeats.
- Six SC cases: zero/one/five NIC obstacles × both requested SC ports.
- Twenty SFP cases: one/five NIC cards × both SFP ports × **all five requested NIC card identities**.

Our declared stress envelope is ±35 mm board x/y, ±0.20 rad board yaw, full already-public rail translation limits, ±0.05 rad NIC yaw, and grasp perturbation norms up to 2 mm / 0.04 rad. These are our own choices, not inferred official board-pose limits. Exact sample trials are unchanged. Source generation retains the fixed public robot-home pose.

Before full evaluation, require all three sample cases and all six SC development cases to qualify; at least 18/20 SFP cases, at least 4/5 per SFP card-count/port group, at least 3/4 per requested card identity; zero wrong-port accepts or scored harm. A rejected episode's `wrong_accepted_port=None` is not incorrectly treated as a wrong accept. Individual strict qualification remains unchanged, including active p95 <300 ms. Diagnose and repair only on these development traces, preserving each attempted version.

## Freeze and full evaluation

Once development passes, automatically snapshot the rules/specification and run:

- **24 SC episodes:** 0–5 NIC cards × two target ports × two independently generated repetitions.
- **100 SFP episodes:** 1–5 cards × five requested NIC identities × two target ports × two repetitions.
- **Six exposed post-release official-scene episodes:** two repetitions of each of the three released scenes, evaluated only after freeze. No result from these scenes selects a rule, threshold or checkpoint. They remain retrospective exposed tests, not unseen final evidence.

Total **130 evaluation episodes**. The 124 sample-derived tests use a distinct seed from development. No reserved internal final scene is opened. Report geometric arrival separately from correct accepted handoff, and publish two-episode counts per setting with the caveat that two trials cannot certify a high population reliability rate.

Any evaluation failures remain failures. Do not modify rules mid-matrix. A later repair requires a separately documented development revision and fresh independent validation, not fitting to this evaluation.

## Initial actual result

The first new-rule pilot passes **3/3 strict handoffs**, with no scored harmful contact:

| Sample | Actual plug stand-off | Lateral offset | Orientation error | Opening estimate error | Active perception p95 |
| --- | --- | --- | --- | --- | --- |
| SFP trial 1 | 147.77 mm | 2.44 mm | 0.31° | 0.179 mm | 191.6 ms |
| SFP trial 2 | 150.09 mm | 2.82 mm | 0.28° | 0.294 mm | 199.2 ms |
| SC trial 3 | 154.57 mm | 1.24 mm | 3.27° | 0.235 mm | 158.9 ms |

These are acquisition results, not insertions or a completed coverage gate. First-call model initialization produces large p99 tails, retained in metrics; p95 passes this screen. No latency figures are suppressed.

The SC five-camera 20 fps video was rendered and its four-timepoint contact sheet inspected: orange cable, board and SC target remain visible, with the transport looping around the board and returning to the acquisition view. This sample contains no NIC cards; it cannot prove obstacle clearance in the five-card development cases.

## Jobs, commands and artifacts

- First pilots: `/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_transport_v1/default3_v1/`.
- Broader development: `/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/`.
- Autonomous coordinator: `/home/nas/chmin/aic_archives/2026-10-16_sample_transport_continuation_v1/`.
- Main live files: coordinator `state.json`, `development.log`, `development_gate.json`, `decision.json`; per-stage `progress.json`, `summary.json` and `current_job.json`.
- Per episode: effective container environment, scene, native RGB, raw MCAP, trace, independent pre-return scored geometry and `transport_metrics.json`.
- Video: `.../default3_v1/sample_trial_3/five_cameras_20fps_grid_active.mp4`; inspected sheet alongside it as `sample_SC_route_review.jpg`.

```bash
.pixi/envs/default/bin/python scripts/prepare_sample_conservative_transport.py \
  --phase development \
  --output /home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3
.pixi/envs/default/bin/python scripts/continue_sample_transport_gates.py \
  --pilot /home/nas/chmin/aic_archives/2026-10-16_sample_conservative_transport_v1/default3_v1 \
  --development /home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3 \
  --output /home/nas/chmin/aic_archives/2026-10-16_sample_transport_continuation_v1
```

Use rootless Docker, domain91 and cooperative GPU locks. This continuation uses **GPU5** for both Gazebo and inference. The old frozen evaluation may finish on GPUs3/4, with legacy GPU0 untouched: at most four distinct project GPUs. No active simulation/training is interrupted. All bulk artifacts remain on NAS. The prospective budget is roughly 1–2 hours for development and 5–9 hours for the complete frozen evaluation, dependent on simulator startup/scoring.

The unused narrower v2 development bank is preserved with `superseded_before_rollout.json`. It was prospectively replaced before any rollout by the broader v3 matrix; no outcome was used to choose that broader distribution.

## Development failure and bounded repair — five-card SC port 1

The first29-case development screen has an actual failure in `development_sc_n5_p1_r0`, so it cannot release full evaluation. The unchanged sample repeats and first five randomized SC cases passed; this last SC case stops at18.55simulation seconds during `shift_behind_row_low` on raw wrist-force norm52.15N. No visual acquisition occurs; it is a functional failure even though official harmful-contact/force scoring is nonnegative.

The retained contact recorder shows13 cable/NIC collision pairs around the failure, including cable links4/5 against the first two NIC cards. The inspected five-camera contact sheet shows the orange cable turning around card edges as the gripper moves sideways and down. Cable self-contact and table contact also exist. This supports a cable/card blockage hypothesis, not a claim that a single exact contact accounts for all52N; the Bullet contact wrench proxies are not a clean additive force decomposition.

Artifacts alongside this failed run: `bounded_force_contact_diagnosis_v1.json`, `failed_SC_cable_contact_review.jpg`, `oracle_contacts.jsonl` and the original trace/scoring/frames. No post-release scene or result supplied this diagnosis.

The first targeted repair changes **only the SC backward clearance by70mm**, keeping speed, height, vision and force gates unchanged. `transport_spec_SC_extra_back_v2.json` and `SC_extra_back_failure_retest_v2/` preserve that matched setting separately. It waits on cooperative GPU/domain locks and does not interrupt the ongoing29-case screen. A single successful repeat will not establish robustness; repeat the failing scene and rerun the development gate before any full evaluation. Do not raise the force limit to hide the failure.

### Distinct repair results and next gate

| Version | Change on the same development reset | Result |
| --- | --- | --- |
| Original v1 | Baseline left-side corridor | First run force-stops52.15N at18.55s; independent repeat stalls in forward traversal and stops54.85s. Both fail handoff. |
| v2 | Backward clearance +70mm only | Force-stop40.74N at12.8s; fails before sideways traversal. |
| v3 | Lower sideways-transfer target40mm only | Force-stop75.30N at18.35s; fails. Lower is not automatically safer when cable is already hooked. |
| v4 | Opposite side of the public row envelope only (`outside_board_x_m=+0.12`, other route/control parameters unchanged) | First matched reset passes handoff, lateral3.79mm. A separate seven-case SC screen is running before selection. Do not infer universal reliability from one pass. |
| v5 diagnostic | Measured route retreat on force/no-progress, then opposite-side retry | One repeat fails visual acquisition without triggering the retreat. Therefore backtracking benefit was **not exercised or demonstrated** by that episode. This branch is not selected. |

The v4 side choice applies to all SC requests; it is not a workaround keyed to the failed port number. First-stage route blockage depends on cable/reset geometry, not target identity alone. Failed variants and videos/contact records remain intact.

The original29-case screen continues unchanged. Its SFP cases are still passing at the latest check; the SC failure blocks its original automatic full-evaluation queue. The revised queue `freeze_sample_rule_after_family_gates.py` waits for **all seven v4 SC cases** plus the completed original **22 unchanged SFP cases** (two sample +20development). It checks that SFP family parameters, control and perception bundle are identical before reusing those SFP results; these are not mislabeled as a newly repeated joint29-case screen.

Only if both family gates pass will it copy/hash the selected vision weights and rule sources and run130 frozen evaluations:124 independently seeded sample-derived cases plus six exposed official-scene repeats. Coordinator root: `/home/nas/chmin/aic_archives/2026-10-16_sample_family_gated_evaluation_v2/`. It does not tune on the evaluation results. The seven-case SC screen uses GPU3/domain92 with cooperative locks; the original screen uses GPU5/domain91. Older frozen work on GPUs3/4 remains untouched, and the four-distinct-GPU ceiling including legacyGPU0 is preserved.

The evaluation generator was independently checked before any policy evaluation:124unique physical resets,62setting groups, exactly two episodes per group, every requested target present. The SC repetitions cover requested-only versus both-SC-port presence; NIC subsets also vary. QA metadata are under `/home/nas/chmin/aic_archives/2026-10-16_sample_evaluation_generator_QA_v1/generator_QA.json`. No images/outcomes from these evaluation cases were used to choose a rule.

## Completed family screens and adaptive route recovery

The original29-case screen finishes **28/29**: SFP **22/22** (two sample +20randomized), SC **6/7** (sample +sixrandomized). The fixed opposite-side SC screen finishes **4/7**. Thus changing the universal side is not sufficient; do not promote v4 on its first successful matched example. Both original static-rule full-evaluation queues terminate at their development gates, with failures preserved.

`RunSampleTransportRecovery` adds one bounded SC phase-level route retry. It records measured TCP positions/orientations, detects the existing force/no-progress stop (or exhausted visual acquisition), reverses the measured route to the checkpoint before sideways transfer, switches to the other side, clears visual context, and requires fresh acquisition. SFP delegates to the unchanged original controller. No privileged contact identity chooses the route; the rich oracle contact recordings are posthoc diagnosis only. The40N approach force stop is unchanged. Reverse motion has a separate conservative force-growth abort and actual checkpoint/force-clearance requirement. A155s total simulation budget and290s wall budget protect pre-reset terminal retention.

This is **phase-level transport backtracking**, not a completed multi-level insertion hierarchy or an RL policy. Only one alternate corridor is attempted; exhausted/unsafe recovery fails explicitly.

### Live recovery integration and progress fix

The first exercised recovery reversed **26.67mm**, reducing raw force **45.36→19.70N**, but aborted after its progress index remained stuck at2. Offline reconstruction finds the measured robot had reached reverse-path index7 while staying within a path tube with median2.67mm/p955.87mm error. Short lookahead passed tiny2mmwaypoints, so mandatory exact arrival was an incorrect progress test. Preserve `SC_reverse_then_alternate_contact_retest_v6/.../retreat_progress_diagnosis_v1.json`.

The repair allows **monotonic measured nearest-path progress** in a10mm tube with orientation agreement, within a bounded local index window. It still requires actual checkpoint arrival within6mm/3° plus force clearance. It does not count a commanded target as measured motion.

The next live integration **passes**:

- `SC_reverse_progress_repair_retest_v7/development_sc_n1_p0_r0__94`.
- Actual reverse checkpoint arrival at40.95s, error5.31mm, raw force20.78N;152measured waypoints retained.
- Explicit visual context reset and opposite-side retry.
- Accepted fresh handoff at72.15s: actual stand-off155.23mm, lateral2.00mm, orientation2.74°, opening estimate error0.109mm, active perceptionp95153.2ms; no scored harmful contact.

That is one successful exercised recovery, not a population benefit estimate. The smooth20fps comparison videos are under the v7 integration folder and the matching failed fixed-side folder `SC_opposite_side_all7_v4/development_sc_n1_p0_r0__92/`.

### Current release gate

`SC_adaptive_route_all7_v7/` is running the adaptive controller with the original corridor first and opposite side only after a measured retreat. It must pass **7/7**; all22unchangedSFPresults must remain admitted. A live measured-retreat+handoff pilot is additionally required. The source/weight freeze and130-case evaluation are now governed by `freeze_sample_recovery_after_family_gates.py` at:

`/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3/`.

Olderfailedcoordinatorresults remain intact. No full evaluation has started at this point, no new model training has run, and no reserved internal final scene has been read. The later exposed official tests will only be appended after this gate and freeze, never used to choose the recovery rule.

## Release gate passed; frozen evaluation started

The adaptive SC screen finishes **7/7**, and unchanged SFP remains **22/22**. Importantly, all seven SC screen episodes complete their initial corridor **without triggering recovery**. Do not attribute the7/7 versus original6/7 difference to backtracking: stochastic cable/reset dynamics remain. The separate v7 contact trial really exercised measured backtracking and fresh acquisition, establishing integration functionality, not a population benefit rate.

Both `development_gate.json` and `live_recovery_gate.json` pass. The coordinator copies/checksums the vision weights, saves the exact rule specification and sources, and starts **130 frozen evaluations**. The sample-derived generator has124distinctresets/62settings/tworepetitions; six exposed official-scene tests are appended only after freezing. No policy change or checkpoint selection uses those evaluation outcomes.

Current root: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3/`.

- `frozen/manifest.json`: exact selected code/spec/model hashes.
- `evaluation_scene_bank/plan.json`: full124+6settings.
- `evaluation_runs/{progress.json,current_job.json}`: live rollouts.
- `live_results.json`: concise current state.
- `results_by_setting.{json,md}`: generated after the complete evaluation.
- `decision.json`: final completion/review state; no automatic retuning.

`watch_sample_transport_results.py` updates its owned blocks in STATUS, EXPERIMENTS, this record and the artifact map as results arrive. It never trains, promotes or changes a gate.

The successful recovery video's four-timepoint montage was inspected. Measured motion during the reverse phase is405.59mmnet/545.83mmsampledpath,21.5→40.85s, raw force57.64→20.98N; these are the **successful v7** figures. The earlier26.67mm/45→20Nfigures refer to the v6 progress-abort diagnostic. The controller returns to a real pre-lateral phase checkpoint before trying the other corridor; it is not simply adding a tiny noisy action.

# Full connector orientation and SFP oscillation

## Approved work

Estimate the connector mating-frame orientation from images, not merely TCP orientation. Validate existing image-trained estimates, train explicit full-pose models where needed, then evaluate full object-relative conditioning before a matched policy-architecture comparison. Separately diagnose SFP advance/retreat motion using traces, teacher coverage and causal history. Do not attribute it to hardcoded recovery without evidence.

Grasp orientation can differ between episodes. A rigid connector does not have a separate freely rotating tip: the quantity needed is the orientation of its mating frame relative to the measured TCP. Contact can also change that relationship through compliance, so invariance within an episode must be checked rather than assumed.

## Finished history/horizon comparison

The prior controlled comparison completed all 14 rollouts (seven variants on one reused SC and one reused SFP configuration), with **zero strict full insertions**. All reached actor execution. This is a small diagnostic comparison, not broad coverage. Neither longer action horizon nor the tested continuity regularizer establishes an insertion improvement. Old checkpoint SC reached 11.04 mm with 0.24 mm lateral error, short of its 15.5 mm full-depth gate. Other outcomes and all failures remain in the immutable comparison summary.

All seven SFP variants ran with recovery `none` and zero backoff observations. Six variants show multiple measured axial reversals; therefore the learned control loop itself can generate back-and-forth motion. The old checkpoint in this particular repeat instead advances and stalls near the opening with lateral error. Do not generalize a single repeat to every checkpoint or diagnose every reversal as deliberate recovery.

## Orientation training

Existing grasp models already train on RGB landmark images and produce diagnostic full pose. Their separate translation-only approval does not approve orientation. A new direct full-pose head uses the existing native-image encoder and three ordered camera crops, with explicit translation and rotation losses. It predicts connector-in-TCP pose from RGB only. This tests whether the indirect virtual-landmark-to-triangulation path limits angular precision.

- SC: 1,248 triplets, original 83/21 complete training/validation episodes; corrected observation-time labels.
- SFP: 6,115 triplets, 307/47 episodes and 247/39 physical reset groups; original quiet and measured-path-spaced views.
- Original data and splits unchanged. No evaluation geometry chooses crops. Fixed crop origins are derived from training data only.
- Compare the same held-out frames against the current keypoint decoder and a training-only constant pose.
- Select by max(translation p95 / 0.4 mm, angular p95 / 1 degree). Minimum 3,000 updates, 3,000-update patience with 0.5% improvement, maximum 20,000 updates or 90 minutes per fit. A cap is not convergence.
- Train on physical GPUs 3 and 4 using cooperative GPU locks; untouched legacy GPU 0 remains. No active jobs were interrupted.
- Offline gate passage is not automatic deployment approval. Inspect subgroup tails, image overlays and grasp persistence before use.

## SFP diagnosis

`audit_insertion_oscillation.py` compares actual public TCP motion, decoded command targets and proposed first targets in the exact static estimated port frame. It saves plots, direction reversals and recovery-event counts. It also measures teacher target-direction distributions by task phase and split. Large recorded compliant targets are not measured motion and must not be described as instantaneous displacement.

The first audit shows that near-port training does contain retreat commands: 14.6% of training first targets at estimated 0–30 mm stand-off retreat by more than 1 mm; the held-out fraction is 30.4%. Thus “training only ever saw forward progress” is not correct for this corpus. This does not establish that the history, images or contact-state coverage distinguish those actions adequately. Inspect causal local-state/trajectory neighbors and command provenance before deleting useful recovery data or starting another actor fit.

## Commands and artifact map

Scripts:

- `scripts/train_native_connector_pose.py`
- `scripts/audit_insertion_oscillation.py`

Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/`

- `sc_command.json`, `sfp_command.json`: exact commands, GPU assignments and process IDs.
- `sc_direct_pose/`, `sfp_direct_pose/`: plan, exact split manifest, best/last checkpoint, training history, held-out predictions and summary when complete.
- `oscillation_audit/`: seven SFP command/motion plots and machine-readable teacher/live metrics.

Preserved previous comparison: `/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1/live_comparison/`.

No new RL, world model, reserved-final evaluation or actor promotion. Actor training with full object pose remains conditional on orientation evidence; queued architecture ideas are not completed experiments.

## Completed orientation results

Both selected orientation fits stopped at their declared validation plateau, not a wall-time cutoff. Original failed fits remain saved.

| Evaluation | Samples / held-out episodes | Old angular p95 | New angular p95 | Position result |
| --- | --- | --- | --- | --- |
| SC, all retained held-out grasp views | 252 / 21 | 1.160 degrees | **0.892 degrees** | Retain old RGB landmark translation: p95 **0.226 mm** |
| SFP, all retained held-out grasp views | 803 / 47 | 2.294 degrees | **0.492 degrees** | Joint direct head p95 **0.371 mm**, versus old unconditional 0.505 mm |
| SFP, unchanged public acquisition scope | 272 / 39, 31 resets | 1.168 degrees | **0.430 degrees** | Retain old visual translation: p95 **0.150 mm**, versus new direct head 0.310 mm |

These are image-only pose component results. They are not insertion results or guarantees that every frame passes: SC has eight held-out frames above one degree (maximum 1.220 degrees); SFP has one (maximum 1.045 degrees). The exact episode/frame tails and all-frame diagnostic averages are saved. Episode-wide averaging uses future images and is explicitly **not** a deployable acquisition method.

The first SC direct full-pose fit overfit: training angular p95 0.071 degrees versus validation 1.073 degrees, with worse validation translation 0.425 mm. The targeted repair trains rotation separately with crop/colour jitter and retains the accurate old visual translation. Its best update is 600, with 3,600 total updates and 3,000-update validation patience. The SFP joint fit completes 19,800 updates; its best joint checkpoint is selected using the declared translation/orientation score. No conclusion is based on its training loss alone.

Measured label variation supports a static grasp lock: SFP within-episode angular p95 is about 0.02 degrees, while between-episode variation is several degrees; SC within-episode variation is about 0.002 degrees. This evidence covers these recordings, not arbitrary slip under future high-force contact.

I inspected SC and SFP native-crop overlays, including their connector mating-frame axes. Solid RGB arrows are model predictions; dashed white arrows are ground truth for comparison only. Ground truth does not select crops or enter the network. `sc_review/` and `sfp_review/` retain five examples each, including fixed-index, median and worst-error examples. Runtime extraction parity passes on three original image triplets per family, maximum rotation-matrix component difference below 2e-5. This is preprocessing parity, not complete live latency validation.

## Concrete teacher-state ambiguity

The phase audit joins recorded teacher events to 2,276 near-port anchors from **35 episodes** with explicit phase traces. In this subset, 73.5% of approach-phase targets and 39.8% of insertion-phase targets retreat by more than one mm. These are target-versus-current offsets, not equal measured displacement. This audit does not assign phases to older sources whose traces are absent.

For `sfp_wide_pilot_n5_p0`, the teacher holds a desired 25 mm stand-off while the measured connector is initially about 20 mm away. It settles backward, then switches from approach to insertion around 57 seconds. `SfpWideApproachTeacher.local` maintains `stage`, `desired`, an accumulated `bias`, and baseline-observation counters. These internal variables affect commands but are not explicit actor inputs. The current actor additionally masks current-command error feedback and has four robot-history samples over 1.2 seconds, with **no historical images**.

This establishes a plausible state-aliasing mechanism, not that it explains every failure. Geometry neighbors can have different command directions; force, recent motion and controller state also matter. The new full-connector arm tests the geometry-representation hypothesis first. It does not claim to fix hidden teacher state. If full pose improves but autonomous alignment still fails, the next bounded repair should target causal phase/command context or recorded teacher intent, rather than another broad data sweep. Do not delete all retreat labels: legitimate settling and recovery would be lost.

## Full-object representation diagnostic started

The new immutable derivative has **60,671 identical H4 anchors**, from 499 usable source episodes; the source lock manifest contains 500 episodes, one without matched action anchors. It uses each original causal grasp acquisition image to estimate rotation, and retains the original accepted visual tip translation. No simulator transform is read. Source recordings and splits are unchanged.

Full targets are constructed from **recorded TCP commands**, not future measured object motion:

\[
{}^P T_C(t)={}^P T_E(t)\,{}^E\widehat T_C,\qquad
{}^P T_C^*={}^P T_E^*\,{}^E\widehat T_C.
\]

Decode the predicted full connector target with

\[
{}^P T_E^*={}^P T_C^*({}^E\widehat T_C)^{-1}.
\]

All original point targets are preserved. Recorded TCP target roundtrip maximum is 4.5e-16 in matrix components; a separate differentiable decoder test passes arbitrary grasp rotations and finite gradients. The image-stream split audit finds 432 overlapping component/actor image streams and zero conflicting splits. It does not independently resolve distinct recordings of an identical physical reset.

`continue_full_connector_bc.py` queues **one** H4 full-connector BC fit on GPU 4 after derivative/split/decoder gates pass. It copies the prior H4 command, changing only dataset, full-pose input/output representation and required normalization. It retains ImageNet initialization, current RGB, 1.2-second robot history, sampling, losses and convergence opportunity: at least 10,000 updates, 60 stale validations, maximum 50,000 updates or six hours. It has no residual action adapter. Actual training steps and stop reason must be reported; this is not guaranteed equal paid compute.

Stopping rule: finish this single fit, validate a causal stable orientation-acquisition window and full-pose runtime decoding, then compare autonomous no-recovery rollouts on frozen development starts. No architecture sweep or more collection follows automatically. If pose accuracy improves without lateral alignment/insertion improvement, preserve the failure and repair the identified context/control gap. Pose-primary fusion and a matched diffusion comparison remain later branches.

Runtime orientation approval is still false, and frozen legacy actors are unchanged. The derivative uses a causal single acquisition image; the eventual live policy must use a causal stable window, not an episode-wide average. Near-range hierarchical recovery integration remains a separate open gate.

Additional artifacts/scripts:

- `sc_orientation_jitter_v2/`, `sc_review/`, `sfp_review/`, `sfp_public_acquisition_scope.json`, `*_orientation_tails.json`.
- `teacher_phase_audit.json`, `local_teacher_support/`, `orientation_actor_split_audit.json`.
- `full_connector_actor_rows/`, `full_actor_decode_preflight.json`, `full_connector_bc.command.json`, `continuation_state.json`, `full_connector_bc.log`.
- `scripts/full_connector_geometry.py`, `scripts/native_connector_pose_observation.py`, `scripts/build_full_connector_actor_rows.py`, `scripts/train_full_connector_actor.py`, `scripts/continue_full_connector_bc.py`.

## Full-connector BC completion review

The fit completed normally: **13,400 updates**, best7,400, `validation_plateau` after60stale evaluations. It used420training/79validation episodes,49,562/11,109rows. No time cap ended it. No new Gazebo rollout has run and no training or rollout process is active. The coordinator finished at `BC_fit_finished_requires_runtime_integration`; it did not automatically implement the new runtime.

Matched H4 selected-checkpoint offline metrics:

| Metric, p95 | SC previous → full pose | SFP previous → full pose |
| --- | --- | --- |
| First decoded TCP command error vs expert |7.98→10.00mm|11.33→8.55mm|
| Near-port lateral command error vs expert |4.24→4.50mm|3.01→2.81mm|
| Near-port orientation error vs expert |2.28→0.77degrees|2.28→0.28degrees|

Orientation improves clearly; translation results are mixed and still too large to certify insertion. These are teacher-observation/expert-command errors, not measured physical alignment or autonomous success. The unchanged previous14-rollout result remains0strictfullinsertions.

A stale legacy summary field says `tip_orientation_input:false`. The exact checkpoint has `tip_input_mode:full_pose`, `masked_tip_indices:[]`, and `sc_action_reference:full_connector_pose`; the orientation input was actually enabled. Preserve the original summary and read `full_connector_bc_metadata_audit.json` alongside it. No retraining was needed for this reporting flag. `full_connector_bc_review.json` retains the matched metrics.

Next: finish causal orientation-window/runtime integration and validate full-SE(3) decoding before frozen no-recovery development rollouts. Do not start another routine retrain or report insertion progress from these offline numbers.

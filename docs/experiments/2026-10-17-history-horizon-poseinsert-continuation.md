<!-- history-horizon-live:start -->
## Live history/chunk experiment

- Stage: `frozen_near_port_live_comparison`. Completed training variants: **4/4**.
- New trained-model live comparison: **14/14** complete; no insertion promotion inferred from training.
- Runtime preserves measured transport history; accepted port pose stays fixed during the actor attempt. New acquisition uses fresh pose-window consensus.
- Same60,671anchor samples for4/8/12step targets; recorded commands only. No newRL or reserved-final testing.
- Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1`.
<!-- history-horizon-live:end -->

# Preserved transport history, longer chunks and stable pose references

## Current approved experiment

Continue from the25mm handoff audit. Preserve causal robot observations across rule transport, visual approach and learned insertion. Compare prediction/execution horizons and cross-plan continuity without tuning on released evaluation scenes or opening reserved final scenes. Maintain the older frozen failures and all original datasets.

The initial25mm controller comparison finishes **4/4near handoffs on two fixed configurations**, one SC and one SFP, each with both actor initializations. Raw checkpoint stand-off24.20–24.77mm; lateral0.22–0.55mm. **None of the four unchanged actors fully inserts.** These are small component/integration results, not broad transport reliability or successful insertion.

## 1. History survives control handoff

`RunHistoryChunkedInsertion` retains a rolling buffer of measured32Drobot states throughout the entire runtime, including transport, acquisition holds and the150→25mm approach. The model's current four history offsets remain1.2/0.6/0.2/0seconds. The buffer covers that window plus0.5seconds. It is reset on a new task, checked for clock reversal and keyed to the task identity.

The locked actor's normal reset still cancels pending actions. It then seeds its **observations**, not old actions, from the same-episode rolling buffer. Historical TCP poses are expressed in the currently accepted port frame when sampled. No future or post-reset state is backfilled.

This model consumes historical robot/relative-pose features and current three-camera RGB. It does **not** acquire historical image input merely because the robot-state buffer is preserved. Native image streams remain recorded for a later explicit visual-history experiment.

The actual reset-method preflight confirms: observations survive, old action generations are invalidated, and another task's history is refused. A separate two-scene live preflight checks seeded history, pose locking and execution before the trained models are used.

## 2. Prediction and execution comparisons

The50ms action period remains fixed. Trained horizons:

| Training model | Predict | Time covered | Continuity objective |
| --- | --- | --- | --- |
|H4|4steps|200ms|off|
|H8|8steps|400ms|off|
|H12|12steps|600ms|off|
|H8-continuity|8steps|400ms|on,weight0.1|

Live variants: **4/4,8/2,8/4,12/4,12/6,8-continuity/4**, plus the existing checkpoint under the repaired history/codec runtime. “8/4” means predicting400ms and executing the first200ms before replanning. Execution count is explicitly forwarded into Docker and audited before engine startup, then checked against the checkpoint horizon.

The four new fits share anchor observations, episode/reset splits, ImageNet initialization, optimizer settings and maximum update opportunity. All have at least10,000updates before plateau stopping;60stale100-update validations support a plateau.50,000updates or6hours per fit is a safety cap, **not proof of convergence**. Actual stop reasons/step counts are retained; these are converged-opportunity comparisons, not falsely equal paid-update counts. The continuity arm has an additional partner forward pass, which is reported rather than described as identical compute.

Training GPUs3/4; sequential live GPU5; legacyGPU0 untouched. The project uses at most four distinct GPUs. Rootless Docker, cooperative GPU/domain locks and private Gazebo networks remain mandatory.

## 3. Recorded actions, not synthetic longer trajectories

`build_recorded_horizon_actor_rows.py` constructs a new derivative from the verified67,039rows. It indexes **recorded future command targets** by their absolute timestamp within the same episode. Targets are transformed from each source row's estimated port frame into base coordinates, then into the anchor row's estimated port frame. Future measured TCP poses are never action labels.

Every horizon uses the same **60,671anchors /499source episodes**. These include420training episodes(102SC/318SFP) and79validation episodes(30SC/49SFP).6,368anchors lack a complete recorded600ms horizon and are excluded from all arms. There are no contradictory overlapping command slots at the declared0.05mm/0.05degree check. The original first four targets independently roundtrip to within2e-6in their stored components. Original image cache entries are reused through immutable image identities; original recordings remain unchanged.

Action horizon is checkpoint metadata and changes the number of transformer query tokens. Four stored targets are not stretched into eight. None of this adds privileged simulator information to actor inputs.

## 4. Cross-plan continuity

Paired training observations come from the same episode, with known50ms-multiple separation and overlapping future command timestamps. Pair labels must independently agree after frame conversion. Each observation's own history is causal; the later observation is a second training example, not future information supplied to the earlier actor.

The consistency loss compares decoded TCP targets for the **same absolute future times**, mapping the second estimated port frame into the first. It includes translation and rotation; invalid or unmatched pairs contribute no loss. This is a soft regularizer alongside recorded-action supervision, not a command to keep moving straight through a new contact.

Tests confirm zero loss for identical time-aligned motion and positive finite gradients after a perturbation. Live reports retain within-chunk versus replan-boundary target changes, actual command indices and timing. A smoother wrong action is not a success: physical insertion, force and identity gates remain.

## 5. What the refiner currently updates

### During rule transport

Requested-port coarse/refiner inference runs approximately every0.3simulation seconds, using all three current cameras and the locator's causal context. New accepted estimates can refine the final acquisition goal. It predicts the requested port, not every port's3Dpose.

### During fresh acquisition

The pipeline gathers repeated port and grasp estimates while holding. The earlier version gates consecutive poses at1mm/1degree, requires three consistent grasp translations and averages two accepted port poses. The new branch additionally requires a **3–5-observation pose window** whose full pairwise position span is<=1mm and angular span<=1degree, with samples at most1second old. Median translation and anSO(3)rotation mean form the window estimate. A burst/outlier/drifting sequence cannot silently approve an old pose.

### During approach and insertion

The accepted **base-to-port pose is locked**. The relative TCP/plug position changes because measured robot pose changes; the refiner is not replacing the reference frame at each actor step. The visual150→25mm controller also uses this locked rigid-fixture estimate. `locked_local_perception` in the trace is misleadingly named: it computes current relative geometry from proprioception and the fixed visual frame, not a fresh refiner call.

After a large retreat, reacquisition may deliberately create a new frame. Pending chunks must be discarded, and measured observation history must be re-expressed in that frame. The current near-range actor comparison keeps recovery off to isolate control; integrating a valid150mm retreat/reacquisition root into this25mm runtime remains a separate recovery task. Refiner flicker cannot explain the earlier intra-attempt jerks because its accepted frame was already fixed.

## 6. PoseInsert algorithm review

The [PoseInsert paper](https://arxiv.org/html/2505.09424v2) uses source-object pose relative to the target, predicts a future object-relative trajectory with diffusion, and augments pose features with goal-conditioned RGBD and gated feature fusion. The [released encoder](https://github.com/sunhan1997/PoseInsert/blob/24641dd3c6cc96fccd1e932a61a150bbb7a115c0/policy/cnn.py) is a useful implementation reference. Its feature residual is not a requirement to predict a residual action.

Our current model is **PoseInsert-inspired port-frame transformer BC**, not a complete RPDP reproduction: deterministic decoder, RGB rather than goal-conditioned RGBD, and no matching pose-primary gated fusion. It uses recorded controller targets rather than assuming future measured object poses are interchangeable actions for a compliant controller.

### Correct transform chain for our declared representation

LetBbe robot base,Pthe accepted opening frame,Ethe measured TCP andcthe estimated tip offset inTCPcoordinates. Current inputs obey

\[
{}^P T_E(t)=({}^B T_P)^{-1}{}^B T_E(t),\qquad
{}^P p_C(t)={}^P p_E(t)+{}^P R_E(t)c.
\]

The model predicts a complete target tip position and TCP orientation inP. Decode its TCP target with

\[
{}^P p_E^*={}^P p_C^*-{}^P R_E^*c,\qquad
{}^B T_E^*={}^B T_P{}^P T_E^*.
\]

The executable TCP-body delta is

\[
{}^{E_t}T_{E^*}=({}^B T_E(t))^{-1}{}^B T_E^*.
\]

Translation and rotation-vector components are bounded and converted to the API's base-frame target. The supervised targets are the recorded expert commands expressed in this same estimated frame. There is no learned residual added to another action.

The saved-trace audit finds no gross numerical error in42,080target conversions. That does not prove every sensor join, action label or model prediction correct.

### Important remaining representation gap

We condition on **tip translation + TCPorientation**, not the connector's fully observed orientation. Actual connector orientation is

\[
{}^P R_C={}^P R_E{}^E R_C.
\]

The grasp rotation\({}^E R_C\)is not approved as an actor input. Both existing grasp helpers produce diagnostic rotations, but their orientation gates have not passed. Randomized grasp angle can therefore matter despite accurate port pose. Do not substitute simulator rotation or enable an unvalidated estimate. Audit the existing diagnostic predictions on held-out observations before a separate full-object-pose experiment. This limitation can affect fine alignment, but does not alone explain20–50degree drift or failure to approach from150mm.

## 7. Additional preprocessing bug found

The new sample-based fresh acquisition JPEG86-normalizes RGB for native port refinement, then returned those images to a grasp helper which applied JPEG86again. Training passed its already-decoded JPEG images with `already_training_jpeg=True`. The history/chunk branch now returns raw RGB to the grasp helper, so each consumer applies its training codec **once**. Port refinement's codec stays unchanged. This is a real preprocessing discrepancy, but no evidence yet makes it the main explanation for large command jumps. It is fixed uniformly across new horizon variants.

## Commands and artifacts

```bash
.pixi/envs/default/bin/python scripts/build_recorded_horizon_actor_rows.py \
  --source /home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/shared_full_causal_grasp_actor_rows_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-17_recorded_horizon_actor_rows_v1

.pixi/envs/default/bin/python -u scripts/continue_history_horizon_experiments.py \
  --output /home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1
```

The tiny factory hooks were installed only after the active four-rollout pilot and video generation finished. Old source snapshots are archived; legacy four-query construction remains the default. History reset, full legacy-checkpoint construction/warmup, horizon shapes, real cached-data pairing and pose-window outlier/drift tests pass.

Artifact roots:

- `/home/nas/chmin/aic_archives/2026-10-17_recorded_horizon_actor_rows_v1/`: matchedh4/h8/h12derivatives and provenance.
- `/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1/`: commands, source snapshots, preflights, training logs/checkpoints/convergence summaries, queued frozen14-rollout comparison.
- `/home/nas/chmin/aic_archives/2026-10-17_history_chunk_integration_v1/`: two-scene existing-checkpoint runtime preflight; not a new training result.

No newRL, dynamics, imagination, final-scene access or automatic promotion. The first unused three-card SC preflight failed in transport before actor execution. The controlled actor comparison is therefore being switched, before its freeze, to the two previously successful25mm development configurations. These are reused development starts; improvement there is not generalization evidence. A separate helper waits for the active preflight to finish before changing its runner, and queues a known-start history test. If no variant inserts, preserve failures and diagnose; do not automatically collect more data or launch another broad retrain. A training time/update cap is reported as incomplete convergence evidence.

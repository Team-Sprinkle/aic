# Autonomous transport/handoff execution: outcome and limits

Status: bounded execution finished with an unresolved full-insertion gate. All new jobs are finished; no final scenes were opened.

## What works

On supported normal-board SC settings, the rule-based Low Lateral Bypass transports the cable around the cards and holds for camera acquisition. It uses public task identity, measured robot state and images. A requested-port check and original-image refinement approve the handoff. Initial matched screens passed 6/6 reused and 6/6 fresh development starts. A later stricter orientation screen passed 5/6. These are small feasibility screens, not full-distribution reliability.

The guarded image-based port orientation repair removed a clipped-search error. Checked port/grasp position estimates are accurate to fractions of a millimeter. The grasp helper supplies position only: its full orientation prediction did not pass its gate. No true simulator pose authorizes autonomous crops, handoffs or actor actions.

## What does not work

The learned local approach actor has not demonstrated autonomous insertion from this handoff. Complete plug-tip action targets reduced held-out near-port decoded command error from 1.010 to 0.629 mm p95, but the live pilot stopped about 130 mm before the opening.

A bounded teacher-continuation screen provided useful corrective labels from actual actor states. The selected small derivative has 1,184 rows from two training episodes and 581 rows from one held-out episode, **added to** the original 83 SC and 234 SFP training episodes. The three selected successes come from two controller versions; they are not three successes from one identical teacher.

The corrective-data actor reduced held-out near-port error from 0.900 to 0.420 mm p95. Nevertheless, the matched autonomous control and treatment both stopped about 141 mm before the opening: 0/1 full insertion each. Better offline imitation did not solve autonomous progress.

A further matched transformer/direct-MLP decoder test used full targets, FP32 output heads and command-motion supervision. Both completed 6,000 updates and failed the frozen command/pose/motion gate. No live rollout was started for these ineligible models.

## Why the last bounded test uses history

Teacher alignment pauses last roughly two to five seconds. The latest single observation may not tell an actor whether the tool is still settling or ready to advance. The last matched history experiment compared identical modules receiving either repeated current state or measured robot-state/force history at 6, 2, 0.5 and 0 seconds ago. Both used the latest three camera images. This isolates robot history; it is not a test of image history.

Training and deployment share the causal sampler. Deployment records observations throughout chunk execution and clears history on reset. A simple held-out linear diagnostic did not show a history benefit, so the nonlinear comparison was run. Both completed 6,000 updates, selected update 100 and failed the frozen gate: stationary false advancement remained 100%. No live history pilot followed.

## Physical success and blocked work

SC promotion requires at least 14 mm actual insertion depth, at most 0.5 mm lateral error, explicit orientation and acceptable contact/force. A reported official score alone is insufficient: shallow plug/port contact can trigger completion and freeze the gripper.

A privileged feedback diagnostic reached 15.571 mm depth and 0.072 mm lateral error. This shows the controller can insert, but it uses true geometry and is **not** an autonomous policy result. The difference between a desired geometric position and the command needed by the compliant controller matters.

The full-insertion gate, broad supported-setting expansion, recovery/SERL and Gazebo transfer remain dependent on ordinary supervised insertion. No new RL, world-model/SEER work or reserved-final-scene evaluation has been performed. SFP local insertion, rotated-board SC and port-0 rail-minimum remain unapproved.

## Where to inspect evidence

- [Execution record](../experiments/2026-10-14-transport-handoff-execution.md)
- [Machine-readable state](../experiments/2026-10-14-transport-handoff-execution.json)
- [Current high-level plan](../../obsolete/docs/2026-10-19-retired-plans/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md)
- [Status](../STATUS.md)

Bulk artifacts, exact commands, frozen gates, selected/last checkpoints, scored streams, failed trials and videos are under `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/`. Originals remain preserved.


## The useful final change: hand the actor a nearer start

A new, separately frozen diagnostic lets an explicit observation-only controller cover the missing approach. It uses the approved camera-estimated port frame, camera-estimated held-plug position and measured TCP; it does not read simulator target geometry or completion events. It stops about30mm before the opening and then runs the **unchanged trained insertion actor**. This is a hybrid pipeline change, not an actor-training improvement.

On one reused normal-board, one-card SC port-0 start:

| Stage | Actual plug position | Lateral error | Orientation error |
| --- | --- | --- | --- |
| Initial camera acquisition |155.58mm before opening |2.515mm |0.549° |
| New near-approach handover |29.12mm before opening |0.148mm |0.548° |
| Actor terminal |11.07mm inside opening |0.413mm |0.938° |

The official score was75, with no scored contact or excessive-force penalty. **It still fails the unchanged14mm depth requirement.** This is substantial progress from the prior far-away stall, but not a promoted full insertion.

The engine logs confirm what stops further movement: the plug touches the port, contact remains continuous for1second, and the toolkit marks completion and makes the end effector static. The first touch was at46.546simulation seconds; the final continuous-contact interval began48.228s. After completion, deeper commands cannot move the frozen robot. The actor also worsens alignment after handover, reaching0.870mm lateral/0.853° at43s. Orientation and lateral errors are plausible contributors to early shallow contact; we have not isolated which is dominant. This run is **not an observed cable snag**.

The five camera intervals were inspected, including side and overhead views. Local actor camera/state/model/result-transfer inference p50/p95/p99 was72.1/92.0/104.1ms. This excludes prior refinement/acquisition and is not a complete pipeline latency certificate.

[20fps H.264 video](/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/observation_approach_llb_1400_v1/five_cameras_20fps_grid_active.mp4)

[Inspected five-view contact frames](/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/observation_approach_critical_review_v2.jpg)

## What remains blocked and the next concrete experiment

Do not broaden collection or start RL on this partial result. The ordinary supervised full-insertion gate remains open after several distinct repairs. Supported normal-board SC transport/acquisition and the new near approach have useful evidence; SFP local insertion, rotated SC and rail-minimum acquisition remain unapproved. Decoder/history fits were capped experiments, not claims of globally optimal or fully converged training.

The targeted next plan is at most **two privileged diagnostic rollouts, approximately20minutes plus recording/scoring**, with the same frozen actor/reset. Replace only orientation in one run while preserving predicted plug-point targets. Replace only lateral compensation in the other while preserving actor orientation/axial targets. This isolates the remaining geometry/control issue. These are diagnostic upper bounds, not deployable policies, and cannot themselves promote a checkpoint.

A successful intervention should identify the specific observation-only representation/target/controller repair to compare next. If neither works, investigate coupled alignment/compliant command semantics before more training. Keep the14mm/0.5mm physical gate; a repaired autonomous pilot must pass before a frozen six-fresh screen. Recovery/SERL, broader transfer and final evaluation stay gated. This is an explicit evidence-backed blocked conclusion permitted by the execution prompt, not a claim that the full pipeline succeeded.

## Short response shown in chat

Finished the bounded comparisons and documented the unresolved gate.

- Decoder and longer robot-history changes failed their frozen gates.
- A new observation-only approach brought the plug to **29 mm before the opening**. The unchanged trained actor then inserted **11.07 mm**, versus previously stalling far away.
- This still fails the **14 mm** requirement. Logs confirm contact-triggered completion locks the robot; the contribution of lateral versus orientation error remains unresolved.
- All new jobs finished. RL remains gated and final scenes sealed.

The next documented experiment isolates orientation and lateral compensation in two bounded diagnostic rollouts; no routine retraining.

[Full explanation and results](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-14-autonomous-execution-outcome.md) · [20 fps video](/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/observation_approach_llb_1400_v1/five_cameras_20fps_grid_active.mp4)

# Experiments and decisions

The completed demonstration uses **rule transport, task-conditioned visual acquisition, and an explicit insertion controller**. Learned perception supplies geometry; learned BC/RL actions do not drive the final videos. This page records the useful findings and failed alternatives concisely. [Implementation, artifacts and qualifications](answers/2026-10-19-final-project-completion-notes.md).

## Recorded results

| Test | Result | Interpretation |
| --- | --- | --- |
| Published qualification config, selected repeat 10 | **276.3383/300; all 3 insert** | Final [videos and GIFs](media/final_evaluation/README.md); selected successful run |
| Same frozen controller, 19 additional scored replays | **15/19 full passes; 53/57 trial insertions** | Best 276.34, median 275.71, mean 256.28 including failures |
| Public sample, current recording | **287.8994/300; all 3 insert** | 1/1/0 cards versus released config's 1/2/3 cards |
| Public sample, earlier V56 selection | **290.19/300; all 3 strict physical gates pass** | Earlier controller configuration; not the selected released replay |

The public configs are exposed development/evaluation material, not unseen tests. The internal reserved final split stays sealed. See [evaluation protocol](EVALUATION_PROTOCOL.md). All run failures and the stopped, unscored twentieth repeat are preserved.

## 1. Audit demonstrations before learning

Recorded success flags alone were insufficient: some trajectories never inserted, and some successful VLM/agent recordings had invalid zero action labels. The verified expert folder now excludes those recordings. Corrected timestamp joins, preserved recorded targets and a copied SC dataset with long scripted pauses removed made later comparisons interpretable. Data cleanup fixed defects; it did not by itself produce a reliable learned transport policy. [Dataset contracts](DATASETS.md).

## 2. ACT and world-model policies

ACT and a Dreamer-style supervised policy were trained on verified demonstrations. In the frozen 20-scene comparison, the world-model policy reached a 25 mm opening envelope in **17/20**, versus **12/20** for ACT, but neither fully inserted. Dynamics failed to beat persistence at 200/400/600 ms; good-looking reconstruction was insufficient evidence for planning. This branch was parked. [Full-data record](experiments/2026-09-19-full-world-training.md).

## 3. Learned transport versus a cable-aware route

Transformer and diffusion transport experiments tested history, action horizons, task-specific fits, overfitting and cleaner route-consistent data. Denser history and intentional fitting reduced visible final-return drift, but no tested learned actor established reliable handoff across the required starts. Those experiments do not prove diffusion is unsuitable. Rule routes became the working transport baseline: SFP lift/transfer; SC descend outside the row, travel beside it, tilt in clearance, then return toward the port. [History/data audit](experiments/2026-10-13-sc-diffusion-code-vision-cadence-audit.md) · [Rule transition](experiments/2026-10-14-transport-handoff-execution.md).

## 4. Geometric vision and precise orientation

A learned position estimate could be good while fixture orientation remained biased. Canny/Hough board-rail lines supplied an independent yaw cue; fixing a clipped search improved one difficult control from **1.37° to 0.017°**, with two reused controls passing. Search-boundary and cross-view checks mattered: agreement at the same clipped boundary was false confidence. The final recorded port estimator instead fits **five native-image landmarks** across calibrated cameras, with a known fixture-normal prior. Dense CAD-edge fitting of the held SC plug was slower and inconsistent and was rejected; it must not be confused with the useful board-edge experiment. [Evidence](experiments/2026-10-14-transport-handoff-execution.md#clipped-board-yaw-agreement-defect-and-bounded-repair).

## 5. Requested-port history and handoff

Predicting every possible port and relying on a late verifier made target identity difficult. The selected coarse model answers one task query, uses all three current views and distance/rotation-spaced history, and proposes current native-image crops. The refiner predicts five landmarks of that requested port; multi-view reprojection, temporal agreement, pose envelope and measured stationary standoff gate acceptance. A projection probability is not calibrated pose confidence. Wider data and corrected container settings repaired specific gaps; held-out component accuracy alone did not certify full insertion. [Execution and data coverage](experiments/2026-10-15-rule-perception-hierarchical-execution.md).

## 6. Connector pose and explicit insertion

Using gripper orientation as connector orientation was inconsistent with varying grasps. Image-trained connector-in-TCP estimates, a fixed port frame per attempt, preserved command references and a gated five-observation update near **25 mm** improved alignment and continuity. Longer BC chunks and cross-plan penalties did not produce reliable full insertion in their controlled tests. The working final controller explicitly aligns and advances in the visual port frame, with antiwindup and a seating dwell compatible with the toolkit's contact requirement. It is not the earlier PoseInsert-inspired BC actor. [Orientation and actor audit](experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md) · [Explicit controller](experiments/2026-10-18-sc-explicit-controller-continuation.md).

## 7. Recovery and scoring gotchas

Force must be compared with a stable initial reference, and retreat must follow measured motion rather than simply reverse command references under impedance control. Short local backoff and orthogonal retry can unload contact. An earlier near-port Isaac recording screen improved **1/8 → 3/8 local seats** with recovery enabled; this was a different policy, simulator and start distribution. A [Gazebo recovery example](media/recovery_demo/README.md) records an 18.4 mm retreat and 3.2 mm sideways retry followed by insertion; it is qualitative evidence, not a matched success-rate ablation. Broad superiority of hierarchical over local recovery is not established. Completion locking could freeze a shallow contact, so physical geometry and pre-reset observations were audited separately from the toolkit award. [Historical recovery screen](experiments/2026-09-22-serl-mixture-recovery.md) · [Transport recovery](experiments/2026-10-19-transport-local-recovery.md).

## 8. Repairing the exposed released configuration

A sample-derived prior rejected a useful SFP estimate **254 mm** from nominal. A guarded visual waypoint plus matched transport/fresh-lock envelope repairs restored SFP trial 2. SC's board differed by roughly **150 mm and 85°** from the nominal route frame; two consistent RGB marker estimates let the route follow the observed row. Scores rose from about **98.7/300 (1/3)** to **191.66 (2/3)** after the SFP repair, then **276.34 (3/3)** with the SC route repair. These were iterative combined changes, not a complete factorial ablation. [Frozen record](experiments/2026-10-19-released-sfp-sc-visual-route-repair.md).

The later 19-run batch exposed four misses: initial force-reference timeout (SFP), native crop/acquisition timeout (SFP), blockage before SC marker acquisition, and missing SC marker acquisition. The initial four clean repeats did not establish reliability. These gaps and the remaining SERL/hierarchical-recovery direction are consolidated in [FUTURE_WORK.md](FUTURE_WORK.md).

## Detailed record

[Completion explanation](answers/2026-10-19-final-project-completion-notes.md) contains the artifact map, runtime boundaries, metrics and media provenance. Dated files under `experiments/` and `answers/` are historical evidence, not active job queues. The [old long experiment index](answers/archive/2026-10-19-documentation/experiments_full_history.md) is preserved; superseded standalone plans are in [obsolete](../obsolete/docs/2026-10-19-retired-plans/README.md).

# Temporal transport and three-model supervised continuation — 2026-10-01

## System and gates

The supervised system has **three models and two actors**:

1. A shared SFP-to-NIC and SC-to-SC **transport actor** reads three RGB cameras, measured 32D robot state, recent causal observations/actions/motion, and the 10D task identity. It moves from the normal start to a stable near-port handoff even while the target is offscreen. It predicts complete TCP-body commands, route progress, and a next complete TCP subgoal; the plan supplies offline labels only.
2. A separate task-conditioned **perception model** estimates the selected port pose only when identifiable, per-camera visibility, and calibrated fused-pose confidence. Simulator geometry supplies labels only. Other visible ports are negatives, and offscreen/occluded targets have masked pose loss.
3. A separate **insertion actor** consumes that observation-only estimated pose, task identity, force, and measured state; it predicts complete port-relative plug/TCP targets. Its action head is distinct from transport's body-frame head.

After independent validation, switch actors only after a stable visibility/pose-confidence window and measured arrival, with hysteresis, action continuity, and safe fallback. Hierarchical backtracking is a later matched addition (none, local, hierarchical), not part of this supervised baseline. RL remains closed until autonomous official full-depth Gazebo insertion passes. Four reserved final configurations remain sealed.

## Frozen prior and current hypothesis

The previous single-observation BC actor uses three RGB images, 32D state, and 10D task code, but no past images/actions. Four variants on 83 SC and 236 SFP training episodes reached **0/16 stable 10 mm handoffs** on two non-final SC layouts at 20/60 mm caps. The best final distance was 75.4 mm. Route choice and drift are observed; their exact cause is unproven. This bounded continuation tests whether causal route history and explicit stage/subgoal supervision resolve that failure. Diffusion was not fairly ruled out for transport. Earlier local 8 mm Isaac SFP seating is neither normal-start transport nor official full-depth Gazebo insertion.

## Gate 1: source and label audit

Command (run completed):

```bash
.pixi/envs/default/bin/python scripts/audit_temporal_transport_labels.py \
  --spec outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json \
  --sc-rows outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/transport_chunks.jsonl \
  --sfp-rows /var/tmp/chmin_aic_20260918_act/sfp_verified_268_port_tcp_waypoints_20260924.jsonl \
  --output outputs/experiments/2026-10-01_temporal_transport/label_audit.json
```

All **104 SC** complete episodes pass the existing **83/21** scene/reset grouped split, with both target ports and card counts 1–5 in each partition. All six recorded Low Lateral Bypass stages occur in order; every included plan has the two named **8 s and 5 s endpoint holds**. No old over-card route was admitted. The first recorded command is within **2.51 mm p95** of the time-aligned MoveIt plan, and the retained body-frame target reconstructs the base-frame target to numerical precision. Three camera timestamps lag the decision by at most **48 ms** (42 ms p95), so temporal inference must use recorded timestamps. The two inspected transition image sequences (`llb_0001`, `llb_0074`) look consistent with route progression, but the center view alone does not prove cable clearance.

The holds are explicitly inserted by `scripts/run_moveit_transport_plan.sh` through `hold_moveit_route_segments.py`, which repeats the endpoint joint target after `sweep_after_row` for 8 s and after `tilt_clear_of_cards` for 5 s. They are scheduled waits, not force-triggered recovery events. The code records no stronger physical reason for their lengths; allowing settling is plausible but unverified. Cleaning preserved boundaries, measured settling/contact rows, and sparse quiet samples rather than deleting the full durations from the temporal clock.

The **268 SFP** episodes split **236/32** by scene/task. Task vectors decode consistently, but coverage is highly skewed: 227/236 SFP training episodes target card 0, port 1; validation has 31/32 of the same target. SFP teacher rows lack MoveIt route-stage labels, so the shared actor can receive SFP full-action BC and generic stage supervision, while SC supplies the six-stage/subgoal labels. This limits SFP target generalization evidence. No new SFP data is being collected as part of this bounded first transport test.

At teacher SC stage changes, measured TCP-to-just-finished waypoint distance is **9.6 mm median and 22.7 mm p95**; the traverse stage is about 22 mm at transition. Therefore a deployment gate that demands exact attainment of every nominal waypoint would deadlock even a teacher-like path. Later stage advancement must use empirically audited measured-progress tolerances, route classifier stability, and force checks; elapsed time alone cannot advance stages. At transport end, only 43/104 measured TCP poses lie within 10 mm of the nominal final waypoint, so the live stable-10-mm requirement is stricter than merely reproducing teacher handoff timing. The local-start event is a handoff *training label*, not proof of a safe insertion switch.

Artifacts: [machine audit](../../outputs/experiments/2026-10-01_temporal_transport/label_audit.json), [transition montage](../../outputs/experiments/2026-10-01_temporal_transport/transition_montage.jpg). The stage labels, nominal subgoals, and selected-port geometry must never be passed to an autonomous actor.

## Gate 2: bounded temporal transport test

The initial candidate freezes the previous three-camera visual backbone/context and caches causal features at four-command (200 ms nominal) decisions. A two-layer 256D transformer consumes recent history with recorded time gaps, prior teacher command proxies, and measured TCP motion. Dense recent samples and sparse older keyframes reach back up to 16 s, covering the long holds. Its heads predict full four-command TCP-body targets, SC route stage, next complete TCP subgoal expressed in the *current* TCP frame, and a local-start/handoff signal. No plan/true port pose enters model inputs. The same architecture without history is the controlled ablation. This is a diagnostic frozen-vision candidate; it does not yet establish that an updated visual encoder would help.

The temporal head has **1,248,693 parameters** (1,252,047 with direct state/task skip). The required frozen ResNet18 backbone, view attention, and context are about **12.44 million** additional parameters, for roughly **13.69 million deployable parameters** before runtime buffers. This is a model-size estimate from instantiated modules; complete live latency was not measured because the offline gate failed.

Use one GPU. Limit each arm to at most 3,000 updates or 60 minutes (total at most 6,000 updates or 2 hours) and stop after five 250-update checks without held-out improvement after 1,000 updates. Sample by episode and SC route stage; retain all held-out episode/reset groups. Require SC held-out stage accuracy at least 90%, next-subgoal translation p95 at most 20 mm, no skipped stage in trace replay, and inspect action error by family/port/card/stage and hold/motion. Offline success is diagnostic only.

If the offline gate passes, freeze the candidate and implement a cached runtime with true causal timestamp/reset masks, full four-command 200 ms execution, per-tick force/stall checks, and measured-progress stage gating. Benchmark complete inference p50/p95/p99 under the 300 ms p95 limit. Evaluate at most the two already opened non-final SC starts for 70 simulated seconds each. Require the correct around-card route, acceptable force/contact, and stable 10 mm handoff, reporting lateral/axial/3D/orientation errors and official insertion score separately. A failure gets one bounded trace/video diagnosis and targeted repair; no broad sweep or routine new teacher collection.

Only after transport passes should the separate perception and insertion models train. The perception gate requires real identifiable visibility labels, wrong-port/occluded negatives, episode-grouped validation, and near-port error/calibration sufficient for the 0.5 mm insertion corridor. The insertion gate requires observation-only predicted pose, both successful SFP and SC local chunks, official full-depth Gazebo scoring, and full latency. No final scenes, RL, or hierarchical backtracking enter this baseline.

## Running record

The rootless GPU-1 cache completed **55,142** four-command decision rows: 33,164 SFP and 12,981 SC training rows; 4,520 SFP and 4,477 SC validation rows. It stores frozen 256D features, full command labels, causal measured state/task/action history, and offline SC stage/subgoal/handoff labels under `outputs/experiments/2026-10-01_temporal_transport/frozen_features/`. The supplementary `camera_offsets.npy` stores three per-camera capture ages and a known-timing bit; all 17,458 selected SC rows have known camera timestamps, with maximum age 48 ms. SFP cache camera offsets are not available and are masked as unknown. The first cache attempt omitted the `/home/nas:ro` mount and failed on a symlinked SC image; the rerun mounted it read-only and completed. No source episode changed.

The frozen single-observation BC head evaluated on these same 200 ms decision rows had held-out first-action p95 **4.33 mm SC** and **7.21 mm SFP**. This is teacher-frame replay, not autonomous transport.

### Completed matched history ablation

Both heads used the same frozen visual cache, architecture, episode split, seed, episode/stage sampling, loss, and 3,000-update/60-minute ceiling. The history arm retained causal dense recent tokens and sparse older keyframes; the control used only the current token. Each checked grouped validation every 250 updates, with a five-check plateau stop after 1,000 updates. Full four-command TCP-body actions, SC stage, current-TCP-frame complete subgoal, and handoff were supervised. No true stage or plan geometry entered inputs.

| Arm | Best / completed updates | SC stage accuracy | SC subgoal translation p95 | SC first action p95 | SFP first action p95 | Stage skips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Causal history | 2,750 / 3,000 | 92.96% | 43.42 mm | 2.84 mm | 3.21 mm | 0 |
| Same model, current token only | 1,500 / 2,750; plateau | 90.97% | 55.93 mm | 2.91 mm | 4.03 mm | 0 |

History improved the held-out subgoal estimate by **12.51 mm p95**, while both arms failed the 20 mm gate. It also improved SFP teacher-frame action error, but the SFP target distribution is skewed and no live behavior is inferred. These offline temporal windows use previous *teacher* targets as action history; autonomous proposals could differ, so rollout shift remains untested. SC subgoal errors concentrate in intermediate routing stages, while the final return-to-port stage groups were roughly 8–17 mm p95. The stage head confused the end of the tilt hold with the return stage in some frames. The stage transition audit found measured TCP lags of up to ~25 mm at stage boundaries, so a runtime progress gate would need tolerance and stability checks, not exact nominal-waypoint attainment. [History summary](../../outputs/experiments/2026-10-01_temporal_transport/history/summary.json), [no-history summary](../../outputs/experiments/2026-10-01_temporal_transport/no_history/summary.json), [frozen baseline](../../outputs/experiments/2026-10-01_temporal_transport/frozen_baseline_offline.json). Checkpoints, code snapshots, manifests, logs, and failure metrics are retained.

**Offline gate failed; no Gazebo actor rollout was launched.** A single targeted repair was tested: expose normalized measured TCP XYZ and the ordinary 10D task code directly to the temporal transformer, because the frozen 256D visual/context feature may discard precise global position useful for TCP-frame route subgoals. The rest of the architecture, cache, split, seed, sampling, loss, and 3,000-update ceiling were fixed. Its completed result is recorded below; it remained above 20 mm p95, so the frozen-vision route-subgoal branch stopped before live promotion. This is a declared repair, not routine extra data collection.

### Completed targeted repair and full held-out check

The raw measured-TCP/task skip finished at 3,000 updates (best 2,750). On the original stage-balanced validation sample, SC stage accuracy was **90.87%**, subgoal translation p95 **41.12 mm**, first-action p95 **2.82 mm**, with no stage skips. It improved the subgoal by only 2.30 mm against the history arm and still failed the 20 mm gate.

All three frozen checkpoints were then evaluated on **every** cached decision in the same held-out complete episodes (21 SC / 32 SFP; 4,477 SC and 4,520 SFP decisions). This is a confirmation set within the existing development split, not a reserved final split.

| Arm | SC stage accuracy | SC subgoal p95 | SC first action p95 | SFP first action p95 | Stage skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Causal history | 91.22% | 41.51 mm | 2.80 mm | 4.16 mm | 0 |
| Current token only | 90.62% | 46.38 mm | 2.85 mm | 4.32 mm | 0 |
| History plus direct measured TCP/task | 87.13% | 39.50 mm | 2.53 mm | 3.79 mm | 0 |

The direct-state repair reduced subgoal error another 2.01 mm but lost the 90% full-validation stage threshold. No checkpoint passes the joint offline gate. A frozen visual-context head alone appears insufficient for the variable intermediate SC route goals. This is an inference from the comparison, not proof that temporal control or a richer encoder cannot work. The prior single-observation actor's **0/16 live handoffs** remain the only live transport evidence; these new numbers are teacher-frame predictions. [Exact full held-out JSON](../../outputs/experiments/2026-10-01_temporal_transport/full_heldout_comparison.json), [stage plot](../../outputs/experiments/2026-10-01_temporal_transport/full_heldout_comparison.png), [machine decision](../../outputs/experiments/2026-10-01_temporal_transport/decision.json).

The auxiliary handoff head was checked separately on the 37 positive local-start decisions among 4,477 held-out SC decisions. At a fixed 0.5 probability threshold, history had **34 TP / 7 FP / 3 FN** (precision 82.9%, recall 91.9%); no history had **36 / 43 / 1** (45.6%, 97.3%); direct-state history had **33 / 3 / 4** (91.7%, 89.2%). History reduced false handoff triggers in teacher replay. This is an uncalibrated, teacher-observation diagnostic, not a safe autonomous switch; the subgoal gate failed. [Handoff check JSON](../../outputs/experiments/2026-10-01_temporal_transport/full_heldout_handoff_check.json).

The failure is concentrated at `tilt_clear_of_cards`: the history arm's stage-4 subgoal p95 is **73.6 mm** and only **74.4%** of those frames receive the correct stage; 299/1,189 are classified as the subsequent return stage. Its final `return_toward_port` subgoal p95 is **12.5 mm**. The no-history and direct-state arms have stage-4 p95 **63.8** and **88.8 mm**. A nondeployable diagnostic using the *true* stage/port and a constant mean base-frame subgoal still has **51.2 mm p95** over SC validation, showing substantial scene-dependent waypoint variation. It is a lower-information reference, not a policy candidate.

At the held-out tilt-hold → return boundary, 21 adjacent frame pairs have only **0.026 mm median measured TCP motion**, **0.160 mm median commanded-target change**, and **0.061/255 mean 64×64 center-image pixel difference**. The three-view [before/after example](../../outputs/experiments/2026-10-01_temporal_transport/tilt_return_boundary_llb_0074.jpg) was inspected: the frames are visually almost identical. The 5 s hold cannot be ended reliably from a single image. Causal route-progress memory must track the measured completion of the tilt and dwell/contact stability as well as elapsed time; elapsed time alone must not advance the stage. [All 21 boundary measurements](../../outputs/experiments/2026-10-01_temporal_transport/tilt_return_boundary.json).

**Gate disposition:** Keep step 5, the live transport handoff, task-conditioned perception promotion, insertion integration, RL, hierarchical backtracking, and final scenes closed. No new autonomous rollout was run because the predeclared offline subgoal gate failed; therefore no complete live p50/p95/p99 claim exists for this temporal architecture. The old actor's p95 under 183 ms cannot be assumed for it.

**Next falsifiable repair, proposed rather than started:** Add an explicit observation-only route-progress state that stores the last accepted stage, measured TCP progress toward a *predicted* subgoal, tilt completion, and hold dwell/contact stability. Condition a route-stage-specific full TCP subgoal head on that state; retain three-camera context and use richer per-view visual tokens if the early waypoints remain above tolerance. Use the same 83/21 SC and 236/32 SFP split and one GPU for at most 3,000 updates or 60 minutes, with the same no-history control. Require full held-out SC stage accuracy ≥90%, subgoal translation p95 ≤20 mm, and zero skips by port/card/stage/motion. Only then benchmark the complete runtime and run at most two 70 s rollouts on the opened SC development layouts. Both must follow the around-card route and stably hold within 10 mm with acceptable force and inference p95 <300 ms before new development starts. If this still fails, stop frozen visual reuse and test a scene-aware visual representation or observation-only planned transport as a separate architecture revision. No new data collection, final scenes, or RL are authorized by this gate.

The execution used the pinned `aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923` rootless image with `--gpus device=1`, no network, and mounts for this checkout, `/var/tmp/chmin_aic_20260918_act:ro`, and `/home/nas:ro` for feature extraction. The Python arguments, image digest, GPU, mounts, code hashes, and logs are in the [command manifest](../../outputs/experiments/2026-10-01_temporal_transport/commands.json). Stdout logs and per-run manifests/checkpoints live under `outputs/experiments/2026-10-01_temporal_transport/`. The original history/no-history trainer source was copied into each run directory before the direct-state repair. The direct-state source was reconstructed after a later evaluation-report-only change; its exact run-time source hash was not captured. Model checkpoints and their hashes remain intact. No job was interrupted.

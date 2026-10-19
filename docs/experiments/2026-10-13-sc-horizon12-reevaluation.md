# SC-to-SC 12/6 autonomous reevaluation and route failure diagnosis

Date: 2026-10-13. The checkpoint is the frozen [12 predicted / 6 executed transport actor](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon12_execute6_best.pt). Each target slot is 50 ms. This is **autonomous transport only**: no teacher, geometry, guide, crop verifier, local insertion actor, or RL supplies runtime actions. Geometry enters only posthoc scoring and the analysis below. The reserved final scenes remained sealed.

## Design and scenes

The first three runs repeat the same nonfinal SC validation configurations used in the [initial matched horizon comparison](2026-10-13-route-action-horizon.md): `llb_0005` (one card, port 1), `llb_0074` (five cards, port 0), and `llb_0183` (four cards, port 1). To avoid relying only on repeated layouts, a frozen second plan uses previously unrun development configurations: `llb_0090` (one card, port 0), `llb_0021` (two cards, port 1), and `llb_0009` (five cards, port 1). Their recorded teachers all scored full insertion with no scored contact. The new layouts are development scenes, not the reserved final split; checkpoint selection had access to this broader development split, so this is not an independent final generalization test.

All six reruns used the same model hash, 12 denoising steps, one physical GPU 3, the same three observation cameras and task code, and the corrected post-inference six-command simulation-clock schedule. Rootless Docker ran sequentially, with no interruption to the standing Isaac container. The plan files, exact scene hashes, scored bags, controller and policy traces, saved RGB, launch logs, and terminal observations are under:

- `/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_repeat_v1/`
- `/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_v1/`

The two posthoc teacher/actor trace analyses are `/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_repeat_analysis.json` and `/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_analysis.json`. [Machine summary](../../outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/summary.json) joins these with the original three runs. [The axial-progress plot](../../outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/axial_progress.png) shows teacher and actor paths on all six configurations; green is the predefined TCP axial approach band of −100 to −70 mm.

## Measured result

**Repeated three scenes: 0/3 buffer visits, 0/3 insertions, 0/3 scored contacts. New three development scenes: the same.** Including the original run, the 12/6 checkpoint has 0/9 buffers and insertions across nine rollouts of six distinct SC configurations. The repeated scenes are not independent samples. Every one of the nine actors had **zero samples in the axial approach band**, so a lateral-in-band percentile is undefined.

| Layout | Cohort | First sustained ≥20 mm departure from teacher | Best TCP 3D distance to opening | TCP axial / lateral at best point | Scored contact |
| --- | --- | ---: | ---: | ---: | --- |
| `llb_0005`, 1 card P1 | repeat | 6.70 s | 181 mm | −178 / 32 mm | No |
| `llb_0074`, 5 cards P0 | repeat | 7.55 s | 212 mm | −189 / 95 mm | No |
| `llb_0183`, 4 cards P1 | repeat | 6.40 s | 169 mm | −167 / 29 mm | No |
| `llb_0090`, 1 card P0 | new dev | 8.90 s | 199 mm | −194 / 41 mm | No |
| `llb_0021`, 2 cards P1 | new dev | 8.50 s | 160 mm | −159 / 23 mm | No |
| `llb_0009`, 5 cards P1 | new dev | 6.20 s | 266 mm | −236 / 124 mm | No |

The quoted distance is **TCP-to-port**, not plug-tip distance. Its purpose here is to compare each actor to the successful teacher's same measured transport standoff. It is not a final insertion error. At about 40 s the successful teachers were typically around −145 mm axially and then advanced to approximately −5 to −33 mm by the end; the repeated actors remained around −195 to −240 mm or moved back there. In the newly added scenes the actor similarly missed the final approach, including the one-card case with little opportunity for a five-card cable snag.

Saved five-camera 10 fps H.264 videos: [repeat one-card P1](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_repeat_v1/rollouts/llb_0005/five_cameras_10fps.mp4), [repeat five-card P0](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_repeat_v1/rollouts/llb_0074/five_cameras_10fps.mp4), [successful teacher on that five-card scene](</data1/chmin/yj/ws_aic/src/aic/outputs/experiments/2026-09-30_sc_llb_collection/continuation170/runs/llb_0074/five_cameras_10fps.mp4>), [new one-card P0](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_v1/rollouts/llb_0090/five_cameras_10fps.mp4), and [new five-card P1](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_v1/rollouts/llb_0009/five_cameras_10fps.mp4). I inspected sampled frames from the one- and five-card actor recordings. They show broad sideways route motion and a cable, but no visually obvious sustained snag or gripper/card block. The official scored contact field is `No contact detected.` in every run. Video inspection cannot prove absence of unscored light contact; the path and contact score together show that these failures arise before a blocked insertion attempt.

## What failed, and what the controlled checks say

**Wrong route transition is directly observed.** Once the teacher completes its side route it advances toward the port. The actor often comes partly beside the row, but its policy keeps proposing targets axially farther from the requested opening. In the six new rollouts, **2,176/2,177 actual issued commands after 40 s** placed their Cartesian target axially away from the opening relative to the command's measured TCP. (The original three runs gave 1,035/1,074.) Measured TCP can lag or occasionally travel differently from one target because this is a compliant controller, but the reference-direction count makes the model's late route decision clear. This does not mean every early backward move is wrong: the expert Low Lateral Bypass route intentionally backs away first. The failure is that the actor does not reliably *leave that route phase* and begin the final approach when the teacher does.

**Execution timing differs from teacher training.** All reruns issued exactly six commands per proposal; no chunk members were silently dropped. Median proposal interval was **0.55–0.60 simulation seconds**, and the gap between the last command of one chunk and the next first command was **0.25–0.30 s**. The 12/6 feature cache was made for decisions every 0.30 s. The remaining inference/hold gap therefore changes causal history and slows route progress. Ignoring a 300 ms latency promotion limit does not remove this control effect. Timing alone cannot explain the continued late away-directed commands, so it is a contributor rather than a complete diagnosis.

**Previous-action feedback is a strong candidate cause of early drift.** I adapted the saved-input audit to this 12-slot checkpoint and its 300 ms cache. On two repeated scenes, reconstructing the first eight model proposals from saved RGB, state, task and history reproduced the logged first targets to **0.00 mm**. This argues against a runtime sampler or frame-conversion bug on those proposals. In frozen one-field interventions at the next decisions, replacing only the *previous actor action* with the time-aligned teacher action reduced first-target gaps to recorded teacher targets:

| Scene | Decision 2 | Decision 3 | Swap only current visual feature, decision 3 |
| --- | --- | --- | --- |
| `llb_0005` | 9.5 → 2.3 mm | 15.2 → 2.5 mm | 15.2 → 14.4 mm |
| `llb_0074` | 4.3 → 1.9 mm | 8.3 → 2.6 mm | 8.3 → 7.1 mm |

These swaps are offline diagnostics, not deployable use of teacher actions. The teacher-time alignment is approximate after states diverge. They show a large causal sensitivity to the previous-command input, consistent with training on previous **teacher** targets in 90% of contexts and previous **actor** targets at runtime. [LLB0005 audit](../../outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/llb0005_input_ablation.json) · [LLB0074 audit](../../outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/llb0074_input_ablation.json). This extends, rather than replaces, the [October 12 frozen-input audit](2026-10-12-route-feedback-and-inference-cadence.md).

**A universal wrong-frame or port-code bug is not supported.** The live SC task vectors select the requested port, and the full training label union contains both port codes (39,544 P0 and 30,137 P1 rows before cadence filtering). The saved-input replay reproduces model output exactly. For five of six new rollouts, controller reference-to-measured TCP position error in the 40–70 s phase had p50 around 8.5–12.2 mm, comparable to or smaller than the successful teacher's errors. The new five-card P1 scene had larger actor tracking error, 23.1 mm p50 versus the teacher's 6.9 mm, and this can worsen its outcome; it does not explain the one-, two-, and other five-card failures. These checks do not prove that the RGB backbone always identifies the selected port correctly; no precise identity output was connected to this transport actor.

**Interpretation:** the dominant observed problem is a learned route/progress decision under its own action history, amplified by the 0.25–0.30 s inter-chunk pause. It fails at transport, before perception handoff, fine alignment, contact recovery, or cable-snag RL is exercised. This is stronger than saying only that it scores zero; the route plot, command references, teacher comparison, and frozen-input swaps identify a specific failure mechanism. A complete causal repair has not yet been demonstrated.

## Next bounded repair, not started here

**Later October 13 continuation:** The SC teacher-pause audit, 100 ms action-spacing fits, denser 70 s context fit, intentional teacher memorization, and matched six-scene autonomous evaluations have now run. None reached the approach band or inserted. See [the follow-up experiment](2026-10-13-sc-pauses-context-and-overfit.md) before using the original proposal below.

First test one frozen-trace repair that represents SC route progress explicitly and trains on safe teacher continuations from **actor-visited** states, while aligning the training history to the actual decision cadence or overlapping inference with command execution. Do not add routine successful-teacher trajectories alone. Freeze a new nonfinal rollout gate before fitting: require a substantial reduction in early actor-state command gap on both saved SC traces, no degradation of grouped teacher-state SC error, and then ≥2/3 safe-buffer visits on new development layouts without scored contact. Keep the precise handoff, insertion actor, RL, and final scenes gated until transport reaches the requested opening. A separate contact/cable-recovery experiment would be premature for these particular runs.

## Reproduction

```bash
DOCKER_HOST=unix:///run/user/1008/docker.sock \
AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK=1 \
AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86 AIC_ROUTE_DIFFUSION_PREWARM=1 \
AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=12 \
.pixi/envs/default/bin/python scripts/run_teacher_distribution_route_diagnostic.py \
  --checkpoint outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon12_execute6_best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_repeat_v1 \
  --gpu 3 --only llb_0005 --only llb_0074 --only llb_0183

# Same environment and checkpoint, previously unrun development scenes:
DOCKER_HOST=unix:///run/user/1008/docker.sock \
AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK=1 \
AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86 AIC_ROUTE_DIFFUSION_PREWARM=1 \
AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=12 \
.pixi/envs/default/bin/python scripts/run_teacher_distribution_route_diagnostic.py \
  --checkpoint outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon12_execute6_best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_v1 \
  --gpu 3 --sc-only --sc-name llb_0090 --sc-name llb_0021 --sc-name llb_0009

# After scored-stream extraction, both route analyses used:
.pixi/envs/default/bin/python scripts/analyze_teacher_distribution_route.py \
  --comparison /home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_newdev_analysis.json
```

The posthoc plot/summary generator is [`summarize_sc_horizon12_reeval.py`](../../scripts/summarize_sc_horizon12_reeval.py); `scripts/audit_route_live_input_parity.py` now accepts `--cache` and the checkpoint action horizon so these frozen 12/6 inputs can be replayed. Every rollout directory also contains its `eval_manifest.json` with the exact environment and image digest.

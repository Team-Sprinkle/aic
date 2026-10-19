# SC teacher pauses, 100 ms action spacing, and intentional BC fit

**Later interpretation correction:** The [independent review](2026-10-13-transport-independent-root-cause-review.md) finds the transport labels stop before the approach band used below, and same-clock separation is not a pure route error. It also identifies lost terminal supervision. Preserve the recorded metrics, but use the revised endpoint analysis and terminal/handoff repair plan before acting on this record's next-step recommendation.

Date: 2026-10-13. This is an SC transport diagnostic on nonfinal Gazebo scenes. No model was promoted, no privileged geometry was passed to an actor, and insertion, recovery, and RL were not activated.

## Question and data audit

The original 104 full-insertion Low Lateral Bypass (LLB) SC teacher episodes are preserved under `outputs/trajectory_datasets/sc_llb_score_verified/`. The active actor did **not** train directly on those raw pauses. Its SC source is the separate `outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/` derivative. It removed **19,570 quiet transport anchors** from the 89,251 raw rows, while keeping measured movement, excess force, edge samples, and at least one anchor per second in each hold. The 83 training episodes were filtered; the 21 episode-grouped validation episodes were left intact. Their original recorded target actions and images were not changed. The planned holds were 8 s after the row sweep and 5 s near tilt; the measured target plateaus in the 83 training source episodes had medians **9.45 s** and **6.48 s** including settling. The source and derivative checks are in [the original cleanup record](2026-10-01-sc-llb-cleaned-bc.md) and [this machine audit](../../outputs/experiments/2026-10-13_sc_pause_models_v1/pause_audit.json).

The prior 12/6 actor already uses **17 sparse history samples spanning up to 60 s**. The span exceeds either teacher pause. The cache for cleaned training episodes has median longest inter-row gap 5.53 s, whereas the untouched validation episodes have a median longest gap 0.33 s. Therefore “context shorter than the pause” is not supported as the main diagnosis. Sparse history and the distribution shift from teacher to actor actions remain possible causes. In live rollout, the old six 50 ms commands produced decisions every ~0.55 s despite a 0.30 s training cadence, leaving a ~0.25–0.30 s between-chunk gap.

The active union remains the verified 533 SFP / 83 SC training and 43 SFP / 21 SC validation episodes. These two new fits deliberately select **SC only** to isolate this failure; they are not deployable shared SFP+SC replacements. Neither uses a true port pose at runtime. The teacher label is the complete recorded TCP-body target, with no delta adapter.

## Three fits and immutable derivatives

1. **100 ms spacing control:** Starting from the frozen 12/6 route-diffusion checkpoint, predict 12 recorded targets at 100 ms intervals (1.2 s) and execute the first six (0.6 s). Keep the already long 60 s / 17-sample observation context. Build the targets from every second recorded 50 ms teacher command, reexpressed in the anchor TCP frame. No action interpolation or padding is allowed; discontinuities invalidate the anchor. The new `/home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2/` is a separate copy of observation features and rows sampled at nominal 0.6 s decisions. It has 2,740 valid 12-slot SC training rows and 1,043 validation rows; all 83/21 SC episodes are represented. The source union, original cleaned copy and prior cache are unchanged.
2. **Genuinely longer context:** Continue that 100 ms model with **27 samples over 70 s**, adding samples around 5–14 s and extending beyond the old 60 s window. The diffusion checkpoint records these ages and the live runtime reads them; the action labels, split and 100 ms controller are identical to the first arm. The 70 s slot is available only very late in these roughly 70 s runs, so the useful extra context is mainly the denser intermediate history. Its selected held-out action p95 was 6.26 mm versus 6.25 mm for the spacing control, so offline fit did not improve.
3. **Intentional 50 ms fit:** Use the same 12/6 architecture and 60 s history, but fit the cleaned SC teacher actions closely. It trained for two 30,000-update stages; the second stage selected by training error. The best second-stage checkpoint was at 29,000 updates. This is a memorization diagnostic, not a selected deployment actor.

All three hide previous teacher-command chunks during training and zero them at runtime.

All use the existing 14.29M-trainable-parameter route diffusion head plus the frozen 11.18M-parameter ImageNet ResNet18 camera encoder. The new controller period check rejects a 100 ms checkpoint if the runtime still runs at 20 Hz. The runtime was set to 10 Hz for the 100 ms actors and 20 Hz for the 50 ms actor. Six command members per decision were verified; none were silently dropped. Median live decision intervals were **0.899–0.900 s** and **0.550 s** respectively, so all still have roughly 0.25–0.30 s of inference/hold time beyond their nominal 0.6/0.3 s training cadence. [Exact measured timing and command counts](../../outputs/experiments/2026-10-13_sc_pause_models_v1/timing_audit.json).

| Fit | SC train rows | SC validation rows | Best update | Stop | Training executed-chunk translation p95 | Held-out executed-chunk translation p95 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 100 ms 12/6, 60 s history | 2,740 | 1,043 | 7,000 | validation plateau at 12,000 | not used for selection | 6.25 mm |
| 100 ms 12/6, 70 s dense history | 2,740 | 1,043 | 500 after warm start | validation plateau at 5,500 | not used for selection | 6.26 mm |
| 50 ms 12/6, first fit | 6,139 | 2,269 | 30,000 | update cap | 1.46 mm | 4.40 mm |
| 50 ms 12/6, continued fit | 6,139 | 2,269 | 29,000 more | update cap at 30,000 more | **0.91 mm** | 4.46 mm |

The held-out teacher error for the intentional fit did not improve with further memorization. Validation includes original hold interiors; train does not, so these offline values should not be interpreted as an isolated architecture comparison. The live test below is matched by exact scene.

## Autonomous six-scene comparison

The six frozen, nonfinal validation layouts were `llb_0005` (1 card, port 1), `llb_0090` (1, port 0), `llb_0021` (2, port 1), `llb_0183` (4, port 1), `llb_0074` (5, port 0), and `llb_0009` (5, port 1). All six recorded teachers fully inserted without scored contact. The actors used three cameras, measured robot state, and the SC selected-port task code. Port geometry was used only after rollout for scoring. The baseline is the previously frozen 50 ms 12/6 checkpoint on exactly these scenes; the older 4/4 checkpoint was also 0/3 approach buffers and 0/3 insertions on its three matching SC scenes.

| Actor | Approach-band visits | Full insertions | Scored contacts | Median closest TCP-to-opening 3D distance | Median live inference p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 50 ms 12/6 | 0/6 | 0/6 | 0/6 | 190 mm | 416 ms |
| 100 ms 12/6, 60 s history | 0/6 | 0/6 | 0/6 | 162 mm | 393 ms |
| 100 ms 12/6, 70 s dense history | 0/6 | 0/6 | 0/6 | **135 mm** | 394 ms |
| Intentional-fit 50 ms 12/6 | 0/6 | 0/6 | 0/6 | 137 mm | 406 ms |

The distance is **TCP to port opening**, not plug-tip insertion error. Smaller is better for transport, but the required approach band is axial −100 to −70 mm; all **24** matched actor episodes had zero samples there. The 100 ms spacing control got closer than baseline on five of six layouts; adding dense 70 s history improved its closest distance on five of six, though this is a continuation rather than an isolated architecture change. The intentional-fit actor got closer than baseline on all six. Per-scene best distances (old / 100 ms 60 s / 100 ms dense 70 s / fit, in mm): `0005` 181/153/141/135, `0009` 266/156/129/110, `0021` 160/137/123/133, `0074` 212/182/190/189, `0090` 199/184/163/157, `0183` 169/169/119/139. None met the 300 ms p95 inference limit.

The models still usually turned away near the final route phase. For the 100 ms spacing control, first proposed axial commands after 40 s pointed away from the opening in 89–100% of sampled decisions across scenes. For the intentional-fit actor, this occurred in **all** sampled post-40 s decisions in all six scenes. This is compatible with a route-transition / closed-loop feedback failure; it is not evidence of a port-lip block or cable snag in these runs. Two additional **training-scene** rollouts of the intentional-fit actor, one-card `llb_0025` and five-card `llb_0019`, also had 0/2 approach-band entries and 0/2 insertions, without scored contact. Their closest TCP 3D distances were 194 and 124 mm. This makes a held-out-scene-only explanation insufficient, although two training scenes are a small diagnostic sample.

## Artifacts and reproduction

- [Machine six-scene comparison](../../outputs/experiments/2026-10-13_sc_pause_models_v1/live_comparison_v2.json), [pause audit](../../outputs/experiments/2026-10-13_sc_pause_models_v1/pause_audit.json), and container-visible [100 ms, 60 s context](../../outputs/experiments/2026-10-13_sc_pause_models_v1/longcontext_100ms_best.pt) / [100 ms, dense 70 s context](../../outputs/experiments/2026-10-13_sc_pause_models_v1/dense70_100ms_best.pt) / [intentional-fit](../../outputs/experiments/2026-10-13_sc_pause_models_v1/overfit_50ms_best.pt) checkpoints.
- Full training manifests, histories and summaries: `/home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_v2/`, `.../2026-10-13_sc_dense70_100ms_v1/`, `.../2026-10-13_sc_overfit_50ms_v1/`, and `.../2026-10-13_sc_overfit_50ms_v2/`.
- Scored bags, exact scene plans, commands, measured traces, RGB frames and scores: `/home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/`, `.../2026-10-13_sc_dense70_100ms_eval6_v1/` and `.../2026-10-13_sc_overfit_50ms_eval6_v1/`. Training-scene diagnostic: `.../2026-10-13_sc_overfit_50ms_train2_v1/`.
- Representative five-camera H.264 videos: [100 ms one-card](</home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/rollouts/llb_0005/five_cameras_10fps.mp4>), [100 ms five-card](</home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/rollouts/llb_0009/five_cameras_10fps.mp4>), [dense 70 s five-card](</home/nas/chmin/aic_archives/2026-10-13_sc_dense70_100ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps.mp4>), [intentional-fit five-card](</home/nas/chmin/aic_archives/2026-10-13_sc_overfit_50ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps.mp4>). I inspected sampled frames from the one- and five-card recordings alongside traces; they show a broad side route, not a demonstrated scored contact.
- The first 100 ms simulator launch pointed Docker at a host-only NAS checkpoint path, failed before any action, and is retained at `.../2026-10-13_sc_longcontext_100ms_eval6_v1/`. The checkpoint was copied into the repository, then all six valid runs were repeated under `..._eval6_v2/`.

The exact training commands and hashes are in each run's `manifest.json`, and exact Docker environment and scene hashes in each rollout's `eval_manifest.json`. The new label/cache commands were:

```bash
.pixi/envs/default/bin/python scripts/build_extended_route_chunks.py \
  --cache /home/nas/chmin/aic_archives/2026-10-13_route_consistent_cache_stride6_v1 \
  --sfp-labels /home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/sfp_labels.jsonl \
  --sc-labels /home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/sc_labels.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_extended_100ms_stride6_v1 \
  --slot-stride 2
.pixi/envs/default/bin/python scripts/derive_route_cache_100ms.py \
  --cache /home/nas/chmin/aic_archives/2026-10-13_route_consistent_cache_stride6_v1 \
  --extended-labels /home/nas/chmin/aic_archives/2026-10-13_route_extended_100ms_stride6_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2
```

Training used GPUs 4 and 3 simultaneously, with the long-standing Isaac job on GPU 0; no more than three project GPUs were active. The representative complete training commands (the first intentional-fit stage used the old 12/6 checkpoint as `--init`, and the shown second stage used its best result) were:

```bash
CUDA_VISIBLE_DEVICES=4 .pixi/envs/default/bin/python scripts/train_route_aware_diffusion_transport.py \
  --cache /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --extended-labels /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2 \
  --output /home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_v2 \
  --action-horizon 12 --execute-steps 6 --train-family sc \
  --init outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon12_execute6_best.pt \
  --updates 20000 --budget-minutes 180 --min-updates 5000 \
  --plateau-patience-updates 5000 --prior-dropout 1.0 --learning-rate 0.0001

CUDA_VISIBLE_DEVICES=3 .pixi/envs/default/bin/python scripts/train_route_aware_diffusion_transport.py \
  --cache /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --extended-labels /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2 \
  --output /home/nas/chmin/aic_archives/2026-10-13_sc_dense70_100ms_v1 \
  --action-horizon 12 --execute-steps 6 --train-family sc --history-context dense70 \
  --init outputs/experiments/2026-10-13_sc_pause_models_v1/longcontext_100ms_best.pt \
  --updates 20000 --budget-minutes 180 --min-updates 5000 \
  --plateau-patience-updates 5000 --prior-dropout 1.0 --learning-rate 0.00005

CUDA_VISIBLE_DEVICES=3 .pixi/envs/default/bin/python scripts/train_route_aware_diffusion_transport.py \
  --cache /home/nas/chmin/aic_archives/2026-10-13_route_consistent_cache_stride6_v1 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --extended-labels /home/nas/chmin/aic_archives/2026-10-13_route_extended_chunks_stride6_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-13_sc_overfit_50ms_v2 \
  --action-horizon 12 --execute-steps 6 --train-family sc --select-on train \
  --init /home/nas/chmin/aic_archives/2026-10-13_sc_overfit_50ms_v1/best.pt \
  --updates 30000 --budget-minutes 180 --min-updates 8000 \
  --plateau-patience-updates 6000 --prior-dropout 1.0 --learning-rate 0.00005
```

The live six-scene runner was `scripts/run_teacher_distribution_route_diagnostic.py --sc-only --sc-name llb_0005 --sc-name llb_0074 --sc-name llb_0183 --sc-name llb_0090 --sc-name llb_0021 --sc-name llb_0009 --max-new 6`; it used `DOCKER_HOST=unix:///run/user/1008/docker.sock`, `AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK=1`, `AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS=1`, `AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=12`, and `AIC_TRANSPORT_EVAL_CONTROL_HZ=10` for the 100 ms checkpoint or `20` for the intentional-fit checkpoint. Each run's `plan.json`, `eval_manifest.json` and launcher snapshot preserve the exact checkpoint, scene hashes, environment and controller command. The two training-scene diagnostics used the same 50 ms settings with `--sc-training-diagnostic --sc-name llb_0025 --sc-name llb_0019`.

## Decision

The pause length is already covered by the observation window. The genuinely longer history improved port proximity but still yielded no approach-band arrivals; closer teacher fitting also failed to reach the opening, even on two training scenes. Do not keep extending pure teacher BC or the 100 ms horizon without a new causal hypothesis. The next bounded repair should target the **first actor-state departure around 6–9 s**: train route-progress or safe teacher-continuation labels on actor-visited states, and match or overlap the live inference gap. A sensible gate is reduction of actor-state command error on the saved first-divergence traces, followed by at least 2/3 new nonfinal SC safe approach-band arrivals with no scored contact and p95 inference below 300 ms. Keep the final split, automatic handoff, insertion policy, and RL gated until this transport gate passes.

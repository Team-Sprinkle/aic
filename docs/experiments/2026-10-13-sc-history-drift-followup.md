# SC actor drift: video teardown, finer history, and route diagnostics

**Later interpretation correction:** The [independent review](2026-10-13-transport-independent-root-cause-review.md) finds the transport labels stop before the approach band used below, and same-clock separation is not a pure route error. It also identifies lost terminal supervision. Preserve the recorded metrics, but use the revised endpoint analysis and terminal/handoff repair plan before acting on this record's next-step recommendation.

Date: 2026-10-13. Development scenes only; the reserved final split remains sealed. These are autonomous Gazebo actor rollouts. Simulator geometry enters scoring and teacher labels, not actor inference.

## Why the board disappears in the recordings

The camera recorder continued after the actor returned. For example, the dense-history `llb_0009` actor's last command and terminal trace are at 72.7 s simulated time; the untrimmed video continues to 82.9 s and the engine log explicitly records deletion of `task_board` and `cable_1` during trial reset. This is not an actor command, cable snag, or scene failure. `scripts/render_transport_five_camera_video.py --active-only` now stops at `terminal_before_return` and writes a separate `_active.mp4`, preserving the original video and frames. I inspected the last frames of both versions: the board and orange cable are present in the active version and absent after teardown in the original.

The development actor deliberately stops after 70 simulated seconds. Matched successful SC teachers took 63.43–100.88 s according to their saved scoring messages. Thus a 70 s diagnostic is useful for diagnosing the route but is not a full-length insertion test on every scene. The example scene's task configuration has a 180 s limit. In the toolkit's `ScoringTier2.cc`, the duration *bonus* runs from 5 to 60 s and requires an inserted plug; duration cannot explain the disappearing assets. Route correctness remains the current gate before compressing the teacher pauses for speed.

## Matched history test

The frozen dense actor used 27 history requests over 70 s. The new `fine70` choice requests 38 ages, concentrating more samples in the recent 0–10 s while keeping the same 12 predicted / six executed 100 ms target actions and the same episode-grouped 83/21 SC train/validation split. It was warm-started from the dense actor and selected by held-out error. It stopped at a validation plateau after 11,000 updates; best update 3,000, held-out executed-chunk translation p95 6.09 mm versus 6.26 mm for the earlier dense checkpoint. Exact manifest, history, checkpoint and log: `/home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_validation_v1/` and matching `.log`. The container-visible selected copy is `outputs/experiments/2026-10-13_sc_pause_models_v1/fine70_100ms_validation_best.pt`.

| Actor | Scenes | Approach-band visits | Median closest TCP-to-opening distance | Median live inference p95 |
| --- | ---: | ---: | ---: | ---: |
| Earlier dense70, 12 denoising steps | 6 | 0 | 135 mm | 394 ms |
| Finer 70 s history, 12 denoising steps | 6 | 0 | 141 mm | 388 ms |

The six finer-model inference p95 values are 380, 393, 361, 383, 417 and 416 ms. The finer model got closer on three of six scenes and farther on three. First sustained 20 mm departures from the teacher were at 6.4–9.05 s. [Exact six-scene post hoc analysis](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_validation_eval6_v1/analysis_v1.json>) and [trimmed five-card video](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_validation_eval6_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>) preserve the evidence.

The new history requests do not all yield new information. At late route times, the 100 ms training cache supplies a median **14 distinct history rows** for both 27 and 38 requested ages, because cleaned training episodes have sparse retained rows around holds. Validation supplies about 20 and 22 respectively. This is a representation/data mismatch, not evidence that 38 distinct observed frames were tested.

## Bounded causal checks

Held-out teacher-frame executed-chunk translation error for the finer model was p95 **5.38 mm early** (3–12 s), **4.75 mm middle** (12–35 s), and **11.32 mm late** (35–70 s). The training cache has 1,244 early, 2,111 middle and only 586 late SC rows before valid-horizon filtering. A separate 27-history phase-balanced fit weighted early rows 1.5 times and late rows four times. It reached its held-out phase-score plateau at 12,500 updates (best 6,500), yet on three matching live scenes its closest distances were 168/198/148 mm for `llb_0005/0074/0009`, versus 141/190/129 mm with the earlier dense model. It entered the approach band in 0/3. Preserve [its scored analysis](</home/nas/chmin/aic_archives/2026-10-13_sc_dense70_phasebalanced_eval3_v1/analysis_v1.json>) and `/home/nas/chmin/aic_archives/2026-10-13_sc_dense70_phasebalanced_v1/` rather than promoting it.

Another frozen-model test cut diffusion denoising from 12 to six steps, to reduce the measured gap between command chunks. On the same three scenes, median inference p95 fell to **296 ms**, but closest distances were **166/194/130 mm**, still 0/3 approach-band visits. [Scored analysis](</home/nas/chmin/aic_archives/2026-10-13_sc_dense70_sample6_eval3_v1/analysis_v1.json>) preserves this latency/route comparison. Reducing inference delay alone did not repair the route.

On five-card `llb_0009`, from about 3–12 s the earlier dense actor's measured TCP path length was **238 mm** versus **304 mm** for the time-matched teacher; the finer-history actor traveled **217 mm** versus **293 mm** for the teacher. This is descriptive evidence of early under-progress, not proof that inference time is the only cause: the six-denoising-step test above reduced latency but still missed the route. Actor proposals shrank after early deviation, so teacher-frame action accuracy does not imply recovery from actor-visited states.

## Deliberate 38-history overfit

The overfit 38-history actor used the same 83 SC training episodes and 21 episode-grouped validation episodes. The first 80,000-update stage selected training update 77,000 with train executed-chunk p95 0.94 mm. A lower-rate 60,000-update continuation selected update 55,000 with 0.60 mm. A still lower-rate continuation stopped at its train-selection plateau after 25,500 updates, selecting update 15,500 with **0.49 mm**. These are cumulative warm-start stages, not a 25,500-update fit from scratch. Exact immutable manifests/histories/checkpoints: `/home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_v1/`, `..._v2/`, `..._v3/`. Normal validation was not used to select these intentional-memorization checkpoints and should not be conflated with train error.

The first-stage actor also ran on two **training** scenes, one-card `llb_0025` and five-card `llb_0019`. It reached no approach band or insertion in either, with closest TCP-to-opening distances **142 and 152 mm** and first sustained 20 mm teacher departures at 7.5 and 6.9 s. [Training-scene analysis](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_stage1_train2_v1/analysis_v1.json>) records these failures.

The second-stage actor was evaluated on all six original development scenes. It reached **0/6 approach bands and 0/6 insertions**; median closest TCP-to-opening distance was **148 mm**, versus **135 mm** for the earlier dense actor. It improved closest distance in two scenes and worsened in four; its median inference p95 was **408 ms**. [Matched six-scene analysis](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_stage2_eval6_v1/analysis_v1.json>) preserves the per-scene measures.

The final plateau checkpoint was checked on the matched one-card and two five-card scenes (`llb_0005/0074/0009`). It reached **0/3 approach bands, 0/3 insertions, and no scored contact**; closest distances were **145/174/144 mm**, with median inference p95 **399 ms**. The earlier dense actor was **141/190/129 mm** on those same scenes. The final overfit stage therefore changed individual paths without solving transport. [Three-scene analysis](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_plateau_eval3_v1/analysis_v1.json>) and [trimmed five-camera video](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_plateau_eval3_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>) preserve the result. I inspected the active video's last frame: board and orange cable remain present, with the plug still outside the approach band.

The container-visible plateau checkpoint is `outputs/experiments/2026-10-13_sc_pause_models_v1/fine70_100ms_overfit_plateau_best.pt`. [Machine-readable cross-arm summary](../../outputs/experiments/2026-10-13_sc_pause_models_v1/history_drift_followup_summary.json) records all rollout paths, checkpoint hashes, scene scores, latencies, and training stop reasons.

### Reproduction commands

Run from this repository with the existing `.pixi` environment; choose a new `--output` directory when reproducing, because existing runs are immutable. Each saved training `manifest.json` records the full inputs, episode counts, selected checkpoint, update budget, and image/action convention. The validation-selected finer-history fit used:

```bash
CUDA_VISIBLE_DEVICES=3 .pixi/envs/default/bin/python scripts/train_route_aware_diffusion_transport.py \
  --cache /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --extended-labels /home/nas/chmin/aic_archives/2026-10-13_route_cache_100ms_v2 \
  --output /home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_validation_v1 \
  --action-horizon 12 --execute-steps 6 --train-family sc --history-context fine70 \
  --init outputs/experiments/2026-10-13_sc_pause_models_v1/dense70_100ms_best.pt \
  --updates 60000 --budget-minutes 90 --min-updates 8000 \
  --plateau-patience-updates 8000 --prior-dropout 1.0 --learning-rate 0.00005
```

The final overfit stage changed `--select-on train`, initialized from `.../2026-10-13_sc_fine70_100ms_overfit_v2/best.pt`, used `--updates 30000 --budget-minutes 30 --min-updates 10000 --plateau-patience-updates 10000 --learning-rate 0.00001`, and wrote `.../2026-10-13_sc_fine70_100ms_overfit_v3/`. The phase-balanced fit reused the dense70 checkpoint and passed `--sc-early-weight 1.5 --sc-late-weight 4 --sc-phase-selection`; its exact remaining arguments are in its manifest. No source trajectory directory was overwritten.

Autonomous evaluation used `scripts/run_teacher_distribution_route_diagnostic.py --sc-only` with the six SC names `llb_0005`, `llb_0074`, `llb_0183`, `llb_0090`, `llb_0021`, `llb_0009`, `AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK=1`, `AIC_ROUTE_DIFFUSION_ZERO_PREVIOUS=1`, `AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=12`, and `AIC_TRANSPORT_EVAL_CONTROL_HZ=10`. The faster sampler changed only `AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=6`. Each run's `plan.json`, `progress.json`, `eval_manifest.json`, `launcher_snapshot.sh`, scored bag, runtime trace and frames preserve the exact scene/model/runtime combination. To make an active-only video:

```bash
.pixi/envs/default/bin/python scripts/render_transport_five_camera_video.py \
  /home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_plateau_eval3_v1/rollouts/llb_0009 \
  --active-only
```

## Decision and speed work

**No checkpoint is promoted.** Extra fine history did not improve the six-scene route median; phase-balanced sampling and six-step inference both missed 0/3 approach bands; deliberate overfitting reached 0.49 mm training error yet still missed 0/3 matched development approach bands and 0/2 first-stage training approach bands. The first departure remains near 7–9 s, far before the 70 s diagnostic budget. The next repair needs **causal labels on actor-visited states**, especially just before and after the 20 mm departure, plus an explicit check of commanded versus measured progress. Collect them only from new training reset groups using a safe LLB route continuation, and keep actor-prefix observations as context rather than mislabeled expert actions. A matched no-correction control and new complete-reset development starts are necessary. Stop that branch unless it reduces early measured route error and achieves at least two of three safe approach-band arrivals with no contact and p95 below 300 ms. Existing SFP corrective-data pilots showed that narrow start/port coverage can overfit, so SC correction groups must vary card count, requested port, and starting geometry.

The current route gate has **not** passed, so I did not speed-edit or regenerate the expert dataset. For a later speed experiment, first copy the existing verified SC set and shorten the measured 8 s / 5 s holds without changing target/image alignment or scene identities; validate the copy against full Gazebo insertion and compare durations. The six matched successful teachers earned zero duration bonus because they took 63–101 s, but full insertion is the higher-priority outcome. Keep the original dataset intact. No final scenes, automatic handoff, insertion policy or RL were opened here.

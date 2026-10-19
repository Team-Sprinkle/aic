# Predicting longer transport chunks: 8/4 and 12/6, October 13

## Question and controlled design

The October 12 route-diffusion actor predicted and executed four 50 ms TCP targets per decision. This experiment keeps the three-camera ImageNet backbone, 384D temporal/diffusion architecture, task code, body-frame full TCP targets, scene splits, and initial four-target checkpoint. It compares:

| Variant | Predicted targets | Executed targets before observing again | Nominal action window |
| --- | ---: | ---: | ---: |
| Control | 4 | 4 | 200 ms |
| Longer prediction | 8 | 4 | 200 ms |
| Longer prediction and execution | 12 | 6 | 300 ms |

The unexecuted targets are training predictions, not commands issued to Gazebo. All targets are **recorded full teacher TCP poses represented relative to the observed TCP frame**, not cumulative delta actions or privileged port poses. At deployment, each selected target is converted to an absolute base-frame Cartesian command from the latest measured TCP state. Only observation images, measured state, task code, and causal history are model inputs.

## Full dataset and label audit

The active [route-consistent union](../ACTIVE_TRANSPORT_DATASET.json) contains **533 SFP and 83 SC Low Lateral Bypass training episodes**, and **43 SFP and 21 SC development episodes**. SC CheatCode paths that go over the cards remain excluded. Training/development scenes are grouped by reset configuration; final scenes stayed sealed.

Each source row carries four recorded targets. To create a longer chunk, [`build_extended_route_chunks.py`](../../scripts/build_extended_route_chunks.py) keeps those four labels unchanged, then takes the first recorded teacher target from the next 50 ms rows and transforms it into the original observation's TCP frame. It requires uninterrupted time, unchanged task, and, for SC, continuous command indices. It does **not** pad an episode ending or jump across a pause. A source-label reconstruction audit matched the original four targets at effectively zero translation error. Every one of the 616 training and 64 development episodes retained usable longer targets.

| Cache decision cadence | Horizon | Usable train anchors, SFP / SC | Usable development anchors, SFP / SC |
| --- | ---: | ---: | ---: |
| 200 ms | 8 | 32,913 / 9,451 | 2,382 / 3,472 |
| 300 ms | 12 | 20,849 / 6,139 | 1,469 / 2,269 |

The 12/6 actor must replan after 300 ms, so it was trained on a new **300 ms feature cache** from the same full episode union. Training it on the existing 200 ms cache would teach the wrong previous-command cadence. An initial 12-target/200 ms diagnostic fit was preserved under `/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_v1/` but is **not** the evaluated 12/6 model. The 8/4 actor uses the original 200 ms cache. Its 8/4 fit used 42,364 train and 5,854 development anchors; 12/6 used 26,988 train and 3,738 development anchors. These are all valid anchors from the full admitted episodes, not a subset of task types.

## Training convergence

Both new models initialized from the same October 12 four-target checkpoint. Only the action-position embeddings grew to 8 or 12 slots; the history input continues to use the first four targets from the previous proposal, matched by the appropriate 200/300 ms data cadence. The model has about 14.29M trainable parameters plus 11.18M frozen ImageNet ResNet18 parameters. Checkpoint selection in the final phases used held-out error over the **executed** prefix, not merely the first target.

| Model | Training sequence | Why training stopped | Selected stage |
| --- | --- | --- | --- |
| 8/4 | 9,500 updates at 3e-5, then 8,500 at 1e-5 | Both phases reached the specified development plateau rule; second phase selected using all four executed actions | 6,000 + 5,000 updates |
| 12/6 | 20,000 at 3e-5, 16,000 at 1e-5, then 4,000 at 3e-6 | First two phases hit update caps while still improving; third reached a development plateau | 19,000 + 15,000 + 500 updates |

These are held-out plateaus under the tested architecture and learning rates, not a claim of global optimality. [The plotted SFP/SC curves](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/training_curves.png) preserve the selected stages and phase boundaries. Full histories, manifests, selected checkpoints, and logs are in the NAS training roots listed below.

## Matched offline performance

The following p95 values are mean translation error over the **first four target slots**, in mm. Each comparison uses identical grouped development rows and a fixed sampling seed within its own decision cadence. The 200 ms and 300 ms columns use different row sets and should not be compared as if paired.

| Decision cadence | Family | Four-target control | Longer-horizon actor |
| --- | --- | ---: | ---: |
| 200 ms, 8/4 | SFP, 516 rows | 3.15 | **2.58** |
| 200 ms, 8/4 | SC, 252 rows | 2.75 | **2.64** |
| 300 ms, 12/6 | SFP, 516 rows | 3.63 | **2.37** |
| 300 ms, 12/6 | SC, 252 rows | 3.73 | **2.53** |

For 12/6, p95 mean error across all **six executed** slots was 2.54 mm SFP and 2.69 mm SC. The offline gains mean better imitation on teacher observations; they do not establish recovery from the actor's own visited states.

The four-target control was originally trained at 200 ms decisions. Feeding it the 300 ms row cache for the 12/6 offline pairing changes its history cadence, so that particular control error may be inflated. The 12/6 absolute error and its within-cadence learning curve remain valid; the autonomous comparison below keeps each model at its intended decision cadence on the same scenes.

## Live execution correction and matched Gazebo result

The original simulation-clock runner anchored all four slots to the *pre-inference* observation and dropped later targets when inference outlasted the chunk. In the first diagnostic 8/4 SFP scene it predicted four but issued **205 commands for 158 proposals**, dropping 427 stale members. This did not test the requested four-command execution. The unmodified six-scene diagnostic is preserved at `/home/nas/chmin/aic_archives/2026-10-13_route_horizon8_execute4_live_v1/` and is excluded from the model comparison.

An opt-in `AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK=1` mode starts execution from a fresh *post-inference* simulation observation and spaces each command by measured 50 ms simulation time. All three variants then issued **exactly four or six commands per proposal**, with zero stale drops, on the same six nonfinal teacher-generator scenes. This fixes the command-count test; inference still creates observation-to-action delay, which is reported rather than hidden. All actors were autonomous, with no teacher, guide, true target pose, or geometry in their policy input. Posthoc geometry and official scoring were used only to analyze their paths.

| Live outcome on three SFP + three SC scenes | 4/4 control | 8/4 | 12/6 |
| --- | ---: | ---: | ---: |
| Safe approach-buffer arrivals | 0/6 | 0/6 | 0/6 |
| Full insertions | 0/6 | 0/6 | 0/6 |
| Scored contacts | 0/6 | 0/6 | 0/6 |
| SFP val04 lateral miss in axial band | 30.65 mm | 47.03 mm | **19.07 mm** |
| SFP val12 lateral miss | 54.36 mm | 57.35 mm | **43.20 mm** |
| SFP val14 lateral miss | 56.74 mm | 50.93 mm | **11.52 mm** |
| SC axial-band entries | 1/3, 214.45 mm lateral | 0/3 | 0/3 |

The 12/6 actor got closer than the control in all three SFP scenes, with the third 1.52 mm outside the 10 mm buffer threshold. It did **not** achieve a valid buffer or insertion and did not fix SC route progress. There was no scored contact to trade against this geometric improvement. On these shared-GPU Gazebo tests, complete inference p95 ranged **380–471 ms** for 4/4, **388–434 ms** for 8/4, and **364–493 ms** for 12/6. The requested 300 ms latency limit was deliberately not a promotion gate in this experiment; the model's delay still affects closed-loop behavior.

Representative synchronized H.264, 10 fps five-camera recordings: [four-target control on SFP val14](/home/nas/chmin/aic_archives/2026-10-13_route_horizon4_execute4_completeclock_v1/rollouts/sfp_trial2_validation_14/five_cameras_10fps.mp4), [12/6 on the same SFP scene](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_completeclock_v1/rollouts/sfp_trial2_validation_14/five_cameras_10fps.mp4), and [12/6 on five-card SC LLB0074](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_completeclock_v1/rollouts/llb_0074/five_cameras_10fps.mp4). I verified the SFP video is H.264/yuv420p and inspected a frame; all five labeled views and the orange cable are visible. The scored streams and traces, rather than visual impression alone, determine the outcomes above.

A bounded four-denoising-step check preserved the 12/6 checkpoint and execution rule. On the same 300 ms development rows, four-step offline errors were at least as good. In one repeated SFP val14 run, inference p95 fell from **459 to 275 ms**, but lateral miss worsened from **11.52 to 51.83 mm**; the paired five-card SC LLB0074 run still did not enter the axial band. The two-run ablation did not justify changing the frozen setting. Gazebo rollout variation means the paired difference is diagnostic, not a precise causal latency effect.

## Decision and next repair

**No checkpoint is promoted.** Longer prediction, including 12/6, improves teacher-state action accuracy and partly improves SFP approach geometry, but autonomous safe transport remains 0/6 and SC still misses the route. The earlier four-scene [feedback audit](2026-10-12-route-feedback-and-inference-cadence.md) identified dependence on the previous teacher command under actor drift. The next bounded repair should use causally aligned route-progress or safe teacher-continuation supervision at actor-visited states, with the existing frozen-frame action-gap gate before another live scale-up. Do not treat more action horizon alone as an insertion fix. Handoff, insertion, RL, and final scenes remain gated.

## Commands and artifacts

The two immutable extended-label manifests are `/home/nas/chmin/aic_archives/2026-10-13_route_extended_chunks_v1/manifest.json` and `.../2026-10-13_route_extended_chunks_stride6_v1/manifest.json`. The cadence-matched feature cache is `/home/nas/chmin/aic_archives/2026-10-13_route_consistent_cache_stride6_v1/`. Training folders are `2026-10-13_route_horizon8_execute4_{v1,final_v2}/` and `2026-10-13_route_horizon12_execute6_{cadence_v2,final_v3,plateau_v4}/`, all under `/home/nas/chmin/aic_archives/`; each has an exact manifest, full history, summary, and checkpoint. Logs beside these folders record the exact CLI invocation. The selected, container-visible checkpoint copies are [8/4](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon8_execute4_best.pt) and [12/6](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon12_execute6_best.pt).

```bash
# The complete source union is the existing immutable v4 manifest.
.pixi/envs/default/bin/python scripts/build_extended_route_chunks.py \
  --cache /home/nas/chmin/aic_archives/2026-10-12_route_consistent_cache_v1 \
  --sfp-labels /home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/sfp_labels.jsonl \
  --sc-labels /home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/sc_labels.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_extended_chunks_v1
# The stride-6 cache was built with cache_temporal_transport_features.py --stride 6,
# then used to build the separate 12-slot label archive and to train 12/6.
AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK=1 \
AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY=86 AIC_ROUTE_DIFFUSION_PREWARM=1 \
AIC_ROUTE_DIFFUSION_SAMPLE_STEPS=12 \
  .pixi/envs/default/bin/python scripts/run_teacher_distribution_route_diagnostic.py \
  --checkpoint outputs/experiments/2026-10-13_route_horizon_comparison_v1/horizon12_execute6_best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_completeclock_v1 --gpu 0
.pixi/envs/default/bin/python scripts/summarize_route_horizon_experiment.py \
  --output outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json
```

[Machine summary with exact per-scene paths, contact messages, command counts, model manifests, and checkpoint hashes](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json). NAS roots `2026-10-13_route_horizon4_execute4_completeclock_v1/`, `2026-10-13_route_horizon8_execute4_completeclock_v1/`, and `2026-10-13_route_horizon12_execute6_completeclock_v1/` retain frozen plans, exact scene YAMLs, bags, saved camera frames, scored streams, runtime traces, and Docker manifests. The bounded faster-sampling root is `2026-10-13_route_horizon12_execute6_sample4_bounded_v1/`.

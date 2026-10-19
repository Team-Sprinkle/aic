# Transport follow-up: SFP check and five-way architecture comparison

Date: 2026-10-01. This record executes the user's requested continuation after the [frame/code audit](../answers/2026-10-01-transport-failure-code-and-frame-audit.md). No reserved final scene, privileged runtime geometry, RL, or recovery supervisor is part of this comparison. Rootless Docker; at most four project GPUs including the old GPU-0 process, which is left untouched.

## Questions and fixed controls

The decisive first question is whether the frozen clean shared body-target BC checkpoint fails autonomous **SFP-to-NIC** transport as it did SC-to-SC. Use one existing, non-final SFP validation trial (`episode_index=23`, card 0/port 1) for a bounded 70-s observation-only Gazebo rollout. Preserve score, trace, terminal frame, latency, and video. This is a development diagnostic, not a final-set evaluation or training episode.

The subsequent requested experiments compare:

1. **Long observation history:** Causal image features from the same three views, measured state, recent executed actions, and time gaps over up to a full route (target 32–60 s), with no reset crossing. Cache visual features so live inference encodes only new images. Compare to no-history with the same encoder/labels/split.
2. **Explicit route representation:** Predict SC route stage, complete TCP-body next subgoal, and progress/handoff state; advance only on measured progress and stable observations. Evaluate the labels and held-out predictions before deploying. Do not feed the teacher plan or true geometry at runtime.
3. **Diffusion transformer:** Predict the same four complete TCP-body target poses with a conditional denoising transformer. Match observations, task code, data, split, and update budget to a deterministic head. Measure full-path latency under 300 ms p95 before any live rollout; no claim that an older local diffusion comparison settles transport.
4. **SFP-to-NIC test:** The frozen shared checkpoint's bounded rollout comes first. Later candidate models must report SFP held-out and autonomous behavior separately; SC results do not transfer by assumption.
5. **Task-specific models:** Train an SFP transport actor and an SC transport actor separately on their verified family data, with matched architecture and aggregate update budget against the shared baseline. This tests interference versus lost cross-task transfer. The deployment task selector reads the public task family only.

All supervised labels are actual recorded target commands converted to complete TCP-body poses at the anchor observation. Compare first and fourth target translation/orientation, direction sign for meaningful moves, SC stage/subgoal/handoff, performance by selected port/card count, and task-code ablations. Use the existing complete-episode 83/21 SC and 236/32 SFP split. Evaluate any valid candidate on the same already-open SC development starts and SFP development start, then new development starts only if it passes stable 10 mm handoff, acceptable force, and p95 latency <300 ms. Keep all negative results. A failed offline representation or latency gate is still a completed comparison, not a live success.

## Running record

- Initial GPU audit: only old GPU-0 PID 468750 occupied compute memory; no current AIC transport training. It was left untouched.
- Rootless daemon reports `name=rootless`.
- SFP frozen-checkpoint scene is copied from verified validation episode 23 into `outputs/experiments/2026-10-01_sfp_transport_decisive/scene/` with source and file hashes. The scene is not a reserved final configuration. `scripts/run_clean_transport_eval.sh` now accepts an explicit `AIC_TRANSPORT_EVAL_SCENE_DIR` while preserving its old default.
- Frozen SFP 20 mm-cap rollout completed on GPU 1, 70 simulated seconds, no guide or privileged geometry. Official score: validation 1, task completion 0. Posthoc port-frame distance from the matched verification episode: 184.4 mm at first proposal, 28.8 mm nearest (26.65 s), 80.0 mm terminal (73.4 s); verified teacher last recorded TCP distance was 11.4 mm. These distances use teacher-derived fixed port geometry only **after** rollout for diagnosis. The actor never received it. Live inference p50/p95/p99: 52.2/94.1/132.0 ms. See `outputs/experiments/2026-10-01_sfp_transport_decisive/decisive_metrics.json` and the five-camera `clean_cap20_run/decisive_contact_sheet.jpg`. This is strong evidence that the shared actor also fails SFP transport, even though its previous held-out offline action error looked better than SC. The transport-only controller never calls insertion, so the score alone was not used to make that judgment.
- The new feature cache uses separate observation-only ImageNet ResNet18 vectors for center, left, and right camera images. This removes the failed policy's 256D fused context as a shared bottleneck in the next architecture comparison. The feature cache is frozen for controlled comparisons; live use would encode each new image once and retain cached history vectors.

## Completed controlled training

The cache contains **55,142** causal rows from the same existing verified 268 SFP and 104 SC episodes, sampled at every fourth retained row. Complete-episode validation comprises **32 SFP and 21 SC episodes**. Matched reporting samples 12 rows per SFP episode and 8 per SC episode *per route stage* (384 SFP and 1,008 SC rows). The long-history model takes **17 causal snapshots spanning up to 60 seconds**, with closer spacing near the current decision; it does not process every 20 Hz frame over that interval. All action labels are the recorded teacher's four **complete TCP-body target poses**, not absolute base poses or residual corrections. ImageNet ResNet18 is frozen and shared across three camera views; the heads see separate 512D features per view, normalized ordinary measured state, public 10D task code, prior four-command chunk, measured TCP motion, image ages, and camera clock ages. Offline prior commands are teacher commands; live previous chunks are the actor's proposals. No teacher route stage, privileged port pose, or scene geometry enters inference. `imagenet_features/manifest.json`, each branch's `manifest.json`, `best.pt`, `summary.json`, and `*-command.txt` preserve exact inputs and budgets.

All deterministic heads use a 2-layer, 256D temporal transformer; each has about 1.59M head parameters plus one shared 11.18M ImageNet ResNet18. The conditional diffusion branch adds a 3-layer, 256D action denoiser with 100 training noise levels and 16 deterministic sampling steps; it has about 3.48M head parameters plus ResNet18. Every branch completed its stated update cap; none ended due to a time cutoff. The 1,500-update task-specific pair matches the shared model's 3,000-update **aggregate** budget; the 3,000-update task-specific pair checks whether a full per-task budget reverses the result.

**Training saturation review (October 1):** Neither shared action-only model demonstrated a validation plateau. Both selected the **last available checkpoint, update 3,000**. On the same grouped held-out validation sample, the current-frame head's SC/SFP first-target p95 went from **2.39/3.21 mm at update 2,000** to **2.32/2.31 mm at update 3,000**. The 60 s history head went from **2.91/3.56 mm** to **2.27/2.62 mm** over the same interval. The history head was still improving on both tasks at the cap; this experiment establishes a fixed-budget comparison, not convergence. The route head selected update 2,750 and the diffusion head update 2,250, but their worse errors and noisy late curves do not establish that the action-only heads have saturated. Longer training might improve them, but the live SC divergence cannot be explained or fixed by an offline error trend alone. A bounded continuation would need a fixed held-out stopping rule and matched SC/SFP development rollouts; no continuation was run for this review.

| Frozen vision + head | Updates / selected | SC first target p95 | SFP first target p95 | SC fourth target p95 | SC port-code effect, late median |
|---|---:|---:|---:|---:|---:|
| Current frame, shared action-only | 3,000 / 3,000 | 2.32 mm | 2.31 mm | 3.34 mm | 0.10 mm |
| Up to 60 s history, shared action-only | 3,000 / 3,000 | 2.27 mm | 2.62 mm | 3.05 mm | 0.14 mm |
| Up to 60 s history + route-stage/subgoal/handoff | 3,000 / 2,750 | 3.42 mm | 3.60 mm | 4.28 mm | 0.37 mm |
| Up to 60 s history + diffusion transformer, one sample | 3,000 / 2,250 | 12.34 mm | 12.71 mm | 15.21 mm | 0.11 mm |
| SFP-only, 1,500 updates | 1,500 / 1,500 | — | 3.66 mm | — | — |
| SC-only, 1,500 updates | 1,500 / 1,500 | 3.24 mm | — | 3.96 mm | 0.18 mm |
| SFP-only, 3,000 updates | 3,000 / 2,750 | — | 2.60 mm | — | — |
| SC-only, 3,000 updates | 3,000 / 3,000 | 2.71 mm | — | 3.10 mm | 0.19 mm |

Source: `matched_heldout.json`. All SC late code effects are the median difference in **first** 50 ms target at 90% of held-out teacher episodes when only the selected-port one-hot code is swapped; they are not endpoint separation. The history gain is small offline and task-specific models do not beat the shared head on this sample. SFP validation is heavily skewed toward card 0/port 1, so the SFP numbers do not establish broad card/port generalization.

The route branch predicts SC stage at 92.3% on matched rows, but the *complete next subgoal* has **44.46 mm p95** translation and **10.23° p95** orientation error, beyond the 20 mm route gate. Its stage 1 and stage 2 subgoal p95 errors are 94 and 120 mm; stage 3 and 4 are 46 and 59 mm. Handoff event precision/recall is 0.55/1.0 on 22 held-out positive rows, so it falsely calls handoff early. No stage skips occurred on the sampled rows. The action-only arms' stage/subgoal heads were deliberately untrained; their numeric stage/subgoal outputs in machine summaries must be ignored. The route branch was therefore not deployed.

The diffusion result is not explained just by sampling different plausible routes. Four-sample spread is 2.25 mm median for SFP and 2.48 mm for SC; even an **oracle** best-of-four choice using the hidden teacher label has 9.89/8.45 mm p95 SFP/SC first-target error, far above the deterministic head. An average of four samples has 12.54/11.24 mm p95. The oracle choice is diagnostic only and could not be used autonomously. See `diffusion_sampling.json`. The first diffusion attempt failed during validation due to a comparison tensor slice of shape `[batch,3,3]` versus `[batch,4,3]`; the failed artifact is preserved as `diffusion_failed_shape/`, and the corrected run completed all 3,000 updates.

Local latency with three image preprocessors, a fresh ImageNet encoding, cached temporal features, and the complete head (60 warmed repetitions on an A6000) is p95 **19.4/19.3/18.1/39.9 ms** for no-history/history/route/diffusion. This excludes ROS transport and scheduling. The actual live p95 numbers below are the deployment-relevant check. See `latency_local.json`.

## Bounded autonomous development rollouts

The original frozen shared actor and both new action-only heads were run with observation-only inputs, 70 simulated seconds, and the same already-open SFP validation scene 23. The new temporal runtime caches image features causally and resets history by episode. The first attempted no-history run exposed a live 9D-versus-36D prior-action shape bug and returned before a valid comparison; it is retained as `live_sfp_no_history/`. The fix supplies the previous **four-action proposal chunk**, matching the offline history field, and the corrected runs completed. All rollouts used a 20 mm per-component translation cap, no local insertion actor, no recovery, no oracle geometry, and no final scene.

For SFP, the selected opening is about 56 mm ahead of the TCP at the teacher's **pre-insertion standoff**. Therefore distance to the opening alone incorrectly calls that standoff a transport failure. We compare terminal TCP pose to the **recorded teacher standoff at 20.0 s** (port-frame TCP xyz about -1.1, +0.6, -55.6 mm). Fixed port geometry is used only after the rollout for measurement.

| SFP actor, same scene | Terminal standoff position | Lateral / axial error | Orientation error | Official insertion | Live p95 | Contact result |
|---|---:|---:|---:|---:|---:|---|
| Original frozen shared actor | 129.9 mm | 47.2 / 121.0 mm | 8.4° | 0 | 94.1 ms | No scored contact |
| New current-frame head | 315.0 mm | 269.6 / 162.9 mm | 18.8° | 0 | 98.1 ms | NIC mount collision, -24 contact points |
| New 60 s history head | **8.0 mm** | **8.0 / 0.05 mm** | **6.9°** | 0 | 98.1 ms | No scored contact |
| SFP-specific 60 s head, 3,000 updates | 34.1 mm | 21.6 / 26.4 mm | 4.9° | 0 | 99.0 ms | No scored contact |

The history head reaches the axial pre-insertion plane and remains near it. It misses the lateral target by about 8 mm and orientation by 6.9°, so a handoff cannot yet be declared safe under the existing tight pose envelope. It also cannot insert: this is deliberately a transport-only runtime, and the actor eventually commanded motion without measurable further progress while wrist force was about 13–16 N. This one-scene result supports **keeping history as a candidate for SFP transport**, not a general insertion claim. See `live_sfp_*/port_distance_metrics.json`, official score files, runtime traces, and five-camera 5 fps MP4s.

To check transfer, both new heads were also run from the same already-open **SC port-0 five-card** development start with a 60 mm action cap. The teacher plan is a posthoc trajectory reference; it was never an actor input.

| SC actor, same scene | Nearest planned standoff | Terminal standoff | Terminal orientation | Official insertion | Live p95 | Scored contact |
|---|---:|---:|---:|---:|---:|---|
| New current-frame head | 21.5 mm | 84.8 mm | 4.3° | 0 | 160.6 ms | None |
| New 60 s history head | 117.0 mm | 158.0 mm | 20.8° | 0 | 186.5 ms | None |
| SC-specific 60 s head, 3,000 updates | 128.5 mm | 181.9 mm | 20.1° | 0 | 164.3 ms | None |

The shared heads initially follow much of the around-card route, then deviate during the late return/tilt portion. The shared long-history actor descends far below the planned standoff; the five-view video and `measured_transport_path.png` show the trajectory. The SC-specific head deviates still farther. This is a **negative SC transfer and task-specific result** despite the positive SFP shared-history standoff result. It is consistent with the route branch's large stage 1–4 subgoal errors. None passed the existing 10 mm standoff gate or autonomous insertion. Live p95 remained under 300 ms. See `live_sc_*/autonomous_summary.json`, official score files, traces, plots, and five-camera 5 fps MP4s.

## Decision

All five requested experiments have been executed, including **matched live tests of both fully trained task-specific heads**. Longer history is retained as a **candidate SFP transport input** because of the controlled one-scene standoff improvement; it is not promoted as a shared actor because SC got worse. The SFP-specific head missed the standoff by 34 mm versus 8 mm for shared history; the SC-specific head missed by 182 mm versus 158 mm for shared history. Explicit route heads, this diffusion implementation, and task-specific heads do not currently justify promotion on these data and budgets. The next targeted repair is to identify the first late-stage SC divergence from the around-card teacher and train an observation-only, task-conditioned route-progress/target representation that passes a held-out **20 mm next-subgoal** gate, then repeat the same SC development start and a new development start. For SFP, a separate orientation and lateral handoff correction should be tested before calling the local insertion actor. No routine new collection, RL, or final-scene evaluation follows from this result.

# Dreamer-v4 for AIC: reviewed proposal

Status: approved and implementation started2026-09-18 at15:52:56UTC. The user confirmed all three recommended choices. The separate6h50m pilot ends22:42:56UTC and uses only physical GPUs2–4. ACT continues separately on GPUs0–1. Pilot records: `outputs/experiments/2026-09-18_dreamer60_pilot/budget.json`. The findings below describe the original read-only review; completed implementation and training stages will be recorded separately.

Reviewed 2026-09-18; repository commit `1163696c628cf53ff230487b7244206cbe3fe909`, cloned to `/tmp/aic_dreamer_review_20260918`.

## Confirmed implementation choices

1. Is the 300 ms limit measured from an already available synchronized observation to the outgoing robot command, on one A6000, including resize, GPU transfer, state processing, model, and command conversion? Recommended: report median/p95/p99/max, require p95 below 300 ms, and aim below 150 ms so the existing four-action/200 ms schedule does not stall. Report sensor age and ROS/simulator delay separately. An absolute worst-case guarantee under a shared server cannot be established by a benchmark.
2. Is a separate 6h50m pilot, starting after approval and including adaptation, training, and evaluation, acceptable? This is an engineering and feasibility budget, not a convergence guarantee. ACT retains its already approved deadline.
3. Start strictly with the same 60 aligned SFP training episodes / 14 validation episodes? Recommended yes for the first comparison. Later, permission to use correctly labeled failures for world/reward learning would be useful, while keeping them outside the verified expert BC dataset. This later expansion must be a separately labeled comparison.

## Findings from the actual implementation

The project contains the expected stages: causal image tokenizer, action-conditioned shortcut dynamics, supervised policy/reward/continuation heads, then policy/value optimization in imagined rollouts. It has executable training, caching, evaluation, checkpoint, and controller entry points. It is a prototype rather than a pretrained robotics model.

Source: [README at reviewed commit](https://github.com/yoonjung0705/dreamer-v4/blob/1163696c628cf53ff230487b7244206cbe3fe909/README.md).

The official paper describes the same broad stages, axial space/time attention, sparse temporal layers, grouped query attention, and freezing the dynamics for policy/value training in imagination. Its reported large setup is 2B parameters (400M tokenizer + 1.6B dynamics), trained on 256–1024 TPU-v5p; that scale is unnecessary for this pilot. The proposed robotics variant is a much smaller adaptation, not a reproduction of those results. [Primary paper, methods and experiments](https://arxiv.org/html/2509.24527v1).

### Checks actually performed

- Both supplied CPU integration tests passed on the existing PyTorch 2.9 environment, taking 1.62 seconds. They exercise tiny synthetic 16x16 data, stage updates, cache equivalence, checkpoint resume, and control. They do not establish learned accuracy or GPU correctness.
- Counted model parameters directly by constructing the models on CPU.
- Confirmed a deterministic-control inconsistency: after resetting the same model and giving the same image, `deterministic=True` yielded different continuous actions. Maximum difference in a tiny random model was 0.000214 normalized action units. The cause is unconditionally sampled noise in `WorldStream.commit`, while the flag controls only the policy distribution. This is a reproducibility issue; its effect in a trained model is not measured.
- No GPU allocation, latency test, learning experiment, dependency installation, or source modification was performed.

### Required changes and implementation risks

| Finding | Required action |
|---|---|
| Dataset/controller accept a single square RGB image; no robot state | Add three camera views, normalized state, shared preprocessing, and a state prediction path for imagination. |
| Default actions are one five-way categorical component | Configure continuous full pose targets and test normalization/round trips. No ACT residual adapter. |
| Task is an integer embedding | Add a projection of the existing 10D AIC task vector; keep it in the policy/reward agent token and preserve the mask preventing task tokens from directly driving world predictions. |
| `Controller.act` assumes each call is one frame/action transition | Define decision timing explicitly. Do not downsample frames while recording only the final command from a four-command interval. |
| Continuous standard deviation is hardcoded to 0.1–1 normalized units | Make bounds configurable and calibrated to physical translation/rotation scales. These exploration scales can be far too wide for insertion. |
| Continuous samples use `tanh`, and BC log probability clamps targets before `atanh` | Do not directly reuse ACT's unbounded mean/std action normalization. Fit a shared bounded physical-action transform on training teacher/executed streams with margin, check round trips and held-out support, and never silently clip demonstration labels. Alternatively, explicitly implement and test an unsquashed distribution. |
| "Deterministic" controller still samples context noise | Propagate a reproducible context-noise mode through reset/control; compare the chosen acting features with training features. |
| CUDA auto backend invokes compiled FlexAttention; only CPU tests exist | Validate BF16, continuous actions, cached GPU attention, and startup/steady-state timing before training. SDPA with softcap=0 is an explicit fallback ablation, not an equivalent silent replacement. |
| Training has no DDP, held-out loop, early stopping, or deadline | Add stage-aware DDP, independent rank sampling, synchronized loss RMS statistics, held-out metrics, deadline handling, and immutable best checkpoints. |
| Model checkpoint includes decoder and reward/value heads during acting | Export an encoder + dynamics feature extractor + policy only module. Do not run video generation or planning during control. |
| Controller image checks synchronize the GPU, and there is no inference autocast | Validate CPU inputs before transfer, use inference mode/BF16 where verified, warm fixed shapes/caches, and profile end to end. |
| Success-only trajectories poorly constrain failure outcomes | Treat imagination as conditional on held-out fidelity; do not interpret high imagined reward as insertion success. |

Specific sources: [controller and imagined rollout](https://github.com/yoonjung0705/dreamer-v4/blob/1163696c628cf53ff230487b7244206cbe3fe909/dreamer4/rollout.py#L115), [context noise](https://github.com/yoonjung0705/dreamer-v4/blob/1163696c628cf53ff230487b7244206cbe3fe909/dreamer4/rollout.py#L16), [continuous action distribution](https://github.com/yoonjung0705/dreamer-v4/blob/1163696c628cf53ff230487b7244206cbe3fe909/dreamer4/actions.py#L74), [training stages](https://github.com/yoonjung0705/dreamer-v4/blob/1163696c628cf53ff230487b7244206cbe3fe909/dreamer4/train.py#L28), [attention implementation](https://github.com/yoonjung0705/dreamer-v4/blob/1163696c628cf53ff230487b7244206cbe3fe909/dreamer4/transformer.py#L36).

## Proposed compact AIC model

### Observations and target

- Pilot scope: SFP, one NIC, card 0 / port 1, using exactly the 60/14 aligned episode split from the previous final training stage. Preserve provenance and frame timestamps.
- Use all three RGB views. Start with 144x144 letterboxing: preserve aspect ratio, resize 256x288 to 128x144, then pad vertically. No three-camera collage that makes each connector tiny. Shared tokenizer weights process each camera as a separate batch element, with distinct cache entries and camera identity embeddings when fusing latents.
- Treat that resolution as provisional. The concurrent ACT audit found only about 0.85–1.11 input pixels/mm at one entrance stall at the original 288x256 resolution. Halving it may discard useful alignment detail. Inspect connector/port reconstructions before accepting the tokenizer; retain a higher resolution or add an observed-image contact crop if the detail is lost, then repeat the latency benchmark. The proposed parameter counts below are for the 144-image base configuration.
- Preserve the canonical 32 robot-state entries, mask controller-error entries 13:19 as in the best ACT recipe, append elapsed simulation time, and supply the existing 10D task vector. Quaternion handling and action conversion must reuse the verified ACT conventions.
- Predict complete absolute TCP targets: xyz and three rotation-vector coordinates. Any conversion to the controller's relative command protocol occurs at the runtime boundary. This is not a residual on an ACT prediction.

### Timing: four commands per decision

The current ACT replans after four 20 Hz commands, so its decision interval is 200 ms. A 300 ms action latency would already miss that cadence. The recommended pilot represents one world-model transition as those four commands and the observation 200 ms later:

`observation(t), [pose(t), pose(t+50ms), pose(t+100ms), pose(t+150ms)] -> observation(t+200ms)`.

- Continuous action dimension becomes 24: four six-dimensional absolute poses.
- Policy MTP offsets 0 and 1 predict two four-command blocks (eight micro-actions), matching the ACT prediction horizon; execute the first block and reobserve.
- The world model operates at 5 Hz decision frequency with context 8 (1.6 seconds). Use ordinary 8-frame batches and occasional 16-frame batches. The raw-data audit below shows longer complete sequences are too scarce for the initial proposal.
- Preserve each actual command within the block. Keep actual executed commands for dynamics and clean teacher commands for BC as separate arrays. Build transitions from original timestamps; exclude missing-command intervals. Sum verified transition rewards over the block and propagate termination correctly. Do not invent missing substeps or interpolate invalid labels.
- Require a practical p95 target below 150 ms and p99 below 200 ms for this cadence, while reporting the user's 300 ms criterion separately. If missed, optimize or shrink before changing the comparison's controller rate.
- Alternative only if measured fast enough: single-command 20 Hz control with a 50 ms deadline; this would change the temporal comparison and must be labeled.

### Joint visual and robot-state prediction

A world model with a policy that sees true robot state cannot then imagine with that state frozen. It needs to predict state as part of each transition.

Recommended modest extension:

- Shared per-camera tokenizer: width 192, depth 8, 4 query / 2 KV heads, 16 latent tokens of width 32 per camera.
- Pack adjacent latent pairs into 64D tokens: three cameras produce 24 visual tokens total.
- Append one token containing normalized robot state, padded to 64D with an explicit valid-dimension loss mask. Masked controller-error entries stay zero. Elapsed time advances deterministically during imagination rather than being freely predicted.
- Dynamics: width 384, depth 12, 6 query / 2 KV heads, temporal attention every fourth layer. Predict visual and state tokens jointly with separate normalized loss statistics so background pixels cannot dominate contact-state errors.
- Task and elapsed-time projection enters the agent token. The world does not receive desired target identity as a way to hallucinate a successful future; its physical prediction remains conditioned on the actual observation history and actions.
- Reuse the BC, reward, continuation, and value heads. Start imagined horizon at eight decision steps (1.6 seconds), four shortcut denoising updates, and strong regularization to the frozen BC policy. Only reduce sampling to two updates after measuring prediction quality.

This needs changes to config, models, data, losses, caches, controller, and training. The existing overall architecture is reusable, but robotics integration is more than editing a config file.

## Size and efficiency

Counts below are CPU measurements of the unmodified base architectures. Multi-camera sharing does not triple parameter count; additional camera/state/task code will add a small amount that must be measured after implementation.

| Configuration | Total parameters | Potential control-only export |
|---|---:|---:|
| Existing ACT | 16.35M | Existing saved model |
| Dreamer repository default, 64x64 single image | 13.03M | 10.42M |
| Proposed base, width384/depth12, 144 image, 24D actions/MTP2 | 27.124M | 22.562M |
| Smaller base, width384/depth8 | 20.829M | 16.267M |
| Larger optional base, width512/depth12 | 42.331M | 36.974M |

The proposed base is about 66% larger than ACT in total parameters; its control-only export is about 38% larger. FP32 weight storage is approximately 108.5 MB total / 90.2 MB control-only, before serialization overhead. These are not measured GPU memory requirements. The control-only counts exclude tokenizer decoder and unused heads; actual code currently still constructs those modules, so export stripping is required.

No need for the larger model initially. Downsizing priority: reduce resolution only if contact details survive; pack spatial latent tokens; reduce dynamics depth; reduce context only after verifying contact memory. Reducing denoising steps speeds imagination training, but does not speed the proposed acting path because acting performs no denoising rollout.

Efficiency measures:

1. BF16 after numerical tests, fused attention after cache-equivalence tests, no gradients through frozen stages.
2. Train the shared tokenizer once; then freeze and cache all training/validation latents separately with a tokenizer fingerprint. Cached latent batches remove repeated camera encoding during dynamics/agent training.
3. Memory-mapped contiguous sequences, background loading/pinned memory, and train-only normalization.
4. Tune microbatch by measured throughput and memory. Use data parallelism for trainable stages; model sharding is unnecessary at this size.
5. Use GPU 2 and 3 for world stages; GPU 4 for the fresh ACT60 comparison, validation, latency, or rollout evaluation. If the third device gives better useful throughput in DDP, switch after the baseline completes. These three are the world-comparison allocation, separate from ACT's GPUs 0 and 1. Check availability and never interrupt another user's processes.
6. Release training devices for parallel final simulator evaluation. Exactly one inference GPU is used in latency measurements.

## Data and rewards

## Additional data audit: teacher labels are not always executed actions

The selected 60/14 split is usable for expert supervision, but its `actions.npy` must not be used as the causal dynamics action sequence. I inspected every raw `frames.jsonl` row in these 74 episodes and compared `teacher_target_pose` with `executed_target_pose`.

| Split/type | Episodes | Raw observations | Held/resampled cache rows | Executed pose differs from teacher |
|---|---:|---:|---:|---:|
| Training, nominal including no-intervention runs | 48 | 20,927 | 26,531 | 0 |
| Training, student corrective | 12 | 5,280 | 7,245 | 1,760 |
| Validation, nominal including no-intervention runs | 13 | 5,538 | 7,173 | 0 |
| Validation, student corrective | 1 | 438 | 626 | 148 |
| Total | 74 | 32,183 | 41,575 | 1,908 |

Every raw row has both pose fields. Differences reach 30 mm translation and 4.58 degrees rotation. The recorded `action` is the clean teacher target converted into a TCP-relative command at that observation. During intervention, it is deliberately not the command that produced the next observation. That is appropriate for corrective BC labels, but wrong for causal transition training.

The Dreamer adapter therefore needs separate arrays:

- `executed_action`: full absolute command actually sent, reconstructed from `executed_target_pose`; used for incoming-action context, shortcut dynamics, and recorded reward/transition prediction.
- `teacher_action`: desired corrective command reconstructed from `teacher_target_pose`; used for BC targets only.
- Separate validity masks, command indices, original observation timestamps, and any controller acknowledgement timestamps available. Preserve physical normalization and quaternion conventions separately from the choice of stream.
- In imagination, the policy's sampled action becomes the executed action for the simulated transition; never substitute an unavailable teacher action.

The unmodified Dreamer code shares `batch['actions']` between incoming dynamics conditioning and supervised BC targets (`train.py:37` and `objectives.py:143`). This split is a required code change, not just new metadata.

### Continuous action support

The policy's continuous actions are squashed to `[-1, 1]`; its BC log-probability
code clamps input labels to that range before applying `atanh`. A read-only CPU
audit reconstructed raw teacher/executed absolute targets using the ACT
X-positive quaternion convention, then applied the previous selected ACT's
saved mean/std normalizer. At least one component lay outside `[-1, 1]` in
**90.22% of the 26,207 training teacher frames**, 90.62% of training executed
frames, and 97.51% of the 5,976 validation frames in either stream.

Those labels are valid physical targets; the proposed direct reuse of their
standardization is incompatible with this bounded distribution. Before any
world-model training, use one invertible physical-action mapping shared by
teacher and executed streams, fitted from training data with explicit margin.
Report held-out out-of-range values and reject silent clipping. The alternative
is a deliberately unsquashed continuous head, which changes the distribution
and its likelihood/KL implementation. The audit created no new model or
normalizer. Evidence: `dreamer_action_range_audit.json` and
`audit_dreamer_action_range.py` in the active September 18 experiment.

### Timestamp and missing-command issues

The original collector observes before issuing each target and records only when the camera timestamp advances by at least 50 ms. `command_count` advances even if a row is not recorded. The existing ACT cache uses causal holds on a synthetic 20 Hz grid. It is not a native, fully observed 20 Hz dynamics dataset.

Across the selected episodes:

- Training: 219 intervals skip an intervening command index; validation: 45. The skipped commands must be recovered from a real command log or their affected transitions excluded from world training. They cannot be assumed equal to the previous command.
- Training: five observation gaps exceed 250 ms; validation: two. Maximum gaps are 2.95 s and 3.4 s. Do not interpret held images/states through these gaps as real stationary dynamics.
- Source timestamps are strictly increasing. All native differences examined align with the 50 ms simulation grid to numerical precision.

Use original rows to build 200 ms world transitions only when both endpoint observations exist and the intervening command sequence is complete. Piecewise holding a verified command until the next actual command is valid; inventing intermediate image/state observations is not. Split sequences at missing commands or excessive observation gaps. Add a test using a student-corrective interval to prove swapping teacher and executed streams changes the dynamics input but not the BC target.

A stricter window audit across all four possible 50 ms phase offsets shows why the initial 24/32-frame training proposal must be shortened:

| Observed 5 Hz frames per sequence | Training windows / episodes represented | Validation windows / episodes represented |
|---:|---:|---:|
| 8 | 4,525 / 60 | 954 / 14 |
| 12 | 1,878 / 59 | 379 / 14 |
| 16 | 788 / 55 | 166 / 11 |
| 24 | 165 / 20 | 38 / 6 |

These are overlapping eligible windows, not independent new episodes. Use **context 8**, ordinary **8-frame batches**, and occasional **16-frame batches** for this pilot. This keeps every training episode represented while sometimes training beyond the context length. Report dynamics transition/window counts separately from the unchanged 60/14 BC episode counts. Shorten imagination horizons if contact predictions deteriorate beyond the supported context.

Persistent audit artifact: `outputs/experiments/2026-09-18_act_all_verified_6h50/aic_dreamer_executed_action_audit_20260918.json`; reproduction script: `aic_dreamer_executed_action_audit.py` in the same directory (original working copies were under `/tmp`). The artifact includes every episode path, counts, examples, command gaps, and the sequence-length sweep. Source collector: `aic_example_policies/aic_example_policies/ros/CollectCorrectiveCheatCode.py:162`; existing resampler: `scripts/merge_corrective_act_cache.py:107`.


The strict pilot uses the same 60 training episodes and 14 held-out episodes throughout tokenizer, dynamics, and agent stages, with per-transition dynamics masks described above. Do not pretrain the tokenizer on held-out episodes. Do not use test scenes for checkpoint selection. All phases retain the episode/scene split.

The prior best ACT's final stage had 60 training episodes, but its inherited weights saw a broader collection. It is a useful historical reference, not an equal-data baseline. Train a fresh ACT on the same 60/14 split with the best current recipe for the primary comparison. Its ImageNet initialization is an explicit difference from the initially random Dreamer tokenizer.

The primary comparison is between practical complete systems: the proposed
Dreamer image resolution and ACT's original 288x256 input also differ. Report
both differences; do not attribute an outcome solely to the world-model
architecture. A secondary resolution-matched ACT check is useful only if the
budget permits. Use the same fixed simulation duration for live comparisons,
with a separate wall watchdog and explicit incomplete-run reporting; a common
wall limit alone gives different physical durations under shared GPU load.

Reward construction must be audited:

- Use recorded official full-insertion success as terminal ground truth.
- Use real target-relative geometry for dense shaping only if correctly synchronized and verified in these episodes. Privileged geometry may provide training labels, never policy observations.
- Do not reuse the fallback in AIC `offline_rewards.py` that treats controller TCP tracking error as task distance, or its <=1 cm distance test as full insertion. Low controller error can occur away from the port.
- If dense labels are unavailable, use sparse verified terminal reward and mark the limitation. Inspect reward predictions on pre-insertion, contact, and terminal frames; terminal score alone gives weak information about failures.
- Later, verified failed transitions could help learn dynamics/rewards. Keep them outside `expert_verified`, never use their commands as successful BC targets, and report that as an expanded-data experiment. This is not part of the strict first60 comparison unless approved.

## Proposed budget after approval: 6h50m total

These are caps and decision points, not guarantees that every learning phase will converge.

| Wall time | Work |
|---|---|
| 0:00–1:20 | Adapt data/state/actions/task interface, add meaningful continuous-action/cache/DDP tests, validate preprocessing, untrained latency/throughput benchmark. Stop and report if integration cannot be trusted. |
| 1:20–2:10 | Train tokenizer; inspect held-out connector/port reconstruction, freeze and cache. Train fresh ACT60 on GPU4 in parallel once data conversion is ready. |
| 2:10–3:40 | Train dynamics with joint visual/state prediction; compare held-out rollouts to persistence and shuffled-action baselines. |
| 3:40–4:40 | Train policy/reward/continuation with BC plus dynamics preservation; retain several checkpoints and evaluate development scenes. |
| 4:40–5:20 | Conditional short imagination stage, only if predictive/reward quality is credible. Otherwise spend this time diagnosing or improving BC and report that RL was not justified. |
| 5:20–6:35 | Freeze models; paired fresh-scene evaluation of fresh ACT60, world+BC, and world+BC+imagination if eligible; latency profiling and videos. Parallel simulator runs use released world GPUs. |
| 6:35–6:50 | Save final artifacts, distributions/metrics, resource cleanup, report. |

Stopping criteria: held-out predictive or BC metrics have no meaningful improvement across several checkpoints and closed-loop screens do not improve, or the phase cap expires. Record which reason applies. A failed fidelity gate is a result, not a reason to spend the remaining budget maximizing imagined reward.

## Evaluation and evidence required

- Before scale-up, overfit a very small set to verify learning, inspect reconstruction at the contact region, test action sensitivity, and verify sequence/reset alignment.
- Held-out prediction: one/four/eight decision-step robot pose and wrench errors; contact-region video reconstruction; action-conditioned versus shuffled-action predictions; persistence baseline. Pixel MSE alone is insufficient in static scenes.
- Policy: physical translation and rotation errors, terminal-region errors, action smoothness, and latency. Compare world+BC before any imagination stage with the fresh ACT60 baseline.
- Imagination: require actual simulator improvements over world+BC. A better learned reward is not evidence of task success. Preserve the pre-imagination checkpoint if imagination regresses.
- Frozen final comparison: same fresh scene seeds and runtime limits, 20 episodes per eligible model, official insertion success plus total score, and paired outcomes. Twenty episodes are a pilot, not high-confidence certification of reliability.
- Save real rollout videos and approximately one-second snapshots; separately label decoded imagined videos so they cannot be mistaken for successful real simulator trials.
- Latency: batch1, one A6000, 100 warm-up calls then at least1000 timed decisions at varied history lengths. Include preprocessing/transfers/command conversion and report cold-start/compile time separately; benchmark at the expected concurrent training load as well as alone. Preserve p50/p95/p99/max, deadline misses, model size, peak VRAM, and exact device/software identifiers.

## Artifacts proposed after confirmation

- New run directory such as `outputs/experiments/2026-09-18_dreamer60_pilot/`, with exact source commit, config, split manifest, normalizers, stage ancestry, latencies, logs, and videos.
- AIC adapter in a dedicated module; reuse common observation/action/task definitions rather than forking them silently.
- Experiment entry in `docs/EXPERIMENTS.md` and a detailed report under `docs/experiments/`; link data provenance in `docs/DATASETS.md`.
- No claim of a trained Dreamer model until each completed stage and simulator result is explicitly recorded.

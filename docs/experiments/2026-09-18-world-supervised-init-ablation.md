# Matched supervised comparison: world initialization

**Status: complete. Training, offline audit and all eight development rollouts finished before the 22:42:56 UTC deadline.** This follow-up started on 2026-09-18 at 22:02:32 UTC. The original pilot deadline remains **22:42:56 UTC**. Both trainers finished by 22:21, followed by the new development set. The earlier frozen 20-scene final results and model selection remain unchanged.

## Question and fixed comparison

Does the selected corrected world checkpoint help supervised control compared with a randomly initialized dynamics transformer of exactly the same architecture?

Both arms share:

- The frozen six-view tokenizer, strict 60 training / 14 validation episodes, native labels and verified executed-action history.
- The same fresh policy heads, untrained task projection, model dimensions, observation-relative TCP/body delta action contract, and four-command control cadence.
- Seed 17, episode-balanced sampling, terminal-label masks, AdamW at 0.00015 with weight decay 0.01, 100-update warmup, and the same cosine schedule.
- **2,500 supervised updates with batch 256**, on one A6000 each. The earlier pilot used batch 512; the two new arms are the controlled comparison. A real four-frame backward/optimizer preflight used 27.84 GiB, so batch 512 on one GPU was rejected before training.
- The existing masked BC loss plus its auxiliary dynamics loss. Reward, continuation supervision and imagination remain disabled.

The pretrained arm starts the dynamics parameters from corrected world step 1,000. The random arm starts those parameters randomly. Every differing tensor was checked to belong to the dynamics component that could have been trained by world pretraining; tokenizer, policy heads and the world stage's untrained task projection are identical. Both use fresh optimizers. The pretrained arm has the additional earlier world-training compute; this is a matched **supervised** budget.

The initialization and exact commands were frozen before training in [plan.json](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/plan.json). The plan pins every Dreamer source file, data/split metadata, initialization checkpoints and component fingerprints.

## Selection and independent reporting

The checkpoint rule is unchanged: at fixed 500-update checks, score the mean of ordinary held-out first-command translation error and final-observation first-command error across all 14 validation episodes. The existing trainer saves a new best when that score improves by more than 0.1%; otherwise it retains the earlier checkpoint. Development outcomes do not select weights.

After selection, a separately declared descriptive audit evaluates every one of the 5,976 held-out native first-command labels. It uses a preceding verified 200 ms edge where available, actual executed command history, and equal episode weighting. It reports all observations and the last three recorded seconds separately against a zero-command baseline. The last-three-second label is a recording-time proxy, not a contact or inserted-state label. This audit does not change selection.

## Live check and limits

Four fresh one-NIC / SFP development scenes were generated and checked against all 74 expert episodes, the earlier four development scenes and the frozen 20 final scenes. Their [manifest](../../outputs/experiments/2026-09-18_world_followup/new_development_scenes/manifest.json) is fixed before evaluation. Each arm uses a fresh simulator per scene, 90 simulated seconds, an initial-state audit, ground truth disabled, and the unchanged worker-based runtime. Rendering and policy inference use separate GPUs. Actual observation-to-command latency was recorded.

The new four-scene set is a small development check. BC sampling and optimizer seeds match, but the random world is a separate fixed random draw, rather than the exact ancestral random weights from which the selected world was pretrained. The [initialization variance audit](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/world_initialization_variance_audit.json) records this and preserves that ancestral checkpoint for a stronger repeat. One such pair cannot isolate a causal pretraining effect from all initialization variance; a repeat should reuse the exact ancestral weights and multiple seeds. Offline command error does not establish reliable insertion. All eight rollouts completed with valid initial-state, task-identity and full-duration audits.

## Results

Both arms completed **exactly 2,500 updates**: pretrained in 1,101.03 seconds and random in 1,052.63 seconds. They stopped at the fixed update limit, before the deadline; this does not establish convergence. The fixed rule selected pretrained step **1,000** and random step **500**. Later checkpoints had worse selection scores, so they were not substituted.

| Held-out metric | World initialized | Random world |
|---|---:|---:|
| Fixed selection score, mm | 2.153 | 2.352 |
| Fixed ordinary first command, mm | 1.828 | 2.277 |
| Final observed first command, mm; 14 episodes | 2.478 | 2.428 |
| All 5,976 native first commands, episode-balanced mm | 2.285 | 2.578 |
| All-native rotation-vector error, degrees | 0.115 | 0.158 |
| Last 3 recorded seconds, episode-balanced mm | 2.486 | 2.734 |
| Last-3-second rotation-vector error, degrees | 0.064 | 0.118 |

The zero-command baseline on the exhaustive audit is 14.545 mm overall and 14.806 mm in the last three seconds. These subsets differ from the **single final observation** metric: the terminal command result slightly favors random initialization. The pretrained initialization lowers the all-native mean by 0.293 mm in this seed; this offline difference did not produce reliable insertion in the live comparison below.

Both models have **27,130,942 total parameters**, with **22,569,344 used for control** (90.28 MB of FP32 weights). Runtime, action cadence and architecture are unchanged between arms. Live latency below measures through actual ROS command publication, including preprocessing, transfer, worker IPC and command conversion. It excludes sensor capture and transport before observation processing starts.

The [offline report](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/offline_comparison.json) records the fixed and exhaustive metrics. [Training curves](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/validation_comparison.png) show all scheduled checks after step 500. Persistent artifacts include both initialization checkpoints, selected weights, resumable final optimizer checkpoints, exports, commands, source/data hashes, validation records and per-episode predictions: [archive manifest](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/archive_manifest.json), 1.15 GB.

### New development results

All four scenes per arm passed the initial-state, task-identity and full **90 simulated second** gates. Both models achieved **0/4 full insertions**. The pretrained arm achieved one official partial insertion, on scene 1; the random arm achieved none.

| Scene | World initialized score | Random world score |
|---|---:|---:|
| 1 | 49.788, partial insertion | 19.008 |
| 2 | 35.691 | 36.684 |
| 3 | 36.674 | 36.368 |
| 4 | 9.265 | 12.906 |
| **Mean** | **32.854** | **26.242** |

| Actual observation-to-command latency | World initialized | Random world |
|---|---:|---:|
| Decisions | 1,789 | 1,786 |
| p95 | 76.26 ms | 79.69 ms |
| Maximum | 146.79 ms | 135.71 ms |
| Calls at or above 300 ms | 0 | 0 |

Both meet the latency requirement, including the practical p95 target below 150 ms. The score advantage comes mainly from the first scene's partial insertion; the remaining three scenes are mixed. The all-camera [pretrained terminal sheet](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/pretrained_world_scene1_terminal.jpg) and [random terminal sheet](../../outputs/experiments/2026-09-18_dreamer60_pilot/supervised_world_init_comparison/random_world_scene1_terminal.jpg) were visually reviewed: the former stays near the opening, while the latter finishes above it. The images do not establish contact forces or precise insertion depth.

The persistent [paired summary](../../outputs/experiments/2026-09-18_world_followup/supervised_world_development_archive/paired_summary.json) and [archive manifest](../../outputs/experiments/2026-09-18_world_followup/supervised_world_development_archive/archive_manifest.json) retain every valid adverse outcome, official score, command/source identity, log, 1 fps video and terminal sheet. Earlier final-scene results did not enter checkpoint selection or scene generation for this comparison.

## Decision

World initialization modestly improves held-out command error in this pair, and its development mean score is higher. **Neither model inserts reliably.** This small development set, one BC seed, a different ancestral random draw, and asynchronous simulator variability prevent a general causal claim about pretraining.

Keep imagination disabled: the earlier dynamics and reward-data gates remain failed. A stronger initialization test should start the random arm from the exact archived ancestral checkpoint, then repeat matched BC seeds and fresh scenes. Before another broad training sweep, resolve the observed near-port alignment/contact failures and improve synchronized near-contact and post-insertion data. The successful latency result leaves room for better visual detail or a contact-aware supervised controller, but does not justify a larger generative model by itself.

# 2026-09-17: direct visual actor and bounded model checks

Status: **implementation checks passed; insertion performance unmeasured**.
Branch: `feat/hybrid-train`, base commit `7534090` plus local changes. This
followed the [evaluation/accounting repairs](2026-09-17-evaluation-curriculum-fixes.md)
and the user's authorization for bounded model runs. No live Gazebo or Isaac
rollout was started. At most **two GPUs** were used concurrently (physical 0
and 1); all runs finished. No sudo was used.

## Question and changes

Can a policy use visual backbone features directly, initialize those features
from ACT if useful, and train without ACT action proposals or residual limits?
This implements the direction in the
[June 17 notes](https://github.com/yoonjung0705/tutorials/blob/master/python/libraries/tutorial_aic.md#617-2026-hybrid-train-branch-model-training-especially-for-isaac).

- The [shared actor](../../aic_utils/lerobot_robot_aic/lerobot_robot_aic/direct_visual_actor.py)
  consumes three camera feature grids and normalized state, predicting the
  complete six-coordinate command. Its backbone trains by default. ACT can
  supply ResNet visual weights only; its action head/decoder is never called.
- Backbone choices are scratch/ACT-initialized ResNet18, ImageNet ResNet18,
  DINOv2 ViT-S/14, and a small convolutional test model. Checkpoints save the
  actor weights, normalization, camera order, action limits, and configuration.
- Offline training defaults to `direct_visual`; Isaac and Gazebo loaders
  accept the same checkpoint without an ACT export. Legacy actor modes remain
  explicit so old experiment artifacts can still be inspected.
- Direct RL uses a one-step action horizon, matching offline transitions.
  State/action statistics use training episodes only. BC errors are scaled
  per action coordinate, avoiding an implicit mixing of meters and radians.
- Fixed offline early-stop/wall-time accounting that could report the requested
  number of steps as completed. Fixed BC-only validation's aggregate actor
  loss to exclude Q, matching its training objective.
- `CheatCode` and `CheatCodeModified` previously targeted a point 20 cm above
  the port at startup. They now preserve initial depth inside a narrow aligned
  final-descent region: lateral error ≤2 mm, angular error ≤0.1 rad, and
  tip-minus-port Z in [−25, +5] mm in the controller's base frame. The normal
  approach remains for other starts. This is an expert handoff rule; it does
  not establish insertion success. Live contact behavior is still unverified.

See [architecture and commands](../DIRECT_VISUAL_POLICY.md). The RL loop remains
a deterministic twin-critic actor-critic; complete SAC or TD3 was not added.
DINOv2 support here is for the actor; critic encoder options are unchanged.

## Check 1: saved-data BC run

Run directory:
`outputs/experiments/2026-09-17_direct_visual_bc_smoke/`.

| Item | Recorded setting |
| --- | --- |
| Data | `outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_h264` |
| Size/schema | 546 episodes, 389,907 frames; state width 82, action width 6, three RGB cameras |
| Split | Last 5% of episode IDs held out: 375,108 training frames, 14,799 validation frames |
| Visual initialization | ResNet tensors from `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/checkpoints/175000/pretrained_model/model.safetensors`; 100 tensors copied, action head excluded |
| Actor | 11,616,838 trainable parameters, including 11,166,912 backbone parameters; zero adapter parameters |
| Controls | Seed 17, batch 4, `bc_only`, reward mode `zero`, horizon 1, learning rate 0.0001, 40 updates, five-minute limit |
| Action limits | Per coordinate: 0.02 m for translation, 0.2 rad for rotation; these are model bounds, not verified task-optimal settings |
| Hardware/software | One physical GPU (0); Torch 2.9.0+cu128, torchvision 0.24.0+cu128, existing Pixi environment |

The run completed **40 actual updates**, stopping at `max_steps`.

| Offline measurement | Result | Scope |
| --- | --- | --- |
| Normalized BC loss at update 20 | 0.69111 | First 8 held-out frames |
| Normalized BC loss at update 40 | 0.37614 | Same 8 frames |
| Saved-actor normalized BC loss | 12.74938 | First 32 held-out frames, strict reload |
| Zero-command baseline loss | 17.83264 | Same 32 frames |
| Mean absolute command change with blank images | 0.000476 | Same 32 frames; mixed physical units, sensitivity diagnostic only |

These tiny sequential samples are not episode/scene-balanced evaluation. Their
different losses show why the eight-frame validation decrease is insufficient
for policy selection. Beating zero command on 32 frames does not demonstrate
closed-loop insertion, robustness, or an improvement over ACT. No task success
rate was measured.

Evidence: `train_config.json`, `warmstart_report.json`, `metrics.jsonl`,
`validation_metrics.jsonl`, `run_summary.json`, `launch.log`,
`checkpoint_latest.pt`, `checkpoint_best_val.pt`, and
`heldout_actor_check.json` in the run directory. `isaac_launch_plan.txt` is a
command preview only.

The initial run predates final metadata/validation-report cleanup. Its raw
summary retains irrelevant legacy `freeze_act=true`, `act_checkpoint="None"`,
and ACT-prior wording. Its architecture config has `freeze_backbone=false`,
`uses_act_actions=false`, and trainable parameter counts above. Also, its
validation `actor_loss` includes Q even though BC-only training excluded it;
the table uses the separately recorded `bc_loss`, which was unaffected. Raw
evidence is preserved; subsequent code corrects these reporting issues.

## Check 2: pretrained DINOv2 forward/backward

Run directory:
`outputs/experiments/2026-09-17_direct_visual_dinov2_smoke/`.

Used the official pretrained ViT-S/14 at pinned source revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, one synthetic observation with three
224×224 RGB cameras and 82 state values, and physical GPU 1. A BC backward pass
produced finite commands/loss and nonzero backbone gradients (norm **0.01268**).
Elapsed time was **8.34 s**, including initialization/download; peak Torch CUDA
allocation was **389,142,016 bytes**. These are a single implementation check,
not a steady-state throughput benchmark. No optimizer update or task-trained
DINOv2 checkpoint was produced. Evidence: `result.json` and `launch.log`.

Torch cached model source/weights under `/home/chmin/.cache/torch/hub/` on this
host. A DINO checkpoint includes learned weights, but inference still requires
the pinned model code in the runtime environment/cache.

## Regression checks

**79 tests passed in 5.34 s** on CPU, with output in
`outputs/local_checks/direct_visual_regressions_20260917.txt`.
Coverage includes image sensitivity and backbone gradients, signed bounded
commands, explicit freezing, strict checkpoint/config/normalizer loading,
Gazebo loading without ACT, an actual BC optimizer update, Isaac launcher
arguments, early-stop counts, expert start geometry, and existing
offline/Gazebo/evaluator/schema regressions. Isaac command generation passed
against the saved smoke checkpoint. Simulator imports and live control were
not exercised by these CPU/model checks.

The run directory also contains `reviewed_source_provenance.json`,
`reviewed_source_tracked.diff`, and `reviewed_source_files.tar.gz`, captured
after the checks. They describe the final reviewed working tree; they are not
an exact snapshot from the beginning of the BC run. Artifacts under `outputs/`
are ignored by Git and need separate backup if retained.

## Decision

Use the direct visual path for new architecture comparisons. Keep this smoke
checkpoint as a wiring/learning-path artifact. First validate reset geometry,
action direction, the expert handoff, and official scoring in live simulation.
Then audit demonstrations and compare ACT-initialized ResNet, ImageNet ResNet,
DINOv2, and scratch initialization under matched data and evaluation. Contact
history, finer visual crops, and a complete TD3-style update with image
augmentation are later controlled experiments; see the
[comparison plan](../DIRECT_VISUAL_POLICY.md#next-comparisons).

# Full verified-data tokenizer and dynamics training

Status: completed 2026-09-19. Started after the bounded world-model follow-up.

This run trained a fresh six-view tokenizer on the canonical verified expert
collection, then trained action-conditioned dynamics and a supervised TCP-delta
policy. Reward and imagination training remained disabled. The existing 20
final scenes stayed sealed until all checkpoint choices were fixed.

## Why this run is needed

The earlier tokenizer was **not shown to be fully trained**. Its main stage
stopped after 5,981 updates at a 50-minute phase cap; its selected validation
checkpoint was update 5,500. The later full-view refinement and native-detail
adaptation each reached their planned 2,500-update limits with their best
validation values at the final update. The matched 74-versus-expanded and
144-versus-224 comparisons were deliberately bounded to 1,000 updates per arm
and also selected their final updates. These runs answer bounded comparison
questions, but do not establish a validation plateau. Longer training alone is
not guaranteed to recover fine contact detail: the prior expanded model still
blurred the connector and port, and the 224-pixel ablation enlarged a source
crop containing only about 160 by 144 pixels.

The compact ACT comparator has **16,354,566 inference parameters** and occupies
65,418,264 bytes as FP32 parameters. Its ImageNet ResNet18 backbone accounts for
11,166,912 parameters. The selected six-view world policy has **22,569,344
acting parameters** (90,277,376 FP32 bytes), while the complete training model
has 27,130,942 parameters. The acting world policy is 1.38 times ACT's size;
the full training model is 1.66 times ACT's size. On the same corrected 60/14
data and frozen 20 scenes, ACT scored 0/20 full insertions, one partial, and a
22.69 mean; the world policy scored 0/20 full insertions, two partials, and a
32.32 mean. The world policy scored higher on 14/20 scenes and approached the
opening more closely, but neither model was reliable. This compares complete
pipelines: the world runtime predicts four commands per visual decision while
ACT replans after one prediction.

## Frozen protocol

- Data source: canonical 289 verified episodes, with the existing grouped,
  scene-disjoint 250 training / 39 validation split. Optimization sees all 250
  training episodes; all 39 validation episodes remain held out.
- Visual sampling: equal probability for historical SFP, aligned SFP, and SC;
  uniform episode sampling inside the selected category; 20% terminal windows.
- Views: three RGB cameras and their three fixed contact crops at 144 pixels,
  using the audited BGR-to-RGB conversion and camera order.
- Initialization: a fresh seeded model. Four canonical validation episodes
  appeared in the old pilot tokenizer's training ancestry, so that checkpoint
  is not a leakage-safe initialization for this split.
- Hardware: at most physical GPUs 0--3. GPU 7 is occupied by another job and is
  excluded.
- Convergence: validate every 1,000 updates. A relative improvement of at least
  0.5% in held-out contact-field MSE resets patience. After three checks without
  such improvement, divide the learning rate by three. Stop after three more
  non-improving checks at the minimum learning rate. The 50,000-update cap is a
  safety bound and does not count as convergence if validation is still
  improving.
- Selection: choose tokenizer checkpoints using held-out contact-field MSE;
  report whole-image MSE/RMSE and contact-view edge error and save held-out
  original/reconstruction sheets.
- Dynamics: use only episodes with native simulator timestamps, native
  observation endpoints, and four actual executed commands per 200 ms edge.
  Commands use the corrected observation-relative TCP-delta contract. Teacher
  targets are reserved for policy supervision.
- Final evaluation: after dynamics convergence and a frozen supervised policy
  fit, run the unchanged policy once on the existing 20 final scenes. No final
  scene can affect selection. The live decision p95 must remain below 300 ms.

Bulk checkpoints and caches live under
`/var/tmp/chmin_aic_20260919_full_world/`; compact manifests, reports, commands,
metrics, reconstruction sheets, and selected artifacts live under
`outputs/experiments/2026-09-19_full_world_training/`.

The pre-training dynamics audit found **148 eligible episodes: 126 train and
22 validation**, containing 68,884 native observations and 9,858/1,486 strict
train/validation 200 ms edges. These comprise 127 SFP and all 21 SC episodes.
The 140 historical video episodes lack native `frames.jsonl`; one additional
aligned SFP episode has native frames but no recorded command indices. All 141
remain useful for tokenizer images but are excluded from action-conditioned
dynamics. No command index or intermediate command was inferred.

## Completed tokenizer training

The tokenizer trained for **78,000 optimizer updates** over four resumable
segments and stopped under the frozen validation rule with
`validation_plateau_at_min_lr`. The segments took 45,151.7 seconds in total
(12.54 hours); the 50k, 60k, and 70k boundaries were safety caps and were not
treated as convergence. The selected checkpoint is update **75,000**, with
held-out contact-field MSE **0.000574803**. This is the first tokenizer in this
work shown to reach the declared plateau rule.

On 156 sampled frames from all 39 canonical validation episodes, the selected
tokenizer improved over the earlier bounded expanded checkpoint:

| Held-out reconstruction metric | Earlier bounded | Full plateau | Change |
| --- | ---: | ---: | ---: |
| Whole-image MSE | 0.00813554 | 0.000712110 | -91.25% |
| Contact-field MSE | 0.00857250 | 0.000574803 | -93.29% |
| Contact-view edge L1 | 0.0182567 | 0.00856489 | -53.09% |

All 39 approach/terminal reconstruction sheets were inspected. Broad scene,
PCB, connector, and colored board details are substantially sharper, including
SC examples, although ordinary autoencoder smoothing remains. The earlier
reference also had four canonical validation episodes in its training ancestry;
it is retained as a disclosed visual reference rather than a leakage-safe
statistical baseline. The new tokenizer was initialized from scratch and no
canonical validation episode entered its optimizer updates.

## Completed dynamics and supervised policy training

Corrected action-conditioned dynamics trained to the same plateau rule and
stopped at update **7,500**; validation selected update **4,500**. The selected
model still fails every numerical fidelity gate and loses badly to a no-motion
persistence baseline:

| Horizon | Episodes | TCP error: model / persistence / shuffled | Orientation: model / persistence / shuffled |
| --- | ---: | ---: | ---: |
| 200 ms | 22 | 11.46 / 2.07 / 12.70 mm | 1.32 / 0.23 / 1.42 degrees |
| 400 ms | 15 | 19.17 / 4.42 / 22.02 mm | 1.91 / 0.44 / 2.42 degrees |
| 600 ms | 5 | 20.12 / 7.24 / 27.08 mm | 3.06 / 1.11 / 5.25 degrees |

The supervised TCP-delta policy then trained from the selected tokenizer/world
weights. It stopped at update **5,500** under the plateau rule and selected
update **1,000**, with 2.815 mm held-out translation error. Reward,
termination, and imagination objectives were never enabled. The exported
controller has 22,569,344 acting parameters and SHA256
`36e9cb8951473642d16a8ecae7430dee69352e9163bd1db4611053766788168d`.

An isolated raw-camera benchmark on an RTX A6000 measured 28.60 ms median,
29.06 ms p95, and 32.75 ms p99 over 1,000 warm calls. The first cold call was
492.7 ms, so the deployed node warms its kernels before accepting a task.

## Frozen 20-scene result

Checkpoint selection finished before the preserved final scenes were opened.
All **20/20** fresh-simulator trials passed source, reset, task identity,
duration, and scoring checks. The unchanged policy achieved **2/20 full
insertions**, **9 official partial insertions**, and mean official score
**45.14**. It scored above the corrected60 ACT comparator on 13/20 matched
scenes and above the earlier corrected60 world policy on 12/20. The comparison
is favorable but does not establish reliability or isolate the contribution of
tokenizer data, dynamics pretraining, and larger supervised data.

Across 8,895 live decisions, observation-to-command latency was 64.27 ms
median, **80.77 ms p95**, 98.29 ms p99, and 146.74 ms maximum, with zero
decisions at or above 300 ms. Twenty one-frame-per-second videos and terminal
contact sheets are archived.

The first attempted final run is preserved separately as an infrastructure
failure. Its first scene inserted, but an older ACT initial-state contract
rejected a subsequently added fail-closed metadata check in the inherited ACT
base class. A separate world contract pins the exact current sources; no
capture or execution-loop code, checkpoint, or scene changed. The complete
20-scene run restarted from scene 1 so every included trial used the same
audited runtime.

## Decision

The visual bottleneck improved enough that another tokenizer scale-up is not
the next priority. Dynamics remains unusable for planning because persistence
is much better at every horizon. Keep reward and imagination disabled. The
next causal policy test should repeat the same architecture and full supervised
data from (a) the exact preserved pre-world random ancestor and (b) the selected
world weights, preferably across multiple seeds and on new development scenes.
Do not use these final 20 scenes for that selection.

Durable selected checkpoints, reports, all held-out reconstruction sheets,
final videos, and hashes are under
`outputs/experiments/2026-09-19_full_world_training/artifacts/`. Bulk caches and
intermediate checkpoints remain under `/var/tmp/chmin_aic_20260919_full_world/`.

## Commands, failures, and provenance

The exact initial tokenizer command is recorded in `tokenizer_launch.json` and
every downstream command and UTC boundary is recorded in
`pipeline_events.jsonl` in the experiment artifact directory. The main forms
were:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone \
  --nproc_per_node=4 train_full_tokenizer.py --steps 50000 --batch 4 \
  --frames 8 --validate-every 1000 --lr 0.00015 --seed 17 ...

CUDA_VISIBLE_DEVICES=0,1,2,3 .pixi/envs/default/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_world_to_plateau.py world \
  --steps 30000 --batch 128 --validate-every 500 --lr 0.0003 ...

CUDA_VISIBLE_DEVICES=0,1,2,3 .pixi/envs/default/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_world_to_plateau.py agent \
  --steps 30000 --batch 128 --validate-every 500 --lr 0.00015 ...

.pixi/envs/default/bin/python run_final20.py \
  --model /var/tmp/chmin_aic_20260918_dreamer60/full_data_control_20260919.pt \
  --output-dir /var/tmp/chmin_aic_20260918_dreamer60/full_data_final20_20260919_v2 --run
```

The continuation directories `tokenizer_full_v2` through `v4` preserve the
resume boundaries. One world-fidelity report attempt failed while generating
image sheets because `images.npy` was an intentional zero-width placeholder;
it is preserved as `world_fidelity_failed_placeholder_images`. Pointing that
dataset entry at the already built six-view native images fixed reporting only;
the trained checkpoint and numerical evaluation definition did not change.
The first administratively invalid final run is likewise preserved at
`/var/tmp/chmin_aic_20260918_dreamer60/full_data_final20_20260919/`.

`aic_commit.txt`, `dreamer_commit.txt`, the two worktree patch files, model and
report hashes, logs, and the selected artifacts provide the source/data/model
lineage. The scripts use only physical GPUs 0--3 and never require sudo.

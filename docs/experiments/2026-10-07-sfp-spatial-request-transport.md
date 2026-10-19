# SFP transport: spatial requested-card evidence

## Hypothesis and bounded design

The earlier frozen three-view transport actor loses within-image card location
when each ResNet-18 camera map is globally pooled to one 512D vector. Its
task-code action diameter was only 0.554 mm against 12.44 mm paired teacher
branching, and it reached zero of three safe live buffered arrivals. A
request-conditioned FiLM plus discrete route-label fit raised code sensitivity
to 3.394 mm but worsened held-out first-action p95 2.153→2.811 mm, so its live
gate was not opened. [Prior result](2026-10-07-sfp-transport-route-return.md).

This distinct test preserves the matched temporal history, full TCP-body
action target, train/validation split, request/paired-scene sampler and
3,000-update cap. A **no-spatial control and spatial candidate both resume
from the same corrected checkpoint for the same update budget and seed**.
The candidate adds a frozen ResNet-18 **4×5 layer3 grid for each of the
three current cameras**. A query containing the requested card/port code and
current temporal context attends to those 60 spatial tokens; the result is
added to the action representation. The old action pathway is initialized
unchanged. No simulator port pose, card coordinates, visibility label,
teacher future, or crop selected by privileged geometry enters inference.
SC rows have zero spatial extras in this SFP-specific diagnostic; this is
not a promoted shared SFP/SC actor.

The layer3 grid comes from the same preprocessed camera frames and frozen
ImageNet backbone as the pooled temporal feature. The cache is on NAS because
the local output disk has only about 8 GB free. The runtime path computes
pool and grid in one ResNet pass. Train on one GPU. Keep the existing Isaac
process untouched and the four final IK configurations sealed.

## Declared gates

1. Verify cache row order and a numerical pooled-feature equivalence on
   sampled old and new source images. Inspect a few feature grids for finite,
   nonconstant spatial content.
2. Held-out complete episodes and paired scenes: first-action translation p95
   must be at most 10% worse than the *matched no-spatial continuation* and
   ≤**2.368 mm** (at most 10% over the corrected original actor's 2.153 mm).
   Same-image request-code action diameter must exceed the matched control
   by at least 2× without collapsing to arbitrary outputs.
   Compare the early paired-request action difference with the 12.44 mm
   teacher reference. These offline checks are necessary but insufficient.
3. If both pass, measure complete live inference latency and run three new
   nonfinal requests with no geometry or guide. Require ≥2/3 safe requested
   buffered arrivals within 10 mm, no extra scored contacts, and p95 <300 ms.
   Then run ten grouped development starts with target ≥9/10. If an offline
   gate fails, stop before new Gazebo runs and diagnose the failure. If live
   progress remains absent, stop the spatial branch and record the first
   divergence rather than collecting or retraining routinely.

Handoff, insertion, RL and final-scene evaluation remain gated.

## Commands and results

The frozen grid was cached with `scripts/cache_sfp_transport_spatial_grid.py`
from `temporal_cache_v4`, `paired_coverage_merged_v2/sfp_labels.jsonl`,
`cache_all_verified_sfp_sc_v3`, and the recorded ImageNet ResNet-18 weights.
The exact invocation used GPU 1, batch size 16, and output
`/home/nas/chmin/aic_archives/2026-10-07_sfp_transport_spatial_cache_v1`.
Host Pixi needed its local `libjpeg.so.8` and `libpng16.so.16` preloaded for
OpenCV; no sudo or Gazebo startup was used. A sampled eight-row audit covering
old cached and newly recorded SFP RGB found nonconstant spatial cells
(per-channel cell standard deviation ≥0.0867); maximum discrepancy from
recomputed features was 0.00391 for the half-precision grid and 0.03125 for
the pooled feature, consistent with the half-precision cache. See
`outputs/experiments/2026-10-07_sfp_spatial_transport/cache_audit.json`.

Both fits used the same starting checkpoint, split, sampler, seed and 3,000
updates. They ran as follows, on one GPU per fit:

```bash
CUDA_VISIBLE_DEVICES=2 .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_v4 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --output outputs/experiments/2026-10-07_sfp_spatial_transport/control_continued_v1 \
  --full-state-skip --long-history --no-route-heads --balance-sfp-targets \
  --paired-sfp-mass .5 \
  --init-temporal outputs/experiments/2026-10-05_transport_identity_audit/paired_coverage_actor_v2/best.pt \
  --updates 3000 --budget-minutes 60

CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/train_temporal_transport_probe.py \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_v4 \
  --checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --output outputs/experiments/2026-10-07_sfp_spatial_transport/spatial_request_v1 \
  --full-state-skip --long-history --no-route-heads --balance-sfp-targets \
  --paired-sfp-mass .5 \
  --init-temporal outputs/experiments/2026-10-05_transport_identity_audit/paired_coverage_actor_v2/best.pt \
  --spatial-cache /home/nas/chmin/aic_archives/2026-10-07_sfp_transport_spatial_cache_v1 \
  --spatial-request-attention --updates 3000 --budget-minutes 60
```

Both selected update 3,000. On the same held-out teacher rows, no-spatial
first-action translation p95 was **1.863 mm** and spatial was **1.828 mm**
(1.9% better). However, the median same-image ten-request action diameter was
**0.883 mm** for the control and **0.903 mm** for spatial, only **1.02×**
versus the declared ≥2×; the paired teacher branch reference is **12.44 mm**.
The new spatial projection did learn nonzero weights, but its small change in
actions does not show that it chose the requested route. Its offline gate
failed. No new Gazebo rollout, latency claim, handoff, insertion or RL test
followed. The final scenes remain sealed.

Artifacts: `outputs/experiments/2026-10-07_sfp_spatial_transport/{summary.json,control_continued_v1,spatial_request_v1,control_task_sensitivity.json,spatial_task_sensitivity.json,cache_audit.json}`.

**Decision:** stop this spatial-attention branch. Better one-step imitation
alone again failed to create the large request-specific route change seen in
matched teachers. The next concrete repair would train on *causally paired
same-reset alternate-request branch labels* rather than hoping the ordinary
action loss discovers rare route choices. Before fitting, audit that paired
frames are aligned before either policy has caused different observations;
after trajectories diverge, same-time images cannot be treated as
counterfactual inputs. Keep this as a separately gated experiment, not an
automatic promotion of the present model.

### Causal pair feasibility check

`scripts/audit_sfp_paired_branch_alignment.py` examined the 35 paired
training reset scenes in `temporal_cache_v4`. It matched alternate requests
within 110 ms during the first 1.5 s and required ≤2 mm measured TCP
difference, ≤2° orientation difference and ≥0.995 frozen-image-feature cosine
before admitting a counterfactual action pair. Of 279 time-matched rows, only
**27 rows across 20 scenes** also had a ≥3 mm teacher-action difference;
**eight rows/eight scenes** had ≥5 mm and **zero** had ≥10 mm. The valid
≥3 mm pairs had 3.42 mm median action difference. Thus the 12.44 mm teacher
branch separation occurs mostly **after the requested trajectories have
already changed the observations**. Naively treating those later images as
the same state would create false counterfactual targets.

This fails a useful-scale paired-supervision preflight. No contrastive fit or
new data collection followed. A future targeted collection could hold the
same starting observation fixed and query alternate requested-route teacher
commands before executing either, then verify that this produces materially
different valid commands across enough independent resets. It needs its own
budget and stop rule. [Machine pair audit](../../outputs/experiments/2026-10-07_sfp_spatial_transport/paired_branch_alignment.json).

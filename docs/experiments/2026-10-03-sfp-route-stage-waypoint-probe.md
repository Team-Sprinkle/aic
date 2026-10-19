# SFP route-stage and next-waypoint probe

**Date:** October 3, 2026. **Decision:** the bounded supervised probe failed its waypoint gate. No route-conditioned actor continuation or new simulator rollout was started. The four reserved final IK configurations remain sealed.

## Hypothesis and frozen gate

The previous full-action temporal actor reached 0/3 safe SFP buffered arrivals despite better offline action predictions. Its measured route split from the same-scene teacher by >10 mm at about 3.85–3.95 s. This experiment asked whether a more explicit route representation could predict **which part of the journey is underway and where that part should end**. The approved cap was one GPU, at most 2,000 probe updates and two hours. A route-conditioned actor could train only if complete-reset held-out route-stage accuracy reached **90%** and 3D next-waypoint translation p95 was **≤20 mm**. If the probe failed, actor continuation and live rollouts would stop.

## Labels and input provenance

The source was the previously frozen **50 successful balanced SFP training episodes and ten disjoint successful development episodes**, covering all ten card/port requests. No new data were collected. [Cache builder](../../scripts/build_sfp_route_stage_probe_cache.py) used the stock CheatCode's 5.5-second interpolation toward a position above the requested port as stage 0; its following descent toward the safe buffered region was stage 1. The code excluded observations within 0.15 s of the stage boundary. The endpoint for each stage was the **measured future teacher TCP pose** in that same episode, expressed relative to the current observed TCP body frame. It used trajectory measurements as offline labels; the actor or probe received no true port position, teacher stage, or waypoint. This representation was a coarse route target, not an absolute-coordinate action and not the precise PoseInsert port estimate.

The labeled cache has **3,828** frozen feature rows: 3,190 train and 638 development. It retains the same three independent 288×256 ImageNet ResNet18 camera features, causal camera history, measured state/motion and public task code. To avoid an offline shortcut absent after autonomous drift, **all previous teacher commands were zeroed** for this probe. Complete episode/reset configurations remain grouped by the original 50/10 split, with no group overlap. A constant-mean baseline and a measured-state/task-only ridge baseline were evaluated on the same 638 development rows. [Exact label/scene manifest](/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/route_stage_probe_cache_v2/manifest.json).

## Bounded training and results

[The probe trainer](../../scripts/train_sfp_route_stage_waypoint_probe.py) initialized the existing two-layer, 256D temporal backbone from the frozen ordinary full-action BC checkpoint. It trained its stage and waypoint heads and representation; the action and handoff heads were frozen. It used one GPU (GPU 2), 2,000 updates, no teacher-command history, and chose its checkpoint by grouped development stage/waypoint score. The cap was reached and update 2,000 was selected; no additional probe sweep ran. [Checkpoint, per-row predictions, history and exact metrics](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/route_stage_waypoint_probe_v1/summary.json).

| Development metric, 638 rows / 10 complete episodes | Constant mean | State/task only | Frozen RGB history + state/task |
| --- | ---: | ---: | ---: |
| Route-stage accuracy | 58.0% | 98.6% | **98.7%** |
| Next-waypoint translation median | 53.8 mm | 15.5 mm | **10.3 mm** |
| Next-waypoint translation p95 | 127.0 mm | 50.2 mm | **48.2 mm** |

The learned stage passes 90%, but waypoint p95 misses 20 mm by more than 2×. By true stage, waypoint p95 is **67.6 mm during the approach** and **19.8 mm during descent**. Even restricting to correct stage predictions, the approach p95 remains **61.6 mm**, so wrong-stage classification alone does not explain the miss. The first second after the teacher's initial command has **83.1 mm p95** waypoint error. Near the 5.5-second transition, the 5.0–5.5 s bin reaches 107.2 mm p95; eight of 638 rows had a wrong stage prediction, but correctly classified approach rows still contain large errors. The plot shows the full time course and each card/port request: [error plot](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/route_stage_waypoint_probe_v1/error_by_time_and_target_v2.png).

The stage-end **measured-versus-commanded** TCP difference at the 5.5-second boundary was 13.0 mm median, 18.1 mm p95, and 20.1 mm maximum over the 60 source episodes. This controller lag contributes label uncertainty, but a change to the commanded endpoint can move any individual target by at most about 20 mm; it cannot by itself close a 48.2→20 mm overall p95 gap. Cached `time_s` exactly matched source timestamp minus episode start on checked training/development episodes. I visually inspected original-resolution three-camera initial frames for one high-error card-4/port-0 scene and one lower-error card-1/port-0 scene: multiple similar cards and ports are visible, including at the initial view. This supports **target selection and spatial precision** as a hypothesis, but two inspected scenes cannot establish a complete visibility result. At 288×256, frozen global features may discard detail; the present experiment does not isolate image resolution from data coverage or architecture.

## Decision and next bounded revision

The predeclared waypoint gate failed, so **do not train the route-conditioned actor** or test this checkpoint live. The earlier action-only actor is also unpromoted. Continue to gate automatic visual handoff, PoseInsert and RL; keep final scenes sealed. The useful finding is narrower: stage recognition is already near-perfect from measured state/task, whereas the early next waypoint remains too inaccurate and gains only ~2 mm p95 from these frozen RGB features over the state-only baseline.

The next proposed diagnostic is a **task-conditioned coarse selected-card/port anchor** from original-resolution wide views, using the existing simulator geometry as **training labels only**. It should predict the requested target's coarse location and abstain when evidence is insufficient; it should not predict every port. Use the same episode-grouped split, include hard neighboring-port requests, and check the selected target identity separately from 3D translation. Freeze new non-final development starts before fitting. Bound the first comparison to **one GPU and at most 2,000 updates**. Its offline gate is **zero accepted wrong requested targets** and coarse selected-target position p95 **≤20 mm** on complete held-out resets. A single capped actor continuation and autonomous transport test may follow only if both pass; live promotion still requires ≥2/3 safe buffered arrivals, no increased scored contact, and complete p95 <300 ms. This is a plan revision, not an authorization to consume the sealed final split or start RL.

## Commands and artifacts

```bash
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic .pixi/envs/default/bin/python \
  scripts/build_sfp_route_stage_probe_cache.py \
  --sfp-labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/balanced_merged_sfp_v1/sfp_labels.jsonl \
  --base-cache /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/balanced_temporal_cache_v1 \
  --output /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/route_stage_probe_cache_v2

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
  .pixi/envs/default/bin/python scripts/train_sfp_route_stage_waypoint_probe.py \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/route_stage_probe_cache_v2 \
  --source-checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --image-checkpoint /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/ordinary_v1/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/route_stage_waypoint_probe_v1 \
  --updates 2000 --budget-minutes 120
```

The image plot was produced with [the plot script](../../scripts/plot_sfp_route_probe_errors.py) from the frozen cache and `best_predictions.json`. [Machine-readable decision](2026-10-03-sfp-route-stage-waypoint-probe.summary.json). Rootless Docker still has only one session-key slot free, but this probe ran in the existing local Pixi/CUDA environment without sudo or any new container. The active old Isaac process on GPU 0 was not interrupted.

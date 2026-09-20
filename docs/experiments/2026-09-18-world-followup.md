# World-model follow-up: frozen final, geometry, and visual fidelity

Status: frozen final diagnosis, matched tokenizer studies, and the bounded
same-architecture supervised comparison completed on 2026-09-18. This
record extends the [pilot](2026-09-18-dreamer-pilot.md) and the
[work sequence](../WORLD_MODEL_NEXT_STEPS.md). It does not change either
frozen final policy or its scene manifest.

## Frozen paired final result

The corrected ACT and world policies each completed the same 20 SFP / one NIC
final scenes, with valid starts, ground truth disabled and 90 simulated seconds.
The [paired result](../../outputs/experiments/2026-09-18_dreamer60_pilot/paired_tcp_delta_final_summary/paired_final_summary.md)
pins both official result hashes and the scene manifest. ACT achieved **0/20
full insertions**, one official partial, mean total **22.69**. The world
policy achieved **0/20 full insertions**, two official partials, mean total
**32.32**; its score was higher in 14/20 paired scenes. ACT incurred seven
prohibited-contact penalties; world incurred one. Neither incurred an applied
force penalty. The world's 8,908 actual ROS decisions had **77.01 ms pooled
p95** processing-to-publication latency, under the 300 ms limit. This is a
complete-policy comparison: ACT replans one command per observation while the
world policy predicts four 20 Hz commands per visual decision.

The [ACT final archive](../../outputs/experiments/2026-09-18_dreamer60_pilot/act60_tcp_delta_final_archive/archive_manifest.json)
has 127 hash-verified files, including 20 one-frame-per-second videos, scores,
start audits and terminal sheets. ACT trial 17 first failed before any scored
rollout because the model lifecycle was inactive. Its original incomplete
attempt and the valid unchanged-model retry are both preserved; the retry
completed the full 90 simulated seconds. The
[world final archive](../../outputs/experiments/2026-09-18_dreamer60_pilot/paired_world_archives/world_tcp_delta_final_worker_v1/archive_manifest.json)
has 426 hashed files and an independent eligibility audit. All 40 terminal
camera sheets were inspected as
[ACT overview](../../outputs/experiments/2026-09-18_world_followup/act_final_terminal_overview.png)
and [world overview](../../outputs/experiments/2026-09-18_dreamer60_pilot/world_final_analysis/terminal_overview.png).

## Where the plug stopped

The [40-rollout geometry report](../../outputs/experiments/2026-09-18_world_followup/paired_final_geometry_v3/summary.json)
uses recorded `/scoring/tf`, measured TCP, published pose commands and wrist
force. It transforms the plug-tip frame into the explicit
`<port>_link_entrance` frame. The SFP opening is **45.8 mm** before the port
reference; score messages alone can make a plug at the opening look 4–5 cm
away. Task-board opening and port-reference positions were constant across
all sampled frames in every trial. Trajectories sample every 0.5 simulated
seconds; official scores supply insertion and prohibited-contact outcomes.

| Final diagnostic | ACT | World |
| --- | ---: | ---: |
| Tip in 25 mm axial/lateral opening envelope | 12/20 | 17/20 |
| Axis-plane crossing within 10 mm lateral | 5/20 | 12/20 |
| Final lateral offset, median | 17.23 mm | 8.71 mm |
| Final lateral offset ≤5 mm | 1/20 | 4/20 |
| Median last-10-second requested/measured TCP gap | 8.22 mm | 39.88 mm |
| Median last-10-second net sampled TCP motion | 1.01 mm | 0.28 mm |

These envelopes are diagnostic thresholds, not official insertion criteria.
The world policy more often approaches the opening, but many endpoints have
5–15 mm lateral error or several degrees of orientation error. ACT has more
far misses and prohibited gripper/card contact. World scenes 1 and 13 are
official partial insertions with final lateral offsets 1.3/1.5 mm and
orientation errors 7.5/4.2 degrees; ACT scene 16 is the sole official partial
at 29.7 mm axial depth, 0.8 mm lateral error and 3.4 degrees orientation
error. None reaches full seating. In several world near-opening trials, the
policy still requests roughly 38–40 mm translations while measured TCP motion
is tiny and the target-tracking gap remains roughly 40 mm. This rejects a
zero-command explanation for those sampled stalls. Contact geometry,
compliance and controller response remain candidate causes; the logs do not
isolate one. The report includes per-trial axial, lateral, orientation,
command, TCP motion, force and official contact messages, with source hashes.

The exact extraction command was:

```bash
.pixi/envs/default/bin/python outputs/experiments/2026-09-18_world_followup/extract_final_geometry.py \
  --act-result /var/tmp/chmin_aic_20260918_dreamer60/act60_tcp_delta_final_v1/result.json \
  --world-result /var/tmp/chmin_aic_20260918_dreamer60/world_tcp_delta_final_worker_v1/result.json \
  --output outputs/experiments/2026-09-18_world_followup/final_geometry_extract_manifest.json
.pixi/envs/default/bin/python outputs/experiments/2026-09-18_world_followup/analyze_final_geometry.py \
  --manifest outputs/experiments/2026-09-18_world_followup/final_geometry_extract_manifest.json \
  --scenes outputs/experiments/2026-09-18_dreamer60_pilot/paired_scenes/final/trials \
  --output outputs/experiments/2026-09-18_world_followup/paired_final_geometry_v3
```

The reader used an owned rootless CPU-only container with the existing AIC
image and `/ws_aic/install/setup.bash`; no sudo or GPU was needed. The
container was stopped after extraction. The command above requires an
equivalent running container named `aic_followup_geometry_cpu`.

## Larger visual-data audit

The [visual manifest audit](../../outputs/experiments/2026-09-18_world_followup/visual_manifest_audit.json)
checked all **289 verified episodes**, **169,710** cached RGB frames and 253
contiguous image shards, including camera order and array shapes. The source
mix is 140 historical BGR episodes and 149 RGB aligned/SC episodes; the
loader converts only documented BGR shards to RGB before the shared
256×288-to-144 preprocessing. The canonical 250/39 episode split has no
scene overlap, but 15 of the original 74 pilot episodes have a different
train/validation assignment there. For the matched comparison, the original
14 pilot validation episodes and their scenes were excluded from both arms.
The original arm trains from the 60 pilot episodes; the expanded arm samples
from 243 eligible training episodes (229 SFP, 14 SC), with 32 additional
validation episodes. It balances historical SFP, aligned SFP and SC source
categories when sampling. A bounded 1,000-update run samples from this whole
pool; it does **not** visit every one of the 140,825 eligible training frames.

Official Tier 3 success and terminal sheets verify eventual insertion for
these demonstrations. The resampled 289-episode image cache does not contain
synchronized plug-tip/port-opening transforms for every frame, so a blanket
near-opening frame count for all 289 is **not established**. The strict 74
native bags support a smaller geometry-linked held-out audit: within 5 mm of
the actual opening, the selected world dynamics misses future measured TCP
position by **13.94 mm** at 200 ms versus **0.57 mm** for persistence (13
episodes, 163 windows). At 400 ms the figures are **20.80 / 0.36 mm** (5
episodes, 12 windows). There are no valid 600 ms windows starting within
15 mm of the opening. See the
[native opening audit](../../outputs/experiments/2026-09-18_dreamer60_pilot/world_opening_geometry_audit/report.json)
for the TF timing/exclusion gates.

## Matched tokenizer comparison

These tokenizer runs do not establish convergence. The original main stage
stopped at its wall-clock cap, and the refinements and matched comparisons
selected their last scheduled checkpoints. They answer bounded comparison
questions; they do not show that additional training has ceased helping.

Both new arms continued from the same six-view initialization for **1,000
updates**, batch 2, four cached frames, identical optimizer/settings,
six 144×144 views and fixed held-out checkpoint selection by contact-field
MSE. They are a bounded continuation test, not a fair comparison of model
architecture or a newly trained control policy. The original common holdout
contains 14 SFP episodes (56 sampled frames); the additional holdout contains
32 episodes, including seven SC. Each sample saves original/decoded full
cameras and three image-coordinate contact crops. Early, one-third,
two-thirds and last-recorded frames are temporal proxies; they are not
geometry-labeled approach phases.

| Held-out group | Original-74 contact-field MSE | Expanded-data MSE |
| --- | ---: | ---: |
| Original 14 SFP | 0.00665 | 0.00770 |
| Additional 32 | 0.01588 | 0.00883 |
| Additional seven SC | 0.02973 | 0.00724 |

Pixels are normalized to 0–1. The expanded model improves unseen source and
SC reconstruction but modestly worsens the original SFP holdout. In the
[paired SFP sheets](../../outputs/experiments/2026-09-18_world_followup/tokenizer_expanded_matched_v1/heldout_sheets/episode_168_terminal_recorded_all_views.png)
and [SC sheets](../../outputs/experiments/2026-09-18_world_followup/tokenizer_expanded_holdout_v1/sc_sheets/episode_253_terminal_recorded_all_views.png),
board shape appears but connector/port boundaries are still blurred. A
[source-colored-feature audit](../../outputs/experiments/2026-09-18_world_followup/sc_color_expanded.json)
examined only visible magenta port-marking and blue connector pixels in the
seven held-out SC episodes. The expanded arm reduced their pixel error
slightly, but **zero** source pixels retained the same thresholded color in
the decoded images; the original arm also had zero recall. These image-only
color masks are local feature proxies, not complete segmentation.

The two matched commands differed only in `--arm` and output path:

```bash
CUDA_VISIBLE_DEVICES=0 .pixi/envs/default/bin/python \
  outputs/experiments/2026-09-18_world_followup/compare_tokenizer_data.py \
  --arm original74 \
  --audit outputs/experiments/2026-09-18_world_followup/visual_manifest_audit.json \
  --cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
  --init /var/tmp/chmin_aic_20260918_dreamer60/tokenizer_six_views_init.pt \
  --output /var/tmp/chmin_aic_20260918_world_followup/tokenizer_original74_matched_v1
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python \
  outputs/experiments/2026-09-18_world_followup/compare_tokenizer_data.py \
  --arm expanded \
  --audit outputs/experiments/2026-09-18_world_followup/visual_manifest_audit.json \
  --cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
  --init /var/tmp/chmin_aic_20260918_dreamer60/tokenizer_six_views_init.pt \
  --output /var/tmp/chmin_aic_20260918_world_followup/tokenizer_expanded_matched_v1
```

The [tokenizer archive manifest](../../outputs/experiments/2026-09-18_world_followup/tokenizer_archive_manifest.json)
hashes selected checkpoints, metrics, image sheets, and exact options for
both arms. Bulk training also remains under
`/var/tmp/chmin_aic_20260918_world_followup/`.

## Higher-resolution crop ablation

After expanded-data blur persisted, two contact-only tokenizers continued
from the same expanded checkpoint for 1,000 additional matched updates. One
retained 144×144 crops; the other interpolated patch queries to accept
224×224 crops. Both were decoded back to the recorded 160×144 contact-field
pixel grid before comparison on the same 14 pilot holdouts. The 224-pixel arm
had **worse source-pixel MSE, 0.01101 versus 0.00828**, and higher edge error,
0.02133 versus 0.02111. Visual
[144-pixel](../../outputs/experiments/2026-09-18_world_followup/contact_crop144_v1/heldout_sheets/episode_168_terminal_recorded_contact_crops.png)
and [224-pixel](../../outputs/experiments/2026-09-18_world_followup/contact_crop224_v1/heldout_sheets/episode_168_terminal_recorded_contact_crops.png)
sheets show no recovered connector detail. The recorded source crop is only
160×144 pixels; the 224-pixel model has more tokens but no new optical data.

An isolated three-camera crop/resize/transfer/encoder benchmark measured
**11.18 ms p95** for 144 and **12.15 ms p95** for 224 across 300 calls each.
These are [component timings](../../outputs/experiments/2026-09-18_world_followup/contact_crop224_latency.json),
not live ROS latency. The 224 candidate was rejected on held-out fidelity,
so it was not integrated into a policy or deployed for a live trial. The
unchanged frozen world's live final p95 remains 77.01 ms; any future
integrated resolution change still needs a fresh live under-300-ms check.

## Dynamics, reward labels, and next supervised control comparison

The [full future-latent audit](../WORLD_MODEL_NEXT_STEPS.md#selected-dynamics-future-latent-and-pose-audit)
compares true-next-latent decoding with actual-command predicted latents,
persistence and shuffled verified actions at strict 200/400/600 ms horizons.
Future measured TCP position errors are 12.06/14.92/21.92 mm for the selected
world dynamics versus 2.29/4.81/9.85 mm for persistence, across 14/9/5
episodes. The real-next-latent decode already blurs contact detail; predicted
latents add drift. The selected dynamics fails the fidelity gate, including
the near-opening subset above. Imagination and reward training remain off.

The [raw-bag audit](../../outputs/experiments/2026-09-18_dreamer60_pilot/post_event_raw_bag_audit/report.json)
found no camera/video topics in any of the strict 74 expert bags, but every
bag retains 0.134–0.922 seconds of controller state after the first correct
insertion event. These state-only tails might support an audited state/event
model after sustained-insertion verification; they cannot create synchronized
post-event RGB examples or justify reward labels on pre-event images.

A separate [all-verified bag metadata audit](../../outputs/experiments/2026-09-18_world_followup/verified_bag_metadata_audit.json)
mapped all 149 aligned episodes to unique ROS bags, reading metadata inside
23 `.tar.zst` archives without altering them. All 149 bags record an insertion
event and controller state, and **none records a camera/image/video topic**.
The other 140 historical episodes link to raw LeRobot video datasets but not
to a ROS bag in the canonical manifest. Their video timing relative to the
official event is therefore not established by this audit. For the additional
75 aligned bags beyond the strict 74, metadata proves topic availability but
does not establish event-relative state-tail duration or sustained insertion.
No new Gazebo collection is justified by treating these pre-event images as
observed post-insertion states.

The next supervised experiment holds the six-view architecture,
strict 60/14 data, delta-command contract, optimizer budget, checkpoint rule,
new development scenes and 300 ms live limit fixed while comparing
**with versus without selected world-dynamics pretraining**. Initialize both
arms with the same frozen tokenizer, train both policy/world trunks on the
same teacher deltas, and keep rewards/imagination disabled. Compare held-out
imitation, live insertion/geometry and actual ROS latency, then compare both
with corrected ACT. This isolates the value of world initialization from
the existing ACT/world size and four-versus-one-command confounds. The
[four new development scenes](../../outputs/experiments/2026-09-18_world_followup/new_development_scenes/manifest.json)
were frozen before checkpoint selection (seed 20260918901) with zero overlap
against the 74 pilot expert scenes, original four development scenes, or 20
final scenes. A separate [full-verified overlap audit](../../outputs/experiments/2026-09-18_world_followup/new_development_scenes/full_verified_overlap_audit.json)
also found zero overlap with the 287 unique scenes in all 289 verified
episodes. The paired evaluation runner and its per-arm rootless profiles are
under `outputs/experiments/2026-09-18_world_followup/`; its exact commands,
model and scene hashes, score, latency and video artifacts are written into
each result directory when the two arms finish. The completed final scenes
are a known test set and cannot tune this choice.

## Bounded supervised initialization comparison

Both arms completed the fixed 2,500-update BC budget on the strict 60/14
native split. The fixed validation rule selected step 1,000 for the arm with
selected world weights and step 500 for the arm with a fresh world trunk. The
[experiment record](2026-09-18-world-supervised-init-ablation.md) pins model,
optimizer, data, source and checkpoint hashes. It also notes a key limit:
the fresh world weights were a separate seeded draw, not the original
pretraining ancestor's exact initial weights. This is one matched-architecture
initialization comparison, not a definitive causal estimate of pretraining.
The tokenizer, fresh heads, task projection, batch schedule and BC budget were
matched, with reward and imagination disabled.

On all 5,976 native held-out labels across 14 episodes, the selected
pretrained arm had **2.285 mm** first-command translation error versus
**2.578 mm** for the fresh-world arm. The last-three-second recording proxy
was **2.486 versus 2.734 mm**. The separate fixed selection metric was
**2.153 versus 2.352 mm**; terminal-only at that selection slightly favored
the fresh arm, **2.478 versus 2.428 mm**. These are imitation errors, not
physical insertion outcomes.

All **4+4 new development rollouts** were eligible and ran 90 simulated
seconds. The [paired live archive](../../outputs/experiments/2026-09-18_world_followup/supervised_world_development_archive/paired_summary.json)
has exact commands, scores, videos, logs, and hashes.

| New scene | Selected-world total | Fresh-world total |
| --- | ---: | ---: |
| 1 | 49.79, official partial | 19.01, no insertion |
| 2 | 35.69 | 36.68 |
| 3 | 36.67 | 36.37 |
| 4 | 9.27 | 12.91 |
| Mean | **32.85** | **26.24** |

Neither arm achieved a full insertion. The difference is concentrated in the
first scene, where the selected-world arm earned the sole partial. Live
observation-to-command latency across 1,789/1,786 decisions was **76.26 /
79.69 ms p95**; maxima were 146.79/135.71 ms and **zero decisions exceeded
300 ms**. These fresh development results support further supervised study,
but they do not justify reward/imagination training or a claim of reliable
insertion. Repeating with the exact ancestral random weights and multiple
seeds would resolve the initialization confound; collecting synchronized
post-event RGB/state and stronger near-opening dynamics is still needed for
world-model control claims.

The [post-run opening-geometry report](../../outputs/experiments/2026-09-18_world_followup/supervised_world_development_geometry/summary.json)
uses the scored bags and the explicit entrance TF, with no privileged geometry
fed to either policy. In scene 1 the selected-world tip ended about **0.2 mm
from the opening**, with **0.1 mm lateral error** and **5.7° orientation
error**, yet the official outcome remained partial. Its last-ten-second
requested/measured TCP gap was **38.5 mm** while sampled net TCP motion was
near zero. In the other near-opening stalls both arms likewise had roughly
40–42 mm requested/measured gaps. Across four scenes, median final lateral
error was **15.4 mm selected-world versus 25.6 mm fresh-world**, but the
fresh-world arm's mean minimum opening distance was smaller (7.8 versus
16.0 mm); the selected-world arm is not uniformly closer. Two scene-4
endpoints moved well past the entrance plane with large lateral errors.
These diagnostics point to alignment, commanded-versus-measured motion and
contact response as priorities for the next supervised policy experiment.

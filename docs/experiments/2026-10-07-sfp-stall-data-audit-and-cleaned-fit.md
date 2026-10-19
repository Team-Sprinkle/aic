# SFP transport stall-label audit and matched data cleanup

## Question and frozen comparison

The user observed long holds in earlier teacher recordings and asked whether
the newer actor training data had received the same cleanup as the SC route
dataset. It had **not**: the SC cleanup targets explicit planned endpoint
holds, while the SFP scored-stream admission joins synchronized measured TCP,
three images and recorded gripper/TCP commands, checks full-insertion score,
and cuts before insertion. It does not remove physical or controller stalls.

The hypothesis for this bounded test is that successful SFP routes can still
teach a harmful command when the arm is effectively stuck. The candidate
changes **only the action-loss sampling of audited sustained stalls**; the
source episodes, images, timestamps, previous-action histories, validation
rows, scene split, model architecture, initialization, seed and 2,000-update
budget remain fixed. The source dataset is preserved. No final scenes or RL.

## Read-only raw data audit

Command:

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_transport_data_quality.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/sfp_labels.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/data_quality_audit_v1.json
```

The 236 **legacy training** SFP episodes averaged 0.90% near-zero first
targets and 0.81% simultaneous near-zero target/measured-motion steps, so
they are *not* dominated by explicit zero-action holds. A different pattern
is substantial: on average **21.65% of raw steps** had measured TCP movement
<0.1 mm despite a next-target translation >5 mm. **68/236 legacy episodes**
had a continuous such interval of at least one second; the worst lasted
3.35 s. One inspected example (episode 0) remained around axial −90 mm,
lateral 6 mm from the selected opening for several seconds while its target
was about 6 mm away. It later moved and belongs to a verified successful
source. This looks like delayed or blocked actuation near the port, not an
invented zero label; the specific physical cause is not proven.
[A checked raw-trace plot](/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/stall_raw_examples.png)
compares that legacy example with a fresh diverse episode. It plots the
recorded first-target distance, actual per-frame TCP movement and axial
distance; the legacy plateau occurs inside the broad transport buffer.

The freshly scored diverse SFP train routes averaged **0.39% near-zero
first targets** and 0.22% simultaneous low command/motion steps. Only one
of 115 diverse train episodes had a sustained high-command/low-motion
interval ≥1 s. The 16 first-round paired train routes had none; the 20
second-round paired train routes had none. Thus applying the old SC-style
zero-hold compaction to these new SFP episodes is not justified by the data.
Source commands use `gripper/tcp`, camera/state joins enforce ≤100 ms age and
camera-triplet synchronization, `read_rows()` checks finite values and
scene-group split disjointness, and a 2,000-row new-source re-expression check
reconstructed port-frame translation from TCP-body labels to <3×10⁻¹³ mm.
No frame-conversion error was found in this bounded audit.

## Immutable cleaned training mask

Command:

```bash
.pixi/envs/default/bin/python scripts/build_sfp_stall_filter.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/sfp_labels.jsonl \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/route_handoff_divergent_round2_v1 \
  --pairs /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/paired_action_manifest_round2_v1.json \
  --output /home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/stall_filter_v1.json
```

The predeclared rule is measured TCP step <0.1 mm and first teacher target
>5 mm continuously for ≥1 s at the original recording cadence, with 0.1 s
padding on both sides. This marks **74 intervals in 70 train episodes**,
excluding **689 of 46,761 training cache rows** (688 legacy, one newer).
It excludes zero validation rows and **zero paired-contrast target rows**.
Among raw rows in the marked intervals, axial distance has median −91.4 mm
and 10th/90th percentiles −116.7/−89.7 mm; most are near the handoff area.
Consequently this cleanup may have little influence on the *earlier* route
drift observed in autonomous tests. It is a controlled data-quality test,
not a claim that the stall was the main failure cause.

`scripts/train_temporal_transport_probe.py --excluded-train-indices` now
removes only these rows from the action-loss sampler and old-SFP preservation
sampler. Historical observations remain at their real timestamps, so no
synthetic across-gap command is created. The original labels and feature
cache are untouched. This is safer than using
`scripts/compact_lerobot_stalls.py`, whose action recomputation is meant for
a different LeRobot dataset and would change the recorded target semantics.

## Other data gaps

- The legacy source is highly skewed: **227 of 236** legacy train episodes
  request card 0/port 1. The actor sampler balances the ten request codes,
  but within this one code old recordings still dominate. Recent diverse
  routes cover other codes more evenly; they do not erase old route behavior
  already present in the frozen warm start.
- The new SFP actor development split has 17 diverse 1–2-card resets, and
  the eight second-round paired development routes also have two cards.
  The older ten-route development set does include five 3-card, four 4-card,
  and one 5-card reset. Training includes varied 3–5-card resets, but the
  *newly diverse* offline validation adds no fresh 3–5-card groups and the
  five-card cell has only one older group. The three fresh live
  starts included 2, 3 and 5 cards and all failed buffered arrival. More
  nominal 3–5-card validation routes would measure this gap but not by itself
  repair the actor.
- All current nominal teacher routes mostly visit successful states. Fresh
  actor traces already differ from the teacher path by 13–22 mm median at
  8–12 s and later sometimes stop moving despite commands. Safe labels for
  **actor-visited off-route states** are the larger likely coverage gap.
  This remains a hypothesis until collision-checked teacher continuations
  are tested.

## Fit and stopping rule

All GPUs were occupied by existing Isaac/other-user jobs, so the matched
fit uses CPU without disturbing them. A 20-update CPU smoke run completed.
`scripts/run_sfp_stall_clean_matched.sh` runs two sequential 2,000-update
fits with the same frozen control initialization, paired loss 0.2,
old-SFP action-preservation loss 1.0, scene-balanced sampler and seed.
The only difference is the immutable exclusion mask above. Artifacts and
exact commands are under
`/home/nas/chmin/aic_archives/2026-10-07_sfp_stall_clean_matched_cpu_v1/`.

```bash
bash scripts/run_sfp_stall_clean_matched.sh \
  /home/nas/chmin/aic_archives/2026-10-07_sfp_stall_clean_matched_cpu_v1
```

The wrapper writes each fully expanded training command to
`baseline.command.txt` and `stall_filtered.command.txt` before execution.
The scoring commands used `scripts/score_sfp_diverse_transport_splits.py`,
`scripts/score_sfp_paired_actor_contrast.py`, and
`scripts/audit_sfp_temporal_task_sensitivity.py` with `--device cpu` and the
two selected checkpoints under that same result directory.

Score grouped old/new SFP and SC action p95, nine held-out same-scene
request pairs, and same-image task-code sensitivity. Do not proceed to live
testing if any grouped action p95 worsens >10% or task sensitivity drops
below 2×. Even if those offline checks pass, a live promotion still requires
≥2/3 safe buffered arrivals on new nonfinal starts, no worse contact and
p95 inference <300 ms. If filtering the near-port stalls makes little
offline difference, stop this cleanup branch and prioritize a bounded
actor-visited correction diagnostic rather than more nominal teacher data.

## Completed matched result and decision

Both CPU fits completed the full **2,000 updates**; the selected checkpoints
were update 1,750 for the unfiltered control and update 2,000 for the
stall-filtered candidate. `grouped_split_score.json`,
`heldout_pair_score.json`, `baseline_task_sensitivity.json`, and
`filtered_task_sensitivity.json` in the result directory score their frozen
weights on identical complete-episode development groups.

| Frozen held-out metric | Unfiltered | Stall-filtered |
|---|---:|---:|
| Old SFP first-action translation p95 | 1.493 mm | 1.498 mm |
| New SFP first-action translation p95 | 1.569 mm | 1.583 mm |
| SC first-action translation p95 | 2.155 mm | 2.193 mm |
| Nine same-reset request-pair error, median / p95 | 1.161 / 3.810 mm | 1.176 / 4.172 mm |
| Same-image ten-code action diameter, median | 1.568 mm | 1.554 mm |

The filtered candidate meets the *no >10% regression* screening threshold,
but shows **no positive signal** on any decisive offline metric and no reason
to spend a fresh live rollout. It is **not promoted**. The earlier live actor
gate remains 0/3 safe arrivals. A cleaned cache is still available as a
separate reproducible derivative, but overwriting the original expert data
or silently excluding valid near-port holds would overstate the evidence.

This negative result also narrows the likely actor problem: most excluded
rows were already near the target, while the live route diverged earlier.
The recorded stalls are real and worth knowing about, but these sustained
legacy windows do not explain the present early-route failure by themselves.
The older October 3 experiment already trained on ten successful
two-second-actor-prefix CheatCode continuations and still got 0/3 autonomous
buffered arrivals; repeating that same short-prefix BC recipe is not the
next justified data collection. A later diagnostic would need to target the
**specific 8–12 s route deviation** and establish safe recovery from that
later actor state before any further fit. A representation/control repair
that explicitly encodes requested route progress is another distinct branch.

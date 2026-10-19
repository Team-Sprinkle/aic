# Actor, pause proposal, and coarse perception continuation

## Starting evidence and limits

The 136 new successful SFP episodes improved held-out first-action error but
only changed same-image requested-code action diameter 0.572→0.620 mm. The
frozen probability head was timely on 26/27 teacher development routes but
made a false first pause proposal on one of three saved autonomous routes and
was overconfident after the pause region. The board-rail coarse label is not
a reliable visual landmark: the 20/40/70-group learning curve improved pixel
p95 only 7%, and a single-reviewer original-image audit marked 0/30 projected
points as unambiguously numbered visual features. These are distinct failures.

No final scenes, true geometry at runtime, insertion or RL are involved in
this continuation. Use rootless Gazebo and at most one GPU for collection;
leave the preexisting Isaac container alone. Model fits may use idle GPUs but
never more than four project GPUs concurrently.

## Bounded experiments and gates

1. **Requested-route data.** Query an alternate far-apart card/port request
   on twenty already verified *training* resets (five each with 2–5 cards),
   leaving scene and robot initialization identical. The original run already
   supplies the first request. Stop after five if fewer than four alternates
   fully insert. Admit only full-scoring, causally joined alternate routes.
   Audit measured TCP, RGB and first action at matched times; accept pair
   supervision only while TCP differs ≤2 mm, orientation ≤2°, frozen RGB
   feature cosine ≥0.995, and next body-frame target differs ≥5 mm. Require
   at least thirty matched row pairs before one paired-action fit. Never force
   late divergent observations to be a shared-observation pair.
2. **Actor fit.** If the pair gate passes, compare one paired-supervision fit
   with its matched sampling-only control, on the same complete episode split,
   initialization, budget and action targets. Require at least 2× same-image
   request sensitivity, no >10% first-action p95 regression on old SFP, new
   SFP or SC development groups, before at most three new nonfinal live
   starts. Require ≥2/3 safe buffered arrivals without extra contacts and
   complete p95 <300 ms to consider the actor improved. The actor still
   outputs complete TCP-body action chunks; no privileged target pose input.
3. **Pause-proposal repair.** Build training negatives from authentic frames
   after or outside the -130 to -60 mm / 30 mm pause region. Keep all existing
   grouped development scenes out of optimization. First test whether
   successful teacher tails supply valid synchronized three-camera examples;
   actor-route negatives are needed if they do not transfer. Fit the head
   only with frozen action weights. Gate on grouped teacher event timing and
   zero false first proposals on separate saved off-route runs; report raw
   and prior-vetoed false rows and calibration. Keep the head shadow-only if
   the off-route gate fails.
4. **Coarse perception.** Stop using projected rail ends as positive visual
   identity labels. Review original three-camera images and causal history
   without projected marks to identify an actually observable slot origin or
   selected-opening cue. Use only independent, episode-grouped training
   labels. A bounded target-specific crop or identity head is warranted only
   if the cue is visible on reviewed sparse and dense card layouts. Require
   zero accepted wrong ports, ≥9/10 early new-group target anchors, and full
   p95 inference <300 ms before linking it to the proposer/refiner. If no
   reliable cue exists, document the observation gap and test a safe view
   change as a separate experiment, rather than scaling ambiguous CAD labels.

## Current execution

The immutable paired collection plan is at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_v1/plan.json`.
Collection uses `scripts/collect_sfp_divergent_pairs.py` on rootless Gazebo
GPU 1. Results, exact commands, decisions, and artifacts will be appended
after each gate. Failure cases remain archived.

## Paired collection and pause-head evidence

The first five alternate requests yielded five full insertions, passing the
pilot gate. The complete twenty-reset collection yielded **16 full insertions**;
four partial/no-insertion attempts remain in the immutable collection archive
and were excluded. The successful streams added **16 training episodes / 4,130
transport rows** to a new dataset at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_v1`.
The original dataset was preserved. A raw same-reset stream audit found 53
matched early command rows with at least 5 mm different requested-action
targets, before visual feature matching and 5 Hz cache thinning. The frozen
feature-pair gate and matched actor comparison are still required before
claiming improvement. Exact collection and admission scripts are
`scripts/collect_sfp_divergent_pairs.py`,
`scripts/extract_sfp_paired_coverage.py`,
`scripts/audit_sfp_divergent_pair_streams.py`, and
`scripts/admit_sfp_paired_coverage.py`; the audit is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_v1/final_stream_pair_audit.json`.

For the pause proposal, balanced successful teacher tails added genuine
post-cutoff negatives: 30 train and ten held-out episodes across all ten SFP
card/port codes. The tail label cache is
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/pause_tail_cache_balanced_v2`.
A linear frozen-action head reduced tail false rows to zero but fell to 9/27
timely proposals on the original teacher development group. A small nonlinear
head on the same frozen actor hidden feature plus public state/task reached
27/27 timely original and 10/10 timely tail teacher proposals, but made six
and four raw false-positive rows respectively. On three *saved autonomous*
actor routes, it made zero false proposals, two timely proposals, and missed
the third short valid window (six valid rows, maximum score 0.72 versus the
predeclared 0.95 threshold). These are shadow replay results, not live
handoff. Checkpoint, grouped summary and per-decision replay are in
`outputs/experiments/2026-10-07_sfp_pause_mlp_hardneg_v1/`.

## Coarse-card and requested-port checks

Fresh diverse-scene RGB generated 119 training and 17 development reset
groups. The frozen occupancy head got 15/17 exact on the 17 sparse development
groups. Training *only* its occupancy modules on 99 train groups gave 17/17
development and 20/20 on twenty separate train-origin untouched groups; the
same frozen head had scored 18/20 there. The repaired head never made a
confident empty-slot vote on these groups. Its artifact is
`outputs/experiments/2026-10-07_sfp_diverse_occupancy_head_v1/`. This is a
card-presence result, not selected-port identity or a handoff result.

The existing frozen selected-port native-crop model, with crop centers from
frozen RGB board predictions, accepted the requested opening early on 17/17
sparse-card groups with zero accepted wrong ports and 2.87 mm accepted 3D p95.
On a predeclared independent eighteen-group **3–5-card** check it reached
15/18 early, zero accepted wrong, and 3.17 mm accepted p95. It failed the
required ≥16/18 early gate. Two missing groups show a confident wrong
neighbor in one camera; the third has near-correct pixels but 7.16 px
cross-view reprojection versus a 5 px consistency gate. See
`outputs/experiments/2026-10-07_sfp_dense_port_crop_frozen_18_v1/` for
per-frame predictions and
`outputs/experiments/2026-10-07_sfp_dense_coarse_board_frozen_18_v1/`
for the card-vote control. Do not tune on those eighteen scored groups or
promote the crop yet. A distinct multi-peak cross-view correspondence repair
is justified on training/development groups, followed by new nonfinal groups.

## Second bounded continuation and stopping rules

The first feature-verified pair manifest found **20 valid ≥5 mm training
pairs, one per scene, and no paired validation scenes**. Stage counts were
3,593 candidate row joins, 3,371 time-aligned, 136 within the measured TCP
pose gate, 107 passing frozen visual-feature cosine, and 20 with ≥5 mm
different teacher first actions. This fails the 30-pair fit prerequisite.
Therefore no paired actor fit started. A second immutable plan uses twenty
previously unused training resets (five per 2–5-card count) and nine unused
two-card development resets, again changing only the public requested
card/port code. The development routes will **never** be optimized. The plan
and saved results are at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_round2_v1/`.
Stop if the same feature/pose/action gate remains below thirty training pairs
or yields fewer than four held-out development pairs. No threshold loosening on those
scored scenes. Collection may run longer than fitting but is bounded to 29
routes. The first training route fully inserted.
The twenty planned second-round **training** alternates all fully inserted.
Eight of nine grouped development alternates fully inserted; the remaining
development run timed out after an official 0-score task failure. Its audit
is preserved and excluded. All 28 successful scored streams were extracted,
causally admitted as **7,249 additional transport rows**, and kept in their
original complete-reset splits (20 train / eight development) in
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/`.
No second-round actor fit started before the frozen-feature pair audit.

The accepted branch rows occur 0.21–0.45 s into the CheatCode interpolation.
Recorded *previous action chunks* already differ across the two teacher
branches (up to 8.6 mm in a fourth future target), even where TCP and images
match. The paired-loss construction deliberately supplies the **same left
history and previous commands** to both actor queries and changes only the
public request code. The alternate teacher target is a local counterfactual
label, not an exact replay of an identical action history. This early teacher
interpolation explicitly resets its XY integrator on every call and computes
its next command from the current gripper/plug pose, requested port and
interpolation fraction; previous command chunks are not read by that rule.
 The remaining ≤2 mm TCP and ≤100 ms time differences are reported for every
pair, and the held-out paired-request score is required before promotion.

If the pair gate passes, `scripts/run_sfp_paired_actor_comparison.sh` fixes
one 2,000-update matched comparison: both arms warm-start from the same
October 6 actor, sample the same merged train rows with balanced requested
identity and 0.5 same-scene paired-scene mass, and use seed 20261007. The
control has paired-contrast weight zero; the sole candidate weight is **0.5**.
Both use full-action TCP-body targets; no delta adapter or ground-truth port
pose is an actor input. The actor output still consists of complete
TCP-body action chunks. The pair manifest may supervise only their first
translation difference because that is where the near-identical observed
branches have a reliable counterfactual target.

The separate native-crop repair tests the concrete hypothesis that the
selected-port model lacks dense 3–5-card training views. It starts from the
frozen October 6 crop checkpoint, uses only the new verified RGB and target
projection labels, and trains on **91** scene groups; **27** disjoint groups
(17 sparse plus ten dense) select the checkpoint. Another **18 dense groups**
remain outside training/selection for the fixed one-shot stress check. These
splits are at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_diverse_crop_fit_split_v1`.
Fit for at most 2,000 updates/120 minutes on one idle GPU. On the stress
groups require ≥16/18 early accepted, zero wrong accepts, accepted 3D p95
≤20 mm, and no regression on the older sparse development groups. If the
gate fails, stop this data-only crop branch and try the distinct multi-peak
cross-view method on fresh development groups. The already scored eighteen
groups are never a source of threshold tuning.

## Dense-view native-crop repair result

The fixed fit used **27,508 request-expanded training crops** from 5,406
three-camera frame groups / 91 reset scenes. It completed the full 2,000
update cap and selected update 1,800 on the 27 grouped development scenes.
The old frozen crop checkpoint accepted early in **26/27** development groups;
the new checkpoint also reached **26/27**, with zero accepted wrong ports and
more accepted decision rows (771→1,084). Its development accepted 3D p95 was
2.05 mm. The selected checkpoint is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_crop_fit_v1/best.pt`.

On the fixed 18-group dense stress set, the original checkpoint reached
**15/18** early anchors, zero accepted wrong, 3.17 mm accepted 3D p95 and
157 accepted rows. The new checkpoint reached **18/18**, zero accepted wrong,
**2.22 mm** accepted 3D p95 and **546** accepted rows. On the older ten sparse
development groups it reached **10/10** early and zero accepted wrong,
versus 6/10 early for the previous checkpoint's original selection result.
These groups are not reserved final scenes; the eighteen dense groups had
previously been scored for the frozen baseline, but were excluded from both
training and checkpoint choice for this single targeted fit. Do not iterate
thresholds on them.

The combined observation-only board → native crop → crop network → three-view
triangulation was profiled for 60 sequential decisions on GPU 2, with native
images preloaded into host memory: **p50/p95/p99 14.4/16.0/17.2 ms**. This
includes image resizing and network inference, and excludes camera capture,
ROS scheduling and transport actor inference. It is well below the 300 ms
model-path budget, but is not a live end-to-end timing claim. Artifacts:
`/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_crop_stress18_v1/`,
`/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_crop_old_sparse10_v1/`,
and `.../2026-10-07_sfp_diverse_crop_fit_v1/latency_60.json`.

This clears the *offline requested-port crop* gate. The upstream actor must
still reach a safe buffered arrival, and the pause proposal must be reliable
on new autonomous routes before automatic handoff can be enabled. Original
RGB is used for crop inference; CAD projections enter only offline labels
and scoring, never crop selection.

## Matched paired-request actor result and one targeted repair

The final feature cache contained 53,541 rows. Its unchanged audit thresholds
admitted **34 training pairs across new/old same-reset scenes** and **nine
held-out pairs from six complete development reset groups**. This cleared
the fixed ≥30/≥4 fit gate. `scripts/run_sfp_paired_actor_comparison.sh` ran
control and paired candidate on GPUs 2 and 3 with the same checkpoint, train
rows, balanced sampler, seed and 2,000 updates. Both finished at the update
cap and selected update 2,000. The scored pair manifest and all commands,
logs, checkpoints and split reports are under
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/`.

The paired loss **did learn a requested-route effect**. On the nine held-out
same-scene alternate requests, median first-action request-difference error
fell **3.22→1.17 mm**; p95 fell **7.71→3.65 mm**. Median predicted branch
magnitude changed **1.05→4.54 mm** against the 4.40 mm teacher branch;
direction sign was correct on 9/9 for both arms. On all SFP development
observations, changing only the public task code increased the median
ten-code first-action diameter **0.686→1.701 mm (2.48×)**, passing the
predeclared 2× task-sensitivity criterion.

The same candidate **failed the action-error regression gate**. Compared
with its matched sampling-only control, first-target translation p95 was
**1.481→1.805 mm on old SFP** (+21.9%), **1.704→1.892 mm on new SFP**
(+11.0%), and **2.359→2.249 mm on SC** (improved 4.7%). The prior frozen
actor had old/new SFP 1.676/1.697 mm and SC 2.032 mm on this exact score
cache. No live rollout or actor promotion followed.

**One distinct bounded repair, chosen after this failure:** The 0.5 pair
loss weight may force the shared action model away from ordinary BC targets.
Train exactly one otherwise matched candidate at weight **0.2**, same 2,000
updates and seed, and compare it with the existing zero-weight control.
Continue toward at most three new nonfinal live starts only if its task-code
sensitivity remains ≥2× and old SFP, new SFP and SC first-action p95 each
regress by no more than 10% against the matched control. If either criterion
fails, stop this pair-loss branch and diagnose representation or route-policy
design instead of sweeping weights. No final scenes or RL.

The fixed weight-0.2 repair finished 2,000 updates and selected update 1,750.
It preserved held-out pair learning (median/p95 request-difference error
**1.14/3.62 mm**, predicted branch 4.75 mm) and 2.44× same-image request
sensitivity. But old SFP action p95 was still **1.820 mm** versus 1.481 mm
for the matched control (+22.9%); new SFP was 1.749 versus 1.704 mm (+2.6%)
and SC improved to 2.123 versus 2.359 mm. The hypothesized global pair-loss
weight tradeoff was therefore **not** the explanation. No live rollout.

**One final supervised trust-region diagnostic:** The pair loss may conflict
specifically with ordinary old SFP behavior, since training examples from
different requested routes pull shared weights apart. Initialize from the
matched zero-weight control, keep that checkpoint frozen as a reference, and
train with the same weight-0.2 pair loss plus **weight-1.0 full-action
preservation** on old SFP *training* scenes without alternate-request pairs.
This is still one full-action actor; it has no delta correction head or
privileged input. Use the same 2,000-update cap and no more than one fit.
Require ≥2× same-image request sensitivity, ≤10% action-p95 regression on
old SFP/new SFP/SC against the control, and reduced held-out paired-request
error. If those fail, stop paired-loss actor training and move to a different
route representation or policy design. Do not tune on final scenes.

The trust-region fit used the same 34 training pairs, with the frozen matched
control as its old-SFP action reference. Its 2,000-update cap completed;
the action-validation-selected checkpoint was update 1,000. On the exact
grouped validation rows, old SFP first-action p95 was **1.459 mm** versus
1.481 control, new SFP **1.678 versus 1.704 mm**, and SC **2.079 versus
2.359 mm**. Held-out pair median/p95 error was **1.23/4.01 mm** versus the
control 3.22/7.71 mm, and same-image task-code diameter **1.565 versus
0.686 mm (2.28×)**. Thus the trust-region actor clears the *offline* gate;
this does not certify autonomous transport. The code, checkpoint, training
manifest and grouped score files are in
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/paired_preserve_old_v1/`
and adjacent `preserve_*.json` reports.

For the live gate, a fixed nonfinal plan at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_v1/`
chooses unused historical reset configurations for one 2-card, one 3-card
and one 5-card start, with up to three teacher candidates per count to
obtain a successful reference. The two actors run on each same scene with
no guide, true geometry, or pause intervention. Require at least **2/3 safe
buffered arrivals**, no loss versus the baseline, no worse official contact
score, and p95 actor inference under 300 ms. Otherwise the new actor remains
unpromoted and the precise crop remains disconnected from handoff.

**Live launch repair:** The first fresh two-card CheatCode reference fully
inserted, but the initial baseline actor did not issue a proposal: policy
initialization failed because the rootless evaluation container mounts the
repo and `/var/tmp`, while both new actor checkpoints were stored on NAS.
That baseline timed out without a runtime trace; the same checkpoint-path
error was visible in the second actor's policy log. No actor motion from
either is a valid test. Failed audits remain under the original
`2026-10-07_sfp_paired_actor_live_starts_v1/` archive. The timed-out,
policy-free containers were cleaned up; the unrelated long-running Isaac
container was untouched. SHA-identical checkpoint copies were staged at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/live_checkpoint_mounted_v1/`;
a rootless-container preflight confirmed both files are readable. The
identical frozen scene plan and successful teacher were copied into
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_mounted_v2/`.
Only checkpoint file placement changed for the restarted matched live test.

## Fresh autonomous gate and bounded failure diagnosis

The restarted matched gate finished on the **same three nonfinal resets**,
with a successful full-insertion CheatCode reference on each. Both actor
checkpoints loaded and issued 301–312 proposals per run. Each actor used only
its three images, public task request and robot observations; the target-port
geometry and teacher route entered posthoc scoring only. Neither actor used a
guide, recovery rule, pause intervention or privileged input. The exact plan,
rootless launcher logs, videos/frames, `runtime_trace.jsonl`, official scores,
`posthoc_score.json` and `summary.json` are under
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_mounted_v2/`.
The policy-free failed-launch attempts remain archived in the original `v1`
directory and are excluded from this comparison.

| Requested scene | Control closest lateral in axial buffer | Trust-region actor | Official contact score, control → candidate |
|---|---:|---:|---:|
| 2 cards, card 3/port 0 | 15.99 mm | 23.69 mm | 0 → 0 |
| 3 cards, card 2/port 0 | 27.33 mm | 13.08 mm | 0 → −24 |
| 5 cards, card 2/port 0 | 36.49 mm | 31.36 mm | 0 → 0 |

The required corridor is lateral **≤10 mm** while axial is between **−100
and −70 mm**. Both actors reached it on **0/3** starts. The candidate helped
the latter two lateral distances but worsened the first; it also contacted a
NIC card mount with the left gripper finger on the three-card run. That run
had an additional official −12 insertion-force penalty: above 20 N for 4.62 s,
maximum 30.21 N. Both actor tier-3 insertion scores were zero. Candidate
actor inference p95 was **102.96 ms**, under 300 ms, so timing was not the
promotion blocker. The ≥2/3 safe-arrival and no-worse-contact gates failed;
**do not promote this actor**.

The one bounded read-only diagnosis used
`scripts/diagnose_sfp_paired_actor_live.py` and wrote
`route_motion_diagnosis.json` next to the live summary. It compares each
measured TCP position to the *nearest point anywhere on the successful
teacher's measured path*, avoiding a false error from different speeds. The
candidate's median nearest-path errors at 8–12 s were **18.3, 13.4 and
21.6 mm** on the 2/3/5-card starts, respectively; by 12–16 s they were
**17.5, 19.6 and 28.5 mm**. The corresponding control errors were **7.9,
22.3 and 23.2 mm** at 8–12 s. Thus a route difference is already present
before the later standstill. At 20–30 s, the candidate issued first-chunk
translation commands of median **35.0, 38.4 and 17.1 mm** in the respective
runs, while the measured TCP advanced less than 0.5 mm between proposals
on **85%, 29% and 100%** of decisions. At 30–60 s those fractions were
**99%, 89% and 99%**. This is a descriptive command-versus-motion mismatch;
the three-card official contact/force report proves a physical collision in
that run, but the other two standstills cannot be assigned uniquely to cable,
card, joint/controller constraints or another cause from these logs alone.

The immediate data gap is **actor-visited off-teacher states** and their safe
continuations. Another ordinary teacher-route dataset or teacher-forced
first-action fit has no supporting evidence after three supervised fits and
the 0/3 live result. The pause MLP is tied to the older frozen actor's hidden
features and remains a shadow checkpoint; it cannot be attached unchanged
to the trust-region actor. The improved crop is an offline, original-RGB
selected-port result, still around 2 mm p95 and disconnected from live
handoff. **Transport is the present bottleneck.**

### Explicit next experiment and stopping rule

Use one **bounded on-policy route-correction diagnostic** on training resets:
select at most six actor rollouts across 2–5 cards, save the first sustained
10 mm deviation from a successful same-reset teacher path, and relabel only
those visited observations with a collision-checked teacher continuation.
Before any fit, require at least four continuations to regain a safe route
and full insertion; reject states for which CheatCode simply drives into the
same obstacle. Train one matched 2,000-update full-action actor with these
corrections and one control with equal update/data budget, while preserving
ordinary SFP and SC behavior. Stop if grouped action p95 worsens >10%, then
run at most three fresh nonfinal autonomous starts only if offline gates pass.
Continue this direction only if safe buffered arrivals improve to **≥2/3**,
contact score does not worsen, and full actor p95 stays <300 ms. If the
teacher cannot recover at least four of six visited states, first isolate the
obstacle/controller cause with contact and reference-pose traces and design
a safe recovery label; do not generate more nominal teacher routes. The four
reserved final configurations stay sealed. Handoff, insertion and RL remain
disabled until the transport gate passes.

**October 7 data-quality follow-up:** The newer SFP scored routes did not
receive the old SC hold-compaction pass, but a raw-label audit found almost
no sustained low-command/low-motion holds in them. It did find 68 legacy
training episodes with ≥1 s of high command and very low measured motion,
mostly around axial −90 mm near the handoff area. A separate immutable mask
excluded 689 affected training-cache rows, and two matched 2,000-update CPU
fits tested it. The filtered actor was slightly worse on every grouped
action p95, held-out requested-route pair error and same-image task-code
sensitivity; it was not run live or promoted. The full result and checked
raw-trace plot are in
[the stall data audit](2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md).
This result reduces the rationale for broad cleanup. Also, the earlier
October 3 two-second-prefix corrective-BC experiment already failed a 0/3
live gate after ten successful teacher continuations. Any next correction
must target the later observed route deviation and pass a distinct safety
and live gate; do not repeat the old short-prefix collection recipe.

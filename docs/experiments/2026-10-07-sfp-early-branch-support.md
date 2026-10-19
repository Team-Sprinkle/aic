# SFP early requested-route branch: causal data and representation gate

## Pre-fit question and fixed budget

The latest later-state BC correction reached the safe transport buffer on
only 1/3 fresh starts. The five-card failure took a route near card 0 instead
of requested card 1. Before fitting another actor, test whether successful
same-reset, different-request teacher episodes supply a causal early branch
label that public observations can predict across card counts.

Frozen archive audit: `scripts/audit_sfp_route_branch_support.py` on
`route_handoff_divergent_round2_v1`, preserving the train/validation scene
split. A same-observation pair requires ≤110 ms time gap, ≤2 mm measured TCP
gap, ≤2° orientation gap and frozen three-view feature cosine ≥0.995. Its
label is the difference between *measured* next-one-second TCP displacement
for the two requested routes. This uses future teacher motion for offline
supervision only. The deployment actor would see public RGB, task code and
measured state, not future motion or simulator geometry.

The audit found 151 same-observation rows in 66 training scenes and 22 rows
in eight validation scenes. Immediate teacher action differences ≥5 mm occur
in only 34 training scenes and four validation scenes; none reached 10 mm.
Yet next-one-second path separation ≥10 mm occurs on 55 training scenes and
six validation scenes. The first measured 10 mm pair route separation appears
at about 1.04 s median in training and 1.16 s median in validation. Thus
one-step action contrast alone misses the early branching consequence.

`scripts/render_sfp_route_branch_review.py` saved original three-camera
paired frames without geometry overlays for every one of the eight paired
validation scenes. They show the row and occupied cards, but the absolute
card index is not consistently obvious by human inspection at this scale.
This is an observability warning, not a blinded accuracy claim.

The first `scripts/probe_sfp_early_route_branch.py` CPU ridge diagnostic
used complete scene-grouped splits, 3,844 training and 278 validation
early samples, and task-balanced training. On these teacher states, next-one-
second motion p95 was 15.59 mm for TCP pose+task, 10.20 mm after adding the
frozen three-view RGB features, and 41.70 mm for a constant mean. Shuffling
task codes raised the visual model's p95 to 25.07 mm; shuffling visual
features across held-out scenes raised it to 27.13 mm. A fuller 32D state
improves the score further, but contains controller error from previous
teacher commands, so it is not clean evidence of independently perceived
target identity. The pose-only comparison is the cleaner causal diagnostic.

**Important split limitation:** all eight existing paired validation scenes
have two cards. The training pairs include 18/11/21 scenes with 3/4/5 cards.
An immutable *probe-only* holdout of five scenes per harder card count was
selected before a separate CPU probe. It yielded 239 samples across 15
complete scenes. TCP pose+task p95 was 15.57 mm and the visual version
10.96 mm; shuffled task/visual p95 was 30.43/22.05 mm. On 26 same-observation
pairs with ≥10 mm future separation, both clean-input probes got the broad
branch direction right, but median branch-difference error was about 6 mm.
The parent actor previously trained on these 15 scenes, so this is a
representation diagnostic, **not** independent actor validation.

### Targeted independent hard-card teacher check

To close that exact split gap, collect at most **six new nonfinal scene pairs**:
two each with 3, 4 and 5 installed cards. Each pair keeps board, robot
start and cameras fixed and changes only the public requested card/port;
select distinct occupied cards far apart in the row. Run ordinary rootless
Gazebo CheatCode for both requests; keep the exact configurations and all
failures. Do not add these scenes to actor training or select a checkpoint
with them. Use GPU 7 sequentially only if it remains free, and leave the
existing Isaac/other-user jobs untouched.

Stop the teacher collection at six pairs; do not replace failures by easier
scenes. The data gate is **at least four pairs** in which both requests
officially fully insert with no contact penalty, have causally joined
three-camera/robot measurements, and differ by ≥10 mm in measured TCP motion
one second after an early same-observation point. If fewer than four pass,
stop the branch experiment and diagnose the teacher/scene or observation
gap before actor training. If at least four pass, evaluate the frozen
pose+task and RGB+pose+task diagnostic without fitting to these scenes.
Only then consider one ≤2,000-update matched *full-action* actor experiment
with a predicted coarse future-motion goal. The actor must predict complete
TCP-body targets, not a delta atop the old action. The existing early-route
FiLM/discrete-class and 9D route-conditioned heads failed action-p95 gates;
a new fit must first show that its 3D branch target actually improves on
the independent hard-card pair metric. Require no >10% old/new SFP or SC
action p95 regression and ≥80% correct, ≥10-mm-separated branch direction
before at most three fresh nonfinal autonomous starts. The live gate remains
≥2/3 safe buffered arrivals, no worse scored contact, p95 <300 ms.
Keep pause/coarse shadow-only and insertion/RL disabled. Final scenes stay
sealed.

## Artifacts so far

- `outputs/experiments/2026-10-07_sfp_route_branch_support_v1/audit_round2.json`
- `outputs/experiments/2026-10-07_sfp_route_branch_support_v1/validation_review/`
- `outputs/experiments/2026-10-07_sfp_route_branch_support_v1/{probe_v2.json,hard_card_holdout.json,probe_hard_card_v2.json}`

## Independent teacher and predicted-goal check

The six frozen new 3/4/5-card paired scenes are at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_hard_branch_teacher_gate_v1/`.
The exact configurations, commands, rollouts and failures are retained there.
Four of six pairs met the predeclared criterion: both requests reached the
official full-insertion tier (75), had no contact penalty, and their measured
one-second early motion differed by at least 10 mm. The other two pairs had
partial insertion on one request (tier 3 about 38 mm), despite 18 and 33 mm
early route separation. They were not replaced. The four passing pairs had
3, 4, 5 and 5 cards.

The frozen ridge probe was scored once on those four untouched passing pairs
using `scripts/score_sfp_hard_branch_goal_probe.py`; see
`outputs/experiments/2026-10-07_sfp_route_branch_support_v1/independent_hard_card_score_v1.json`.
One-second future TCP motion p95 error was **18.57 mm** using current TCP
pose plus public task code and **9.28 mm** when adding the three original
camera features. There were 72 early samples and nine same-observation
route pairs across four independent scenes. Broad branch direction was
100% for both baselines, so it does not discriminate them; branch-difference
median error was 9.15 vs 7.47 mm. This is a small representation diagnostic,
not autonomous route success.

## Matched actor fit and phase support repair

`scripts/train_temporal_transport_probe.py` now optionally supplies the
actor a **predicted 3D one-second TCP-motion goal**, derived from public
three-camera features, measured TCP pose and public task code by the frozen
ridge. It is rotated into the TCP frame; the actor still outputs the full
24D TCP-body action chunk. Isaac/Gazebo geometry and future motion are
training labels only, never actor inputs. Control and candidate share parent
checkpoint, cache, split, sampler, seed and 2,000 updates. The control
receives zero in that goal channel.

The first candidate at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_route_goal_matched_v1/`
incorrectly used the early (0–1.6 s) goal probe throughout the route. A
separate phase audit in `goal_by_phase_v1.json` found held-out goal p95
error rose from **10.20 mm** in its training phase to **71.56 mm** at 1.6–3 s
and above 140 mm later. This candidate is retained as a failed
out-of-support ablation and was not sent to live evaluation.

The corrected candidate at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_route_goal_matched_v2/candidate_progress10/`
zeros the goal after the measured TCP has moved more than 10 mm from its
episode start. That progress criterion, rather than elapsed time, matches
the probe's observed support. The same rule is embedded in deployment.
Both corrected candidate and matched control selected update 2,000. On
ordinary held-out rows, first-action p95 was **1.438→1.451 mm** for SFP and
**1.813→1.825 mm** for SC, a 0.85% and 0.66% regression, within the 10%
cap. `paired_actor_contrast_progress10_v1.json` checks nine paired rows
across six existing two-card validation scenes under the same image/history
with changed public task code. Broad action-change direction was 100% for
both actors. The median error in their action difference was 2.025→2.000 mm;
p95 was 5.880→5.832 mm. This small change does not establish live benefit.

The fresh nonfinal live start plan is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_route_goal_live_v1/plan.json`.
It excludes the existing train/validation labels, prior actor live groups and
the six new teacher-gate scenes. The matched autonomous comparison is the
next decisive gate; no insertion actor or RL is enabled.

## Fresh autonomous result and next diagnostic

The fixed three-start test completed at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_route_goal_live_v1/`.
All three matching CheatCode references fully inserted. The matched control
reached the safe approach buffer on **1/3** starts; the corrected predicted-
goal actor reached it on **0/3**, failing the ≥2/3 live gate. Neither actor
autonomously inserted. Both incurred a −24 contact score on two cards; the
candidate avoided the control's −24 on five cards but remained far from the
requested opening. Worst rollout p95 inference was 89.53 ms control and
90.26 ms candidate, under 300 ms. The candidate is **not promoted**.

**Split correction:** A later complete-source audit found that this three-
start live plan accidentally included two scenes already present in the
actor's train groups and one in its validation groups. The live failure is
real, but **0/3 versus 1/3 is not an unseen-scene generalization estimate**.
The previous scene picker excluded an older label list, not the full actor
label source. It now accepts all label sources. The replacement nine-scene
candidate plan at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_route_goal_unseen_live_plan_v2/plan.json`
has **zero** overlap with the full actor SFP labels; three groups are set
aside for correction-state development and six for a later autonomous live
gate. None has been used to select a checkpoint. The three successful
takeovers on the earlier live plan remain **diagnostics only**, excluded from
the new training and development merge.

Read-only route diagnosis on the six actor rollouts is archived as
`route_motion_diagnosis.json` and `first_causal_route_error.json` next to the
live plan. First >10 mm departure from the successful teacher path occurred
at 5.15–6.00 s for the control and 5.15–5.65 s for the candidate. Candidate
closest lateral distance to the requested opening was 4.9, 23.1 and 38.3 mm
for 2/3/5 cards; the corresponding control values were 2.9, 26.1 and
39.3 mm. Both arms had high early command-direction cosine against a teacher
at nearby measured TCP states (median 0.92–0.99, only 4–6 matches per run),
yet drifted and later stalled. The candidate's goal channel switched off
after 10 mm measured progress at 4.45–4.90 s, shortly before route error,
but the control had similar early error timing. This timing alone does not
establish the switch as the cause. A five-camera candidate review video is
at `rollouts/fresh_cards5_card3p1_n400__ec2_02_sfp_hard_trial_000070_candidate/five_cameras_10fps.mp4`.

**Predeclared next check:** On these same three already-open development
resets, run the unchanged control actor for 5 s then hand over to privileged
CheatCode. This is a teacher recoverability diagnostic, never an autonomous
success. Cap at three runs, no replacements or routine new scene collection.
Require at least two official full insertions with nonnegative contact score
and valid actor-prefix/teacher-transition logs. If it passes, the next
specific repair is to preserve the actor's camera/action/motion history in
training examples at the **first** 5–10 mm route departure and train a matched
full-action actor on separate training resets; the old correction materializer
began history at takeover and had only port-1 targets. A new development
split and fixed ≥2/3 live safe-buffer gate would then be required. If teacher
recoverability fails, stop imitation relabeling and diagnose controller/
contact constraints in the exact takeover window. Final scenes stay sealed.

### Recoverability check passed; bounded training-source gate

`/home/nas/chmin/aic_archives/2026-10-07_sfp_first_departure_takeover_v1/summary.json`
records **3/3** official full insertions with zero contact penalty after the
unchanged actor ran for five seconds, including the held-out port-0 request.
These are explicitly privileged teacher takeovers, not autonomous insertion.
The first attempt to materialize actor-prefix history was preserved at
`history_context_validation_v1/` and stopped when mixed command frames were
detected: the actor issued absolute `base_link` targets, while CheatCode
issued `gripper/tcp` deltas. The materializer now converts each according to
its recorded frame and rejects unknown frames. The corrected isolated
validation materialization at `history_context_validation_v2/` contains 93
actor-prefix context chunks and 682 teacher-only supervised transport chunks
across the three scenes. The old six safe port-1 takeovers were rematerialized
train-only with 389 context and 1,383 teacher chunks at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1/history_context_train_v1/`.

Before fitting, test request-port diversity on **four** additional fixed
train-only source resets, one each with 2/3/4/5 cards and a port-0 request,
using the unchanged actor for five seconds then CheatCode. Do not replace a
failure. Require at least **3/4** official full insertions with no contact
penalty and valid causal joins. These are solely training-source checks; the
three development takeovers above stay out of training. If fewer than three
pass, stop this corrective fit and inspect the failed source. If the gate
passes, admit only successful teacher continuations plus actor-prefix history
context, audit the loss mask, then compare one matched full-action fit against
a zero-correction control on a separate episode-grouped development split.

The port-0 gate passed **4/4**, with full insertion and zero contact penalty
on 2/3/4/5-card train-only starts. The output is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_port0_first_departure_train_gate_v1/`.
Before actor fitting, run one fixed three-scene, truly unseen 2/3/5-card
actor-prefix-to-teacher diagnostic from the replacement candidate plan.
Require at least **2/3** safe full teacher insertions; admit only successful
ones as episode-grouped **development** correction examples. Reserve the
other six unseen groups for autonomous live comparison. Do not promote on
the three previously leaked scenes, and do not count any teacher takeover as
autonomous success.

All three unseen 2/3/5-card validation takeovers passed safe full insertion;
their record is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_history_context_dev_takeover_v1/`.
The first history materializations failed the command-frame and repeated-
timestamp audits; the episode-ID range then collided with the base labels.
These failed versions remain archived. The admitted **v3** materializations
use separate 7-million episode IDs and collapse same-timestamp actor commands
to the final effective target. A strict original-image/causal-order audit
passed on all ten train and three development episodes, with 447 actor-prefix
context and 2,289 teacher rows for training, and 94 context plus 696 teacher
rows for development. `scripts/merge_sfp_actor_prefix_context_labels.py`
verified zero train/development reset overlap and no actor command admitted
to teacher supervision. The rootless GPU-7 cache at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_context_v1/`
contains 54,427 rows, including 116/571 cached train context/teacher rows
and 25/174 cached development context/teacher rows. A read-only history
check found six actor-prefix context keyframes in the first teacher row's
causal window in both train and development.

**Fixed matched-fit gate:** start both action-only actors from the same
unchanged control checkpoint, use the same merged cache, grouped split,
sampler seed, 2,000 updates and checkpoint-selection rows. The control
excludes the new corrective teacher targets; the candidate gives them 10%
of SFP sampling mass. Both exclude actor-prefix context from supervised
loss. On untouched development correction rows require ≥10% lower first-
action translation p95, ≥80% useful-action direction agreement without a
>2-point drop, and no >10% p95 regression on old SFP, new SFP or SC. If
this passes, evaluate at most three reserved unseen autonomous starts,
requiring ≥2/3 safe buffered arrivals, no worse official contact and
p95 inference <300 ms. Keep handoff/insertion/RL off in this test.

## Actor-prefix-history fit: offline gain, unseen live gate failed

The full matched 2,000-update training archive is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_actor_prefix_history_matched_v2/`.
The first `_v1` launch stopped before any update because the chosen warm-start
checkpoint included the unused goal-projection layer from the preceding
ablation; `_v2` uses the actual action-only control in both arms. Both selected
update 1,750 using ordinary validation rows. On **three truly unseen**
correction episodes (174 cached teacher rows), first-action translation p95
improved **2.897→2.338 mm** (19.3%). Broad useful-action direction remained
100% for both. Ordinary old/new SFP p95 worsened 5.5%/5.0%, SC improved 0.5%; all
stayed within the 10% cap. The offline gate passed.

The independent autonomous 2/3/5-card live comparison is preserved at
`/home/nas/chmin/aic_archives/history_context_live_scenes_v1/`. All three
matched teachers fully inserted. Safe buffered transport arrivals were
**0/3 control and 0/3 corrected**, below ≥2/3. Neither actor inserted. The
corrected actor avoided the control's −24 official contact penalty on 2 and
5 cards (minimum contact 0 versus −24); p95 inference was 87.33 versus
90.63 ms. This is a safety improvement on these three starts, **not** a
transport-success promotion. The corrected checkpoint remains experimental.

Read-only route and port-relative traces are `route_motion_diagnosis.json`,
`first_causal_route_error.json`, and
`axial_lateral_approach_with_teacher_v2.png` beside the live plan. At the
safe axial band (−100 to −70 mm), the corrected lateral miss was 23.8,
20.6 and 19.1 mm for 2/3/5 cards; control was 25.9, 18.8 and 55.1 mm.
On five cards the corrected actor got to 1.4 mm lateral at axial −177 mm,
but drifted to ~19 mm while moving forward into the safe axial band. A
strict nearby-teacher-state command comparison had **no** matched rows at
12–20 s, so it cannot establish whether the late error was solely the policy
command or the resulting mechanics. The actor's port-frame commands carried
a sustained lateral component during that interval, and the rollout had no
official contact penalty; this particular failure is not evidence of a cable
snag. The successful teacher stayed near the lateral opening line.

**One bounded late-state discriminator:** reuse exactly these three already-
opened development scenes, run the unchanged corrected actor for **15 s**,
then switch to privileged CheatCode. Cap at three runs, no replacements;
require at least **2/3** official full insertions, nonnegative contact and
valid takeover logs. This is diagnostic only. If it passes, one final
supervised correction cycle may collect at most six separate train-only
late takeovers, balanced across port requests, and compare a matched fit on
the remaining untouched candidate scenes. Stop that cycle if the offline
gate fails or live safe arrivals remain below 2/3; then change architecture
or use selected-port perception for route feedback rather than scaling the
same BC data. Keep automatic handoff, insertion and RL disabled until the
transport gate passes. Final scenes remain sealed.

## October 7 continuation: late takeover and final bounded correction cycle

The 15-second takeover on the same three already-open 2/3/5-card development
resets passed **3/3**: each privileged CheatCode continuation fully inserted
with zero official contact penalty. The first two-card launch was invalid
because the diagnostic policy rejected actor durations above nine seconds
before it ran. It is preserved in
`/home/nas/chmin/aic_archives/2026-10-07_sfp_late15_recoverability_v1/`;
the identical two-card reset was rerun after extending the diagnostic-only
duration bound to 20 seconds at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_late15_recoverability_repair_card2_v1/`.
Only these three valid takeover events count. The frozen actor checkpoint and
scene configurations did not change. This is recoverability evidence, not
autonomous actor success.

The single permitted follow-up cycle uses at most six **train-only** late
takeovers from the separate, episode-grouped paired-request round-two plan:
one each with 2/3/5 cards and selected port 0, and one each with 2/3/5
cards and selected port 1. No source scene from its validation subset may be
used. Admit only officially full-inserted, nonnegative-contact episodes.
Before fitting, require at least four of six valid safe insertions and at
least two per requested port, plus causal frame/history and split audits.
Training controls and candidates will share the existing corrected actor
checkpoint, architecture, base labels, split, seed, update budget and
checkpoint selection. The control excludes only the new late teacher rows;
both retain the earlier five-second correction source. Offline continuation
requires ≥10% lower late-state first-action translation p95 on held-out
episodes, no useful-direction regression >2 points, and no >10% old/new
SFP or SC action-p95 regression. Only if that gate passes, use the three
still-unopened nonfinal candidate resets, one per card count, for one matched
autonomous live comparison. Promotion requires ≥2/3 safe buffered arrivals,
no worse contact and complete inference p95 <300 ms. If either gate fails,
stop further same-recipe corrective BC and test explicit selected-port route
feedback or another distinct architecture. Final scenes, automatic handoff,
insertion and RL remain closed throughout.

The six fixed train-source runs passed **6/6** safe full insertion with zero
official contact penalties: 3/3 port 0 and 3/3 port 1 across 2/3/5 cards.
The runs, exact selected resets and scores are at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_late15_train_port{0,1}_v1/`.
The first extractor attempt on held-out bags failed because those new bags
had not yet been decoded into `scored_stream.json`; the corrected extraction
used the existing host MCAP reader, with the original bags untouched.
The causal materialization produced **713 actor-context rows and 1,334
teacher-label rows** across the six train episodes. The three previously
opened, now valid held-out takeover episodes yielded **394 context rows and
662 teacher rows**. Both port-specific strict audits passed synchronized
three-camera image decoding, increasing timestamps, actor/teacher boundary,
and zero train/development reset overlap; see
`outputs/experiments/2026-10-07_sfp_route_branch_support_v1/late15_port{0,1}_context_audit.json`.
The combined immutable label manifest is
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_late15_context_merged_v1/manifest.json`:
it retains the original source and earlier five-second corrections, adds
only the six new train and three held-out late episodes, reports zero
train/development scene overlap, and verifies actor-prefix actions are never
supervised targets. The feature cache and matched fit are complete.

## Late-state matched result and decision

The immutable 55,206-row, 1,536D cache is
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_late15_v1/`.
It includes 333 cached new train teacher rows, 181 cached new train context
rows, 166 held-out late teacher rows and 99 held-out late context rows. The
loss mask excludes context rows. The matched fit used the same action-only
checkpoint, architecture, grouped split, seed, 2,000 updates and ordinary
validation selection in both arms; both kept the prior five-second corrective
data with 10% corrective sampling mass. The control excluded the new late
teacher actions; the candidate included them. Both selected update 2,000.
Exact train command lines, logs, checkpoints and held-out scores are in
`/home/nas/chmin/aic_archives/2026-10-07_sfp_late15_matched_v1/`.

On **three** held-out late takeover episodes, 166 cached teacher decisions,
first-action translation p95 changed **3.086→2.978 mm** (3.5% better), below
the predeclared 10% gate. Useful-action direction agreement was 100% in both
arms (160 rows). Per-episode p95 changed 3.153→3.312 mm on two cards,
2.858→2.731 mm on three, and 2.949→2.618 mm on five. Ordinary old SFP p95
rose 2.0%; new SFP and SC p95 fell 3.3% and 2.0%. The regression/direction
parts passed, but the improvement gate **failed**. [Machine decision](../../outputs/experiments/2026-10-07_sfp_route_branch_support_v1/late15_decision.json).

The remaining three untouched development starts are prepared at
`/home/nas/chmin/aic_archives/history_context_late15_remaining_live_v1/`,
with zero scene-hash overlap against the merged labels, but **were not run**.
The independent previous live comparison already found zero of three safe
buffered arrivals for both the parent actor and its matched control. More
late teacher actions under the same architecture did not clear the offline
gate, so the corrected checkpoint is not promoted. The trajectory evidence
is consistent with errors accumulating before and during approach: action
error on teacher continuations is about 3 mm p95 while the prior autonomous
actor missed the requested opening by roughly 19–24 mm in the safe axial
band, and its measured route first departed by >10 mm around 5–8 s. This is
an inference from separate offline/live measurements, not a causal proof
that a single action error causes the entire drift. A strict same-state
teacher/actor match had no comparable rows at 12–20 s.

**Next distinct gate, before another actor fit:** shadow the already trained
observation-only selected-port model on the saved off-route actor RGB/state
traces. Freeze its checkpoint and decision thresholds; select the requested
port from public task code and three camera images, with simulator geometry
used only after predictions for scoring. Require the correct requested port
to be accepted in at least 2/3 of the previously failed live scenes before
the actor first reaches the safe axial band, with zero wrong-port accepts,
reported pose error, and complete pipeline p95 <300 ms. Cap this at the three
archived scenes and one frozen checkpoint. If it fails, improve observability
or camera/identity supervision before policy training. If it passes, test a
small explicit relative-route-error input to the full-action actor under a
matched offline and new-scene live gate. No privileged target coordinates
may enter that actor. Automatic handoff, insertion, RL and reserved final
scenes remain gated.

### Frozen selected-port replay on saved off-route actor views

The distinct shadow prerequisite above was executed once, with the frozen
RGB board checkpoint and frozen native crop checkpoint, on the **actor-prefix
images only** from the three already-open 15-second takeovers. Sampling was
fixed at one three-camera observation per second (42 moments total). The
requested card/port code chose the crop; the board and crop models saw only
RGB, public measured TCP state and fixed camera calibration. True Gazebo port
geometry was loaded only after each prediction for scoring. Exact source,
checkpoint paths, per-moment predictions, acceptance flags, geometry errors
and latency are in
`outputs/experiments/2026-10-07_sfp_route_branch_support_v1/offroute_port_shadow_v1/`;
the runner is `scripts/shadow_sfp_offroute_selected_port.py`.

Before the actor reached the −100 to −70 mm safe axial band, a correct
requested-port estimate was accepted in **3/3** scenes with **0 wrong accepted
ports**: 12/14, 6/14 and 2/14 sampled moments for 2/3/5 cards. Accepted
3D position error p95 was **3.10 mm**. The first correct estimate occurred
at simulation time 3.69, 9.33 and 5.95 s, respectively. The actor's route
first exceeded 10 mm error at 6.15, 7.15 and 6.90 s: visual feedback was
available before that drift in two scenes and after it in one. All three
correct estimates preceded the actor's first >25 mm route error. The fixed
15-second prefix ended at axial −137/−114/−138 mm at its closest in these
scenes, so this check is about early transport visibility, not near-opening
pose or insertion accuracy.

Reading saved JPEGs plus board inference, original-image crop, crop network
and triangulation took **42.1/91.9/2575 ms p50/p95/p99** across 42 calls.
The first cold call alone took 4289 ms, dominating p99; the warmed saved-image
p95 meets the 300 ms model-path requirement, but complete live ROS/camera
timing remains unmeasured. This shadow gate is a useful *prerequisite* for
explicit visual route feedback, not autonomous transport or handoff evidence.
One bounded route-feedback actor fit may now be designed with the frozen
predicted relative target pose/validity as input to a full-action head;
train/evaluate it against an action-only control on grouped episodes and
new nonfinal starts. Keep true port geometry entirely outside policy input.

**Fixed route-feedback ablation before seeing its fit result:** Cache the
frozen board/crop estimate on the existing temporal-cache rows that have
original-resolution three-camera images. Convert an accepted predicted base
point to a 3D offset in the measured TCP frame, scaled by 0.2 m; use zero
when the frozen perception rule rejects it. Legacy 288×256 expert videos
cannot provide the same original-image crop, so exclude their SFP rows from
*both* training arms; retain SC rows with zero offset and retain previously
audited native SFP episodes. Exclude the six 15-second train takeovers from
both arms because their matched BC test already failed. Require at least
5,000 native-image cache rows and ≥20% accepted frozen predictions in each
split before fitting. Before training, score a fixed, episode-spread sample
of 100 accepted train/development predictions with simulator geometry used
only *after* prediction; require zero wrong-port accepts and ≤20 mm p95
position error. Train the same full-action temporal transformer twice
from the unchanged five-second corrected parent, for exactly 2,000 updates,
with identical sampler, split and ordinary-validation selection. One arm
receives zero offset; the other receives the frozen predicted offset. No
ground-truth port coordinate enters either actor. The offline gate requires
≥10% better held-out 15-second continuation first-action translation p95,
useful-direction agreement ≥80% without a >2-point drop, and no >10%
old/new SFP or SC p95 regression. If it fails, stop before any new live
rollout. If it passes, implement the *same* frozen perception path at runtime,
verify complete warm inference p95 <300 ms, and compare the two actors on
the three still-unopened nonfinal starts; require ≥2/3 safe buffered arrivals
and no worse official contact to promote. This tests a full-action policy
conditioned on an explicit visual goal, not an action-delta adapter.

The first pose-input fit completed but failed: held-out late-state p95 was
**2.849 mm with zero pose versus 2.906 mm with predicted pose**, a 2.0%
regression. Old/new SFP and SC shifts were each under 1%. On the 50 held-out
late teacher rows where the frozen perception accepted a pose, zeroing that
input changed the candidate's first action by only **0.014 mm p95**; the
learned branch was effectively unused. The first fit deliberately excluded
the new 15-second teacher recovery source, so a concrete remaining
interaction hypothesis is that explicit pose becomes useful only with those
actor-visited recovery targets present. One final bounded test will train
zero-pose and predicted-pose arms on the **same** native-image rows including
the six late teacher continuations, with all other settings unchanged. It
must meet the same ≥10% held-out action-p95 gain and regression/direction
gates, and additionally change candidate actions by ≥0.1 mm p95 on accepted
held-out pose rows. If it fails, stop pose-conditioned BC under this
architecture; do not open the remaining live scenes or adjust thresholds.

### Route-feedback ablation result: the pose is visible, but this actor ignores it

The frozen perception cache has **19,955 native-image SFP rows** aligned to
the 55,206-row temporal cache. It accepted 7,251/17,188 train rows (42.2%)
and 1,005/2,767 validation rows (36.3%), passing the fixed coverage gate.
A fixed post-prediction check of 100 accepted rows across 90 train/development
episodes found **zero wrong requested-port accepts** and 2.39 mm 3D position
p95. This sample cannot certify the uninspected rows. The three-scene visual
sheet [first accepted prediction versus truth](../../outputs/experiments/2026-10-07_sfp_route_branch_support_v1/offroute_port_shadow_v1/first_accepted_prediction_vs_truth.jpg)
was inspected: green predicted dots and red projected true-opening dots lie
nearly on top of each other in the saved center/left/right views. The dots
are posthoc scoring overlays; true geometry was not used to choose a crop or
predict a point.

The first matched 2,000-update actor test used identical native-image SFP
and SC training rows in both arms, excluding the failed 15-second BC source.
On the three held-out late episodes, **zero-input p95 2.849 mm versus
pose-input p95 2.906 mm**; the pose input worsened the error by 2.0%. On 50
accepted held-out pose rows, setting the pose to zero changed the candidate's
first action by only **0.014 mm p95**. The actor learned almost no dependence
on the explicit input. Ordinary old/new SFP and SC p95 changes were all
under 1%. The exact mask, command lines, checkpoints and scores are at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_port_route_feedback_matched_v1/`.

The single planned interaction test then included the **same six 15-second
teacher continuations in both arms**, with all other settings unchanged.
Held-out p95 was **2.95322 mm zero-input versus 2.95296 mm pose-input**,
an immaterial 0.009% gain against the required 10%. Action change when the
accepted pose was zeroed increased only to **0.032 mm p95**, still below the
0.1 mm mechanistic gate. Old/new SFP and SC changes again stayed under 1%;
useful action direction remained 100% in both. Both pose-input experiments
failed their gates. The second archive is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_port_route_recovery_matched_v1/`;
[machine decision](../../outputs/experiments/2026-10-07_sfp_route_branch_support_v1/route_feedback_decision.json).

**Decision:** stop this simple additive pose-projection branch. The frozen
perception model supplies an accurate requested-port offset on many saved
actor views, but ordinary supervised action loss lets the transformer
continue imitating the demonstrated route without using that offset.
Only 112 of 333 cached new train late-teacher rows and 50 of 166 held-out
late-teacher rows had accepted offsets; actor-prefix rows still have no
expert action targets. These are plausible reasons for weak route-feedback
learning, not a proven single cause. The next architecture must make route
error/progress an explicit supervised control objective or provide many more
valid actor-visited corrective decisions; it must still output full actions,
avoid a delta adapter, and be compared against a matched action-only actor.
Do not collect routine copies of the same late-state data or change the
frozen perception thresholds based on this held-out result. Automatic
handoff, insertion, RL and final scenes remain closed.

Reproduction entry points (the rollout `plan.json` files and training archive
retain full scene/checkpoint paths):

```bash
python scripts/run_sfp_first_departure_takeover_gate.py --live-plan /home/nas/chmin/aic_archives/history_context_live_scenes_v1 --checkpoint /var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt --actor-prefix-s 15 --output /home/nas/chmin/aic_archives/2026-10-07_sfp_late15_recoverability_v1 --gpu 7
python scripts/run_sfp_first_departure_takeover_gate.py --train-source /home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_round2_v1 --checkpoint /var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt --actor-prefix-s 15 --card-counts 2 3 5 --request-port 0 --output /home/nas/chmin/aic_archives/2026-10-07_sfp_late15_train_port0_v1 --gpu 7
python scripts/run_sfp_first_departure_takeover_gate.py --train-source /home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_round2_v1 --checkpoint /var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt --actor-prefix-s 15 --card-counts 2 3 5 --request-port 1 --output /home/nas/chmin/aic_archives/2026-10-07_sfp_late15_train_port1_v1 --gpu 7
bash scripts/cache_sfp_balanced_temporal_features.sh /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_late15_context_merged_v1/sfp_labels.jsonl /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_late15_v1 7
bash scripts/run_sfp_late15_matched.sh /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_late15_v1 /home/nas/chmin/aic_archives/2026-10-07_sfp_late15_matched_v1 7 /var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt
```

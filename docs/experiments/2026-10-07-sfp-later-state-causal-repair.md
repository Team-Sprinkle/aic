# SFP later-state transport repair gate (2026-10-07)

## Why this experiment

The frozen paired actor comparison on three fresh nonfinal SFP starts ended in
0/3 safe buffered arrivals for both candidate and control. The candidate had
an official contact penalty on the three-card start. This experiment follows
[the approved actor repair plan](../answers/2026-10-07-next-actor-repair-plan.md).
The reserved final scenes remain sealed. Actor observations contain the three
RGB cameras, public task code and measured robot state; true port geometry is
used only for retrospective diagnosis and privileged teacher continuation.

## First causal route diagnosis

Read-only script: `scripts/diagnose_sfp_first_causal_route_error.py`.
Inputs: `/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_mounted_v2/posthoc_score.json`
and the referenced saved actor `runtime_trace.jsonl` and teacher
`scored_stream.json` files. Output:
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_mounted_v2/first_causal_route_error_v1.json`.
The actual three-camera montage at actor simulation seconds 5 and 8 is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_mounted_v2/first_route_rgb_montage.png`.

The comparison matches measured TCP position and orientation to the teacher
path, then compares the next TCP-body target in the base frame. Matching by
equal clock time would mislabel a slow actor as taking the wrong route. The
candidate's early (before 8 s) median command-direction cosine with the
teacher was 0.981/0.999/0.979 on 2/3/5 cards. Its three cameras had zero
measured age at those sampled decisions, and all three task one-hot codes
matched the requests. The actor's body targets are converted via the fixed
decision-pose anchor into base-link targets, and teacher and actor Cartesian
gain vectors are both `[90,90,90,50,50,50]` stiffness and
`[50,50,50,20,20,20]` damping. The runtime sends the four absolute targets
from the decision anchor, so later chunk members do not accidentally compound
body deltas. These checks found no gross frame or input-contract error.

Nevertheless, the actor gradually leaves the path. Candidate median nearest
teacher-path distances at 5–8 s are about 19.6, 7.0 and 17.0 mm on 2/3/5
cards. Matching path progress, its measured 0.2–0.25 s TCP motion at 6–7 s
is about 10.6/11.7/7.8 mm versus the teacher's 12.5/14.3/10.6 mm. Motion
directions remain almost parallel (cosine about 0.99) at that stage. Thus a
small but sustained course and speed error, rather than an immediate wrong
direction or early collision, is the working hypothesis. The 20 N-ish force
channel at this stage reflects a large baseline/tare component; it is not by
itself evidence of contact. Official contact scoring and later contact traces
remain the contact evidence. This is a diagnosis, not proof that more BC data
will solve the drift.

The measured controller TCP-reference error over 5–8 s was median
31.3/31.6/27.8 mm for the actor versus 35.9/34.6/31.4 mm for the teacher at
matched route progress (2/3/5 cards). Therefore a larger actor tracking
error does not explain its route gap. The teacher itself commands future
targets far enough ahead that both have nonzero reference error.

## Bounded later-state teacher gate

Six complete **training** reset configurations, two each with 2, 3 and 5
cards, are taken from
`/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_v1/plan.json`.
Their original CheatCode routes scored full insertion. The frozen actor runs
for 6 or 8 s of elapsed simulation time (takeover near absolute simulation
time 9 or 11 s); then a privileged CheatCode teacher gets at most 60 s to
finish. A run counts only with official tier-3 full insertion and nonnegative
official contact score. At least 4/6 safe full insertions are required before
any correction labels are admitted or a matched actor fit starts. This is
different from the failed 2 s prefix collection.

Code: `scripts/run_sfp_later_takeover_gate.py` and diagnostic runner
`aic_example_policies/aic_example_policies/ros/RunTemporalThenCheatCodeDiagnostic.py`.
Exact plan and progressive scores:
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1/manifest.json`
and `summary.json`; each run has its own rootless Gazebo audit and launch log.
Command:

```bash
.pixi/envs/default/bin/python scripts/run_sfp_later_takeover_gate.py \
  --output /home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1 \
  --checkpoint /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/live_checkpoint_mounted_v1/candidate.pt \
  --gpu 7
```

An evaluation bookkeeping bug affected the first two already launched runs:
the diagnostic returned `False` even when its teacher routine returned
success, so the official scorer had no task end time and reported “Task not
completed.” The wrapper now returns the teacher's completion result in
full-insertion mode. The two affected resets were rerun; their initial
official scores cannot be used for the gate.
The old buffered-arrival diagnostic still returns `False` as intended.
The session restarted after the fourth run completed; the resumed read-only
reconciliation is in `summary_reconciled.json`. The third and fourth unique
resets officially scored 75/75 with zero contact penalty. The first two are
marked invalid for official scoring and are not counted as failures or
successes. The fifth and sixth unique resets and two corrected scoring
repeats subsequently completed. Another user's job occupied GPU 7 at reconciliation time;
no shared GPU job was interrupted.

## Result and decision

The gate passed **6/6** safe full insertions after the two corrected scoring
repeats: two each on 2, 3 and 5 cards, all tier-3 score 75, all official
contact scores zero. The exact audited results are in
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1/summary_reconciled.json`.
These are privileged teacher continuations after the frozen actor prefix, not
autonomous insertions. The original two incorrectly scored runs remain in the
archive and are named in `invalid_scoring_replays`.

The six scored streams were extracted with
`scripts/extract_moveit_training_stream.py`. The correction admission code is
`scripts/prepare_sfp_later_correction_labels.py` and the resulting labels and
manifest are at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/later_correction_labels_v1/`.
`scripts/materialize_sfp_scored_transport.py --corrective-full-insert` joins
only commands **after** the teacher takeover to causal images and measured TCP,
converts recorded gripper/TCP-frame commands into complete body-frame targets,
and cuts the transport prefix before the insertion region. The six runs yielded
1,383 transport chunks. Four complete correction episodes / 930 chunks are
train, and two complete episodes / 453 chunks are withheld from the optimizer.
For a new reset-grouped train/validation split, 1,038 ordinary training rows
on the two withheld scenes were removed. There is zero overlap in the *new*
split. The prior checkpoint had already seen original teacher episodes on
these reset scenes, so the withheld correction episodes test unseen recovery
states and labels, not globally unseen visual scenes. The new live development
starts remain the independent scene test.

All six sampled corrections target SFP port 1. This limits the diagnosis:
the correction fit cannot establish port-0 recovery coverage. Its offline and
live comparisons must report port identity and the scope of any improvement.
The selected teacher command-to-controller reference-target translation p95
errors were 0.59–1.16 mm across the six materializations; causal image joins
and full-scoring gates were enforced before admission.

Frozen ImageNet three-view feature cache completed on GPU 7 at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/later_correction_cache_v1/`:
53,628 rows, 1,536 visual features per row. The source label and model hashes
are in its `manifest.json`; exact command is in the adjacent `.command.txt`.
The five-camera 10 fps video for the corrected two-card teacher continuation
is `/home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1/episode2400050_card0p0_to_card3p1_actor6s_teacher_full_scorefix/five_cameras_10fps.mp4`.
It displays overhead, side, center, left and right views. The planned
matched comparison (`scripts/run_sfp_later_correction_matched.sh`) uses the same
parent checkpoint, 2,000 updates, seed, architecture, split, ordinary SFP/SC
data and reference preservation in both arms. The candidate adds 10% of SFP
sampler mass from four later-state correction episodes. The control excludes
those correction action rows. Two withheld correction episodes are excluded
from checkpoint selection in both arms. The ordinary validation split selects
checkpoints; correction-state and grouped task metrics are checked afterward.
The first attempted comparison is preserved at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_matched_v1/`.
Its control finished but the candidate refused to start because the trainer
does not allow paired-request and corrective SFP sampler masses together.
The matched repair removed paired-request resampling from **both** arms and
restarted them from the same prior at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_matched_v2/`.
Both completed 2,000 updates; both selected update 2,000 using ordinary
validation examples only.

The first offline evaluator incorrectly grouped all episode IDs above 2.6M
as corrective and is retained as `offline_gate.json` for audit. The corrected
evaluator uses the explicit `corrective_teacher` field; its output is
`offline_gate_v2.json`. On two withheld correction episodes (114 cached rows),
first-action translation p95 fell **3.77→2.22 mm**. Broad direction agreement
(cosine ≥0.7 on actions ≥5 mm) was already 100% for both arms, so this is a
precision gain, not evidence of a newly learned direction. Old SFP, new SFP,
and SC p95 action errors changed 1.535→1.531, 1.780→1.773, and 1.626→1.618 mm
respectively. The revised gate requires no >10% ordinary-group regression,
≥80% useful-row direction agreement without >2-point loss, and ≥10% lower
correction p95. These thresholds were written after finding and correcting
the evaluator grouping error; the fresh live test, rather than this offline
gate alone, determines whether the actor is promoted.

Fresh nonfinal 2/3/5-card starts are frozen at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_live_v1/plan.json`.
The chosen requests use port 1 / port 0 / port 1 respectively, so the middle
start tests transfer to a port identity absent from corrective supervision.
The prior three live groups and all train/validation scene hashes were excluded
when making this plan. `scripts/run_sfp_paired_actor_live_three.py` finished
sequential matched control/corrected Gazebo trials on GPU 7; the live gate
was ≥2/3 safe buffered arrivals, no worse official contact, and p95 actor
inference <300 ms. No automatic handoff or insertion actor was active.

## Fresh autonomous live result: gate failed

The frozen plan, official scores, posthoc geometry analysis and complete
summary are in
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_live_v1/`:
`plan.json`, `posthoc_score.json`, `summary.json`, `route_motion_diagnosis_v1.json`,
`first_causal_route_error_v1.json` and `rollouts/`. Every teacher control
fully inserted. The autonomous actors had **no guide, true port geometry or
privileged teacher input**. Geometry was used only afterward to score their
paths. Reserved final scenes were not opened.

| Fresh request | Control safe buffer | Corrected safe buffer | Control / corrected closest lateral offset in scored axial band | Contact |
| --- | --- | --- | --- | --- |
| 2 cards, card 4 port 1 | no | no | 31.4 / 37.4 mm | both −24 official contact score |
| 3 cards, card 2 port 0 | no | yes | 24.8 / 5.28 mm | neither penalized |
| 5 cards, card 1 port 1 | no | no | 55.8 / 68.9 mm | neither penalized |

Thus the corrected actor improved from **0/3 to 1/3** buffered arrivals, but
missed the predeclared ≥2/3 gate. It did not autonomously insert; the local
insertion actor was never invoked. Its accepted three-card buffer visit was
at axial −98.68 mm, lateral 5.28 mm, and approximately 10.91° orientation
error relative to the teacher standoff. These are approach measurements,
not insertion accuracy. The corrected actor's aggregate inference p95 was
94.3 ms (control 118.3 ms), so latency passed. Contact was no worse than
control but the two-card −24 penalty remains unacceptable for promotion.

The five-card measured XY route plot, `cards5_routes_xy.png`, and original
five-camera frames `cards5_candidate_12s.png` and
`cards5_candidate_20s.png` show the corrected actor branching near card 0
while the requested card is 1. It remained roughly 69 mm lateral from the
requested opening. The candidate's median nearest-teacher-path distance
grew from 18.8 mm at 4–8 s to 55.3 mm at 8–12 s and 68.2 mm at 20–30 s.
Four five-camera videos are in `rollouts/`: all three corrected actor starts
and the five-card control. The remaining runs retain their scored bag,
camera captures, runtime trace and launch log without a rendered video.
This is a first-route/goal-selection failure on the five-card start, not an
observed cable snag. On the two-card start the actor moved toward the
requested port but missed the safe buffer and eventually contacted the
scene. On the three-card start the corrected actor did reach the buffer but
later stalled. The causal history behind the wrong branch is not proven:
task identity, visual grounding, and route progress are plausible factors.
Correction labels start at privileged takeover and omit the actor's earlier
prefix history, whereas live actor inference retains its previous history;
that is a data-contract difference, not a demonstrated cause.

### Decision and one distinct next experiment

**Stop this later-state BC correction recipe.** The offline correction p95
gain did not translate into reliable autonomous transport. Do not admit more
similar correction episodes or retrain this same candidate. The next bounded
test should target the *initial requested-card route branch*: make a
task-conditioned spatial goal token for the requested occupied card and a
coarse teacher-route branch/progress target, both using public task code,
three RGB views and measured robot history. Teacher geometry and route stage
may label training rows, but cannot enter the deployed actor. This differs
from the earlier FiLM/discrete-route fit, which improved task sensitivity
while worsening action p95, because the proposal must first demonstrate
correct **visual requested-card grounding on the actor's early views** and
correct branch selection before it may influence action generation.

Budget: reuse archived data and current frozen nonfinal development splits;
at most one ≤2,000-update matched actor comparison after a read-only label
and observability audit, then at most three *new* nonfinal matched autonomous
starts only if offline gates pass. Stop before fitting if requested-card
identity or teacher branch is not causally labelable on the early views.
Stop before live trials unless complete-episode-held-out early branch
accuracy is ≥90%, same-image alternate-request branch changes follow the
teacher on ≥80% of separable pairs, and no old/new SFP or SC action p95
regresses by >10%. Stop/pivot after live trials unless ≥2/3 safe buffered
arrivals, no worse contact and p95 inference <300 ms. These proposed branch
thresholds apply to the **next** experiment, not retroactively to this one.
If visual grounding is unobservable on early views, investigate an
information-gathering view or bounded route-planning baseline rather than
multiplying BC data. Keep pause/coarse in shadow, insertion and RL disabled,
and final configurations sealed.

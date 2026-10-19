# SFP transport: data-diversity test and pause-proposal head

## Why this test

The current 326 verified SFP training episodes include all five card counts
and ten requested card/port codes, but the coverage is very uneven: 235/326
are card 0/port 1. The one-card set has 130 card 0/port 1 episodes and no
port-1 episode for cards 1–4; the two-card set is similarly sparse. All
current routes are successful teacher paths, so the actor sees little of its
own off-route states. Balancing the sampler cannot create missing images or
routes. The matched actor still reached zero of three safe buffered regions.
The spatial model improved teacher-frame action p95 only 1.9% and did not
improve request-dependent route choice. This record tests a *data* hypothesis
with a bounded new collection, not an assumption that more episodes suffice.

The transport stop head already existed: an SFP linear head on frozen action
features was timely in only 2/10 held-out teacher routes at its strict
standoff threshold and three-decision streak, even after the earlier 29
paired episodes were admitted. It has never been deployed. This continuation
trains a distinct **pause proposal**, meaning “close enough to ask precise
vision whether to hand off,” not “safe to insert.” A proposal must still pass
requested-port identity, pose, force and posture checks later.

## Collection plan and gates

Use only historical **scene configs** as sources, then run fresh ordinary
CheatCode in rootless Gazebo with normal collisions. Do not use old agent/VLM
action arrays. Select nonfinal scenes disjoint from the existing train and
grouped validation scene hashes. Fill each of 50 cells (1–5 installed cards ×
five requested card slots × two ports) toward at least **five successful
complete episodes**, excluding already sufficient cells. The current deficits
sum to about **137** successes. The immutable plan lists the exact trials,
requests, scene hashes, commands and source family. Preserve failures.

Pilot the first **20** new requests, prioritizing sparse card counts and
underrepresented ports. Continue only if at least **16/20** fully insert,
with correct task/request, synchronized three-camera images and causal
scored TCP-body commands. Thereafter stop if fewer than **23/30** of a
rolling block fully insert. Each failed cell can receive one distinct
source-scene retry in a separate documented batch; do not silently accept
partial insertion as expert BC. Cap this first collection at the immutable
plan size and use at most one Gazebo GPU while leaving the active Isaac job
untouched. Archive raw scored streams and sample videos. Keep original
expert labels unchanged; create a new verified derivative only after score,
frame, reset and split audits.

Before fitting, reserve the first 17 planned scenes (the first new request in
each sparse one- and two-card cell) as **new development validation**. They
are not final scenes and never supply optimizer examples. This gives a direct
test of whether the actor generalizes to the card counts missing from the old
ten-scene validation set. The remaining successful scenes join training;
failed or noncausal scenes join neither split. Report both old and new
validation subsets rather than averaging away a regression. Reserving one
scene per sparse cell means the initial batch targets **four training
successes plus one development validation success** in those cells. The
bounded retry may restore a cell to four training successes after one failed
attempt; it does not expand the entire collection to chase five training
examples in addition to the held-out scene.

Fit a matched pair of full-action temporal transport actors on the same
complete-episode split and update budget: old verified data versus old plus
new diverse successes. Preserve the full TCP-body target and public task
code; do not add a delta action or privileged geometry. Test held-out
teacher-action error and same-image task-code sensitivity, then at most
three new nonfinal autonomous starts if offline checks improve without
>10% action-error regression. Require ≥2/3 safe requested buffered visits,
no extra contact and complete p95 <300 ms before expanding. If the live
gate fails, stop routine teacher collection and diagnose the first divergence.
For this fixed comparison, proceed to live only if the new one/two-card
development subset improves first-action translation p95 by at least 10%,
the old SFP and SC subsets each avoid a >10% p95 regression, and same-image
requested-code action diameter grows by at least 2×. These are screening
metrics, not insertion success; report each separately.

## Pause-proposal training

The former positive label described a narrow *safe handoff* region. This
continuation labels a wider **pause-for-visual-check** region, provisionally
selected before fitting as TCP in requested port frame with axial position
between **−130 and −60 mm** and lateral distance ≤**30 mm**. It is supervision
only; true port geometry never enters the runtime actor or proposer. The
existing request-specific base-coordinate region can veto impossible
proposals but cannot create one. Freeze the transport action weights and
train only the probability head on grouped training episodes; keep grouped
validation and newly collected nonfinal checks separate. Report episode
timely/early/late/missed events and row-level false positives, separately
by request/card count. The preliminary pause head is shadow-only unless it
achieves at least **8/10** timely held-out teacher events, zero early events
and no false accept on saved off-route traces; no automatic insertion can
follow from this score alone.

## Commands, artifacts and results

The collection planner is `scripts/collect_sfp_diverse_teacher.py` and the
output root is `/home/nas/chmin/aic_archives/2026-10-06_sfp_diverse_teacher_v1`.
The immutable plan contains 137 distinct, nonfinal scene configs. The
**20/20 pilot full-insertion gate passed**. All twenty were materialized into
5,241 causally joined transport chunks, had the correct selected-port task
vector, and had no scene-hash overlap with the old train or validation data.
The maximum episode p95 command-to-controller-reference discrepancy was
3.86 mm; the maximum measured lateral offset at the transport cutoff was
9.96 mm. Full per-episode reports are in
`/home/nas/chmin/aic_archives/2026-10-06_sfp_diverse_teacher_v1/pilot_causal_audit.json`.
The first fresh one-card/card-1/port-1 run alone had 439 joined command rows,
258 usable transport chunks, and a 3.51 mm lateral cutoff offset. A
center-camera sample at 10 s retains the orange cable and visible NIC
opening. The remainder of collection and scored-bag extraction continue on
GPU 1 and CPU respectively. The first 50 attempts were also **50/50 full
insertions**, passing the first rolling 30-run gate. This is a
teacher-collection result, not an autonomous policy score.
The complete initial plan finished at **132/137 full insertions**. Five
partial-insertion scores are preserved in `progress.json` and remain outside
BC. Every successful bag was extracted (132/132). Causal admission accepted
all 132 successful episodes, producing **34,350 transport chunks** and a
17-episode new development holdout; all five partials were excluded. The
immutable derivative is
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_diverse_merged_v1`.
The per-cell retry decision follows, with any replacement scene required to
have a new scene hash.

**Final data decision (October 7 UTC):** Five distinct-scene retries were
planned for the five partial cells. Three fully inserted, one one-card/card-4/
port-1 retry scored **no insertion**, and one two-card/card-4/port-1 run hit a
Gazebo component-container `exit -11` before scoring. The crashed container
and raw logs were preserved; the same exact scene was rerun once in a new
audit folder and fully inserted. The original progress was snapshotted at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_retry_v1/progress_before_infra_retry.json`;
the causal outcome and both artifact paths are in `infra_retry_summary.json`.
The retry admission accepted **four** and excluded the genuine no-insertion
run. No further scene was sampled for that cell.

The final derivative is
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_diverse_final_v1`:
**136 new full-insertion episodes**, **35,383 new causal transport chunks**,
and the five initial partials plus one genuine retry failure excluded. The
new episodes contribute 119 train and 17 grouped development validation
scenes, giving **445 SFP train and 27 SFP validation episodes** with the old
data retained. All 50 card-count/request cells have at least five combined
new/old train plus new development examples except one-card/card-4/port-1,
which has three train and one new development example. The split and this
shortfall are preserved rather than filled by repeated sampling. The final
admission manifest records every episode and exclusion.

For the separately fitted probability head, the exact commands were:

```bash
.pixi/envs/default/bin/python scripts/prepare_temporal_route_handoff_cache.py \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_v4 \
  --sfp-labels /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/paired_coverage_merged_v2/sfp_labels.jsonl \
  --output /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/pause_proposal_cache_v1 \
  --sfp-axial-min-m -.13 --sfp-axial-max-m -.06 --sfp-lateral-max-m .03
CUDA_VISIBLE_DEVICES=2 .pixi/envs/default/bin/python scripts/train_sfp_frozen_handoff.py \
  --cache /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/pause_proposal_cache_v1 \
  --base outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --actor outputs/experiments/2026-10-05_transport_identity_audit/paired_coverage_actor_v2/best.pt \
  --prior /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/pause_proposal_cache_v1/handoff_prior.json \
  --output outputs/experiments/2026-10-06_sfp_diverse_transport/pause_proposal_head_2k_safe_v1 \
  --updates 2000
```

The 2,000-update checkpoint achieved **9/10 timely first proposals, one
miss, zero early proposals, and zero raw false-positive rows** on the old ten
grouped teacher validation episodes. Action weights were frozen. A selection
bug that previously preferred an update with two false-positive rows was
fixed before the definitive fit. The saved checkpoint and per-episode metrics
are in `outputs/experiments/2026-10-06_sfp_diverse_transport/pause_proposal_head_2k_safe_v1`.

The observation-only replay at
`outputs/experiments/2026-10-06_sfp_diverse_transport/pause_proposal_offroute_replay_v1.json`
used three previously saved autonomous development runs. Replay fidelity for
the actor's first command was 0.15–0.27 mm p95, with three missing JPEGs
replaced by the nearest frame within 100 ms in one run. The **first** proposed
pause landed in the broad true region on two runs; the third had no
three-decision trigger. If the actor had continued after a pause, many later
high-probability rows would have been outside the region. This head therefore
remains **shadow only**. The replay does not establish safe deployment or
insertion, and the broader pause window is not an insertion corridor.
Its sigmoid output is currently an **uncalibrated proposal score**: 0.95 is a
decision threshold, not a demonstrated 95% chance that the requested port is
correct or safe.

The old teacher transport prefixes stop at the near-port cutoff, so their
training rows contain few or no **post-window** negatives. That is a concrete
reason the head can stay high on an autonomous route that continues past the
window. More successful teacher prefixes alone cannot be assumed to repair
this off-route failure. A later proposer repair needs saved actor-route
negative states and a separate grouped off-route development test; it must
not re-label geometry as a runtime input.

## Matched actor and final proposer result

The final shared frozen three-view cache contains **50,679 decisions**:
31,448 SFP training rows and 1,773 SFP validation rows, plus unchanged SC
rows. Camera ages were 0–48 ms. The two temporal actors started from the same
prior checkpoint and used the same sampler, seed, split and **3,000 updates**;
the control excluded new training episode IDs ≥2,400,000 and the candidate
included them. The training command was:

```bash
bash scripts/run_sfp_diverse_matched_train.sh 1 1 2
```

The best checkpoints were at updates 1,500 (control) and 2,500 (diverse),
selected on the same development episodes. On the fixed per-episode sampled
score, first-action translation p95 changed as follows:

| Group | Old-data control | Diverse-data actor | Change |
| --- | ---: | ---: | ---: |
| 17 new one/two-card development episodes | 1.856 mm | 1.668 mm | 10.09% lower |
| 10 old SFP development episodes | 2.033 mm | 1.676 mm | 17.55% lower |
| 21 unchanged SC development episodes | 1.899 mm | 2.032 mm | 7.04% higher |

Those satisfy the error-screening parts of the offline gate. However, on
the **same saved image/state/history**, swapping only the requested card and
port code changed the ten-code first-action diameter from **0.572 to 0.620
mm**, only **1.08×** rather than the predeclared 2×. The nearest alternate
request changed even less (0.156→0.155 mm median). The 0/1/2/4/8 s
breakdown remains under 0.9 mm in both actors. The added 136 scene groups
each had one request; they increased visual and reset diversity but supplied
**zero new same-reset alternate-request action pairs**. This is consistent
with improved imitation error without learning a meaningful requested-route
branch. It is an inference from the data design and matched behavior, not a
claim that data diversity can never help.

**Decision:** the requested-code gate failed. No new autonomous rollout,
handoff promotion, insertion evaluation, RL, or final-scene opening followed
from this actor. The action-error gain is retained as an offline comparison,
not treated as evidence of full insertion. Exact split metrics are in
`outputs/experiments/2026-10-06_sfp_diverse_transport/split_score_v1.json`;
same-image records are in `control_task_sensitivity_v1.json` and
`diverse_task_sensitivity_v1.json` in the same folder. The machine-readable
gate is `decision_summary.json`.

The final probability-head fit used the **445 SFP train episodes** while
freezing the earlier actor's action weights. Its selected update 1,800
produced **26/27 timely first proposals and zero raw false-positive teacher
rows** on the 27 grouped SFP development episodes. The earlier head, scored
on the exact same cache and train-derived region veto, also produced **26/27**:
the diverse teacher data did not improve that event count. On three saved
autonomous runs, the new head's first trigger was inside the broad true
pause region in **two** and falsely triggered in **one** (−68.6 mm axial,
36.8 mm lateral, outside the ≤30 mm label). It also stayed confident on many
post-window decisions. This is a **failed off-route safety gate**. The
probability head remains shadow-only and uncalibrated; no actor or verifier
automatically accepts its proposal. Checkpoints and event rows are in
`pause_proposal_head_diverse_v1`, `pause_proposal_split_score_v1.json`, and
`pause_proposal_diverse_offroute_replay_v1.json` in the experiment folder.

The next bounded repair should target *causal route choice* rather than
another unpaired teacher sweep: construct same-reset, alternate-request
examples at a shared observation where expert next actions materially
diverge, or revise the observation/goal representation if such states cannot
be produced. Separately fit the pause head with real actor-route post-window
negatives and test on new grouped off-route starts. Do not use true port
geometry in runtime decisions; keep the autonomous gate and final split
sealed until these targeted checks pass.

One bounded next test is ≤20 new nonfinal reset configurations and ≤40
teacher route queries on one Gazebo GPU, with alternate requests queried at
the **same measured branch observation**. Require at least 30 causally
aligned next-action pairs differing by ≥5 mm in TCP-body translation before
one matched actor fit. If the data gate fails, stop this BC branch and test
an observation-based selected-goal representation. If it passes, require a
≥2× same-image requested-code action-diameter gain without >10% old/new
action-error regression before at most three matched autonomous starts.
Keep post-window proposer-negative collection and its grouped off-route
evaluation separate.

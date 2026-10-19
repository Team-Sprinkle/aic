# Later-state actor repair: result and next decision

I completed the approved SFP transport repair plan through its autonomous
gate. The revised actor **did not pass** that gate, so I stopped this
correction recipe and did not connect automatic handoff, insertion or RL.
The four reserved final configurations remain sealed.

## What I did

1. Compared the first actor route decisions with successful teachers at
   matched measured robot poses. The cameras, task code and TCP-body command
   conversion checked out. Small early speed/course errors accumulated
   before the routes separated; an initial controller-frame bug was not found.
2. Let a privileged CheatCode teacher take over after 6–8 seconds of actor
   motion on six training resets (two each with 2, 3 and 5 cards). **All
   six fully inserted with zero official contact penalty.** These are
   teacher recoveries, not autonomous policy successes.
3. Trained two otherwise matched full-action transport actors for 2,000
   updates: control without those later-state examples and corrected actor
   with them. The corrected actor's withheld correction-state action p95
   improved from 3.77 to 2.22 mm without a >10% grouped old/new SFP or SC
   offline regression. The six correction runs target port 1 only, and the
   prior checkpoint had seen ordinary teacher episodes on the withheld
   correction scenes; the fresh live scenes are the independent check.
4. Ran both actors autonomously on the same three new nonfinal 2/3/5-card
   starts, with no guide or privileged geometry fed to either actor. All
   three teacher controls inserted. The control reached the safe approach
   buffer **0/3** times; the corrected actor reached it **1/3** times.
   Promotion required at least **2/3**. Both actors incurred a −24 official
   contact penalty on the two-card start. The corrected actor's p95
   inference was 94 ms, below the 300 ms limit.

The three-card improvement is real but narrow: the revised actor came
within 5.28 mm laterally at about 99 mm before the opening, then stalled.
It never attempted autonomous insertion. On the five-card start it took a
path near card 0 although card 1 was requested, ending about 69 mm lateral
from the requested opening. This is an early route/goal-selection problem,
not an observed cable snag. The two-card run moved toward the requested
opening but missed the buffer and eventually made penalized contact. The
cause of the wrong branch is not yet proven; visual requested-card grounding
and route-progress representation are the next specific hypotheses.

## Decision

The offline gain did not produce reliable live transport. The corrected
checkpoint is **not promoted**. I stopped additional collection and
retraining of this same later-state BC recipe. The coarse selected-port
crop and pause proposer remain shadow-only. Automatic handoff, insertion
and RL remain disabled.

## One bounded next experiment

First audit archived early three-camera views, public task codes and teacher
routes to see whether the requested occupied card and initial route branch
can be labelled and distinguished without runtime simulator geometry. If
that is feasible, test one explicit task-conditioned spatial goal/route
representation, with a matched full-action control and at most 2,000 updates.
Unlike the earlier FiLM/discrete-route trial, this must establish correct
requested-card visual grounding and branch choice *before* actor training.

Stop before training if the early views do not support a causal label. Stop
before live rollout unless complete-episode-held-out early branch accuracy
is at least 90%, alternate-request branch changes agree with the teacher on
at least 80% of separable pairs, and old/new SFP and SC action p95 each
regress by no more than 10%. If those checks pass, allow at most three new
nonfinal matched autonomous starts. Continue only with at least 2/3 safe
buffered arrivals, no worse official contact and p95 inference below 300 ms.
Otherwise change the observation or route-planning approach rather than
adding more of the same correction labels.

## Artifacts

- [Full experiment record](../experiments/2026-10-07-sfp-later-state-causal-repair.md)
  and [machine summary](../experiments/2026-10-07-sfp-later-state-causal-repair.summary.json).
- Teacher gate: `/home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1/summary_reconciled.json`.
- Matched checkpoints and offline result: `/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_matched_v2/`.
- Frozen live plan, official/posthoc scores, route plot, screenshots, traces
  and four five-camera videos: `/home/nas/chmin/aic_archives/2026-10-07_sfp_later_correction_live_v1/`.

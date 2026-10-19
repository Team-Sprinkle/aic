# Is the current SFP-to-NIC training data unsafe, or is the training method the problem?

**The successful teacher data is not inherently an unsafe route.** Most
admitted SFP trajectories fully inserted, and six privileged CheatCode
continuations from later actor states fully inserted without an official
contact penalty. But **a policy trained only to imitate those examples is
not safe to promote yet**: on fresh starts it missed the safe transport
buffer, and some autonomous tests incurred contact penalties. The latest
independent matched test was 0/3 safe arrivals for each actor.

This is **both a coverage problem and a learning/control problem**, with no
single root cause proven:

1. **Coverage:** The 236 legacy training episodes were highly skewed:
   227 requested card 0 / port 1. New data broadened requests, but most
   examples still show successful teacher states. Once an actor drifts
   1–2 cm away from the teacher route, it sees states with much less
   corrective supervision. The old data also has near-port periods where
   commands are large but measured motion barely changes. These are real
   recordings, not necessarily wrong action labels; their exact physical
   cause is unresolved.

2. **Learning/control:** Matching the next teacher action does not directly
   reward taking the correct *route* over several seconds or correcting
   accumulated error. Early actions for different requested cards can look
   similar even though their paths separate a second later. In one fresh
   five-card rollout, an actor branched near card 0 when card 1 was
   requested. A pose-conditioned version barely changed its actions when
   its predicted pose was removed, so that implementation did not make
   effective use of the target estimate.

3. **Checks that did not reveal a simple bug:** The task one-hot codes,
   three-camera timing, TCP-body action conversion, and actor/teacher
   command frames were audited. A 2,000-row label conversion check had
   numerical reconstruction error below `3×10⁻¹³ mm`. Masking audited
   sustained stalls in a matched fit did not improve the actor. These
   checks narrow the likely problem; they do not prove the rest of the
   system is bug-free.

So I would **not** say “the dataset teaches the robot to bump into things.”
The safer interpretation is: the nominal demonstrations are useful, but
they do not sufficiently teach how to choose the requested route and
recover after the learned actor leaves it. The current imitation objective
also does not force the actor to use the available route/pose information.
Some SFP actor rollouts *did* receive contact penalties, so running this
policy unattended as though it were validated would be unjustified. That
is different from saying SFP has the same multi-card cable-snag problem as
SC-to-SC.

The next meaningful comparison should keep a fixed episode-grouped test
set and vary **one cause at a time**: add verified actor-visited correction
states while holding the architecture fixed, and separately use an
explicit requested-route/progress objective while holding data fixed.
Measure safe buffered arrival, correct-card choice, scored contact, and
insertion separately. More routine successful teacher trajectories alone
have not been shown to fix the failure.

Evidence: [data-quality audit](../experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md),
[later-state actor diagnosis](../experiments/2026-10-07-sfp-later-state-causal-repair.md),
[early route-branch test](../experiments/2026-10-07-sfp-early-branch-support.md),
and [pose-use training diagnostic](../experiments/2026-10-08-sfp-pose-actor-training-method.md).

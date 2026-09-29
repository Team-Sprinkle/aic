**Recommendation as of September 25, 2026: continue the project, finish the current controlled experiment, and put a firm limit on further repetition of the same BC approach. Keep step 5 open. A small RL experiment may become justified, but broad RL training is premature.**

This assessment is based on the [handoff](2026-09-25-step5-shared-policy-blocker-handoff.md), the [execution plan and subsequent log](2026-09-24-shared-actor-sc-serl-execution.md), and the linked correction-data reports. It is a recommendation from the recorded evidence; I did not run new experiments or independently analyze the videos and raw bags.

The robot has learned enough to approach the socket more safely in the tested body-target runs, but it has not demonstrated that it can finish the job by itself. Small errors on saved expert examples do not guarantee success when the robot controls the whole episode: its own mistakes change what it sees next. The repeated gap between good offline numbers and zero autonomous insertions is the main reason to change how progress is judged.

The earlier work still has value. You have validated useful parts of the simulator and command pipeline, collected real scored teacher successes, and found a more promising action representation. However, the prerequisites are only partly settled: the perception result is strongest near the socket, some individual scenes fail its accuracy target, and demonstration coverage is thin. Completing a checklist cell with one successful example does not establish robust coverage.

**There is a newer development beyond the handoff.** The one-second teacher-rescue test also produced only 1/4 insertions. A subsequent method gave the actor brief 0.2-second turns, with the teacher correcting between them. Across 24 attempts, 18 inserted and 16 passed the stricter data-admission checks. Those 16 episodes yielded 10,042 training chunks, including 347 actor-owned anchor chunks. The execution log says a matched control/treatment training comparison is underway; it does not yet record an autonomous result for that comparison. These are teacher-assisted successes. See the [collection report](2026-09-25-sc-body-interleaved-rescue-batch20-gate.json) and [label audit](2026-09-25-sc-body-interleaved-rescue-combined-labels.json).

I would take the following steps, in order:

1. **Finish that existing comparison before collecting another batch.** It tests a meaningfully different correction source, and the data already exist. Preserve the same starting checkpoint, training budget, and evaluation conditions for both arms. Check that corrective targets really describe what the teacher would do from the actor's encountered state, rather than simply reinforcing the actor's mistaken command. The 10,042 chunks overlap within only 16 episodes; they are not 10,042 independent recovery examples.

2. **Judge the result through autonomous behavior.** Use offline validation to catch invalid or clearly regressed candidates, but a tiny improvement in average target error should not be the sole reason to evaluate or reject a candidate. For a valid candidate, compare both arms on the two already-open starts under fixed limits, logging full insertion, actual motion, force, contact, and complete inference latency. Repeat promising outcomes because the log shows that identical scene definitions can produce different takeover states. If there is reproducible improvement, freeze the candidate and test a small, declared batch of previously unused development starts, including multi-card SC and SFP. Success on the two familiar starts would be a reason to expand evaluation; step 5 still requires new-start evidence in both families.

3. **Use one bounded diagnosis to locate the first failure.** Start with the existing recordings and synchronized traces. At the first sustained divergence, determine whether the socket is observable, whether the actor requests an appropriate full target, and whether the robot physically follows it. Use training scenes for any teacher-assisted phase tests. Compare actual state at takeover, since elapsed time alone does not ensure equivalent starts. A short test beginning near a visible socket can help distinguish a transport problem from a local insertion problem, but its success would remain a local diagnostic.

   | What the diagnosis shows | What to investigate next |
   | --- | --- |
   | The selected socket is hidden or incorrectly located | Visibility, observation history, and target selection; assess whether the available observations contain enough information. |
   | The socket is correctly observed, but the requested target is wrong | The action mapping, correction labels, and sampling around the failure state. |
   | The target is sensible, but measured motion stalls or deviates | Controller tracking, collision, and grasp behavior before more policy training. |
   | Nominal motion works, but small deviations repeatedly defeat the actor | Targeted recovery learning, potentially a short RL experiment. |

   These are hypotheses to distinguish, not diagnoses already established. In particular, replacing a frozen model's learned pose input with true pose is an unreliable causal test because its action head was trained around the original input. The body-target actor also does not directly consume the auxiliary predicted pose, so that pose head's error alone cannot explain its action.

4. **Set a stopping rule for this BC branch now.** My proposed limit is the current matched comparison plus the bounded diagnosis. If both valid arms still achieve zero autonomous insertions and the diagnosis identifies no concrete repair, stop routine data expansion and extra-update runs. If a specific fault is found, allow one targeted repair and matched retest. Further work should answer a new question. Another small offline gain or a slightly higher partial score is insufficient justification for repeating the cycle. This is a proposed experiment budget, not an existing plan requirement.

5. **Consider RL as an explicit revision to the plan if recovery is the remaining bottleneck.** The present plan requires step 5 before RL. Keep that failed status visible and document a separate diagnostic exception if this route is chosen. Start with short training-side incidents close enough to completion to produce useful feedback, initialize from the BC actor, and compare with a frozen BC baseline under matched starts and rollout budgets. Verify that the chosen simulator reproduces the actual failure being addressed. Reward full insertion, use progress feedback that cannot be collected indefinitely by hovering nearby, and account for force, contact, stalls, and time. Log executed actions and terminal states. Predeclare a finite interaction budget; continue only if there are repeatable autonomous insertions on separate development incidents without unacceptable force or SFP regression. A local RL success would still need full-start validation. If the actor cannot observe the target or the controller cannot execute the command, address that first.

6. **Keep broad recovery and final evaluation gated.** The latest failures ended centimeters away with no scored contact, so the evidence does not yet establish a blockage that a retreat supervisor would solve. Develop the relevant recovery mechanism once measured traces show its trigger and purpose. Keep the four final configurations sealed; development incidents used to select rewards or thresholds are development data, never final evidence. Preserve the requirement for one shared actor, both task families, and full inference p95 below 300 ms.

The immediate decision is therefore to finish the interleaved-correction comparison and identify where autonomous behavior first breaks. The evidence supports that bounded investment. It does not support an indefinite sequence of similar BC runs, a claim that step 5 has passed, or a large RL campaign yet.

## Executed update and revised next experiment, September 25

The controlled comparison is complete. Whole-episode interleaved replay
regressed held-out SC metrics at two declared weights. A narrower 0.5 s
post-pulse correction selection improved combined offline selection error
from **2.645 to 2.621 mm**, but worsened SC initial-command p95 from
**1.760 to 2.315 mm** and near-port lateral p95 from **0.312 to 0.439 mm**.
Its same-parent matched live comparison was **0/2 full insertions in both
arms**, with no scored contacts. The treatment's higher single-run score
(45.64 versus 28.17) is inconclusive: identical control weights previously
scored 43.80 on the same scene definitions. See the [offline gate](2026-09-25-shared-body-bc-window0p5s-offline-gate.json)
and [live gate](2026-09-25-shared-body-bc-window0p5s-matched-live-gate.json).

The frozen cadence comparison also finished. Executing all four predicted
targets or replanning from target one both gave **0/2 insertions**. In the
first opened scene, a scored-bag diagnosis found that between 20 and 30 s
the chunk actor commanded movement away from the opening and the physical
plug followed (motion-direction cosine 0.995); the distance grew from 94
to 151 mm. Replanning also briefly commanded and followed outward motion.
There was no scored contact. The port was within the calibrated camera
frustums, but actual image visibility and model recognition cannot be
established: the bags did not save synchronized RGB or pose-head output.
See the [scored diagnosis](2026-09-25-step5-body-cadence-scored-diagnosis.json)
and [plot](figures/2026-09-25-step5-body-cadence-first-divergence.png).
This is a concrete command-route failure, with its upstream cause open.

**Plan revision: one targeted command-generation repair gate, not another
routine BC run.** Hypothesis: the actor gives an outward full target while
the opening is actually visible and it has enough information to approach.
First instrument the *frozen* actor on at most **four training-only SC full
starts**, spanning the already used two-/five-card variants. Retain
synchronized original RGB from all cameras, actor pose-head output,
four-target proposal, executed target, measured TCP and physical plug,
force, and contact. Use simulator opening geometry only in posthoc scoring.
Allow at most **one GPU and 90 minutes** for these runs plus trace analysis;
no training, new development starts, or privileged runtime crop/actor input.
Stop this gate if complete causal traces cannot be retained in four starts.
At the first sustained ≥20 mm distance regression, classify actual port
visibility/occlusion, pose prediction, commanded direction, and measured
tracking for every affected start. Require at least **three of four** traces
to support the same actionable cause before a repair. The four scenes are
diagnostic, not an insertion-rate estimate.

If visible-port **wrong-command** events dominate while TCP tracks the
command, allow **one** matched repair of the full-target action head on
existing admitted training data, with a train-only near-approach
command-direction/goal-consistency objective. The deployable actor must use
observation-only inputs, not true geometry, and must still output the full
TCP-body target rather than an added delta. Cap this repair at **1,500
updates on one GPU plus four autonomous runs** (same two already-open
development starts per arm), about **three GPU-hours and two simulator
hours**. Proceed only if grouped offline SC initial and near metrics do
not regress, full inference p95 is below 300 ms, and the matched treatment
achieves at least one **scored full insertion without off-limit contact**
while control does not. Repeat any apparent gain on unused development
starts before promotion; the four reserved final IK configurations remain
sealed. If RGB shows the opening is hidden or pose prediction is wrong,
replace the action repair with one explicitly scoped visibility/perception
repair; if a sensible command is not followed, repair controller or
collision fidelity instead. These alternatives require their own frozen
comparison and budget before work begins.

If attribution is mixed, or the one targeted repair fails the stated gate,
stop this BC branch. A short RL diagnostic would then require a separate
written exception to the step-5-before-RL plan, training-only near-completion
incidents with verified visibility/actuation, finite interaction budget,
frozen-BC matched baseline, actual insertion reward, force limits, and
new-start autonomous validation. This result does not trigger RL training.
**Step 5 remains open; no recent experiment established an
autonomous full insertion.**

# Current gaps after the later-state SFP actor test

**Later October 7 update:** This ranked list was the input to the ordered
continuation. The early-route visual probe found some usable branch signal;
the first route-goal actor test was later found to overlap existing labels,
so it is diagnostic only. A replacement unseen-scene live test found **0/3
safe buffered arrivals for both arms**. Three late privileged teacher
takeovers and six balanced train-only takeovers fully inserted, but the
matched late-label actor improved held-out action p95 only **3.5%**, below
the fixed 10% gate. The transport failure and automatic handoff gates remain
open. [Current experiment and next distinct test](../experiments/2026-10-07-sfp-early-branch-support.md#late-state-matched-result-and-decision).

**Latest continuation:** Frozen selected-port perception then passed on
3/3 saved actor routes, but two matched full-action actor fits with its
predicted pose missed the offline gate. The actor barely used that input
(0.032 mm p95 action change when accepted pose was zeroed). The remaining
gap is an explicit route-error/progress control objective with useful
actor-visited full-action targets. The remaining live starts were not
opened. [Result and saved overlay](../experiments/2026-10-07-sfp-early-branch-support.md#route-feedback-ablation-result-the-pose-is-visible-but-this-actor-ignores-it).

## The main gap: getting to the *requested* port

The transport actor still does not reliably reach a safe area near the
requested port from a normal start. In the latest matched Gazebo test, the
revised actor reached that area on **1/3** new starts; its control reached it
on **0/3**. The required gate was 2/3. No autonomous insertion was attempted.
On the five-card scene, the revised actor branched near card 0 although
card 1 was requested, ending about **69 mm** lateral from the requested
opening. On the two-card scene it moved toward the request but missed the
safe buffer and eventually received a contact penalty. The one three-card
success only reached a broad approach buffer, about **99 mm axially before**
the opening and **5.28 mm laterally** off, with about **10.9°** orientation
error from the teacher standoff. It later stalled. These are different
failure modes; one universal “more BC data” explanation is unsupported.

## Gaps, ranked by what currently blocks progress

| Gap | What the evidence says | What remains unknown |
| --- | --- | --- |
| **1. Requested-card route choice** | The five-card actor visibly took the branch near a neighboring card. Prior paired-request supervision improved code sensitivity, but the actor still failed fresh live routes. | Whether early three-camera images plus public task code/history uniquely identify the requested card and correct branch, and whether the actor's representation preserves that information. |
| **2. Closed-loop route stability** | Early commands generally pointed roughly along the teacher's path, yet small speed/course differences accumulated into 10–25 mm route errors by roughly 5–8 s; later deviations were much larger. Measured controller-reference error was not higher for the actor than for the teacher at matched progress. | Whether the actor lacks a usable progress/error signal, whether its history helps correct drift, or whether the commanded chunk cadence/magnitude creates an avoidable feedback delay. A gross frame or camera timestamp bug was not found. |
| **3. Useful training signal at actor-visited states** | Privileged teachers fully recovered from all six later actor states, and correction examples improved withheld action p95 **3.77→2.22 mm**. Fresh live safe arrivals rose only 0/3→1/3. | Whether those labels begin too late to repair the *initial* wrong branch. All six target port 1. Their materialized histories start at takeover rather than including the actor prefix; this train/live context difference has not been isolated as a cause. |
| **4. Offline tests predict live performance poorly** | Correction-action p95 improved and broad direction agreement was already 100% in both arms, yet live transport still failed. A previous route-choice auxiliary head achieved 92.5% held-out direction-class accuracy while action p95 worsened 2.153→2.811 mm. | An offline measure of correct *requested-card branch and later route progress* on complete held-out episodes is needed before another live actor test. Single-step imitation error and coarse direction class are insufficient gates. |
| **5. Contact and stalling** | Both latest actors received a −24 contact score on the two-card start. The revised actor reached the three-card buffer but later stalled. Fresh scored teacher routes have few sustained holds; removing 689 legacy stall rows slightly worsened offline results. | The specific two-card contact object and whether the late three-card standstill is geometry, controller saturation, or policy persistence. These matter for safety, but the five-card wrong branch happened earlier. |
| **6. Handoff and insertion remain unverified together** | The selected-port crop reached 18/18 early correct anchors on separate dense-card development scenes with 2.22 mm accepted 3D p95; the pause head was shadow-only. Neither was connected to the latest autonomous actor. | Whether the pipeline stops at the right time on actor-generated views, preserves requested-port identity, achieves insertion-grade pose/orientation, and safely hands control to the local insertion actor. Offline coarse accuracy does not demonstrate full insertion. |
| **7. Generalization across tasks/layouts** | The latest decisive live gate covered only three SFP starts (2, 3 and 5 cards). Its correction examples targeted port 1. Newer offline SFP development data underrepresent fresh 3–5-card groups; SC was monitored for offline action regression but was not part of this live gate. | Reliable rates by requested card/port, occupancy pattern, starting pose and SC-to-SC task. The reserved final scenes remain sealed. |

## What this is *not* currently evidence for

- A broken TCP-body to base-frame conversion or stale camera triplet: the
  bounded causal audit checked those and found no gross error.
- A cable snag in the five-card live failure: the measured route diverged
  toward the wrong card before the late standstill. A snag could still occur
  in other starts.
- A need to scale the same later-state correction dataset: its matched
  offline gain did not pass the live gate, and it omits the initial branch.
- A failure of the insertion actor: it was not invoked in the latest test.

## Most useful next discriminator

Use archived **early** observations from same-reset, different-request
teacher routes. Ask whether an observation-only model can identify the
requested occupied card and classify which *coarse branch* the successful
teacher takes, using complete episode-grouped held-out scenes. Inspect the
original images and task code before fitting. Compare against a state-only
and task-code-shuffled control. If the requested card or branch is not
visually/causally distinguishable, an explicit goal token cannot repair the
problem; change the view or supply a safe route-planning mechanism. If it is
distinguishable, one matched spatial-goal actor fit can test whether using
that information improves full actions and fresh autonomous arrivals.
The [experiment record](../experiments/2026-10-07-sfp-later-state-causal-repair.md#decision-and-one-distinct-next-experiment)
sets its budget and stopping rules.

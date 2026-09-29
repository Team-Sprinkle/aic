# Hierarchical recovery for cable insertion

Status: proposed core controller; not yet validated as an improvement (2026-09-24)

The [September 24 Isaac SC repair](experiments/2026-09-24-isaac-sc-fidelity-fix.md)
now reaches the Gazebo-derived SC target in one normal-collision five-card
scripted scene and reproduces two local plug/port blockages. It also shows
that the earlier gripper/card item-2 collision vanished on the centered route
after calibrating the grasp. [Deliberate corrected-asset route probes](experiments/2026-09-24-isaac-sc-contact-routes.md)
then reproduced named gripper/card and plug/card jams; cable/card contact
occurred only alongside a gripper jam. These traces support specific contact
mechanics, not an isolated route-level cable snag or autonomous recovery. A
container-local RTX driver-check override later passed a short three-wrist-camera
SC smoke test, with an orange cable visible in a separate diagnostic view.
Longer moving-cable, observation-contract, and policy gates remain open.

The [first read-only trace replay](experiments/2026-09-24-hierarchical-recovery-continuation.md#trace-diagnostic-command-and-interpretation)
found that a correctly vector-tared two-sample force-plus-stall trigger fires
on all seven retained 20 Hz repeats, including six partial insertions. A longer
force-history and motion-progress rule also fires on all seven. Subtracting
force norms instead misleadingly misses most, because it discards direction.
The controller code is a tested prototype, **not** a deployable recovery
policy. Named contact and controller-limit diagnosis plus independent Gazebo
scene calibration are required before a trigger can own motion.
In a broader scored set of 17 retained traces, the three no-insertion runs
reached at least 89 mm peak TCP command-tracking error, while fourteen
full/partial runs stayed below 73 mm. Three new scene variations added a full
and two partial controls that also stayed below 80 mm. Because that line was
chosen after seeing the saved outcomes and no new failure validated it, it is
only an incident-search hypothesis. Tracking error also cannot tell whether
the cable, gripper, port, IK, or controller caused the stop.

## Why this is a project direction

The robot can command motion toward a port while its measured TCP and plug
hardly move and wrist force rises. The failure may be a local port contact, a
gripper/card collision, cable tension, or a controller/workspace limit. A
single small retreat can clear a port lip; it cannot be assumed to undo a
cable route across several cards. We therefore propose **hierarchical
backoff**: a force-and-progress supervisor that retraces measured motion to a
verified safe checkpoint, then escalates to an earlier route checkpoint and a
new corridor when a local retry repeats the same obstruction.

The [five-card Gazebo route probe](experiments/2026-09-23-fixed-five-card-route-probe.md)
is the motivating example. In one across-card run the cable lay close to a
card, a cable link became nearly stationary, and the TCP stopped despite
advancing commands. A later 20 Hz rerun visibly draped the cable over cards
and stopped without insertion, but its larger lateral error and moving cable
link mean **its exact contact pair and root cause are unproven**. The official
[friction and physics notice](https://discourse.openrobotics.org/t/note-to-participants-gazebo-bug-fix-changes-to-friction-properties-of-the-cable/54098)
describes related simulator issues; our fresh single-trial runs use the
post-friction official image and do not match the old between-trial ghost
collision symptom. Do not label either run as a proven cable snag.

## Controller contract

```text
RGB + robot state + wrist force + measured motion
                    |
             learned BC/RL actor
                    |
             normal action proposal
                    |
          force/progress supervisor
             |              |
           clear         blocked
             |              |
        execute actor   stop forward push
                            |
                    retrace measured poses
                            |
                    safe local checkpoint B
                            |
                  actor chooses new retry
                            |
                    repeated blockage?
                         |       |
                        no      yes
                         |       |
                    continue   earlier route checkpoint D
                               + different observed corridor
```

The supervisor owns immediate unloading and safe reversal. The learned actor
owns normal transport, alignment, lateral/rotational retry, and choice among
observation-supported route options after clearance. No simulator ground-truth
plug, port, cable, or card pose may enter the deployed policy or route selector.
Simulator geometry and named contacts can be used for training labels and
post-run diagnosis.

### Detecting a real blockage

Do not use force alone: correct insertion is contact rich. Trigger on
persistent force plus meaningful command with little measured TCP/plug progress,
or an independently validated tension signal. Use a lower force threshold to
declare release. Record the blocked direction and the first stall time. During
reversal, stop if force grows, the cable tightens visibly, or motion diverges
from the stored path.

### Checkpoints, escalation, and memory

- Save measured **position and orientation**, force, image/visibility summary,
  timestamp, and route phase at verified low-force checkpoints. Index by
  traveled distance and phase, not only by decision count.
- Level 1: retreat to a free near-port checkpoint, then attempt a different
  lateral/rotation choice. This addresses port-lip or alignment contacts.
- Level 2: after repeated blockage in the same spatial corridor or insufficient
  net progress, retrace to a pre-obstacle checkpoint and choose a different
  card/cable route. The route must differ meaningfully from the failed one.
- Level 3: if neither route releases load within a bounded force, distance,
  time, and retry budget, stop safely and mark the episode unresolved. An
  endless cycle of local backoffs is not recovery.
- Track local attempts, route resets, tried corridors, maximum and integrated
  force, measured clearance, progress since the last checkpoint, and whether
  the same obstacle was revisited. Expose compact recovery state to the actor.

If a new contact at C follows substantial safe progress from B, a local retry
may be appropriate. If C repeats A's position/blocked direction or cable
tension remains after reaching B, escalate to earlier checkpoint D. The
thresholds, number of retries, and route candidates must be selected on
development episodes, then frozen before held-out comparison.

The [current Isaac recovery code](../aic_utils/aic_isaac/aic_isaaclab/scripts/serl/measured_path_recovery.py)
stores only recent TCP positions and implements a local 1 mm minimum/10 mm
maximum retreat with a 12-decision near-lateral retry. It counts backtrack
events but has no orientation history, safe-checkpoint stack, route-level
escalation, or Gazebo deployment parity. The [matched SFP result](experiments/2026-09-22-serl-mixture-recovery.md)
is inconclusive for recovery and has no proven cable snag.

## Learning and evidence boundary

Keep every complete failure and recovery transition, including executed
commands, measurements, recovery mode, actor ownership, force, terminal
observation, and scorer outcome. Hard-coded retreat actions can update a critic
through their outcomes but must be masked from actor gradients. Successful
scripted recoveries can supervise the learned post-clearance decision; failed
traces alone do not reveal the successful alternative. Compare the same actor
with no recovery, local recovery, and hierarchical recovery on identical
episode-grouped starts and budgets. Measure **full insertion**, not just force
release, plus peak/integrated force, recovery count, time, and p95 inference
under 300 ms. Preserve the sealed final configurations until development
gates pass.

## Related work and claim boundary

- [ARCH](https://arxiv.org/html/2409.16451v2) uses a high-level policy to
  select motion-planning and RL assembly primitives. Its recovery trials teach
  retries after failed skills; the paper gives a failed grasp followed by a
  new grasp attempt as an example. Its *hierarchy is in skill selection*. It
  does not describe retracing measured motion, local and earlier route
  checkpoints, or escalation based on repeated blockage.
- [Hierarchical DLO Routing](https://arxiv.org/html/2510.19268v2) uses a VLM
  to choose Insert, Pull, or Flatten skills for routing a flexible object
  through clips. It counts consecutive failed insertions; when the cable head
  is twisted or poorly aligned, Flatten reorients it into an insertable state
  before another attempt. Pull and Flatten can move the cable away from a clip.
  This *is* cable-specific recovery within a hierarchical skill system, but
  the described method does not reverse the measured gripper path to a local
  checkpoint and then escalate to an earlier route checkpoint or new corridor
  after another blockage. We should not cite it as prior implementation of
  that particular two-level backtracking rule.
- [Multi-Stage Cable Routing](https://arxiv.org/abs/2307.08927) learns both
  sequencing and motor skills for routing through clips. Its routing setting
  differs from a gripped connector inserting beside five NIC cards.
- [SERL](https://arxiv.org/abs/2401.16013) demonstrates sample-efficient
  off-policy robotic RL, including cable routing and insertion, with prior
  data and recovery behavior. It does not validate our exact checkpoint
  escalation scheme.
- [Recovery RL](https://arxiv.org/abs/2010.15920) learns safe recovery zones
  for contact-rich tasks. It motivates a recovery boundary, not a novelty
  claim for simple retreat.

A defensible **candidate** paper contribution is the *specific, measured-path
checkpoint escalation coupled to cable-aware route changes and learned
post-clearance insertion*, if matched ablations show better full insertion
than both no recovery and local-only recovery. The bounded literature search
above did not find this exact combination for the AIC wire task. Neither ARCH
nor Hierarchical DLO Routing establishes that particular mechanism, though
both are relevant context. This is not an exhaustive novelty review, and no
performance claim is warranted yet.

## Execution

The [September 24 continuation plan](experiments/2026-09-24-hierarchical-recovery-continuation.md)
contains the ranked failure matrix, data requirements, BC/RL sequence,
simulator gates, matched ablations, and Gazebo transfer plan. Current failure
artifacts are mapped in [outputs_README.md](../outputs_README.md).

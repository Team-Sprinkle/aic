# Step 5: SC correction data, vision, RL, and teacher routes

The added data were **all SC-to-SC**, but the policy being trained was a
**shared SFP-to-NIC and SC-to-SC actor**.

## 1. What data and failures did we test?

The latest correction collection ran 24 SC-to-SC episodes. The actor
controlled brief **0.2-second pulses every 4 seconds**; a privileged
CheatCode teacher controlled the robot between pulses and tried to finish
insertion. The teacher-assisted runs inserted in **18/24** episodes. Only
**16** passed the stricter force, grasp, and recording checks. From those
16, the focused training comparison used **1,458 image-and-action rows**:
actor-controlled moments and the following 0.5 seconds of teacher
corrections.

Both policies started from the same BC checkpoint and trained for 1,500
updates. The control had **318 training episodes**; the treatment had
those same episodes plus the **16 SC correction episodes**. Neither policy
inserted autonomously on either of the two opened SC development scenes.
**Backtracking was not enabled** in collection or evaluation. This
comparison therefore says nothing yet about whether the backtracking idea
helps SC-to-SC.

The two trained-policy failures ended **110 mm and 70 mm from the port**,
with no scored contact. They were failures to complete the approach and
insertion, not observed high-force blockages that would trigger retreat.
In a separate frozen-policy trace, the actor commanded motion away from
the port and the plug followed. Earlier naive SC teacher failures often
ended roughly **30–80 mm away**, but their recordings do not establish
whether cable tension, an obstacle, or another route problem caused each
stall.

Existing recordings were rendered into three-camera videos:

- Teacher-assisted, successful: [SC port 0](../outputs/experiments/2026-09-25_step5_review_videos/teacher_interleaved_trial_1003101.mp4) and [SC port 1](../outputs/experiments/2026-09-25_step5_review_videos/teacher_interleaved_trial_1003102.mp4).
- Trained policy alone, failed: [two cards, SC port 0](../outputs/experiments/2026-09-25_step5_review_videos/trained_window0p5s_trial_999001.mp4) and [three cards, SC port 1](../outputs/experiments/2026-09-25_step5_review_videos/trained_window0p5s_trial_999003.mp4).

The [video notes](../docs/experiments/2026-09-25-step5-review-videos.md)
explain the scores and recording limits. These teacher and policy clips
use **different scenes**, so they show the behaviors but are not a paired
video comparison. The policy clips have roughly one saved frame per
second; the teacher clips play collected frames in order and compress
some simulator time.

## 2. Is vision finding the target reliably?

**Not throughout a full episode.** The current actor reads three camera
images, robot state, elapsed decision time, and a task vector. It has a
pose-prediction head, but in this **body-target version**, that predicted
pose is an auxiliary training output: the action head does **not** consume
it. The actor predicts full movement targets relative to the current TCP.

We also trained a separate, task-conditioned visual port locator. It can
work well when the opening is visible, especially near the port. At the
initial position, however, the selected opening was outside **both
side-camera images in six of eight** checked starts. Its tested offscreen
prediction did not solve that case. A task label tells the model *which*
port is wanted; it does not reveal the port's 3D position when the cameras
cannot see it. The robot may first need an observation-based move to
obtain a useful view. We have not established a reliable full-start
strategy for that.

## 3. Could we start RL from this BC policy?

The task vector does identify the family and selected port. The shared
actor receives two family bits and two selected-port bits; SFP tasks also
encode the target NIC card. The separate SC visual locator receives the
selected-port bits too. So the point about **task conditioning is
correct**.

The part to correct is the PoseInsert assumption: **this current
body-target actor does not use an estimated target pose as an action input
or express its actions in the port frame.** An earlier port-frame actor
did, but it also failed the autonomous gate. Even with a reliable pose
estimate, knowing where the port is would not by itself teach a safe
cable route, alignment, insertion, and recovery.

We *can* initialize RL from the current BC checkpoint. The concern is
practical: on these full starts it finishes centimeters away, without
insertion or a useful contact event, so full-start insertion rewards
would be very rare. A bounded RL test becomes more informative on
**observable, near-completion incidents** where the robot can execute
its commands and success or failure supplies feedback. It would still
have to prove that what it learns transfers to full starts. Backtracking
should be tested as its own controlled condition, not assumed to be part
of these BC results.

## 4. Did we try a detour for the SC expert?

Yes, **a vertical pre-route lift**, though we have not established a
general left-or-right detour policy. On one four-card scene, an 8 cm lift
changed a failed teacher run into a full insertion. In a matched
six-scene five-card test, the unchanged teacher inserted **0/6**; adding
an 8 cm lift inserted **3/6**, all for port 0. The other three port-1
scenes still failed. Those successful lift runs supplied expert data,
but the lift was a **teacher collection intervention**, not learned or
deployed backtracking.

The route idea remains plausible. What we know is that the naive teacher
can stall short of the opening and that lifting fixes *some* starts. We
do **not** yet know the physical cause of every stall, nor whether moving
left or right around the cards would fix the remaining ones. That
distinction matters before training an RL recovery rule around a presumed
cable snag.

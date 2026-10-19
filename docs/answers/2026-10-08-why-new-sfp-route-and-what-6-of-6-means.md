# Why we tried a new SFP route

Our transport actor still reached the safe area near the requested port on
**0 of 3** independent autonomous test starts. We wanted better examples of
what it should do after it goes off course. The existing CheatCode teacher
could sometimes recover, but its recorded short paths included pauses and
detours that were hard to turn into clear training targets. An earlier
MoveIt test planned to a *future teacher joint position* and sometimes cut
straight across the teacher's detour. That did not give us trustworthy route
labels.

So I tried a different **offline route-label candidate**. At a saved actor
state, I gave MoveIt the actual requested card and port from the simulator
and asked for a path to a point **85 mm in front of that port**. I compared
three simple path shapes: go directly, align from farther away, or retreat
and then align. This was a diagnostic use of privileged scene geometry. It
was **not** the visual actor choosing or following a route.

## What the numbers mean

- **6/6 arm plans:** On all six saved actor states, MoveIt found a path for
  the rigid robot arm to the requested port's approach point while avoiding
  the rigid scene objects represented in its planning scene. This means the
  planner returned paths. It does **not** mean the robot executed them, the
  cable stayed clear, or the plug inserted.

- **3/6 cable-screen passes:** I took the cable shape observed at each
  saved starting state and, as a rough calculation, moved that shape along
  the planned path as if it were rigidly attached to the gripper. Only three
  of six scenes had a candidate for which this *entire simulated cable
  shape* stayed outside a conservative NIC-card collision bound. This is a
  warning about possible cable interference, **not three successful
  insertions out of six**. The real cable bends and moves differently. In
  fact, this same screen flags recorded CheatCode runs that later inserted,
  so a failed screen does not prove a snag.

- **No live result:** I did not execute the new planned paths in Gazebo.
  There is therefore **no 6/6 or 3/6 insertion result** for this planner.
  The last independent autonomous transport result remains 0/3 safe
  arrivals.

## Was a new actor trained?

**No.** I wrote a planner and an offline cable audit, then checked saved
scenes. There is no new actor checkpoint from this work. The privileged port
coordinates were used only to generate and inspect possible training routes;
they were not supplied to a deployable actor. We stopped before training
because these paths have not yet been shown to be safe route labels.

The detailed measurements and exact artifacts are in the
[experiment record](../experiments/2026-10-08-sfp-requested-port-corridor-audit.md).

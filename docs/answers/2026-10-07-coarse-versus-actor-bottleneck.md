# Is coarse perception working, and is the actor the bottleneck?

**For the current SFP transport-to-handoff test, the actor is the immediate
bottleneck.** On three fresh autonomous starts, neither the revised actor nor
its matched control reached the safe area where perception could take over
(0/3 each). Improving the port crop cannot fix a route that never reaches
that area.

The coarse model has passed a **specific offline test**, rather than the
complete deployment test. Given original-resolution camera images and the
requested card/port code, it found an early correct requested-port crop on
18/18 separate dense-card development scenes, with no accepted wrong ports.
Its accepted 3D position error was 2.22 mm at the 95th percentile. This is
promising for identifying a nearby port and making a crop. It does not yet
prove reliable live handoff on actor-generated views, SC-to-SC coverage, or
the submillimeter accuracy needed for final insertion. The camera-to-model
path took 16 ms at the 95th percentile with images already captured; complete
live latency still needs measurement.

The actor's problem is broader than the number of nominal demonstrations.
We added 136 varied, fully inserted SFP teacher episodes, then more
same-reset alternate-request episodes. A matched actor learned a clearer
effect of the requested port and improved held-out *teacher-observation*
action error. Yet on fresh rollouts it was already about 13–22 mm away from
the successful teacher's measured path by 8–12 seconds. Later it sometimes
issued large movement commands while the measured TCP barely moved. The
official scorer confirmed gripper-to-card contact in one run. The exact
cause of the other two standstills is not established by the current logs.

That pattern suggests a **coverage mismatch**: supervised training mostly
shows the actor what to do while it is *on a successful teacher route*. A
small early error changes its next view and robot state; those actor-visited
states are poorly represented by more ordinary teacher routes. The actor
then has little demonstrated guidance for correcting its route or avoiding
an obstacle. We should treat this as a hypothesis, not a proven sole cause:
controller execution, collisions, and representation or history could also
contribute. The command-versus-measured-motion traces are why the next test
must check safe continuations from actual actor-visited states before fitting
another policy.

The bounded next test is at most six off-route training rollouts. If a
collision-checked teacher can recover and fully insert from at least four
visited states, use those corrections for one matched actor fit and require
at least 2/3 safe arrivals on fresh nonfinal starts, without worse contact.
If it cannot recover, diagnose the physical or controller blockage first.
Do not infer improvement from another lower recorded-action error alone.

Details: [October 7 experiment record](../experiments/2026-10-07-actor-proposer-coarse-continuation.md).

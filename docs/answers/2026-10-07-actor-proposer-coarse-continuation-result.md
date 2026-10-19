# October 7 actor, proposal head, and coarse perception update

I collected additional varied SFP teacher routes and trained a matched
requested-route actor comparison. The new actor learned to respond more
strongly to the requested card and port, and its held-out recorded-action
errors improved. But on three fresh autonomous Gazebo starts, **both the
control and new actor reached the safe transport handoff area 0/3 times**.
The new actor also hit a card mount with its gripper in one run. It is not
promoted.

The route diagnosis shows why the recorded-action score was misleading:
the new actor was already roughly 13–22 mm from the successful teacher path
by 8–12 seconds. Later, it kept commanding sizable moves while the TCP
barely moved. The three-card collision is confirmed by the official score;
the exact cause of the other two stalls remains unresolved. More ordinary
teacher routes alone would not address those actor-visited states.

The perception work did improve. The original-image selected-port crop now
found the correct requested port early on **18/18** separate dense-card
development scenes, versus 15/18 before, with zero accepted wrong ports.
Its accepted 3D pose p95 is **2.22 mm**, so it is useful for coarse handoff
but not yet accurate enough to establish the 0.5 mm insertion corridor. The
small pause-proposal head was timely on **2/3** saved actor routes without a
false first proposal; it remains shadow-only and is tied to the older actor.

The next bounded test is to gather at most six **actor-visited** route
deviations and check whether a collision-checked teacher can safely recover
and fully insert from at least four of them. Only then should a matched
2,000-update correction fit be tried. It must preserve grouped action error
within 10%, then reach the safe buffer on at least 2/3 fresh nonfinal starts
without worse contact and with inference under 300 ms. If safe teacher
continuations fail, the obstacle or controller cause needs direct diagnosis
before more training. Automatic handoff, insertion and RL remain disabled;
reserved final scenes remain sealed.

Full methods, scores and exact artifact paths are in
[the experiment record](../experiments/2026-10-07-actor-proposer-coarse-continuation.md).
The [machine summary](../experiments/2026-10-07-actor-proposer-coarse-continuation.summary.json)
and [output map](../../outputs_README.md) locate the checkpoints, traces,
videos and measured-route diagnosis.

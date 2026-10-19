# Transport return: result

I audited the three frozen SFP actor failures against their same-reset
CheatCode teachers. The actor starts moving later, then gradually drifts away
from the teacher route. The first point more than 10 mm from the teacher's
*path* occurs at 4.9, 6.55 and 10.35 simulation seconds. Camera timestamps
match the policy proposals, and the robot generally moves in the direction
the actor commands. This points mainly to route/progress decisions, with
startup delay also contributing.

I trained one matched transport actor on the existing data. It uses the
requested card/port code to modulate camera features and learns a discrete
future travel direction while still predicting the full TCP-body action.
The route-direction score reached **92.5%** on held-out teacher frames
(50% majority baseline). Changing the request on the same image changed its
action more than before: **3.394 mm** versus **0.554 mm** median. Matched
teachers differed by **12.44 mm** at the early route branch.

The actual action metric got worse: held-out first-action p95 increased from
**2.153 to 2.811 mm**. That exceeded the declared 10% regression limit, so
I did **not** run or promote this checkpoint in Gazebo. The previous live
transport result remains **0/3** safe buffered arrivals. No final scene was
opened and no RL or insertion training was started.

The next distinct test should give the actor spatial evidence about the
requested card and nearby route, such as camera feature maps or an
observation-selected original-image card-row crop. It should use the same
paired scene split and full-action target, and must improve request-specific
route behavior *without* worsening held-out action error before any live
test. Selected-port identity is still a separate handoff gate.

Full commands, checkpoints, metrics and trace paths:
[SFP transport experiment](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-07-sfp-transport-route-return.md).

# SFP actor-state recovery labels: three-run feasibility pilot

## Decision

**The preset pilot gate failed.** Three new, train-split, nonfinal SFP scene
resets were run with the unchanged frozen transport actor for five seconds,
then the privileged CheatCode teacher took over. The teacher fully inserted
on **2/3** and had no official contact penalty on all three. However, only
**1/3** yielded a clean short-route label: after 20 mm of measured TCP path,
its net displacement from the takeover state was at least 10 mm. The pilot
required at least **2/3 both safe full insertion and valid label**. No new
waypoint actor was trained, no autonomous rollout was opened, and the reserved
final scenes remain sealed.

## Why a single shadow CheatCode query is not a route oracle

The current `CheatCode.insert_cable()` initializes a 5.5 s interpolation
from its *current* gripper pose toward the target. At interpolation fraction
zero, `calc_gripper_pose(..., position_fraction=0,
slerp_fraction=0)` returns the current pose and `_send_delta_pose_target`
issues essentially a hold command. A one-shot query at an actor observation
therefore cannot provide a meaningful obstacle-aware waypoint. Later
interpolation commands depend on elapsed teacher stage and the changed robot
state; replaying them without the simulator is not a causal teacher query.
The three actual takeovers measured this behavior instead.

## Exact pilot results

| Cards/request | Official outcome after teacher takeover | Contact score | First 20 mm path: lookahead | Net displacement | Valid label? |
| --- | --- | ---: | ---: | ---: | --- |
| 2 / port 1 | Full insertion, tier 3 = 75 | 0 | 2.22 s | 9.17 mm | No |
| 3 / port 0 | Partial insertion, tier 3 = 38.62 | 0 | 3.74 s | 7.66 mm | No |
| 5 / port 0 | Full insertion, tier 3 = 75 | 0 | 1.84 s | 15.73 mm | Yes |

The saved [measured TCP path plot](</home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1/takeover_tcp_paths.png>)
shows the loops: the 2-card successful case had traveled 20 mm but was only
9.17 mm from its takeover pose. The three-card case took over four seconds
to reach 10 mm net displacement and ultimately did not fully insert. This
supports a **label-definition / controller-settling issue**, and also shows
that teacher success from actor-visited states is not guaranteed. It does not
show a cable snag or identify the precise cause of the partial insertion.
The CheatCode return value alone is not a success label: it returned normally
even on the three-card partial insertion. The **official scored insertion
result** determines success in this table and pilot gate.

A read-only alternative-label count on the existing 16 train / six
development actor-boundary episodes found a point at **first 10 mm net TCP
displacement within four seconds** in all 22 episodes. This is a possible
*new* label hypothesis, but it has not been independently checked for safe
route direction or used to train a model. A stricter read-only straightness
audit found only **11/16 train and 4/6 development** paths with measured
travel at most 1.5 times the net displacement. In two development cases the
first 10 mm net target was reached only after 1.84× and 2.74× as much actual
travel; commanding a straight shortcut to it could omit the teacher's detour.
This fails the intended six-group label-quality gate even before contact or
card-side checks. Changing the label after seeing
this pilot would not make the predeclared 20 mm-path pilot pass. The three
new runs contain two safe takeovers, so even counting them would provide
only 18 distinct train-state sources, short of the planned 20-source
training minimum.

## Provenance and reproduction

The exact scene hashes, requests, command environments, three launch logs,
official scores, complete scored bags and extracted `scored_stream.json`
files are under
`/home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1/`.
The immutable `manifest.json` fixes the three sources. `summary.json` records
the predeclared gate; `path_diagnosis.json` and `takeover_tcp_paths.png` are
the posthoc measured-path diagnosis. `first_net10_audit.json` preserves the
read-only alternative-label lengths and path/net ratios for all 22 existing
actor boundaries. The executables are
`scripts/audit_sfp_first_net_waypoint.py` and
`scripts/run_sfp_actor_recovery_label_pilot.py`; the pilot command was:

```bash
.pixi/envs/default/bin/python scripts/run_sfp_actor_recovery_label_pilot.py \
  --output /home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1 \
  --gpu 7 --actor-prefix-s 5
```

The frozen actor received normal observations only; the teacher's port TF
was available **after** takeover for diagnostic control and labels. These
teacher-assisted insertions are not autonomous policy successes.

## Next decision

Keep the actor-state route-label gate open. Do not gather more identical
takeovers or retrain the waypoint head under either failed label. The
first-net-10-mm idea was checked and missed a conservative straightness gate.
The next distinct hypothesis should come from an explicit obstacle-aware
route planner or a recovery controller that can be queried at the actor's
current state and whose proposed path can be checked for collision and
force. Budget an offline six-scene feasibility check before collecting new
teacher data or fitting an actor. Require a safe proposed path on at least
five of six episode-grouped nonfinal actor states, no wrong requested card,
and no direct shortcut across the teacher's detour. If this fails, revisit
the transport observation/control design instead of scaling the same labels.
The repository has a MoveItPy collision-scene backend and an SC-specific
corridor planner (`scripts/plan_moveit_sc_fixed_route.py`), but **no ready
SFP actor-state route planner**. Any adaptation must load the measured
takeover joint state and the exact randomized Gazebo collision scene, then
check the full rigid path. MoveIt rigid collision success alone does not
validate a trailing cable or force; a later bounded Gazebo replay would
still be required.
The [subsequent six-scene offline probe](2026-10-08-sfp-actor-state-moveit-feasibility.md)
confirmed that MoveIt can plan to five available privileged teacher future
states, but it straightened two long development detours. That smoke test
does **not** pass the safe-route-label gate.
Do not promote visual handoff, insertion or RL while independent autonomous
transport remains 0/3 safe arrivals.

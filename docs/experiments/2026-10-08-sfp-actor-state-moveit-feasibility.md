# SFP actor-state MoveIt path feasibility

## Decision

**Rigid arm planning works as an offline smoke test; the recovery-route gate
failed.** Five of six saved SFP actor takeovers had a teacher state at least
10 mm net TCP displacement away within four seconds. MoveItPy loaded the
exact randomized Gazebo collision assets and found a rigid collision-checked
joint path from the measured takeover state to all five *privileged teacher
future joint states*. Start-pose FK agreed with measured TCP to numerical
precision and 124–233 collision objects were loaded. The sixth case was the
three-card partial-insertion pilot: it had no 10 mm net target within four
seconds.

On **two of three development scenes**, the teacher had traveled 22.13 and
27.65 mm to reach a 10 mm net endpoint. MoveIt planned a roughly 10 mm
straight shortcut to the same endpoint. The planner did not model the cable,
did not choose the requested port, and was never executed in Gazebo. These
shortcuts cannot be accepted as safe route labels. No actor was trained or
deployed; the independent autonomous transport baseline remains 0/3 safe
arrivals for both earlier actors, and reserved final scenes remain sealed.

## Data and protocol

Six distinct nonfinal reset groups were checked offline: three earlier
episode-grouped development takeovers (two, three and five cards) and the
three train-split five-second takeover pilots from the
[label-feasibility record](2026-10-08-sfp-actor-recovery-label-feasibility.md).
The three later development takeovers retained scores but no raw scored bags;
they cannot provide measured joint states for this probe. An initial
six-development run stopped after finding the missing streams. The switch to
three development plus three train groups is explicit; this result is not an
independent six-development generalization score.

For each saved bag, the target was the **first teacher measured TCP state at
least 10 mm net from takeover, within four seconds**. Its recorded joint
state was supplied to MoveIt as a *label-only goal*. The start was the
recorded actor takeover joint state. The exact scene collision objects came
from `eval_config.yaml` through the existing `MoveItPy` backend. A small
current-time joint-state publisher was necessary for MoveIt's scene monitor;
the first smoke test without it failed initialization and changed no scene.

| Group | Rigid plan | Measured teacher path to 10 mm endpoint | Planned TCP path | Interpretation |
| --- | --- | ---: | ---: | --- |
| Development, 2 cards | Yes | 22.13 mm | 10.00 mm | Shortcut across teacher detour |
| Development, 3 cards | Yes | 27.65 mm | 10.02 mm | Shortcut across teacher detour |
| Development, 5 cards | Yes | 14.65 mm | 10.02 mm | Smaller difference |
| Train, 2 cards | Yes | 22.02 mm | 10.03 mm | Shortcut across teacher detour |
| Train, 3 cards | No target | — | — | Expert later only partially inserted |
| Train, 5 cards | Yes | 12.42 mm | 10.00 mm | Smaller difference |

The [inspected measured-versus-MoveIt plot](</home/nas/chmin/aic_archives/2026-10-08_sfp_actor_state_moveit_six_v5_paths/measured_vs_moveit_paths.png>)
shows XY and XZ paths. The blue plan goes almost straight to the orange
teacher endpoint; in the first two development cases it bypasses a visible
loop in the measured teacher path. The loop might include controller settling
as well as purposeful clearance; we have not established why the teacher
took it. Because cable geometry and force were unmodeled, the shortcut's
safety is unknown. A MoveIt success bit alone cannot certify it.

## Artifacts, commands and failed iterations

The final six-case archive is
`/home/nas/chmin/aic_archives/2026-10-08_sfp_actor_state_moveit_six_v5_paths/`.
It contains six exact command files, per-case logs and JSON, the full planned
joint/TCP points, `summary.json`, `shortcut_audit.json`,
`measured_vs_moveit_paths.png`, and the scripts used to plot and audit.
Source scripts are `scripts/probe_sfp_actor_state_moveit_path.py` and
`scripts/run_sfp_actor_state_moveit_six.sh`. Run with:

```bash
bash scripts/run_sfp_actor_state_moveit_six.sh \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_actor_state_moveit_six_v5_paths
```

Earlier immutable archives preserve the diagnostic progression:
`.../2026-10-08_sfp_actor_state_moveit_six_v1/` attempted a 20 mm goal and
found none on the five-card early-development takeover;
`.../2026-10-08_sfp_actor_state_moveit_six_v2_net10/` switched consistently
to 10 mm but found the missing later-development raw bags;
`.../2026-10-08_sfp_actor_state_moveit_six_v3_train_dev_net10/` exposed the
partial pilot's missing 10 mm goal;
`.../2026-10-08_sfp_actor_state_moveit_six_v4_train_dev_net10/` retained that
case as a structured no-target result; v5 saved the actual planned paths.
No Gazebo job or active training process was interrupted.

## Next bounded experiment

An SFP-specific **path** planner must propose the requested-port route from
the current actor state using training-time scene geometry, rather than
receiving a future expert joint state as its goal. First run a six-scene
offline comparison of alternative route corridors and reject any path that
cuts through the teacher detour without a verified clearance explanation.
Require at least five of six scene-grouped plans with the correct requested
port, rigid collision checks, a documented cable-clearance proxy, and no
unexplained shortcut. Then replay at most three accepted plans in Gazebo to
check measured cable contact, force and progress. MoveIt currently has an
SC-specific corridor script, not an SFP actor-state corridor generator; it
would need adaptation. Do not collect more routine teacher takeovers or fit
the waypoint actor until a safe route-label source passes these gates.

**October 8 follow-up:** The [requested-port corridor audit](2026-10-08-sfp-requested-port-corridor-audit.md)
implemented that SFP adaptation and found rigid routes to the selected port
on 6/6 saved groups. The full-cable screen cleared only 3/6, so the safe
route-label gate is still open and no Gazebo replay occurred.

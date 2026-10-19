# SFP short route waypoint: bounded feasibility test

## Decision (October 8, 2026)

**Do not connect this waypoint model to the live transport controller.** Its
ordinary teacher-route predictions improved, but the proposed 10 mm command
pointed along the teacher's subsequent recovery motion on **0/6** saved
actor-to-teacher boundaries. A distance-based target did not repair the
boundary error. The independent autonomous transport result remains **0/3
safe buffered arrivals** for both prior actors; this experiment opened no
new rollout and makes no insertion claim. Final scenes remain sealed.

## Why this experiment was run

The [overall recommendation](../answers/2026-10-08-overall-recommendation-after-pose-actor-reset.md)
proposed an explicit, short obstacle-aware transport target. Earlier
full-action behavior cloning reproduced teacher actions but did not recover
after the actor drifted off its teacher route. Appending predicted port pose
to that actor barely changed its actions. The older
[two-stage endpoint probe](2026-10-03-sfp-route-stage-waypoint-probe.md)
already failed, so this test used full-phase short future motion and directly
checked the command that would be executed.

The input was the frozen three-camera vision feature, sparse history, robot
state and requested task code in the existing temporal cache. The target was
the **measured TCP displacement from the current TCP frame to a future point
on the teacher route**. This is route supervision, not the absolute port pose
and not an action delta added to the existing actor. Simulator geometry was
used only for posthoc scoring. No true port position entered a policy input.

## Data and controlled fits

The one-second labels contained 11,672 training teacher rows and 2,469
development teacher rows (41 complete development episodes), plus 16/6
training/development actor-boundary rows. Only the last actor observation
before teacher takeover could receive a teacher-future target. Splits were by
complete scene/reset; a transition-level random split was not used.

The first ridge probe had 16.83 mm p95 error on moving teacher rows, better
than a 33.54 mm state-only control, but 31.64 mm p95 at actor boundaries. A
same-budget frozen-feature neural comparison then fit:

| One-second target; 2,000 updates each | Moving teacher error p95 | Actor-boundary error p95 |
| --- | ---: | ---: |
| Teacher-only | 12.82 mm | 26.40 mm |
| 15% actor-boundary batch mass | 13.51 mm | 17.86 mm |

The second arm cleared the *numeric* error gate (≤15 mm moving teacher and
≤20 mm actor boundary). That boundary number comes from only six episode
groups, and the 15% arm was chosen after inspecting teacher-only performance
on them. It is therefore a diagnostic, not an independent generalization
result.

The decisive offline controller audit kept the existing actor's orientation
command and replaced translation with a step capped at 10 mm toward the
predicted waypoint. On moving ordinary teacher observations, its direction
had cosine ≥0.7 with subsequent measured motion in **97.6%** of rows versus
66.6% for the action-only control. At the six actor-to-teacher boundaries,
**both were 0/6**. Several early takeovers had only 2.4–7.1 mm net measured
motion during the first second while the teacher/controller settled; their
waypoint predictions pointed substantially the other way. This is why an
acceptable Euclidean error did not make the command safe. The saved record
contains all six vectors and cosines.

We tested one distinct label repair: choose the first point after **20 mm of
measured path length**, within four seconds, and require at least 10 mm net
displacement. It produced 26,719/2,332 train/development teacher rows but
only **13/4** actor-boundary rows. With the same architecture and update
budget:

| 20 mm path target; 2,000 updates each | Moving teacher error p95 | Actor-boundary error p95 |
| --- | ---: | ---: |
| Teacher-only | 14.77 mm | 28.98 mm |
| 15% actor-boundary batch mass | 17.01 mm | 27.37 mm |

Both failed the actor-boundary ≤20 mm gate, and the second also failed the
ordinary ≤15 mm gate. No controller audit or live rollout followed. This
rules out these two label definitions under the current small set of
actor-visited recovery states; it does not establish that route waypoints in
general cannot work.

## Reproduce and inspect

The exact probe command and logs are under
`outputs/experiments/2026-10-08_sfp_short_waypoint_feasibility_v1/`:
`run.command.sh`, `run.log`, `probe_with_labels.json`, `labels.npz`,
`score_servo_diagnosis.command.sh`, and `servo_offline_diagnosis.json`.
The path-label source, immutable arrays and manifest are in its
`arc20_labels/` subdirectory. The one-GPU rootless training wrapper was
`scripts/run_sfp_short_waypoint_neural_matched.sh`; the two archives are
`/home/nas/chmin/aic_archives/2026-10-08_sfp_short_waypoint_neural_matched_v1/`
and `.../2026-10-08_sfp_arc20_neural_matched_v1/`. Each archive has exact
commands, checkpoints, histories, grouped metrics and manifests. See also
the machine-readable `decision.json` beside this record.

Source scripts: `scripts/probe_sfp_full_route_short_waypoint.py`,
`scripts/train_sfp_full_route_short_waypoint.py`,
`scripts/score_sfp_short_waypoint_servo.py`, and
`scripts/build_sfp_arc_waypoint_labels.py`.

## Next targeted repair and stopping rule

The missing evidence is a route label **at states the actor actually visits**.
First audit whether a privileged teacher can be queried in *shadow* at saved
actor observations without relying on its internal stage from a different
trajectory. Verify proposed labels by short, controlled teacher takeovers on
nonfinal scenes. A static CheatCode call is not automatically a valid oracle:
its stage and controller settling must be accounted for. Budget this as one
bounded label-feasibility test, not another generic teacher-route collection:
at most 30 actor-state queries and six validation takeovers, with no policy
training if fewer than 20 distinct training reset states and six held-out
episode-grouped states have causally verified safe recovery directions.
Require ≥5/6 held-out proposed directions to agree with safe measured teacher
continuation (cosine ≥0.7 after meaningful movement), no wrong requested
port/unsafe card route, and a documented label latency. If this gate fails,
revisit explicit route planning or a safer recovery controller instead of
scaling this waypoint network. Only after it passes should a matched
waypoint-versus-action-only actor be fit, then tested on three new nonfinal
starts with ≥2/3 safe buffered arrivals, no worse contact, and complete
inference p95 <300 ms. Handoff, insertion and RL stay disabled meanwhile.

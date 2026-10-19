# SFP pose actor: training-method and partial-restart diagnostic

## Question and split

The frozen requested-port estimator could locate the target on saved off-route
actor views, but a pose-conditioned full-action actor barely used its input.
Test whether the previous actor initialization, old-policy preservation loss,
and weak corrective sampling explained this. Use the nine existing 15-second
actor-to-CheatCode takeover episodes: six separate train configurations and
three development configurations. Keep actor-prefix rows as causal history,
supervise only recorded teacher actions after takeover, and never pass true
simulator geometry to the actor. No simulator, live, final or RL run was made.

The first teacher action joins to the preceding same-reset actor observation
in **9/9 episodes**, with a 0.196–0.482 s gap and zero train/development
scene overlap. However, the frozen pose was accepted at that *first* decision
in only **3/6 train** and **1/3 development** episodes. Over all later teacher
decisions, accepted pose covered 112/333 train and 50/166 development rows.
One training episode had no accepted pose at any teacher decision, and one
development episode had only two. This is a substantial supervision coverage
gap. [Causal join audit](../../outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/causal_join_all_nine.json).

## Small capacity and pose-use check

On the 112 accepted train rows and 50 accepted development rows, fit the
same full-action temporal actor for 500 steps from its old checkpoint,
first retaining and then resetting the action head. Both used frozen
predicted poses, not true poses; complete episodes remained split.

| Diagnostic | Train action error p95 | Development action error p95 | Development action change when pose is zeroed p95 |
| --- | ---: | ---: | ---: |
| Warm action head | 0.603 mm | 2.564 mm | 0.025 mm |
| Reset action head | 0.671 mm | 7.386 mm | 0.035 mm |

Both fits learned the small training actions but barely responded to pose.
The reset generalizes much worse. This does not prove that the port offset is
unusable: with only a few small routes, the model can copy actions from
history and task code. It does show that resetting the decoder alone does not
make the current imitation loss exploit pose. Exact commands and logs:
`outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/{run.command.sh,run_reset_head.command.sh,run.log,run_reset_head.log}`.

## Matched full-data partial restart

To remove two likely suppressors, run two more 2,000-update arms with
identical native-image SFP and SC rows, the same frozen pose cache and prior
temporal weights. Give corrective SFP rows 40% of SFP sampler mass (20% of
all draws), remove the old-policy preservation term, and include corrective
development action error in checkpoint selection. The only arm difference is
whether the final full-action decoder is reset. The frozen visual feature
cache and pose estimator are unchanged.

| Selected actor | Late correction action p95 | Pose-zero action change p95 on 50 accepted rows | Old SFP p95 | SC p95 |
| --- | ---: | ---: | ---: | ---: |
| Warm decoder | 2.833 mm | 0.044 mm | 1.446 mm | 1.610 mm |
| Reset decoder | 3.694 mm | 0.038 mm | 1.891 mm | 1.820 mm |

The reset worsened late correction by 30.4% and ordinary SFP/SC by about
31%/13%; neither arm passed the predeclared 0.1 mm pose-use test. Both ran
the full 2,000 updates; selected updates were 1,000 warm and 1,750 reset.
The late development episodes **were used for checkpoint selection** here,
so their action errors are a diagnostic comparison, not an untouched
generalization estimate. We did not open new live scenes because even this
development comparison failed. [Machine score](../../outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/partial_restart_score.json) ·
[decision](../../outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/decision.json).

## Decision and next gate

Do **not** start a whole actor from scratch now. The smaller reset lost
ordinary behavior and still did not use pose. More iterations of the same
four-step imitation loss are also poorly motivated by the pose-ablation
result. The next bounded design should supply an explicit target-relative
route-error, waypoint, or short progress objective, then decode a complete
TCP-body action chunk. Check pose acceptance at the first corrective action
and keep one new episode-grouped development split independent of checkpoint
selection. Compare with a matched action-only actor; require a meaningful
pose intervention effect and held-out correction improvement before any
autonomous Gazebo start. No action-delta adapter or privileged pose input.
Automatic handoff, insertion and RL stay disabled; final scenes remain sealed.

Reproduction entry points: `scripts/diagnose_sfp_pose_actor_training.py`,
`scripts/run_sfp_pose_partial_restart_matched.sh`, and
`scripts/score_sfp_pose_partial_restart.py`. Full fit commands, checkpoint
manifests and histories are archived under
`/home/nas/chmin/aic_archives/2026-10-08_sfp_pose_partial_restart_matched_v1/`;
the exact scoring command is
`outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/score.command.sh`.

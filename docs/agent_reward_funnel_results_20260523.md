# Agent Reward Funnel Results - 2026-05-23

## Comparison

| method | source | strict success | final / best geometry | module consistency | contact/force | video sanity | conclusion |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| ACT baseline eval | prior docs, Gazebo/Isaac | 0 | ACT reaches near gate but does not reliably insert | not seated | no reliable insertion | prior videos show module outside | baseline remains unsolved |
| Online SERL existing hand reward | prior docs | 0 | best prior semantic-progress samples around `s=7.9mm`, `r=0.13mm`, `theta=0.042rad` | improved but not strict | not enough visual confidence | noisy video, no strict claim | promising but orientation residual remains |
| Guarded privileged servo | `outputs/agent_reward_funnel/servo_sweeps/20260523_phase2_privileged_servo_sweep` | 2/60 synthetic | strict only for `lat1/2mm`, `ax3mm`, small theta | required in success predicate | 36 contact-spike failures | image snapshots only, no Isaac video | useful controller smoke, not task solved |
| Phase reward funnel hand-tuned | `outputs/agent_reward_funnel/reward_audits/20260523_phase3_hand_tuned` | N/A formula audit | zero bad-forward surfaces | included in gate | contact not simulated | reward plots only | reward avoids known false positives |
| Phase reward funnel auto-from-servo | `outputs/agent_reward_funnel/reward_audits/20260523_phase3_auto_from_servo` | N/A formula audit | zero bad-forward surfaces | included in gate | contact not simulated | reward plots only | stricter than hand baseline after audit clamp |

## Training Stability

No new Isaac online SERL training was launched in this pass. The implemented scripts are designed as fast gates before training:

1. Run formula and servo sweeps.
2. Run one-env Isaac guarded servo smoke with videos.
3. Only then launch online SERL variants.

## Reward Exploit Check

The new reward audit explicitly rejects positive near/inside axial reward when `r/theta` are bad. The servo sweep also rejects positive `s` when module consistency is low. This directly targets the prior exploit where `sfp_tip_link` looked inserted while the visible SFP module stayed outside.

## Recommendation

This remains viable only if the guarded servo transfers to Isaac contact with visible insertion. The paper direction should be framed as agent-assisted reward and controller-guard synthesis for contact-rich insertion, not as solved learning yet.

Minimum next controlled experiment:

- ACT-only no-learning Isaac eval.
- Existing `cheatcode_insertion_v1` guarded guide.
- New guarded servo settings from `docs/agent_reward_funnel_audit_20260523.md`.
- Same near-gate episode set, same cameras, same strict success parser.

Do not claim success without videos and strict `s/r/theta/module` checks.

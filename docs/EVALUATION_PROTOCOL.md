# Evaluation scope and provenance

The final media are local replays of the **published qualification** configuration. They are selected successful demonstrations, not new hidden-test results. The separate internal reserved final configurations remain sealed.

## Two different public configurations

| Trial | Task | Predeadline `sample_config.yaml` | Published `eval_config.yaml` |
| --- | --- | --- | --- |
| 1 | SFP to NIC | 1 card, rail 0 | 1 card, rail 2 |
| 2 | SFP to NIC | 1 card, rail 1 | 2 cards, rails 3 and 4 |
| 3 | SC to SC | No NIC cards | 3 cards, rails 0, 1 and 2 |

The default sample was public before the May 15, 2026 deadline; the exact qualification leaderboard configuration was released on May 28 after submissions. [Upstream release PR](https://github.com/intrinsic-dev/aic/pull/558). The archived [publication audit](answers/archive/2026-10-19-documentation/evaluation_protocol_full_history.md) retains dates, task assumptions and discussion links.

| File | SHA-256 |
| --- | --- |
| Pinned predeadline sample | `21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d` |
| Released qualification config used in final recordings | `9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5` |

## What the reported results establish

The user explicitly authorized inspecting released-evaluation videos and repairing their failures on October 19. The final **276.34/300** result was selected after that diagnostic work. A subsequent fixed batch yielded **15/19 complete passes and 53/57 trial insertion awards**; it did not establish universal success. The simpler public sample's **287.90/300** is a separate result. The earlier **290.19/300** sample used an earlier selected configuration.

Runtime inputs are wrist RGB, public task identity, measured robot state and wrist force. Camera calibration, static asset dimensions and fixture assumptions are explicit priors. Simulator poses/contact identities may supervise training or score an offline audit; they must not steer the autonomous controller or choose its crops. Diagnostic overview cameras are for videos, not policy input.

Preserve failures, frozen plans, effective Docker settings, hashes and pre-reset terminal observations. Distinguish toolkit insertion awards from separate strict depth/lateral/orientation audits. Any future generalization claim requires independently generated scene groups frozen before model/route selection; published scenes cannot become unseen again.

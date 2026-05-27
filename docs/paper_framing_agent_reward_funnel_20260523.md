# Paper Framing: Agent Reward Funnel for SFP-to-NIC Insertion - 2026-05-23

## Related Work Snapshot

- SERL provides a sample-efficient robotic RL software stack with controllers, rewards, resets, and real-world manipulation examples including PCB assembly and cable routing. Source: https://huggingface.co/papers/2401.16013
- HIL-SERL combines demonstrations, binary reward classifiers, and human interventions for precise manipulation. Source: https://hil-serl.github.io/
- Eureka uses LLMs to write and iterate reward code, with in-context evolutionary/reflection loops. Source: https://arxiv.org/abs/2310.12931
- RL-VLM-F uses vision-language foundation model feedback to generate rewards from task text and visual observations. Source: https://arxiv.org/abs/2402.03681
- Recent LLM reward-design variants add dynamic feedback loops. Source: https://arxiv.org/abs/2410.14660
- Contact-rich insertion literature repeatedly emphasizes force/contact handling, compliance/admittance, and tight geometric constraints; recent peg-in-hole work combines RL with compliance or demonstrations. Examples: https://arxiv.org/abs/2008.10224 and https://huggingface.co/papers/2305.17110

## Proposed Contribution

Agent-assisted synthesis of reward funnels and controller-aware servo guards for final cable insertion:

- Uses privileged semantic geometry during training/evaluation.
- Synthesizes phase-conditioned reward gates over `s`, `r`, `theta`, action-axis alignment, and module consistency.
- Audits reward surfaces before training to reject false positives.
- Uses guarded servo rollouts to generate successful or near-success trajectories and tune reward gates.
- Requires visual and module/body consistency checks, not reward alone.

## Novelty

Relative to Eureka-style reward design, the core novelty is not simply LLM-written reward code. It is the closed loop between semantic insertion geometry, controller realization diagnostics, and strict physical success predicates.

Relative to SERL/HIL-SERL, the proposal targets the narrow final insertion bottleneck with privileged geometric funnels and servo guards before online RL. Human interventions are replaced or augmented by privileged servo trajectories where practical.

Relative to VLM reward methods, visual feedback is not used as the primary reward source yet. Visual sanity is used as a hard validation layer because the repo already showed that semantic tip depth can be misleading.

## Strong Baselines

- ACT 175k TorchScript baseline in Isaac and Gazebo.
- Existing online SERL hand reward.
- Existing `cheatcode_insertion_v1` / `cheatcode_alignment_v1`.
- Guarded guide without auto-tuned reward.
- Reward funnel without guarded action/servo residual.
- Optional offline imitation from privileged servo trajectories, then online SERL.

## Ablations

- No module consistency gate.
- No action-axis gate.
- No scheduled lateral/orientation widths.
- No rotation-induced tip-sweep compensation.
- Hand-tuned vs servo-tuned funnel.
- Reward-only vs reward plus guarded residual.
- Tip-frame orientation vs gripper orientation.
- Isaac-only vs Gazebo transfer validation.

## Risks

- Privileged servo may exploit simulator geometry that does not transfer.
- Synthetic geometry-level success may disappear under Isaac contact.
- Reward gates may become too sparse for online exploration.
- The controller may not realize small semantic tip corrections.
- Visual evidence may remain ambiguous at current camera resolution.
- A paper claim is weak unless Gazebo or real transfer validates strict success.

## Minimal Workshop Result

- Demonstrate that the audited reward funnel prevents known false positives.
- Show one-env Isaac guarded servo strict success with synchronized `s/r/theta/module/force` logs and video.
- Show online SERL with reward funnel plus guard improves strict near-gate success over ACT and existing hand reward.

## Strong Conference Result

- Statistically meaningful strict success improvement across randomized card/port starts.
- Gazebo transfer validation with no tip-depth-only successes.
- Ablations proving module consistency, action-axis gating, and controller-aware rotation/lateral guard each matter.
- Evidence that the learned policy performs insertion rather than delegating entirely to a privileged controller.

## Next Steps

1. Run the Isaac guarded servo smoke command from `docs/agent_reward_funnel_audit_20260523.md`.
2. Parse `audit_log.jsonl` into the same failure categories as the synthetic sweep.
3. If strict Isaac success appears, collect trajectories for auto-tuning and imitation.
4. Launch controlled online SERL comparisons with identical near-gate episodes and strict success metrics.

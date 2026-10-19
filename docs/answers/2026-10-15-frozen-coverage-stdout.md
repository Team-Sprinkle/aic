## Summary

**The earlier evaluation was insufficient. I have now completed 140 frozen-model Gazebo rollouts. The broader tests did not demonstrate reliable full insertion.**

### What we tried and what worked

- **Transport-label repairs:** offline prediction improved, but autonomous handoff remained **1/6 versus 1/6**. No demonstrated live improvement.
- **Rule-based SC transport:** earlier screens achieved **6/6 reused and 6/6 fresh visual acquisitions**. These were restricted acquisition tests—not insertion successes across all settings.
- **Perception repairs:** a matched orientation estimate improved from **1.37° to 0.017°**. Useful component improvement, with limited coverage.
- **BC target/data/history changes:** some offline errors improved, but they did not establish autonomous insertion.
- **Explicit visual approach before insertion:** one earlier pilot inserted **11.07 mm**, below the **14 mm** physical success gate. The larger evaluation did not reproduce reliable insertion.

### Exactly what ran

| Component | SC candidate | SFP tests |
| --- | --- | --- |
| Transport | **Rule-based** Low Lateral Bypass | **Trained diffusion actor** |
| Coarse camera check and crop refinement | Yes | Not connected in this runtime |
| Extra approach toward 30 mm stand-off | Explicit visual/proprioceptive controller | No |
| Insertion | **Existing frozen BC actor**, when readiness passes | Not invoked |
| RL or backtracking | Neither | Neither |

The SC insertion actor was **reused, not retrained**. It uses a ResNet18 image encoder and transformer action head, conditioned on images, robot state, task code and estimated port/plug geometry. It predicts complete targets in the estimated port frame—not residual corrections on another actor.

An important version distinction: this SC prototype uses the **older coarse locator/identity stack**, with internal 2D port candidates. It does **not** use the proposed target-only camera-history coarse model. Only the requested candidate is refined into a 3D opening estimate.

Autonomous commands did not receive simulator ground-truth port poses. Those were recorded separately for posthoc scoring.

### Evaluation coverage

- **50 SC candidate episodes:** 1–5 cards × both ports × five fresh episodes.
- **50 SFP transport episodes:** the same matrix, cycling requested NIC card locations.
- **20 matched SC controls:** without the extra approach controller.
- **20 alternate-board-pose stress episodes:** both families and ports, five per group.

That is **28 groups with five episodes each**. Layouts, rails, grasp and actual robot starts vary. This does not cover every combined card-index/layout/start setting, and it is not the sealed final evaluation.

### Results

| Test | Visual approvals | Insertion actor ran | Full insertion |
| --- | ---: | ---: | ---: |
| SC candidate | 23/50, including **3 wrong ports** | 3/50 | **0/50** |
| SC far-start control | 7/20, including 1 wrong port | 7/20 | **0/20** |
| SC alternate board pose | 0/10 | 0/10 | **0/10** |
| SFP transport | Not connected | Not invoked | **Not evaluated** |

For SC, **17 additional episodes got near the correct opening**, but failed the tighter approach-readiness test, usually because lateral error remained above 0.5 mm. Thus, many failures were handoff/convergence failures, not simply failure to reach the port area.

The three insertion-actor runs reached **1.98, 5.58 and 1.94 mm depth**; none passed the full gate. Only three actor starts also means we cannot independently rate that actor across settings.

When SC identity was correct, opening-position error was promising: **median 0.17 mm, p90 0.23 mm**. However, three approvals precisely localized the **wrong opening**. Accurate refinement does not solve identity.

SFP transport had **zero sampled visits to the broad diagnostic approach region across 60 runs**. Its primary closest-distance median was **71 mm**, p90 **109 mm**. This frozen transport checkpoint is inadequate on this matrix; it is not a full SFP insertion evaluation.

**Conclusion:** neither handoff nor insertion is ready to call reliable. Port identity, approach convergence and SFP transport/integration need targeted repairs. No further training or RL was launched.

[Detailed explanation](docs/answers/2026-10-15-frozen-coverage-explanation.md) · [Per-setting results and video paths](docs/experiments/2026-10-14-frozen-pipeline-coverage.md).

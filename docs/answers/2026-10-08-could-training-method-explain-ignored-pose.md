# Could the training method explain why the actor ignores predicted pose?

Yes. The current result does **not** prove that pose conditioning is unhelpful. It shows that this particular supervised training recipe did not make the actor use it.

## What the code actually does

- The frozen three-camera features and history pass through a two-layer temporal transformer. A zero-initialized linear projection of the three-dimensional predicted port offset is then **added to the final hidden state**, just before the action head. This is a weak, optional route by which pose can affect the action; the transformer has already summarized the scene without it. See `scripts/train_temporal_transport_probe.py`, `TemporalTransport.__init__` and `forward`.
- The matched runs optimize a four-step, full-action smooth-L1 imitation loss. They do not directly penalize failure to correct route error, and the pose-input runs have no active SFP route/subgoal auxiliary loss. An additional weight-1 preservation loss keeps actions near the earlier checkpoint on old SFP observations.
- The sampler assigns half of the total mass to SC, 45% to ordinary SFP, and **5% to all corrective SFP rows** (`--corrective-sfp-mass .1`). The six new late takeovers are only part of that corrective pool. Among their cached teacher-action rows, frozen perception accepted pose on 112/333 training rows. Actor-prefix views, where the port was often visible and the actor was drifting, provide history context but no supervised teacher action at that instant.
- Checkpoint selection uses ordinary SFP plus SC validation and explicitly excludes corrective validation (`--selection-exclude-corrective`). It therefore does not select for late recovery, although the separate held-out late score was reported afterward. Both matched runs stopped at the fixed **2,000-update cap**, and each selected its final update. They did not establish a convergence plateau.

## Why this can produce the observed result

On familiar teacher routes, the task code, camera features and past motion may already predict the next action reasonably well. Ignoring a sometimes-zero pose channel can minimize the average imitation loss. The relatively few accepted late corrective examples supply a weak gradient to the new zero-initialized projection, while the old-policy preservation term resists changing familiar actions. Also, a correct port pose does not by itself specify which collision-free detour to take around cards; the target action still depends on route and contact history. These are plausible mechanisms, **not individually proven causes**.

The direct test is telling: zeroing accepted pose estimates changed the trained actor's first action by only **0.032 mm p95**, and held-out late-action p95 was **2.95322 mm without pose versus 2.95296 mm with it**. This is much less than the declared 10% improvement gate. It measures lack of *use* of pose, not lack of visual pose accuracy.

## What I would test next

1. First audit the exact causal actor-prefix-to-first-teacher-action join. If it works, create a small grouped training set of *actor-visited* states with accepted predicted poses and corrective full-action labels. Keep a separate episode-grouped holdout.
2. Run a cheap capacity/optimization check: train on that small set and compare true pose, zero pose, and pose shuffled across requests. Require the model to fit the training targets **and** change its actions appropriately when the requested pose changes. Inspect projection gradients and action sensitivity by phase. If even training data cannot be fit, fix the architecture or optimization before gathering more data.
3. For the bounded full experiment, supervise an explicit target-relative route-error or short waypoint/progress variable and decode a **complete TCP-body action chunk** from it. Match the action-only control on data, update budget and split. Measure late corrective action error and autonomous safe arrivals under the gates already documented in the [ordered continuation result](2026-10-07-ordered-sfp-transport-continuation-result.md).

Simply extending the same 2,000-update imitation run is a weak next bet: both arms had the same training budget and nearly identical action sensitivity to pose. A changed training signal and a direct pose-use check will tell us more. Automatic handoff, insertion, RL and final scenes remain closed.

Detailed experiment: [SFP route branch record](../experiments/2026-10-07-sfp-early-branch-support.md#route-feedback-ablation-result-the-pose-is-visible-but-this-actor-ignores-it). Matched-run manifest and logs: `/home/nas/chmin/aic_archives/2026-10-07_sfp_port_route_recovery_matched_v1/`.

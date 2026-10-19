> Historical snapshot preserved on October 19, 2026. Dated running/queued statements are not current instructions. See the [current status](../../../STATUS.md) and [single future-work plan](../../../FUTURE_WORK.md).

# Development settings versus post-release qualification settings

## October 19 diagnostic exception

The user explicitly authorized inspecting videos from the already published `eval_config.yaml` and iterating on its failures. The October 19 SFP and SC repairs therefore use that **exposed, post-release configuration for diagnosis and selection**. This overrides the earlier no-tuning instruction below for this work. Its score measures a repaired known configuration; it is **not** an unseen generalization or a historically valid competition submission. Preserve all intermediate failed variants, and keep the separate reserved internal final scenes sealed. Broad reliability still requires independently generated development starts and a newly sealed test after the design is frozen.

## Current rule — October 16, 2026

**Develop transport rules, models and thresholds using the pre-release public sample and task specification. Do not use the subsequently released leaderboard scenes, or derivatives of them, to design the solution.** This supersedes earlier recommendations to work backwards from the released exact qualification configuration.

Existing results are preserved. This protocol does not turn an already inspected test into an unseen holdout.

## What the organizers published, and when

### Public default available during qualification

The launch default is `aic_engine/config/sample_config.yaml`. It contains three example trials and was already public during the competition. The local file exactly matches the version at upstream commit `ced3678cadaa446911a7d14c11b6fd1399509f08` (May 11, before the May 15 submission deadline). GitHub history shows the latest change to the sample itself was February 28, PR [#387](https://github.com/intrinsic-dev/aic/pull/387), which moved the sample SC target into view.

Pinned sources:

- [Pre-deadline sample configuration](https://github.com/intrinsic-dev/aic/blob/ced3678cadaa446911a7d14c11b6fd1399509f08/aic_engine/config/sample_config.yaml).
- [Pre-deadline qualification task description](https://github.com/intrinsic-dev/aic/blob/ced3678cadaa446911a7d14c11b6fd1399509f08/docs/qualification_phase.md).
- Local immutable reference: `configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml`.
- Source URLs, dates and SHA-256 hashes: the adjacent `provenance.json`.
- Sample SHA-256: `21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d`.

| Default trial | Manipulation task | NIC obstacles | Requested target |
| --- | --- | --- | --- |
| 1 | SFP to NIC | One card, rail 0 | Card 0 / SFP port 0 |
| 2 | SFP to NIC | One card, rail 1 | Card 1 / SFP port 0 |
| 3 | SC to SC, reversed cable | No NIC cards; SC port 1 present | SC port 1 |

The sample uses concrete scene values. Repeating it does not by itself explore all layouts described in the task specification.

### Exact qualification leaderboard configuration released afterwards

PR [#558](https://github.com/intrinsic-dev/aic/pull/558) added `aic_engine/config/eval_config.yaml`, merged **May 28, 2026, 00:29 UTC**. The qualification documentation explicitly says this file was released following the qualification phase and contains the exact cloud leaderboard settings. The [May 28 results announcement](https://discourse.openrobotics.org/t/aic-qualification-phase-results-advancing-teams/55138) explains that it was released so teams could analyze and reproduce their results after submissions closed.

This is the **qualification** evaluation. The organizers separately call the later June stage “Phase 1”; calling both “round 1” can cause confusion. No claim is made here about an unreleased later-stage evaluation configuration.

The post-release file is not the default sample and did not replace the launch default. An explicit `aic_engine_config_file` override selects it.

## What information is allowed for development?

The pre-deadline specification already disclosed:

- Both SFP-to-NIC and SC-to-SC tasks, selected target identity in the public task message, five NIC rails, and the available component assets.
- Variation in board position/yaw, rail translations, component arrangements and SFP card orientation.
- Fixed configured robot start rather than an unspecified random robot-home distribution.
- Approximately 2 mm / 0.04 rad grasp deviations and the opposite cable end being free.
- The requested target being in view at the intended initial trial setup.
- Numeric rail limits in the public sample.

Therefore, we may generate diverse SC/SFP development scenes from those public definitions, including multiple NIC obstacles around SC tasks. We must choose and record our own development distributions and seeds. Where the old specification gives no numeric board-pose distribution, label our choices as our own stress distribution—not the organizers' official distribution. Enforce feasibility and the published initial-view assumption when claiming a variant represents the described task.

We must not copy exact post-release board poses, grasps, layouts or target combinations into development generators, or optimize waypoint offsets/crop gates after examining their failures. Sharing a rail limit or robot-home value with the later file is not itself leakage when the same value was already public; provenance matters.

## Treatment of the work already performed

The October 16 batch `/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1` uses the post-release exact scenes. Its 100 stress cases also inherit post-release board/grasp/task templates. **The entire batch is excluded from development**, including training, route design, threshold selection and checkpoint selection. `development_use_restriction.json` records these restrictions without changing the frozen plan, models, source hashes or result files.

The existing frozen job may finish as descriptive evaluation, without interruption or tuning. Do not start follow-up collection/repair based on that batch. Its results are exposed retrospective evaluation, not untouched final-test evidence. The separately reserved internal final split remains sealed; it is different from the now-exposed public qualification scenes.

Earlier notes and October 12–16 experiments also used “official-anchor,” “qualification-family” and similar development templates. Audit manifests and ancestors before asserting that any existing corpus, prior or checkpoint meets this stricter protocol. Do not assume all old experts are contaminated: some were collected from the public sample, and public rail/home values were available before the final release. Conversely, passing a train/validation reset-overlap check does not prove independence from the post-release configuration.

No retraining or deletion is authorized merely by this classification. First identify concrete lineage contamination, preserve existing results, and construct a clean manifest if needed.

## Next development sequence

1. Audit existing route templates, perception priors, scene generators and training manifests against the pinned pre-deadline sources. Mark post-release-derived and unresolved lineages separately.
2. Reproduce the **three default sample trials** with frozen current components. These are development diagnostics, not clean historical results for a checkpoint whose lineage is unresolved.
3. Define a balanced development matrix using only the old sample/specification: SC and SFP, both target ports, requested NIC identity, rail ranges, present/absent alternatives, card obstacles and independently chosen board/grasp variation. Include the default SC **zero-NIC-card** case; the earlier 1–5-card sweep omitted it.
4. Split complete reset configurations into training and development validation/test groups before collection or tuning. Choose transport and handoff criteria in advance. Preserve all failures and do not use post-release results for promotion.
5. Improve routes/perception on these development groups. Test generalization on independently generated, sealed groups after freezing the method. The previously inspected public leaderboard scenes cannot become unseen again.

This changes which settings guide development; it does not change the observation-only deployment requirement, the shared full-approach insertion actor goal, or the hierarchical reacquisition design.

# Default development settings versus the later official evaluation settings

**You are right. These are two different configurations, and I should not use the later-released leaderboard scenes to design the transport rule.** My earlier suggestion to work backwards from those exact scenes needs correcting.

## What was available before submissions closed?

The default is **`aic_engine/config/sample_config.yaml`**. The launch file still selects it unless explicitly overridden. I checked the pre-deadline repository version, and our local sample is byte-for-byte identical.

Its three examples are:

| Default trial | Task | NIC cards | Target |
| --- | --- | --- | --- |
| 1 | SFP to NIC | One, rail 0 | Card 0 / port 0 |
| 2 | SFP to NIC | One, rail 1 | Card 1 / port 0 |
| 3 | SC to SC | None | SC port 1 |

[Pinned pre-deadline sample](https://github.com/intrinsic-dev/aic/blob/ced3678cadaa446911a7d14c11b6fd1399509f08/aic_engine/config/sample_config.yaml).

The public task description already said that board pose, card layout, rail positions and requested ports could vary. Therefore, we may make our own diverse development scenes using that information. We should not limit ourselves to memorizing these three examples. [Pre-deadline specification](https://github.com/intrinsic-dev/aic/blob/ced3678cadaa446911a7d14c11b6fd1399509f08/docs/qualification_phase.md).

## What was released afterwards?

**`aic_engine/config/eval_config.yaml`** contains the exact qualification leaderboard settings. Submissions closed May 15; this file was merged May 28 in [PR #558](https://github.com/intrinsic-dev/aic/pull/558). The organizers explicitly announced its release so participants could reproduce and analyze their results afterwards. [Results announcement](https://discourse.openrobotics.org/t/aic-qualification-phase-results-advancing-teams/55138).

This was the **qualification round**; the organizers use “Phase 1” for a subsequent stage, so those names should not be mixed.

## What changes in our work?

1. **Development:** use the old sample and pre-release specification/assets/ranges, with independently chosen variations and grouped splits.
2. **Exclude the latest official-setting sweep from tuning:** both its exact scenes and its derived stress scenes inherit post-release settings. It may finish unchanged as retrospective evaluation, but must not supply training data, waypoint constants or threshold choices.
3. **Audit existing data and checkpoints:** some earlier official-anchor experiments also used post-release settings. We cannot claim clean separation until their lineage is checked. This does not mean every old expert episode is invalid.

Because we already inspected the released scenes, they cannot now count as an untouched final test. The separately reserved internal final split remains sealed.

I pinned the historical sample and source hashes under `configs/evaluation_protocol/qualification_public_predeadline/`, documented the boundary in [EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md), and updated the main status/planning documents. No model or route was changed and no training was started.

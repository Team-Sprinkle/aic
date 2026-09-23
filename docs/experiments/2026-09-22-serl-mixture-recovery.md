# Probabilistic RPDP SERL and measured-path recovery

**Date:** 2026-09-22
**Status:** Complete; safe warm-start retention achieved, no RL improvement, no promotion
**Reserved final split:** Sealed

## Question

Can the selected deterministic port-trajectory BC policy become a natural SAC/RLPD actor that explores coherent recovery strategies, and does deterministic measured-path backtracking improve those recoveries?

The experiment deliberately keeps predictive world models, SEER, Gazebo adaptation, reward-model training, and imagined rollouts parked.

## Design

The actor predicts a distribution over the complete four-waypoint connector trajectory in the port-opening frame. Each waypoint has 3D translation and a rotation vector, for 24 stochastic coordinates. A deterministic calibrated SE(3) adapter converts the selected trajectory into the same four TCP-body-frame commands used by the successful supervised controller.

This is a full trajectory policy. It does not predict a correction to a frozen BC action. See [the architecture explanation](../SERL_PROBABILISTIC_ACTOR.md).

The distribution is a four-component mixture of low-rank Gaussians. Component 0 reproduces the BC mean exactly. For exploration, it starts at 70% probability while three coherent lateral alternatives receive 10% each. A chosen component is held for three 200 ms decisions so that exploration remains correlated for 600 ms.

Two critics consume the causal observation and actual 24D executed command chunk. SAC optimizes the expected mixture objective. Perception remains frozen. A separate immutable prior replay can provide 50% of each batch, matching the central RLPD replay idea.

## Measured-path recovery

A state machine watches measured TCP motion and force. Persistent force or commanded motion without measured progress triggers a backtrack. It retraces stored measured TCP positions until at least the configured clearance and a lower force are observed. It then gives control back to the policy while projecting a short exploration window near the plane perpendicular to the blocked direction.

Replay stores an `actor_owned` flag for every macro decision. Backtracking and abort chunks train the critic through their outcomes but are excluded from actor gradients. A deliberately low-threshold Isaac test produced four controller-owned chunks, a critic update, and exactly zero actor gradient, confirming the mask.

## Supervised initialization

Command:

```bash
aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train_rpdp_serl_actor.py \
  --dataset outputs/experiments/2026-09-22_rpdp_dppo/bc_repair_dagger_precontact/dataset/rpdp_dataset.pt \
  --bc-checkpoint outputs/experiments/2026-09-22_rpdp_dppo/bc_repair_direct/training/aic_rpdp/checkpoint.pt \
  --output-dir outputs/experiments/2026-09-22_serl_recovery/actor_bc \
  --updates 8000 --patience 1600 --batch-size 128 --device cuda:0
```

Training stopped through validation patience at update 2,500; update 900 was selected. Episode-grouped development adapter error was 0.022 mm median and 0.339 mm p95 for translation, and 0.0021 degrees median for rotation. The probabilistic actor has 2,049,564 trainable parameters. Frozen perception adds 536,631 parameters, for 2,586,195 total.

A complete Isaac smoke test performed one critic and actor update successfully.
Across 298 prefetched decisions in matched collection, complete perception plus
policy inference was 20.43/22.67/23.78 ms p50/p95/p99, with a 26.18 ms maximum.
This is below the 300 ms requirement.

## Runtime checks

- Eight focused unit tests pass for mixture probability/gradients, exact BC-primary preservation, lateral mode separation, blocked detection, measured path reversal, lateral projection, and reset isolation.
- The rootless Isaac runtime loaded frozen perception plus the mixture actor and completed a full SAC update.
- A forced recovery test retraced 2.34 mm before its deliberately small 2 mm safety limit and entered abort as configured.
- All forced-recovery replay chunks were `actor_owned=false`; actor gradient norm was 0 while the critic still updated.

## Development collection

Twenty new, nonfinal SFP-to-NIC near-port development starts were generated across eight observed cable templates. The run is bounded to 1,200 simulator steps, so the number of complete episodes can be smaller than 20 when failures time out.

The first matched branch uses correlated mixture exploration without backtracking. Its replay contains 301 transitions from seven complete episodes plus one incomplete episode. Terminal geometry gives **3/7 local 8 mm seating successes and 4/7 failures**. The incomplete episode is excluded from outcome-relabeled replay. The four failures supply natural lateral drift and timeout behavior that is absent from CheatCode demonstrations.

The first backtracking branch produced **2/7** local successes versus **3/7**
for correlated exploration. Inspection found an implementation problem rather
than evidence against retreat: force alone triggered on valid insertion and a
safety abort then held zero commands until timeout. This run is retained as a
failed comparison.

After requiring simultaneous force, a commanded move, and measured TCP stall,
and changing abort into a four-decision safety hold, the revised five-episode
branch produced **1/5** successes versus **2/5** in the corresponding correlated
prefix. No revised episode triggered backtracking. Those failures were lateral
drift/timeouts and did not meet the blockage definition. The result says that
measured-path recovery did not help this sample; it does not test recovery from
a cable snag because no snag occurred.

## Prior replay and critic warm-up

Outcome relabeling retained the actual terminal observation before reset and
formed 12 complete groups: seven success and five failure episodes, 387 causal
macro transitions. The train/validation split is by complete episode. A bounded
critic-only warm-up used ten groups and kept one success and one failure held
out. Mean Q was 1.792 on the held-out success and 1.011 on the held-out failure.
The ordering passes the smoke check, but the two-episode validation is too small
to claim a robust value function.

## Online update

The correlated and backtracking arms each use the same critic checkpoint,
50/50 prior-to-online batches, 250 updates, actor learning rate $2\times10^{-5}$,
and BC-anchor weight 0.1. The actor begins updating after 128 simulator steps.
The correlated run finished after 190 steps and one completed training episode.
At update 250, actor and critic losses were finite, actor gradient norm was
0.0484, BC anchor loss was 0.0100, mixture entropy was 0.895, and the largest
component probability was 0.507. These numbers establish a functioning update,
not policy improvement. The matched held-out run provides the outcome test.

The first deterministic held-out comparison was **3/8 for frozen BC, 0/8 for
the correlated SAC actor, and 0/8 for the backtracking SAC actor**. The first
online objective preserved only the primary component's mean. It did not
preserve component probabilities. On all 327 fixed prior-audit transitions,
the correlated actor changed from selecting primary component 0 to component 3,
and the backtracking actor changed to component 2. This is policy collapse onto
unvalidated exploration modes, not evidence that the alternatives improved
control.

The repaired objective adds forward categorical KL to the frozen mixture and
anchors every component mean in proportion to its frozen probability. With
anchor weight 1.0, probabilities remained at approximately
0.700/0.100/0.100/0.100 and component 0 stayed the argmax on 327/327 rows.
Nevertheless, a 0.017 mean normalized shift in its primary trajectory, with a
0.388 worst coordinate, was enough to score **0/8**. A final tighter trust
region rerun reduces the actor learning rate tenfold and raises behavior
regularization tenfold. A second silent issue then became visible: restoring
the warm-up optimizer also restored its `1e-4` learning rate and overrode the
requested `2e-6`. Restore now reapplies the explicit actor and critic rates
after loading momentum and other optimizer state.

The corrected tight-trust run kept component 0 as argmax on 327/327 audit rows,
kept mean probabilities at 0.6999/0.1000/0.1000/0.1000, and reduced primary
normalized mean drift to 0.00098 average and 0.0234 maximum. It then scored
**3/8 on exactly the same successful episodes as frozen BC**. The bounded RL
path therefore preserves the warm start but supplies no measured improvement.
It is retained as the selected mechanical checkpoint and is not promoted as a
better policy.

### Matched final table

| Actor | Recovery controller | Successes | Decision |
| --- | --- | ---: | --- |
| Frozen BC mixture, deterministic mode | Off | 3/8 | Baseline |
| Initial SAC, correlated | Off | 0/8 | Reject: categorical mode collapse |
| Initial SAC, backtracking branch | On | 0/8 | Reject: categorical mode collapse; confounded recovery comparison |
| Full-mixture KL, requested LR silently overridden | Off | 0/8 | Reject: primary trajectory drift |
| Tight trust region, LR restore fixed | Off | 3/8 | Retains baseline; no improvement, no promotion |

All arms used the same episode IDs `serl_dev20_09` through
`serl_dev20_16`. Frozen BC and the final tight-trust actor succeeded on 09, 11,
and 14. Autonomous evaluation disabled guide, insertion guard, stochastic
sampling, policy updates, and privileged geometry input.

An attempt to run three evaluations concurrently failed before scene loading
because the established rootless container exposes only its assigned GPU.
Evaluation therefore runs sequentially on its GPU 0. This failure changed no
checkpoint or scene.

## Matched video reruns

On September 23, all three selected comparison arms were rerun on the same
eight development configurations with three-camera recording enabled:

| Recorded arm | Checkpoint | Backtracking | Video-rerun result |
| --- | --- | --- | ---: |
| Frozen BC | supervised mixture | Off | 1/8 (`serl_dev20_09`) |
| Tight-trust RL | selected trust-region actor | Off | 1/8 (`serl_dev20_09`) |
| Tight-trust RL | selected trust-region actor | On | 3/8 (`serl_dev20_09`, `11`, `14`) |

Guide input, insertion guard, stochastic action sampling, policy updates, and
privileged geometry policy input were disabled. Frames were retained every four
simulator steps and encoded at 5 fps. This produces 72 per-episode H.264 clips:
eight configurations, three arms, and three cameras. All clips passed an
`ffprobe` decode/metadata check and representative final frames were inspected.
Open the [local review page](../../outputs/experiments/2026-09-22_serl_recovery/videos/index.html)
or the [committed per-episode failure analysis](2026-09-23-serl-video-failure-analysis.md).

These new recording-enabled outcomes do not replace the frozen metrics-only
comparison above. BC and tight-trust RL each scored 3/8 there but 1/8 in the
recorded rerun. Simulator/contact nondeterminism and recording overhead are
plausible causes, but this run does not isolate them. The 3/8 backtracking result
is therefore useful visual evidence and not a promotion claim.

## Limits

These are local Isaac SFP-to-NIC starts near the opening with an 8 mm seating target. They are neither normal full-episode evaluations nor official Gazebo full-depth insertions. The four reserved final configurations remain unopened.

Only 12 complete episodes were available for prior critic learning, and critic
selection used two held-out episodes. The online update observed one completed
training episode before reaching its update budget. That evidence is too small
to support useful actor improvement. Another run requires more recoverable
blocked/contact outcomes and a broader episode-grouped critic check; repeating
gradient updates on this replay is not recommended.

## Artifacts

- Machine summary: `outputs/experiments/2026-09-22_serl_recovery/summary.json`
- Artifact map: `outputs/experiments/2026-09-22_serl_recovery/artifact_map.json`
- Exact launch scripts and commands:
  `outputs/experiments/2026-09-22_serl_recovery/commands.md`
- Selected safe-continuation checkpoint and hash:
  `outputs/experiments/2026-09-22_serl_recovery/online_selected/`
- Five compact evaluation summaries and compressed raw metrics:
  `outputs/experiments/2026-09-22_serl_recovery/evaluation/`
- Actor-shift reports: `policy_shift.json`, `policy_shift_kl1.json`, and
  `policy_shift_trust10_lrfix.json` in the experiment root.
- Matched three-camera review page and per-episode clips:
  `outputs/experiments/2026-09-22_serl_recovery/videos/index.html`
- Durable per-episode geometry and failure interpretation:
  `docs/experiments/2026-09-23-serl-video-failure-analysis.md`
- Bulk replay and failed checkpoints remain at the container paths listed in
  `artifact_map.json` because `/data1` had less than 1 GB free.

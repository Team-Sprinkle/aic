# Why transport looked better, and the official-setting recheck

## 1. Why was transport successful this time?

The recent tests used **rule-based transport plus trained visual localization**, rather than asking the diffusion transport actor to learn the whole route:

- SC follows the Low Lateral Bypass route around the cards.
- SFP follows staged CheatCode-style motion.
- Measured robot movement determines when a stage finishes. Camera estimates can adjust the final acquisition position.
- The coarse model identifies the requested port region; the original-image refiner and multi-camera checks decide whether its pose is reliable enough for handoff.

This removes learned route drift and incorporates real perception repairs. However, several things changed together, so I cannot attribute the gain to one model alone. **The recent strong results were also mostly on a familiar board placement. They were transport/handoff successes, not insertion successes.**

## 2. Was that evaluation representative enough?

**No.** The published official settings move and rotate the board substantially. The earlier alternate-board test already failed 0/10. I re-downloaded the [official qualification configuration](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/eval_config.yaml); its checksum is unchanged.

I started a frozen **120-rollout evaluation**:

1. Five repeats of each of the three official scenes: **15 runs**.
2. Five matched official SC runs with the previous refiner: **five controls**.
3. SC and SFP × 1–5 cards × both ports × five repeats: **100 additional runs**, with varied card subsets and rail positions. These are clearly labeled derivatives of official settings.

The exact public scenes are evaluated separately in fresh simulator launches, preserving their scene, grasp, robot home and task settings. This is not an unchanged sequential cloud submission. The insertion actor and backtracking are disabled so that transport/handoff performance is measured separately. Reserved final scenes remain sealed.

### First results — only one repetition per official scene so far

| Official scene | Handoff | What happened |
| --- | --- | --- |
| SFP, one card, card 2 / port 0 | Passed | Correct visual acquisition and strict gate pass. |
| SFP, two cards, card 4 / port 1 | Failed | Reached about **177 mm before the opening, 2.3 mm sideways**, but the multi-camera pose fit was rejected. This was a verification failure despite useful geometric arrival. |
| SC, three cards, SC port 1 | Failed | Did not reach the correct acquisition corridor; ended about **204 mm sideways** from the target. |

No scored harmful contact occurred in these three runs. **These initial results already show that familiar-board success does not generalize automatically.** Remaining repeats are running without tuning. Estimated batch budget is 6–12 hours, using GPU3 for rendering and GPU4 for perception.

## 3. Was the refiner trained further? What else was trained?

**Yes, during this continuation; no new training is running now.**

| Component | What was trained |
| --- | --- |
| Native-image refiner | Substantial SC+SFP training on **72,857 three-camera samples from 497 source episodes**, followed by SFP acquisition and SC rail-coverage repairs. These fits reached validation plateaus. |
| SC and SFP coarse locators | Broader requested-port crop/in-frame training; current-image versions were selected because the tested history variants were not better. |
| SC identity/association checker | Additional wrong-port and absent-port examples. |
| SFP plug-geometry helper | Image-based held-plug geometry, with a limited free-approach translation approval; full orientation prediction is not approved. |
| Shared insertion actor | Two matched **30,000-update BC runs**, using **500 usable SC+SFP episodes**: ImageNet initialization versus reused transport visual weights with fresh action heads. Completed-model insertion remains untested. |

No new learned transport actor, diffusion model, RL actor or world model was trained in this round.

I also found a pass/fail bookkeeping error in the previous 20-run screen: a safely rejected episode was incorrectly counted against the no-wrong-accept condition. Its 9/10 SC and 9/10 SFP meet the declared numerical screen, but the actual failures remain failures. The correction is recorded separately; it does not resolve the official-setting coverage problem.

[Full protocol, training inventory, commands and artifact paths](../experiments/2026-10-16-official-transport-recheck.md).

Live results: `/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1/progress.md`.

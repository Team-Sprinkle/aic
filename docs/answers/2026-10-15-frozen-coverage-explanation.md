# What we tried and what the broader evaluation showed

**The earlier evaluation was too small. I have now completed 140 Gazebo rollouts with frozen models. The broader results do not show reliable full insertion.**

## 1. What did we try, and what worked?

| Change | What the evidence actually showed |
| --- | --- |
| Repair diffusion transport labels | Offline action error improved, but autonomous handoff stayed **1/6 versus 1/6**. No demonstrated live improvement. |
| Replace SC learned transport with rule-based Low Lateral Bypass | Earlier small screens reached visual acquisition in **6/6 reused and 6/6 fresh starts**. These tested acquisition, not insertion, and did not cover all settings. |
| Repair board-orientation estimation | One matched case improved **1.37° → 0.017°**. Useful targeted perception repair, not proof of overall reliability. |
| Improve BC action targets and add corrective examples | Offline errors improved; the autonomous actor still stalled far from the opening. Decoder and robot-state-history variants did not pass their gates. |
| Add an explicit visual approach before the existing insertion actor | One earlier pilot reached approximately 30 mm before the opening, then inserted **11.07 mm**. It still failed the 14 mm physical-depth gate. The broader evaluation below did not reproduce reliable insertion. |

So: some individual components improved, but **we have not demonstrated a reliable complete system**.

## 2. Did we evaluate five episodes per setting?

**Yes, for the declared groups:**

- **50 SC candidate runs:** 1–5 cards × two requested ports × five fresh episodes.
- **50 SFP transport runs:** the same card-count/port matrix.
- **20 matched SC controls:** one/five cards × both ports × five episodes, omitting the added near-port approach.
- **20 board-pose stress runs:** three cards × both ports × both task families × five episodes.

Total: **140 rollouts on 120 distinct development configurations**. Starts, grasp, rails and card layouts vary. SFP requested card index cycles through all five locations, but this is **not five episodes for every combination of card index, card count and port**. It is also not exhaustive production coverage. Final scenes stayed sealed; five episodes per group are a useful screen, not proof of high reliability.

## 3. What actually ran?

### SC-to-SC candidate

```text
Rule-based transport around the cards
                  |
                  v
Trained coarse port locator + identity check + high-resolution refinement
                  |
                  v
Explicit visual/proprioceptive approach toward 30 mm stand-off
                  |
                  v
EXISTING frozen BC insertion actor
```

- **Transport was rule-based**, not the learned diffusion actor.
- **Coarse perception and the refiner were used.** This frozen prototype uses the older coarse model with 12 internal 2D candidate slots; the requested SC code selects one. It is not the proposed target-only camera-history model, and it does not estimate 12 complete 3D poses.
- The identity check uses current images, robot state and requested port. Board-orientation memory is different from full image history.
- The **existing trained insertion actor was reused**, not retrained. It uses ResNet18 and a transformer action head. Images, robot state, task code and estimated port/plug geometry condition complete targets in the estimated port frame; four 50 ms targets are predicted and executed.
- **No RL or backtracking was enabled.** True simulator geometry was used only afterward for measurement, not to command the autonomous policy. Extra overview cameras were recording views only.

### SFP-to-NIC

This used the **frozen learned diffusion transport actor**, with task family, requested NIC card, requested port, robot state and image history. It predicts twelve 100 ms targets and executes six. **The selected SFP runtime did not invoke coarse approval or an insertion actor.** Its full insertion performance is therefore untested, not “0%.”

## 4. What were the results?

| Test | Visual approvals | Insertion actor started | Full insertion |
| --- | ---: | ---: | ---: |
| SC candidate, 50 primary runs | 23/50, including **three wrong ports** | 3/50 | **0/50** |
| SC far-start control | 7/20, including one wrong port | 7/20 | **0/20** |
| SC alternate board pose | 0/10 | 0/10 | **0/10** |
| SFP transport, 50 primary + 10 stress runs | Not connected | Not connected | Not evaluated |

The **20 matched configurations produced zero full insertions with or without the added approach**.

For SC, 27 candidate runs failed before visual approval. Another 20 stopped in the approach controller. **Seventeen of those 20 nevertheless got near the correct opening:** within ±5 mm of the intended 30 mm stand-off and within 10 mm laterally. They failed the tighter readiness test, usually because lateral error remained above 0.5 mm. This is different from never reaching the port area.

Only three entered the insertion actor. Their depths were **1.98, 5.58 and 1.94 mm**, with lateral errors **2.50, 0.27 and 0.99 mm**, and orientation errors **1.32°, 1.23° and 1.91°**. None passed the frozen depth ≥14 mm / lateral ≤0.5 mm gate. Three conditional actor starts are too few to judge that actor independently across settings.

**The camera check also needs repair:** three SC candidate approvals selected port 0 when port 1 was requested. Their estimates were within 0.32 mm of the wrong opening. Conversely, on the 20 correctly identified approvals, opening-position error was median **0.17 mm**, p90 **0.23 mm**. Precise localization does not guarantee correct identity.

SFP transport entered the declared broad diagnostic near-port region in **0/50 primary and 0/10 stress runs**. Its primary closest sampled distance to the requested opening was median **71 mm**, p90 **109 mm**. This frozen transport checkpoint is not adequate on this matrix. It does not establish that every historical SFP checkpoint is equally poor.

## 5. What does this mean?

We should not call the current handoff or insertion reliable. The immediate gaps are **correct port identity, approach convergence/readiness, and SFP transport plus handoff integration**. The older small-screen successes remain useful diagnostics, but do not outweigh these broader failures. I stopped after this evaluation; no further retraining or RL was launched.

[Complete per-setting results, commands, measurement repairs and video paths](../experiments/2026-10-14-frozen-pipeline-coverage.md).

Artifacts: `/home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1/`, especially `coverage_cells.csv`, `summary.json`, `decision.json` and `runs/`.

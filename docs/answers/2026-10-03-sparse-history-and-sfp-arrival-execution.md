# Sparse approach history and handoff: execution update

I worked through the next safe part of the eight-step plan. **The handoff gate has not passed, so the automatic switch to PoseInsert remains off.** No RL, new BC training, or final-scene evaluation was run.

## What changed and what worked

I added a causal, request-specific memory of the SC opening seen during approach. At roughly every 20 mm of measured travel, it can retain a clear three-camera observation and measured TCP pose. At a later stop, it compares a fresh opening found from original-resolution RGB crops with the earlier observation-derived opening. It does not use simulator port coordinates at runtime.

On seven saved one-port development trajectories, the original single-frame identity check accepted 235 of 367 target-projected three-camera samples. The original check **or** the history match accepted 363 of 367. This is correlated, partly projection-based evidence, not 363 independent visible-port successes. An absent alternate-port request produced no anchor in 444 samples. Four saved two-port scenes had the two observed openings 68–83 mm apart, so the 5 mm track rule kept them distinct in those scenes.

I also ran three live rootless Gazebo shadow tests on the same SC development reset. The first inadvertently omitted the new flag inside Docker and served as a frozen baseline. One history run missed the stop because its transport route had already diverged before history started. A bounded repeat reached the opening; after the existing posture correction, the old identity check rejected the real port, while the **shadow** history check would have accepted two fresh inspections. It never changed the actual robot action, so there was **no autonomous insertion**. Policy plus periodic history processing measured 196.25 ms p95; complete ROS-to-first-PoseInsert-command latency remains unmeasured. The SC experiment is documented in [the sparse-history record](../experiments/2026-10-03-sc-sparse-history-handoff.md).

## SFP diagnosis and failed gate

The temporal transport cache had no SFP arrival label; the prior SC-only stop head was inappropriate for SFP. I trained one bounded SFP-specific diagnostic on the existing frozen three-camera image features. A positive label meant the teacher TCP was 70–100 mm before the selected port opening and within 10 mm laterally. The label came from true geometry **for training and scoring only**. The model input was image features, measured robot state, and public task code. It was only a possible *pause proposal*, not a selected-port verifier.

On 32 complete validation episodes, the image-only probe proposed a stop in the labeled region for **28**, proposed **2 early**, and **missed 2**. State-only was worse (10 correct, 22 missed), and combining RGB plus state did not fix it (23 correct, 2 early, 7 missed). A two-of-three history rule increased early proposals; three-frame averaging left the two early/two missed result. One early proposal had the TCP **17.8 mm** laterally from the port, outside the 10 mm label region. These numbers are exploratory because the checkpoint was selected on that validation split.

The dataset audit exposed a stronger limit: **258 of the 268 SFP episodes request card 0, port 1**. Only ten episodes cover other target combinations. Train/validation scene hashes do not overlap, but the split cannot test broad target generalization. An arrival classifier trained on this set is not a reliable switch for the other cards and ports. The existing SFP visual pose system also has known wrong-port locks, so we need a selected-port identity check in addition to a stop proposal. Full details, commands, checkpoints, and episode metrics are in [the SFP diagnostic](../experiments/2026-10-03-sfp-arrival-probe-and-coverage.md).

## Where the eight steps stand

1. Existing SC traces, resets, and models were audited. **Done for the bounded SC diagnostic.**
2. Sparse measured-motion RGB history was implemented and run offline and live in shadow. **Done as a diagnostic, not promoted.**
3. Request-specific SC identity tracking was implemented with frozen heads and original-RGB geometry. **Promising but incomplete on independent hard occlusions and coherent wrong-port histories.**
4. Automatic handoff validation **has not passed**. We still need reviewed clear-to-occluded and wrong-port complete resets, SFP identity/arrival coverage, and full attached-actor latency.
5. Existing PoseInsert relabeling was checked: it converts recorded complete teacher TCP targets into the estimated port frame, with true geometry only on the supervision side. **New offset-data training has not begun because the handoff gate is open.**
6. The new actor's 0°/3°/6–7°/10° correction envelope **has not been measured**.
7. Full-start transport-to-PoseInsert autonomous runs **have not begun**.
8. Recovery/RL and reserved final scenes **remain gated**.

**Correction during the October 3 continuation:** The older 180-episode agent/VLM collection *does* have exact task identity in numbered trial YAML files, even though its selected replay folders lack individual engine configs and its LeRobot task text is generic. All 180 selected replays map to a card/port pair and have score ≥90. These images and measured states can support a new task-conditioned arrival probe; their invalid zero-action arrays must not be treated as BC targets. The ongoing bounded experiment uses these recordings to test the stop proposal on independent complete scenes. The stop condition remains zero accepted known wrong/absent ports, at least 95% recall on reviewed clear requested-port arrivals, and complete p95 inference below 300 ms. In parallel, SC history needs new clear-to-occluded and coherent wrong-port challenge resets. Only after those pass should we attach and evaluate a retrained PoseInsert actor.

The current saved artifacts are under [SC sparse-history outputs](../../outputs/experiments/2026-10-03_sparse_history/) and [SFP probe outputs](../../outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/). The final scenes remain sealed.

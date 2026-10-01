# SC-to-SC collection status — 2026-09-30 20:20 UTC

The 200-setting Low Lateral Bypass teacher collection is **still running**. The initial pilot passed with **17 full insertions in 30 attempts**. The varied-start continuation has committed **26/170 attempts with 17 full insertions**, for **56/200 attempts and 34 full insertions overall**. Its extra 10-of-30 gate already has enough successes numerically (17 in the first 26), but the collector will formally evaluate it after all 30 continuation pilot attempts. `llb_0056` is currently running and is not included in those counts.

There are **54 valid scored replay bags** among the 56 committed attempts. One pilot case failed model startup and one continuation case failed MoveIt planning. All **25 scored continuation starts** audited so far passed the robot-start contract; there are no invalid-start or archive errors. Runs continue sequentially on one GPU, with continuation artifacts archived to NAS. No BC training or final-scene evaluation has started.

Live counts and per-case outcomes: `outputs/experiments/2026-09-30_sc_llb_collection/summary.json` and `continuation170/collection_progress.json`. The detached finalizer will write `collection_final_report.json` and a score-verified success manifest when collection stops. [Experiment record](../experiments/2026-09-30-sc-llb-collection.md).

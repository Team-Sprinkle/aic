# SC-to-SC collection status — 2026-09-30 20:33 UTC

The Low Lateral Bypass teacher collection remains active. It has committed **59/200 attempted settings** and **35 official full insertions**. The initial 30-attempt pilot passed with **17 full insertions**. The varied-start continuation has committed **29/170 attempts** with **18 full insertions**; case `llb_0059` is still executing and is not in the committed count. Its additional 10-of-30 gate is numerically assured but will be recorded only after case `llb_0059` completes.

There are **57 valid scored replay bags** across the 59 committed attempts. Every scored continuation start audited so far passed (**28/28**), with no archive errors. One pilot attempt failed model startup, and one continuation setting failed MoveIt planning. Collection uses one GPU and archives scored continuation runs to NAS. No BC training or reserved final-scene evaluation has started.

Current per-case scores: `outputs/experiments/2026-09-30_sc_llb_collection/summary.json`. Collector progress: `outputs/experiments/2026-09-30_sc_llb_collection/continuation170/collection_progress.json`. [Experiment record](../experiments/2026-09-30-sc-llb-collection.md).

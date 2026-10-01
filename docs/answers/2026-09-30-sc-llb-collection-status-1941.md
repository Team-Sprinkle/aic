# SC-to-SC collection status — 2026-09-30 19:41 UTC

The 200-setting Low Lateral Bypass teacher collection is **still running**. The first 30-setting gate passed with **17 official full insertions** (required: 10). The continuation has completed **14/170** settings with **9 full insertions**, making **44/200 attempted settings and 26 full insertions total**. The next run, `llb_0044`, is being archived and is not included in those committed counts.

Of the 44 committed attempts, **42 have valid scored replay bags**. One pilot run failed model startup and one continuation setting (`llb_0035`) failed MoveIt planning; neither is an expert trajectory. Every scored continuation run audited so far has a valid varied robot start (**13/13**); there are no start-contract or archive errors. The active collector is using one GPU and the scored continuation runs are being archived to NAS. The extra 10-of-30 gate for the varied-start continuation is still pending.

No BC training or final-scene evaluation has started. Exact live counts and per-case outcomes are in `outputs/experiments/2026-09-30_sc_llb_collection/summary.json` and `continuation170/collection_progress.json`. The collection record is [here](../experiments/2026-09-30-sc-llb-collection.md). The detached finalizer will create a score-verified success manifest and final report when the collector stops.

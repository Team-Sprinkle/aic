# SC Low Lateral Bypass collection status — 2026-10-01 04:30 UTC

The requested collection **completed all 200 attempted settings**. The official scorer recorded **104 full insertions (52%)**. There are **198 valid scored replay bags**; the other two attempts remain counted as attempts, not successes. The first 30-case gate passed with 17 full insertions, and the 170-case continuation produced 87. All 169 scored continuation cases had valid measured robot-start checks and verified NAS archives; there were no archive errors.

Successes by card count (40 attempts each): 1 card **18**, 2 cards **22**, 3 cards **25**, 4 cards **15**, 5 cards **24**. By selected SC target port (100 attempts each): port 0 **43**, port 1 **61**. These are privileged teacher replay results, **not autonomous policy performance**.

The automatic finalizer wrote `outputs/experiments/2026-09-30_sc_llb_collection/collection_final_report.json` and the score-verified dataset manifest `outputs/trajectory_datasets/sc_llb_score_verified/manifest.json`. That manifest links exactly **104 successful episodes** with scored bags and action traces. The main status and experiment docs were updated. **No BC training has started from this collection**, and reserved final scenes remain sealed.

See [collection experiment record](../experiments/2026-09-30-sc-llb-collection.md), [current status](../STATUS.md), and the machine-readable `outputs/experiments/2026-09-30_sc_llb_collection/summary.json` for per-case outcomes.

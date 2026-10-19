# Latest diffusion retrain: episode count

The final collision-free dataset for the public-layout diffusion retrain had **602 training episodes**: **497 SFP-to-NIC** and **105 SC-to-SC**. They represent **529 distinct reset groups** (424 SFP, 105 SC), because some SFP episodes share a reset configuration. The training set includes **38 newly admitted episodes** from the public-layout collection (16 SFP, 22 SC); the rest came from the earlier verified sources.

An additional **65 development episodes** (39 SFP, 26 SC) were kept out of training. The frozen feature cache contained 49,023 training rows and 7,319 development rows sampled from these episodes. The model trained for 6,000 updates, drawing rows with per-episode and task-request balancing; the new public-layout rows received 25% of each task family's training probability.

Source of truth: `/home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_union_v1/manifest.json` and the [training record](../experiments/2026-10-11-public-evaluation-coverage-retrain.md). An earlier intermediate count of 593 was from an archived union with duplicate episode IDs and is superseded by the corrected 602-episode union.

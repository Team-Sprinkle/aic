# Is diffusion worse than the previous transformer?

**For reaching the port on the six matched public-layout development starts, no.** The latest coverage diffusion checkpoint got closer than the older deterministic long-history transformer. **For speed, yes:** diffusion was much slower and exceeded the 300 ms limit on SC-to-SC. **Neither solved transport:** each reached the required axial −100 to −70 mm and lateral ≤10 mm safe buffer on 0/3 SFP-to-NIC and 0/3 SC-to-SC starts.

| Saved policy | SFP safe arrivals | SC safe arrivals | SFP closest lateral error in axial band | SC closest lateral error in axial band | Complete inference p95 |
| --- | ---: | ---: | --- | --- | --- |
| Older deterministic transformer | 0/3 | 0/3 | 45.8, 155.4, 132.4 mm | No band on two; 113.8 mm on one | SFP 59–67 ms; SC 133–145 ms |
| New coverage diffusion | 0/3 | 0/3 | 16.7, 24.8, 26.2 mm | 68.4, 29.4, 45.3 mm | SFP 250–273 ms; SC 302–352 ms |

The unchanged-data diffusion control also reached 0/3 safe arrivals in each family. The transformer and coverage diffusion had no scored contacts in these six starts. An earlier matched single five-card SFP-to-NIC test likewise found diffusion closer laterally (about 18–20 mm) than the older temporal transformer (about 56–69 mm), but neither arrived safely. On older *teacher-observation* validation, the deterministic transformer had lower first-action imitation error than the first diffusion fit; that metric did not predict reliable autonomous routing.

This is a comparison of **saved checkpoints**, not a clean architecture experiment. The diffusion checkpoint was retrained with new public-layout scenes, while the older transformer was not. A fair architecture comparison would retrain a deterministic transformer on the same corrected dataset and test it on newly reserved nonfinal starts. The immediate blocker for the current diffusion policy is live lateral drift plus SC latency, so no checkpoint is promoted.

The latest diffusion retrain used **602 training episodes**: 497 SFP-to-NIC and 105 SC-to-SC, representing 529 distinct reset groups. It included 38 new public-layout training episodes. Another 65 development episodes stayed out of training. See the [episode count note](2026-10-12-diffusion-training-episode-count.md), [full experiment record](../experiments/2026-10-11-public-evaluation-coverage-retrain.md), and [machine-readable three-arm comparison](../../outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/official_anchor_retrain/three_arm_comparison.json).

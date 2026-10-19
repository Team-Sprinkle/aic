# Longer transport chunks: result

I trained the two requested transport variants on the **full admitted dataset**: 533 SFP-to-NIC and 83 SC-to-SC training episodes, with 43 and 21 separate development episodes. Every episode retained valid recorded longer-action labels. A target slot is 50 ms. The first model predicts **8 slots and executes 4** (200 ms before replanning); the second predicts **12 and executes 6** (300 ms). The 12/6 model used a newly built 300 ms feature cache so its previous-command history matched its deployment cadence. Both models predict full TCP-body target poses; neither receives true port geometry.

The 8/4 model used 42,364 valid training anchors and continued through 9,500 updates at 3e-5 plus 8,500 at 1e-5, stopping at held-out plateaus. The 12/6 model used 26,988 valid training anchors; its first 20,000- and 16,000-update phases still improved at their caps, so I continued at 3e-6 until the held-out six-command error plateaued after 4,000 more updates. [The development curves](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/training_curves.png) show the selected checkpoints. This supports convergence under the tested setup, while it cannot prove a global optimum.

**Held-out teacher observations improved.** For error averaged over the first four predicted commands, p95 translation error changed:

| Comparison on identical rows | SFP | SC |
| --- | ---: | ---: |
| Four-target control → 8/4, 200 ms rows | 3.15 → **2.58 mm** | 2.75 → **2.64 mm** |
| Four-target control → 12/6, 300 ms rows | 3.63 → **2.37 mm** | 3.73 → **2.53 mm** |

The 12/6 model's error averaged across all six *executed* commands was 2.54 mm p95 SFP and 2.69 mm SC. The two row cadences are different comparisons; the numbers should not be directly compared across them. The four-target control was trained for 200 ms decisions, so its offline error on 300 ms rows may be inflated by that history mismatch. The live comparison keeps each model at its intended cadence.

The first live 8/4 run exposed a runtime bug: inference was slower than the 200 ms chunk window, so the controller often executed only one of the four predicted commands and dropped later ones. I preserved that diagnostic. I then added an opt-in schedule that begins the chunk after inference and spaces commands by measured 50 ms simulation time. In the matched evaluation, command traces confirmed **exactly 4, 4, and 6 commands per proposal**, with zero stale drops, for the control, 8/4, and 12/6 respectively.

**Autonomous transport did not pass.** On the same three SFP and three SC nonfinal development scenes, all three variants scored **0/6 safe approach-buffer arrivals, 0/6 full insertions, and zero scored contacts**. The 12/6 model was closer to the SFP port on all three scenes: its lateral misses were **19.1, 43.2, and 11.5 mm**, versus **30.6, 54.4, and 56.7 mm** for the four-target control. The safe SFP buffer requires at most 10 mm, so even the closest case missed it. The 8/4 model's corresponding misses were **47.0, 57.3, and 50.9 mm**. In SC, only the four-target control entered the axial approach band on one scene, still 214 mm laterally away; neither longer-horizon actor entered it.

We ignored the 300 ms latency limit for model selection as requested. Complete live inference p95 was roughly 364–493 ms for 12/6. This still creates observation-to-action delay in Gazebo, even though the corrected scheduler executes every command. A two-scene four-denoising-step diagnostic lowered latency but did not improve transport: the previously closest SFP miss grew from 11.5 to 51.8 mm; the tested SC scene still missed the axial band. I did not promote that sampler.

**Conclusion:** predicting farther ahead helps imitation on teacher states, and 12/6 may help SFP approach geometry, but it did not achieve reliable handoff and did not repair the SC route. No new checkpoint is promoted. The next targeted work remains causal route-progress or safe teacher-continuation supervision from actor-visited states, addressing the previous-command feedback problem documented in the [earlier audit](../experiments/2026-10-12-route-feedback-and-inference-cadence.md).

Details, commands, paths, and limits: [experiment record](../experiments/2026-10-13-route-action-horizon.md) · [machine-readable per-scene summary](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json).

Representative five-camera videos: [4/4 SFP control](/home/nas/chmin/aic_archives/2026-10-13_route_horizon4_execute4_completeclock_v1/rollouts/sfp_trial2_validation_14/five_cameras_10fps.mp4) · [12/6 SFP](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_completeclock_v1/rollouts/sfp_trial2_validation_14/five_cameras_10fps.mp4) · [12/6 five-card SC](/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_completeclock_v1/rollouts/llb_0074/five_cameras_10fps.mp4).

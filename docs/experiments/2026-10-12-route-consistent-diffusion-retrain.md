# Route-consistent diffusion transport retrain, October 12

**Follow-up:** the [held-out teacher-generator diagnosis](2026-10-12-teacher-distribution-transport-diagnosis.md) found 0/6 autonomous insertions despite 6/6 successful original teachers. It also found that the 10 mm SC TCP buffer below excludes 10/21 fully inserted held-out LLB teachers, so the SC buffer count here must not be treated as a universal route success gate. The first released-family SFP JPEG-86-named recheck reached 1/3 buffers, but [the launcher audit](2026-10-12-route-feedback-and-inference-cadence.md) found that codec setting was never forwarded to Docker. Those runs are unchanged-policy repeats, not evidence of a JPEG-86 benefit. Correctly forwarded codec pilots did not establish a reliable gain.

## Question and data

Does retraining the shared SFP/SC diffusion transport actor after removing the SC CheatCode route conflict improve autonomous transport? The active [dataset manifest](../ACTIVE_TRANSPORT_DATASET.json) has 533 SFP and 83 SC Low Lateral Bypass (LLB) training episodes, plus 43/21 complete-episode development episodes. The 22 SC CheatCode prefixes from the retired mixed union were excluded. Forty new SFP sources have verified varied first-trial starts; two partial-insertion sources contribute only contact-free transport prefixes. The rotated public SC board remains outside LLB teacher coverage.

The training cache has 57,424 sampled rows: 50,129 train and 7,295 development, grouped into 543/56 reset configurations. It retains the three ImageNet camera features, measured state, task code, and causal history. Actions are four recorded TCP-body pose targets; the offline one-second TCP displacement is an auxiliary label. No true port transform, geometry or teacher target is a live actor input.

## Fit and matched checks

The 14.29M-trainable-parameter route diffusion actor (plus 11.18M frozen ImageNet ResNet18 parameters) initialized from the **pre-mix** `fit_pair_ft` checkpoint, which was trained on the LLB SC route. It did not initialize from the retired mixed-SC checkpoint. One GPU ran 6,000 updates at batch size 48, learning rate 3e-5, 12 inference denoising steps; best grouped-development score was at update 5,500. Training stopped at the update cap, not a demonstrated plateau. No anchor sampling mass or pair loss was used. The selected checkpoint is [best.pt](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt).

The same-row offline replay compared both checkpoints using identical held-out observations and sampling seed:

| Group; first-action translation p95 | Pre-mix control | Retrained |
| --- | ---: | ---: |
| SFP public-anchor rows, 96 | 4.21 mm | **2.64 mm** |
| SFP early rows, 287 | 2.28 mm | **2.01 mm** |
| SC LLB rows, 252 | 2.90 mm | **2.74 mm** |

The frozen autonomous check used three SFP and three SC nonfinal development starts, each run with both checkpoints, no guide or privileged target. The SFP starts overlap the source development distribution used for checkpoint selection; these are development checks, not independent final generalization. The SC starts include the rotated released board without matching LLB training labels. The preset safe transport buffer requires axial −100 to −70 mm and lateral ≤10 mm relative to the selected port.

| Live result | Pre-mix control | Retrained |
| --- | ---: | ---: |
| SFP safe-buffer arrivals | 0/3 | **0/3** |
| SC safe-buffer arrivals | 0/3 | **0/3** |
| Scored contacts, SFP / SC | 0 / 1 | **1 / 0** |
| SFP closest lateral offset while in axial band | 53.45, absent, 92.64 mm | **68.86, 25.33, 10.97 mm** |
| SC axial-band entries | 0/3 | **0/3** |

The retrained SFP gripper contacted card 4 on the closest 10.97 mm near miss. Candidate complete inference p95 by SFP scene was 211.6, 220.7, 251.2 ms; by SC scene it was 287.2, **354.7**, 239.0 ms. One SC scene therefore violates the 300 ms limit. All twelve official insertion scores were zero, as expected for transport-only rollouts; this is not an insertion evaluation.

**Decision:** the new checkpoint is a useful offline diagnostic, but it fails the autonomous transport gate and is **not promoted** for handoff, insertion, or RL. No final scene was opened. The observed gains on teacher states did not solve route drift on actor-visited states. The strongest bounded next test is causal correction from actor-visited starts with route-consistent teachers (LLB for SC) and a stop rule requiring safe, collision-checked corrections for both families before fitting. The rotated SC layout first needs a valid LLB route; do not reintroduce its CheatCode shortcut as an expert label. A separate denoising-step latency ablation is justified only after route behavior improves.

## Reproduction and artifacts

```bash
bash scripts/run_route_consistent_transport_retrain.sh
.pixi/envs/default/bin/python scripts/compare_route_diffusion_offline.py \
  --cache /home/nas/chmin/aic_archives/2026-10-12_route_consistent_cache_v1 \
  --base-checkpoint outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --control outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/fit_pair_ft/best.pt \
  --candidate outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt \
  --selected outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/selected_validation_rows.json \
  --output outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/offline_same_rows.json
.pixi/envs/default/bin/python scripts/run_official_anchor_route_match.py \
  --general /home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_coverage_v1 \
  --exact /home/nas/chmin/aic_archives/2026-10-11_official_anchor_sc_exact_v1 \
  --control outputs/experiments/2026-10-11_route_aware_diffusion_transport_v1/fit_pair_ft/best.pt \
  --candidate outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-12_route_consistent_live_match_v1 \
  --gpu 3 --policy-gpu 4 --max-pairs 6
.pixi/envs/default/bin/python scripts/score_official_anchor_route_match.py \
  --comparison /home/nas/chmin/aic_archives/2026-10-12_route_consistent_live_match_v1 \
  --output outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/live_comparison.json
```

The immutable feature cache and all twelve Gazebo bags, camera frames, and runtime traces are on NAS at the paths above. The local checkpoint folder contains `manifest.json`, `history.json`, `summary.json`, `selected_validation_rows.json`, [same-row metrics](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/offline_same_rows.json), [live metrics](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/live_comparison.json), and [machine decision](../../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/decision.json).

Representative H.264 20 fps replays made from the saved side and overhead Gazebo frames: [SFP near miss, side](</home/nas/chmin/aic_archives/2026-10-12_route_consistent_live_match_v1/rollouts/candidate_sfp_trial2_validation_13/side_20fps.mp4>) · [SFP near miss, overhead](</home/nas/chmin/aic_archives/2026-10-12_route_consistent_live_match_v1/rollouts/candidate_sfp_trial2_validation_13/overhead_20fps.mp4>) · [rotated SC, side](</home/nas/chmin/aic_archives/2026-10-12_route_consistent_live_match_v1/rollouts/candidate_sc_trial3_exact_validation_06/side_20fps.mp4>) · [rotated SC, overhead](</home/nas/chmin/aic_archives/2026-10-12_route_consistent_live_match_v1/rollouts/candidate_sc_trial3_exact_validation_06/overhead_20fps.mp4>).

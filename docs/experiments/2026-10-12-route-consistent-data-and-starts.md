# Route-consistent training data and real randomized starts

Status: current data correction complete, 2026-10-12; broader SC orientation coverage remains open. No new actor fit or autonomous promotion. Final scenes sealed.

## Why the October 11 union was retired

The frozen coverage experiment used 83 older SC Low Lateral Bypass (LLB) training episodes plus 22 new SC CheatCode transport prefixes. The new collector never invoked LLB and its admission gate never checked the cable path around/below the cards. Those 22 SC prefixes are useful diagnostic recordings but are **not LLB training data**. The SFP public-anchor recordings did use the intended CheatCode route.

`scripts/merge_official_anchor_route_labels.py` now excludes SC rows from all public-anchor label folders. The first route-consistent union, `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v2/`, held **497 SFP train / 39 development** and **83 LLB SC train / 21 development**. Its manifest records source SHA256 hashes and the exact excluded SC label paths; the SC output SHA256 is identical to the original cleaned LLB SC source. The old mixed union/cache have `RETIRED_FOR_TRAINING.json`, and the mixed checkpoint has `RETIRED_FOR_PROMOTION.json`. Its historical training script requires an explicit reproduction override and is no longer the normal training recipe. Raw bags, images, scores and the old checkpoint remain preserved for diagnosis.

**Intermediate union after varied-start admission:** `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v3/` added the 16 newly admitted SFP training episodes and four grouped development episodes. Its counts were **513 SFP train / 43 development** and **83 LLB SC train / 21 development**. The SC output SHA256 remained exactly `c2a984ce522d30ec84e116036fa87f838ecf86b51eb9b9b1deaf5ffd464da653`; no new SC CheatCode rows entered. The tracked machine-readable active pointer is [`docs/ACTIVE_TRANSPORT_DATASET.json`](../ACTIVE_TRANSPORT_DATASET.json), with a local mirror under `outputs/trajectory_datasets/`. The v2/v3 unions are immutable intermediates.

**Final active union after the bounded all-request pilot:** `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/` adds 20 more SFP training episodes, one for every requested card/port on each released board orientation. Counts are now **533 SFP train / 43 development** and **83 LLB SC train / 21 development**. The same SC SHA256 remains unchanged. The v2/v3 unions are immutable intermediates; v4 is current. The subsequent [route-consistent actor retrain](2026-10-12-route-consistent-diffusion-retrain.md) failed its autonomous promotion gate.

Rebuild command:

```bash
.pixi/envs/default/bin/python scripts/merge_official_anchor_route_labels.py \
  --base-sfp /var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/sfp_labels.jsonl \
  --base-sc outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/transport_chunks.jsonl \
  --anchor-labels /home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_labels_v1 \
  --anchor-labels /home/nas/chmin/aic_archives/2026-10-11_official_anchor_sc_exact_labels_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v2
```

## First-trial robot start repair

The previous public-anchor generator randomized the board and rails but copied the official robot home joints unchanged. Gazebo's *first* trial starts from the robot xacro; changing only `robot.home_joint_positions` would not vary that first start. The collector now has `--vary-starts`: it perturbs each of six requested joints by up to 0.02 rad, writes a per-scene `ur_llb_start.xacro`, hashes it in the plan, and passes it to the rootless Gazebo launcher. A scored-bag joint audit must show the requested offset before any row from such a scene can be admitted. The label materializer explicitly rejects an unverified varied start.

The bounded 20-scene SFP public-request set is `/home/nas/chmin/aic_archives/2026-10-12_sfp_start_variants_v1/`. It completed **20/20 official full insertions** with **20/20 valid first-trial start audits**. The first two trial-1 pilots' measured joints matched expected varied-start predictions within **0.00442/0.00389 rad** maximum, while differing from the old fixed start by **0.01793/0.01857 rad**. Sixteen scenes are training and four are grouped development.

The causal label pass subsequently admitted **all 20/20** episodes, with 16 train/four development and zero rejects. It stored only action chunks before each measured safe transport cutoff. Labels and the per-episode admission report are at `/home/nas/chmin/aic_archives/2026-10-12_sfp_start_variant_labels_v1/`. No actor was retrained.

Generation and pilot commands:

```bash
.pixi/envs/default/bin/python scripts/collect_official_anchor_route_coverage.py \
  --output /home/nas/chmin/aic_archives/2026-10-12_sfp_start_variants_v1 \
  --seed 20261012 --plan-mode sfp_only --vary-starts --plan-only
DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock \
  .pixi/envs/default/bin/python -u scripts/collect_official_anchor_route_coverage.py \
  --output /home/nas/chmin/aic_archives/2026-10-12_sfp_start_variants_v1 \
  --seed 20261012 --plan-mode sfp_only --vary-starts --max-new 2 --gpu 3
```

## Broader requested-card/port coverage gate

A separate **planned, not admitted** 120-scene SFP grid is at `/home/nas/chmin/aic_archives/2026-10-12_sfp_start_grid_v1/plan.json`. Each of two released board sides has five training card counts for every one of ten requested card/port codes, plus one held-out reset per code. NIC presence always includes the requested card. Every scene varies the actual first-trial robot joints as above. This grid is an attempt to cover the released task combinations; no finite local grid can prove coverage of undisclosed final randomization.

One pilot from each board side was run. Both varied starts passed the scored-bag check (maximum expected-joint errors **0.00401/0.00380 rad**). Trial-1 card0/port0 scored **38.61 partial insertion** with no scored contact; trial-2 card0/port0 scored **75 full insertion**. Thus the original two-full-insertion pilot gate failed, and broad collection stopped. A separate pre-insertion transport-prefix check on each showed a safe measured TCP buffer: **−69.99 mm axial / 3.28 mm lateral, 257 chunks** for the partial run, and **−69.07 mm / 5.36 mm, 255 chunks** for the full run. The partial run's insertion tail is not an expert label. `pilot_transport_revision.json` records the original failure and transport-only evidence. `materialize_sfp_scored_transport.py` now permits an explicitly marked `transport_only_prefix` that must reach the measured safe buffer; it never calls a partial rollout a full-insertion teacher. Each new label still requires no scored contact and verified actual start. Expansion, if any, must be staged with an admission-rate check; the 120 scenes are **not** currently a dataset.

The predeclared next stage, after the two-request set finishes, is at most **18 more grid attempts**: one-card settings for the nine remaining card/port codes on both board orientations. Stop there and materialize only measured, no-contact, verified-start safe transport prefixes. Continue beyond this stage only if at least **16/20 attempted grid scenes** admit such a prefix and **all ten requested card/port codes** have an admitted example on at least one board orientation. Full insertion scores remain separately counted. This is a coverage check, not actor training or evidence of autonomous success.

That 20-attempt stage completed: **18 official full insertions, two partial insertions, zero scored contacts, and 20/20 valid first-trial start audits**. There are exactly two scenes for each of the ten requested SFP card/port codes, one on each released board orientation. Causal transport-only labels are at `/home/nas/chmin/aic_archives/2026-10-12_sfp_grid_pilot_labels_v1/`.

The causal label pass admitted **20/20** safe transport prefixes with zero rejects. Both partial-insertion sources have `transport_only_prefix=true`, and their failed insertion tails are outside the action chunks. The frozen [gate report](2026-10-12-sfp-grid-pilot-gate.json) says 20 admitted, ten of ten requested codes, 20 valid actual starts, 18 full source insertions, **gate passed**. Across both new collections, the worst start-prediction discrepancy was **0.01102 rad** and the smallest maximum-joint deviation from the old fixed start was **0.01256 rad**. At this point further bulk SFP collection is deferred: the active set already spans all ten requests, one to five cards in older verified sources, and both released board orientations in the new pilot, while the unresolved rotated-board SC LLB route is the larger missing cell. The factor combinations are still sparse (the new all-request pilot used one card), and neither the pilot nor the union proves autonomous transport or coverage of an undisclosed final distribution.

## Remaining coverage limit

SC has 83 verified LLB training episodes spanning one to five cards and both requested ports; its varied-start continuation was measured in Gazebo. Its older board yaw range does **not** include the released rotated SC anchor around −1.8 rad. A direct transform of the old LLB corridor to that anchor was tried as a *planning-only* diagnostic at `/home/nas/chmin/aic_archives/2026-10-12_sc_llb_official_candidates_v1/`; MoveIt could not find a valid state for the second low waypoint. No Gazebo replay or label resulted, and the established LLB planner code was restored. A scene-specific collision-checked route is needed before rotated-board SC data can be added. The removed SC CheatCode labels must not be silently used as a substitute.

The practical status is therefore: the active dataset is route-consistent, and actual randomized SFP starts have been verified. Full task/reset coverage is still an open data gate, especially for rotated-board SC. The completed retrain improved held-out teacher-state actions but failed autonomous transport, so do not claim final-evaluation readiness or repeat the same fit without a distinct hypothesis.

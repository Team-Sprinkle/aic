> **October 18 score target achieved:** **290.19/300**, all three public-sample insertions pass independent depth/lateral/orientation checks; all used-stage p95 inference is below300ms. The analyze/repair/evaluate loop stops after18 completed full samples. V56 preserves three quiet SCcontext frames while sharing initial grasp observations, and seats1.5s above the public1s requirement. Rule transport and explicit RGB/public-state insertion use unchanged perception weights; no learned BCactor or RLactions. Failures and unqualified fits are preserved. This is one selected public-sample pass, with small SCdepth margin and unresolved unusual-SFP-grasp coverage; final scenes stay sealed. The32new SFPcaptures are perception-only, not insertion experts; verified BCexpert sources are unchanged. Earlier sections are historical. [Results, commands and videos](experiments/2026-10-18-five-repair-cycles.md).

## October 18: gated close-up pose update and SC insertion result

V34 passes **4/4 matched SC development cases** (zero/five obstacle cards, both ports), followed by a passing SC round in the unchanged public sample: **15.510 mm depth, 0.141 mm lateral error, 0.469° orientation**, toolkit completion, no harm. Five fresh agreeing images support one bounded plug-pose update at 25 mm; the port frame stays fixed. Tighter lateral centering and stronger lateral tracking are also needed. All used-stage p95 timings pass; these are explicit RGB/public-state control results, not BC or RL.

The complete sample is **170.57/300, 2/3 strict** because SFP trial 2 has a 3.53° biased grasp estimate before contact. Its actual roll exceeds the original SFP training range; close-up predictions remain wrong. Repeats stop, and the next targeted repair is SFP orientation coverage rather than a stiffness-only change. No new training or collection is running. The SC model's original 1.026° qualification failure remains explicit; four successful development cases do not establish general reliability. Final scenes remain sealed.

[Detailed evidence, commands, images and next repair](experiments/2026-10-18-sc-explicit-controller-continuation.md).

## October18: SCheld-plug close-view supervision

**October18 near-grasp corpus:** bounded additions are complete at3,446 RGB triplets /294 complete episodes (234train,60validation), including80 near training grasps and four unchanged near validation episodes. Each triplet has center/left/right native crops. Extra resets are **perception supervision only**, not successful insertion or BC demonstrations. Original sources and rejected captures are preserved. No new collection is queued. [Current record](experiments/2026-10-18-sc-explicit-controller-continuation.md#completed-80-grasp-fit-and-bounded-fine-feature-revision).

A bounded36episode acquisition-view set and12episode25mmset are under `/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/{acquisition_captures,near25_captures}`. These are **perception labels, not BCexperts**. Both collections are complete: near8train/4validation and acquisition24train/12validation. The far derivative has2,441three-camera samples/210episodes(154train/56validation); addingnear yields2,582samples/222episodes(162train/60validation). Counts reflect only valid image-time rows, not12assumed frames per episode. Complete episode/reset splits, native three-camera pixels, causal image-time relative-grasp labels and visual QA are retained. One reviewed acquisition validationepisode retains four quiet valid triplets rather than six; all failed retries and the explicit exception remain. Do not admit unsteady/stale frames or count this as a stable-hold/insertion success. Episode-balanced and frame-weighted metrics both enter model selection. Original supervision is immutable; merged derivatives and both far-only/far+near fits have separate paths. See [SCcontinuation](experiments/2026-10-18-sc-explicit-controller-continuation.md).

## October 18: targeted SC grasp coverage, separate from BC

The current full-connector BC derivative still has **60,671 anchors from 499 usable episodes**: SC132 episodes and SFP367, with420training/79validation episodes overall. Targets remain recorded complete TCP commands, expressed through the same RGB-estimated port/grasp frames as the inputs. No new perception hold is an expert insertion demonstration.

The SC plug-in-gripper estimator has a concrete initialization gap: the public sample's nominal cable roll, pitch and yaw all lie outside the original132SC source ranges. A bounded36-reset perception-only collection varies nominal-centred grasps, board/card layouts, both SC targets and0/1/5NIC obstacles. It uses24training/12validation reset groups, fixed original-image crops and strict image-time raw-TF labels. All36holds completed with429valid triplets; three frames lacking past TF were rejected. The merged grasp dataset has2,039triplets/174episodes(130training/44validation episodes), with unchanged episode/reset splits. First-six and final crop QA passed. Originals are preserved; derived crops/labels live on NAS. A completed hold with a teardown crash retains no official success score and is explicitly excluded from BC expert admission.

Initial expanded-source SC fits reached LR-floor validation plateaus but failed the new-grasp accuracy screen. New fits will use the checked nominal-coverage data. If an estimator is selected, actor input/target frames must be rebuilt with that exact estimator before using it with a retrained actor. [Data, commands, QA and gates](experiments/2026-10-17-insertion-control-and-grasp-repair.md).

# Expert trajectory locations

<!-- current-continuation:start -->
## Current approved continuation — October 16

Rule transport brings either SC or SFP to a visual acquisition hold about 150 mm before the opening. A fresh requested-port pose and RGB-derived plug offset then initialize **one shared learned actor for the entire 150 mm approach and insertion**. Large hierarchical retreats return along measured poses to that acquisition hold and require fresh perception. The old 30 mm controller is historical.

**Training is running:** matched ImageNet versus visual-only transport initialization, same 13.725M full-target transformer, 30,000-update budget per arm. The fresh arm is active; the warm arm follows automatically. Actor data contain **67,039 examples from 500 usable source episodes / 377 physical reset configurations**: SC 102/30 train/validation episodes, SFP 319/49. All wider-source admission and estimated-frame audits passed. Predictions are complete recorded command targets in the visually estimated port frame, with causal robot/pose history and no privileged geometry input. Training error is not autonomous insertion success.

**Mandatory refiner retraining is complete**, including SC and SFP: 72,857 native RGB triplets from 497 original source episodes, followed by a targeted 20-episode SFP acquisition extension. Both base and repair reached validation plateaus. The strict repaired SFP offline component gate passes with the 0.95 projection threshold; this projection flag is not calibrated confidence or certified physical visibility.

**Launcher audit correction:** several newer host environment settings were not forwarded into Docker. The recent SFP live10 result remains 10/10 correct acquisitions, but actually used projection threshold **0.90**, not the intended 0.95. The active SC broad60 uses the **new repaired native refiner**, not the separately specified older SC checkpoint. Preserve these results under their actual configuration; they do not certify the intended family bundle. Let the frozen SC job finish. A separate corrected launcher records the effective container environment and forwards these settings for subsequent matched tests. The original launcher/source hashes remain intact during active evaluation.

The hierarchical runtime is implemented, including retained pre-return RGB/state, coherent lateral supervisor retries, measured route retreat and reacquisition. **28 focused contract/recovery checks pass; live actor/recovery validation is still pending.** Supervisor exploration is not RL. No new SAC/SERL, reward model, imagination or reserved-final evaluation has run. Keep final scenes sealed and at most four project GPUs including untouched legacy GPU0.

Next: finish matched BC fits and current frozen SC acquisition evaluation; verify corrected runtime settings; run fresh autonomous full-depth development rollouts, diagnose failures, then compare no/local/hierarchical recovery with identical actors and scenes. Advance to RL only after the supervised insertion gate.

See [execution record](experiments/2026-10-15-rule-perception-hierarchical-execution.md) and artifact `state.json` for exact commands, failures, comparisons and paths.

**Additional SC coverage repair:** the29 existing rail-expansion episodes added4,526 RGB-proposed native crops (22/7 complete train/validation episodes) after label/split/visual QA. The joint SC+SFP repair reached a validation plateau at4,000updates. It passes9/9 previously rejected SC probes versus0/9 for both baselines, and improves new-rail accepted components166→211. Overall accepted components3722→3711 is a small regression; zero wrong/absent accepts remain. The original strict gate failure is preserved. A separately documented20-scene live diagnostic is running; the new checkpoint is **not promoted**. Black-gripper-only crops explain some large pixel-error tails and are not evidence of visible targets. Active frozen models/scenes remain unchanged.

<!-- current-continuation:end -->


## October 15 autonomous continuation: current evidence

### October 16: acquisition repair and actor-data readiness

**No new insertion BC or RL checkpoint has been trained yet.** Mandatory SC/SFP native port refiner retraining is complete to plateau. The approved learned actor still owns the entire 150 mm approach, with starts buffered to 180 mm. Large retreats must return to a measured acquisition checkpoint and obtain fresh port/grasp locks.

- **SC:** the old checker run stopped under its failure rule at 36 qualified / 41 completed of 60 planned, zero wrong ports/harm. Its code is archived with all eight frozen source hashes verified. The new single-request native-five pipeline, bounded 140 mm visual correction and measured hold gate achieved **7/7 correct acquisitions**, opening errors 0.077–0.363 mm and zero wrong/harm. All global inference p95 values are below 128 ms; one perception-active subset is 306.8 ms. Thus its stricter latency gate is **6/7, still open**. A same-scene diagnostic separates Gazebo rendering from policy inference while respecting four project GPUs. Models/scenes are unchanged. A single-point 64 px relaxation remains rejected because it accepted an absent target; the replacement five-landmark geometry rejects that case.
- **SFP coarse:** full-data recorded-image performance improved substantially, but the unchanged live six-case gate remains 3/6. Existing image history did not repair the three failed traces. A bounded twenty-reset acquisition-state repair is collecting ten training / ten validation configurations across 1–5 cards, both ports and all requested card codes, then fitting source-balanced current/history heads while retaining the full old corpus. The first two all-camera label overlays were inspected. It predicts projected-in-frame status, not certified physical visibility or learned confidence. Require at least 9/10 fresh correct acquisitions and zero wrong/harm before promotion; otherwise change the representation instead of routine recollection.
- **Grasp estimate:** a separate native SFP plug-offset model plateaued after 15,000 updates on 6,115 triplets from 307/47 train/validation sources. Across all phases its position p95 is 0.505 mm, failing the 0.4 mm gate; worst images hide the plug inside the port. In the actual runtime acquisition scope—after a causal visual port lock, with public measured TCP at least 120 mm before the opening—position p95 is **0.150 mm on 272 observations / 39 held-out episodes / 31 resets**. Translation is approved only for that scope and is locked before insertion. Orientation remains diagnostic and is not an actor input. Do not update this estimate from an inserted, hidden plug.
- **BC data:** 454 qualified original expert sources now have causal RGB port locks. Requiring a legal three-observation grasp lock leaves **432 sources / 51,667 rows**: SC 80/20 train/validation episodes and SFP 293/39. The other 22 successful SFP recordings are preserved, not called bad experts; they lack an approved early free-grasp lock. Earlier rows are never backfilled. Original data remain untouched, with a separate exact actor-pixel cache on NAS.
- **Wider demonstrations:** SC first-thirty admission passed: 22/30 physically qualified episodes, from ten original training reset configurations. All three SFP pilots at 90/150/180 mm, including lateral/8-degree offsets, fully inserted at 46.776 mm depth and 0.247–0.382 mm lateral error without scored harm. These are privileged teachers, not learned-policy successes. SFP first-thirty collection reuses those three pilots. Ten held-out wider starts per family are queued before the matched BC comparison.

Preflight fixes for the upcoming actor: complete port-frame targets for both families, exact recorded force/trajectory profile, plug-based near-port phase weighting, command-feedback masking in BOTH current and historical state, and identical native-JPEG preprocessing during training and deployment. The old 15 mm command cap would clip about 11% of SC and 27% of SFP expert chunks; the new controlled runtime must use the audited 60 mm component cap. Visual warm starts now reject silently empty/incompatible transfers; the comparison retains fresh current-data state normalization. Four real history offsets are planned, with the history-masking ablation OFF.

Rootless runs use private bridge networks and unique Gazebo partitions. Cooperative per-GPU locks serialize future rollout/training workers without interrupting active jobs. Legacy GPU0 is untouched. Current work uses GPUs3/4/5 plus that legacy allocation. Final scenes remain sealed; reward/imagination and new actor-Q training remain gated. See the [execution record](experiments/2026-10-15-rule-perception-hierarchical-execution.md) for commands, failed comparisons and artifact paths.

### Latest: rail coverage, clean simulation isolation and rule pilots

**Current gates remain open.** Mandatory native refiner retraining is complete to plateau on both families, but live acquisition still needs repair before the new 150 mm insertion actor can be trained and promoted. No new insertion BC or RL actor has been trained in this continuation.

The SC coarse failure has a concrete data gap: old SC0 rail translations cover roughly −2 to +18 mm and SC1 −60 to −43 mm, whereas the released range spans −60 to +55 mm. Even the 95 failed original collections do not fill that gap. An old-layout current/history locator fit cannot establish generalization to these missing port arrangements. The bounded repair collects 29 new physical reset configurations across the rail range, both requests, 1–5 cards and absent-other-port negatives. These are perception-only recordings, not successful insertion experts. Native overlays from the first two runs were inspected and passed causal-label QA. Current/history fits use grouped splits and balance original versus expanded-layout sources; live correct-port acquisition remains the promotion gate.

The first six SFP rule-transport cases yielded **1/6 fully qualified acquisitions, zero wrong-port handoffs**. The one pass stopped about 148 mm before the port, 5.7 mm laterally off-axis, with 0.324 mm visual opening error. Several failures waited at the elevated acquisition plane without a usable visual lock; another reached the region but could not sustain the stability gate. A bounded matched pilot tests acquisition at the nominal 150 mm plane. It does not relax the visual handoff gate. This is rule transport only; no insertion actor ran.

Three repaired SC teacher starts now pass independent raw-bag terminal checks, including 150/180 mm lateral/orientation offsets. Depth is 15.57–15.59 mm, lateral error 0.15–0.46 mm, with no scored harm. A smaller reset integral gain fixes measured oscillation, and 6 mm/s axial insertion reaches full depth before the simulator completion event freezes motion. These are privileged teacher demonstrations. Restart only the bounded 30-attempt teacher gate with this controller; require at least ten independently qualified successes before expansion.

BC preparation now has **53,921 command rows from 454 qualified source episodes**, including SC and SFP. All sources have matching completion events; 4,822 later commands are excluded. Episode IDs include source-path hashes to avoid basename collisions. Four labels are actual active complete TCP targets at +50/+100/+150/+200 ms. Original data are preserved. Body-command conversion uses preceding header-time TCP and is explicitly an approximation to RT-consumption state; a four-episode audit measures p95 discrepancy of 0.07–0.12 mm. Causal RGB-derived port locks are still needed before this becomes an actor-ready dataset.

A confirmed infrastructure bug affected two later runs: concurrent host-network containers shared Zenoh port 7447 and Gazebo discovery despite separate ROS domains. Both launchers now use private bridge networks and unique Gazebo partitions. Audits found no conflict warnings in the 454 admitted original sources or the frozen 12-run SC comparison. Failed infrastructure runs remain preserved and excluded from policy evidence.

See the [execution record](experiments/2026-10-15-rule-perception-hierarchical-execution.md) and artifact-root `state.json` for commands, exact counts, failures and subsequent updates. Earlier October 15 paragraphs below are intermediate history where they describe running jobs or older derivatives.

### Latest acquisition gate and BC-data preparation

The completed frozen SC comparison failed promotion: old stack **3/6** correct acquisitions, new point refiner plus association **1/6**, zero wrong accepts in either arm. No insertion actor ran. The bounded replacement of the legacy presence veto then accepted the neighboring port in **1/1** diagnosis (56.21 mm opening error): **rejected**, not a safe fix. Exact-frame replay reproduces the association scores to within 0.000041 and finds contradictory identity evidence in one camera; there is no established simple state-range or frame-conversion explanation.

A distinct crop-proposal repair is being tested: a public-request-conditioned locator chooses one SC region per camera from all three current RGB feature maps and optional 40 mm / 10 degree motion-spaced context. It omits absolute TCP coordinates and retains relative motion for history. On 83 training / 21 validation episodes, current-only and history heads both reach about 10.9 px coarse p95 with zero nearest-wrong-port views among 14,235 labeled in-frame views. Both initial fits hit an update cap; separate weight continuations seek a plateau. Fresh live diagnosis remains necessary. This predicts projected-in-frame status, not certified physical visibility.

The first BC command derivative contains 46,910 rows from qualified original sources, but 78 sources had stale archived image paths. A corrected derivative is rebuilding from the audited native-image provenance fallback. Original datasets and failed derivatives remain intact. Robot state is reconstructed as 32 dimensions from preceding controller/joint/wrench observations; actual public task codes have 10 dimensions. Four labels are the recorded active TCP targets at +50/+100/+150/+200 ms. Targets held between message publications remain genuine command supervision. Static pauses are thinned in the copy without joining actions across removed time. No future measured pose supplies an action label. Six QA panels were saved and SC/SFP examples visually inspected. Estimated port locks must still be produced before actor training.

All three initial wider SC teacher successes passed a separate raw-MCAP, pre-return audit: depth about 15.568 mm, lateral 0.082–0.117 mm, orientation about 0.012–0.013 degrees, no scored harm. A bounded 30-attempt batch covers both ports and 1–5 cards with 90/150/180 mm starts and offset perturbations. Some supervisor resets fail; these are not BC experts. Diagnose the completed batch before expansion; at least ten independently qualified successes are required. A successful teacher video is `wide_approach_pilots/wide_sc_pilot_02_event_v5/five_cameras_20fps_grid_active.mp4` under the artifact root.

The approved pipeline is rule transport → requested-port acquisition/refinement → one shared learned insertion actor owning the full 150 mm approach. Training should buffer starts to 180 mm. Large hierarchical retreats return to rule transport and require a fresh visual lock. The old 30 mm controller is a historical control, not part of this pipeline. BC and recovery comparisons remain open; no new RL has been trained in this continuation.

**Refiner retraining is now complete to a validation plateau.** The corpus contains 72,857 original RGB triplets from **497 source episodes / 633 named image streams / 422 physical resets**. These are different counts; image-stream IDs must not be counted as independent episodes. Source episodes: SC 83 train / 21 validation, SFP 340 train / 53 validation. Physical reset groups are disjoint across splits.

A matched SFP component test combines the stronger requested-port coarse locator with the new native refiner. The five-landmark pose variant accepts 356/1,120 sampled triplets with zero wrong-port accepts, versus 324 accepted by the old component. Conditional position/lateral/orientation p95 is 0.447 mm / 0.189 mm / 0.708 degrees. It fits translation and yaw using a training-only fixture normal prior; arbitrary tilted fixtures are not established. The set has 70 image-stream IDs but only 53 source episodes / 45 physical resets. These are component results, not autonomous insertion success. Preloaded-image perception p95 is 20.46 ms; complete live latency remains unverified.

A diagnostic global orientation head previously used simulator card count, which is absent from the public task. Those old results are quarantined and must not support deployment claims. Its corrected public-input control has now plateaued: SC/SFP angular p95 2.33/3.27 degrees, versus state-only 4.80/4.08 degrees. The native landmark fit above remains the more precise component candidate.

Independent geometry, official insertion score and contact/force records admit **454 source episodes** for subsequent BC preparation: SC 80 train / 20 validation, SFP 307 train / 47 validation. This is an admission index, not yet a materialized action-chunk dataset. Ownership, image-time state joins, stall cleaning and predicted-frame targets remain mandatory. Original data are untouched. Failed or shallow recordings can remain useful for perception without becoming successful BC experts.

The wider SC teacher now demonstrates full insertion from all three diagnostic starts, including a 180 mm approach and lateral/orientation offsets. The third case originally returned before the simulator contact timer completed; waiting for the teacher-only completion event fixes its official score to 75. These are privileged teacher demonstrations, not actor successes, and require independent raw-bag geometry/force checks before expansion. The first 30 wider attempts must contain at least 10 qualified full insertions before collection scales.

The frozen six-configuration/two-arm SC acquisition comparison has completed and failed its gate. The legacy-veto replacement also failed a wrong-port test; see the latest update below. No final scenes have been opened.

Artifacts and exact commands: `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/`. Detailed chronology: [execution record](experiments/2026-10-15-rule-perception-hierarchical-execution.md).


## October 14 grasp-label timestamp repair

The original104SCepisodes and actiondatasets are preserved. Grasp-probe labels mistakenly paired command-timeTIPTF with earlier image-synchronizedcontrollerTCP. A derivative verifies each row against its recordedcontrollerstamp and joins TIPTFat that stamp; command andimage times are retained separately. The p95label shift is1.931mm, and correctedwithin-episodegrasptranslationvariation p95 is0.179mm. This repair changes probe geometry labels, **not recorded expert actions**. Earlier probe failures remain archived and cannot establish a perception-capacity ceiling.

Artifact root: `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/`; corrected labels: `sc_grasp_observation_labels_v3/`, same1248triplets and83train/21validationepisodes. Native192crop pixels are reused immutably from the earlier probe. [Execution record](experiments/2026-10-14-transport-handoff-execution.md).


**October 14 timestamp repair:** The versioned SC target derivative is `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/repaired_targets_v1`. Original recordings are unchanged. It restores 1,906 complete 12-slot chunks and uses per-slot masks at terminal/ownership boundaries. All 104 episodes retain final-return supervision and all 61 stop examples are retained independently; SFP targets remain unchanged. This does not restore dense visual history to the sparse feature cache. [Audit and execution](experiments/2026-10-14-transport-handoff-execution.md).

**October 13 independent transport review (analysis only; supersedes the earlier immediate correction-collection recommendation):** The SC training/evaluation boundary was mismatched: **0/104 admitted transport prefixes** enters the old −100 to −70 mm TCP approach band; their median terminal TCP axial coordinate is −168 mm (plug tip −148 mm). Later teacher approach actions were excluded. The 12-step filter also leaves **29/83 training episodes without nominal final-return-stage supervision** and removes every cached handoff-positive row from its action subset. The diffusion branch has no stop head and sets handoff probability to zero. Rescoring existing runs at the actual taught endpoint gives fine70 **5/6** versus dense70 **3/6** transient visits within 20 mm/5°, followed by drift; these posthoc visits are **not autonomous handoffs or insertion successes**. Same-clock early separation also confounded speed and route error. Next: validate the component boundary, repair timestamp/terminal supervision and test the executor before another model/data sweep. No new training, simulation, or GPU inference; final scenes stay sealed. [Evidence, research and bounded plan](experiments/2026-10-13-transport-independent-root-cause-review.md).

**October 12 active transport union:** `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/` contains **533 SFP CheatCode-style training episodes** and **83 SC Low Lateral Bypass-only** training episodes, with 43 and 21 grouped development episodes respectively. Forty newly scored SFP randomized-start episodes were causally admitted (36 train/four development): 38 source rollouts fully inserted, two partials supplied safe transport-only prefixes with insertion tails excluded. The bounded all-request pilot covers all ten card/port codes on both released board orientations. The 22 newer SC CheatCode training prefixes were removed because their cable route was not verified as LLB; raw recordings remain under the October 11 public-anchor archives for diagnosis. The older mixed union/cache and checkpoint are marked retired. [Exact sources, exclusions and checks](experiments/2026-10-12-route-consistent-data-and-starts.md).

The first route-consistent actor fit used this v4 union to build `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_cache_v1/` (57,424 sampled rows). Its checkpoint is retained as a failed autonomous comparison, **not** an active promoted policy: [fit and gate](experiments/2026-10-12-route-consistent-diffusion-retrain.md).

**October 7 stall-label audit:** The SFP dataset below was not modified or
time-compressed. An optional train-only action-loss mask at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/stall_filter_v1.json`
marks 689 sustained high-command/low-measured-motion cache rows, 688 from
legacy SFP data. It excludes no validation or paired-contrast rows. A
matched fit found no improvement, so this is a diagnostic derivative rather
than the new default expert dataset. [Audit and result](experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md).

## October 7 same-reset alternate SFP requests and coarse RGB

The new paired-request derivative
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_v1/`
adds **16 fully inserted train episodes / 4,130 transport rows** from twenty
alternate-request attempts on the *same reset configurations* as earlier
successful SFP teachers. Four partial/no-insertion attempts remain archived
at `/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_v1/` and are
excluded. The strict frozen-image and measured-TCP pair audit leaves twenty
≥5 mm paired action rows; its manifest is adjacent to
`temporal_cache_divergent_v1` under the same `/var/tmp/.../sfp_paired_request_20261005/`
parent. A completed second collection of unused train/development resets is
under `/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_round2_v1/`.
It added **20 fully inserted training alternates and eight fully inserted
development alternates / 7,249 admitted transport rows**. One development
timeout was excluded. The complete-reset, frozen-feature audit admitted
**34 training and nine held-out action pairs from six development groups**.
The successful episodes are in the separate derivative
`sfp_divergent_merged_round2_v1/` under the same `/var/tmp/.../sfp_paired_request_20261005/`
parent. Their actor fit passed offline action metrics but failed fresh
autonomous transport (0/3 safe arrivals); these rows are not evidence of
live route success. The original 289-episode expert set is unchanged.

The **coarse RGB data are perception labels, not actor actions**. Their
91-train/27-development/18-reserved-stress scene split is at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_diverse_crop_fit_split_v1/`.
The 27,508 request-expanded training crop rows and 1,609 development rows
are in `sfp_diverse_crop_fit_materialized_v1/` alongside that split. Native
images come from scored Gazebo RGB; CAD projected openings train/score the
crop model offline, while frozen RGB predictions alone center its crops.

## October 7 diverse SFP transport derivative

The original `expert_verified/` collection and its 289 canonical episodes
are unchanged. A separate verified **transport-only** SFP derivative at
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_diverse_final_v1/`
adds **136 fresh, officially full-insertion Gazebo episodes / 35,383
causal TCP-body transport chunks** to the prior paired SFP labels. The
initial 137 attempts yielded 132 successes; a fixed five-cell replacement
batch yielded four more. Five initial partials and one genuine retry
no-insertion remain in the raw NAS archives and are excluded. An infrastructure
crash before scoring was rerun on the identical scene with both audits kept.
The new episodes are split by complete scene/reset: **119 train / 17
development validation**, so the merged SFP counts are **445 train / 27
development validation**. The 17 held-out scenes cover sparse one- and
two-card requested-port cells; none train the actor. One cell (one card,
requested card 4/port 1) has only three train plus one new development
example after its allowed replacement failed. The raw images, original
scored bags, scene plans and failure scores are under
`/home/nas/chmin/aic_archives/2026-10-06_sfp_diverse_teacher_v1/` and
`/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_retry_v1/`. Historical
agent/VLM scene YAML was used only to choose reset configurations; its action
arrays were never imported. [Exact admission, command-frame check and actor
result](experiments/2026-10-06-sfp-diverse-transport-and-proposal.md).

**October 6 paired-port fit use of existing data:** The 70-group SFP training manifest supplied 21,262 request rows, deduplicated to 10,631 selected-card/frame native crops and balanced by requested card. The ten checkpoint-development groups stayed separate; 119 requests scored the three-point model and 355 in-frame requested camera views scored the rail/TCP coordinate prior. No new scenes or reserved final scenes were opened for these failed branches. Original image paths, frozen RGB rail predictions, CAD projection labels and per-step outputs remain in the materialized manifests and NAS `2026-10-06_sfp_local_card_pair_fit_v2/` and `2026-10-06_sfp_rail_to_port_prior_v1/`. [Record](experiments/2026-10-06-sfp-local-card-pair-repair.md).

**October 6 joint-candidate fresh evaluation:** Ten unused source-development SFP reset configurations, one per card/port request, were frozen at `/home/nas/chmin/aic_archives/2026-10-06_sfp_joint_candidate_fresh10_v1/plan.json`. They do not overlap the 70 crop-fit train groups, ten checkpoint-development groups or earlier additional ten groups; reserved final scenes stayed sealed. Sequential rootless Gazebo CheatCode controls all fully inserted and provided 528 synchronized 0.5 s triplets; 527 passed board-label synchronization. Images, scored bags, labels, frozen RGB rail predictions and crop proposals are under the matching `2026-10-06_sfp_joint_candidate_fresh10_*` NAS directories. These groups have now been scored and must not be treated as an untouched tuning set. [Record](experiments/2026-10-06-sfp-joint-camera-candidates.md).

**October 6 SFP crop-fit manifest:** Frozen RGB board predictions for 3,690 train frames from 70 complete reset groups were joined to original synchronized images and expanded to 21,262 physically present requested card/port examples. Ten development groups (119 recorded requests) selected the checkpoint. Ten additional disjoint nonfinal teacher-route groups (581 recorded requests) were scored once after selection; they are now an inspected stress set, not a fresh blind test for threshold tuning. Exact row/source manifests: `/home/nas/chmin/aic_archives/2026-10-06_sfp_slot_crop_training_materialized_v1/` and `/home/nas/chmin/aic_archives/2026-10-06_sfp_slot_crop_independent_materialized_v1/`. Projected CAD positions are training/scoring labels only; no simulator pose places a crop at inference. [Record](experiments/2026-10-06-sfp-slot-conditioned-crop-feasibility.md).

## October 5 SFP board-reference perception derivative

The [fixed-rail label manifest](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/summary.json>) uses synchronized native 1152×1024 three-camera RGB from **70 complete train reset groups**, ten disjoint checkpoint-development groups and three separate saved sparse autonomous actor groups. The labels are five scene-YAML occupied-slot bits and CAD rail projections used **only for fitting and posthoc scoring**; projected-in-frame does not mean visually identifiable. A further [ten-reset nonfinal stress manifest](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_labels_v1/summary.json>) contains 581 synchronized triplets with no reset overlap with the fit or checkpoint selection. These are **perception labels, not expert action trajectories**. The slot-local fit's 10/10 exact-map results on both ten-group sets do not certify requested-port identity or safe handoff. [Full experiment](experiments/2026-10-05-sfp-slot-local-occupancy-ablation.md).

## October 5 sparse-card camera-label addition (not policy actions)

The initial strict camera-check train source had **no one- or two-card scenes**. The [v3 all-request manifest](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3/all_requests.jsonl>) adds only four newly scored **train** episodes with official full insertion: two one-card pilots and two matched two-card original-target controls. The separate pilot stopped at 2/4 because both attempts to redirect its two-card CheatCode task to the second card failed full insertion; those two attempts remain diagnostics and are **excluded** from v3. After the 10 ms timestamp filter, the addition is 2,026 native three-camera moments and 20,260 correlated request-image pairs. V3 totals 3,690 training moments / 70 complete reset groups and retains the previous disjoint development/diagnostic groups. All new physical-presence and recorded-request labels match the saved scene YAML; [audit](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3/sparse_scene_independent_audit.json>). These are geometry and task labels, **not** reviewed RGB visibility, numbered-slot identifiability, safe-pause or actor-action labels. No new camera-check head was trained. [Full record](experiments/2026-10-05-sfp-camera-check-data-quality.md#sparse-card-coverage-follow-up-october-5).

## October 5 SFP camera-check label candidate (not expert actions)

The [strict native-RGB request manifest](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/all_requests.jsonl>) expands 1,664 training triplets across 66 resets to 16,640 request-image examples, including 1,460 physically absent requested ports. Ten grouped-development resets, ten harder diagnostics and three saved actor-route diagnostics remain separate. Original three-camera 1152×1024 JPEGs and source hashes are retained. The builder excluded 73 stale train, 17 stale development and two mixed-camera train triplets (10 ms maximum state/image lag). The [blind review queue](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/blind_review_queue.jsonl>) contains 166 stratified examples. `projection_training_label_only` is **not** a visual visibility label; projected near-port points can lie behind the gripper. `safe_pause_label` and `visual_identifiability_label` are deliberately null, so no new camera-check or handoff model was trained from these fields. [Audit and examples](experiments/2026-10-05-sfp-camera-check-data-quality.md).

## October 4 SFP coarse-perception derivative

The selected-port coarse experiment uses **66 complete source-training episodes** and ten disjoint grouped-development episodes, not a transition-level random split. Six newly collected, previously unused five-card source-training configurations all fully inserted with ordinary CheatCode and yielded 327 additional causal 0.5 s native-RGB/state rows. These augment 60 existing training episodes; the ten development and ten previously inspected diagnostic scenes were not added to training. The [combined labels](</home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1/rows.jsonl>) join requested identity, each of three original images, measured state and projected port geometry; geometry is **training/evaluation label only**. [Original six collection](</home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_targeted_coverage_v1>), [observation-only all-channel training proposals](</home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_targeted_coverage_v1/corrected_train66_proposals/rows.jsonl>), [grouped-development proposals](</home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_targeted_coverage_v1/corrected_dev10_proposals/rows.jsonl>). Projected-in-frame must not be treated as actual visual visibility; occluded/absent and safe-stop labels still need curation. These episodes do not replace the canonical expert trajectory dataset for policy actions.

## October 3 balanced SFP transport derivative (separate from canonical experts)

The old 268 verified SFP BC episodes are valid but target-skewed: **258 request card 0 / port 1**. The selected 180 older agent/VLM replays contain balanced scene/task YAML and useful RGB/state, but their Cartesian action arrays have invalid zeros during motion. They are **not** an action-label source. A new rootless ordinary CheatCode collection instead re-runs selected `train` scene configs and records scored `gripper/tcp` commands and native center/left/right images. The predeclared three-per-target gate passed **30/30 official full insertions** and **30/30 causal command/observation label audits**, giving 7,777 TCP-body transport chunks. The smallest command join fraction was 98.6%; the worst episode's controller-reference p95 was 3.67 mm / 0.54°. [Collection progress](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_teacher_train/progress.json), [gate](/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/balanced_train30_gate_v2.json), [materializer](../scripts/materialize_sfp_scored_transport.py).

Expansion to at most five `train` scenes per selected card/port and one disjoint `development` scene per request is in progress. This derivative will use old SFP training episodes only through their first buffered pre-insertion arrival, and preserve old validation episodes outside the new checkpoint selection. The selected port geometry is used **offline** to mark that boundary and labels; it is never supplied to the transport actor. This derivative is transport supervision, not a verified local-insertion dataset or an autonomous insertion result. The original canonical dataset below remains unchanged. [Full running record](experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md).

The October 1 SC Low Lateral Bypass BC derivative is **training-ready** at `outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/`. It has 104/104 officially inserted episodes in an 83-train/21-validation episode and reset-grouped split. The independent [audit JSON](../outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/audit_motion_safe_v2.json) verified exact recorded targets, all three camera paths at episode boundaries, untouched validation rows, no split leakage, and preservation of measured TCP motion and excess-force rows. This preferred derivative retains 69,681 of 89,251 raw transport chunks after removing 19,570 quiet, nearly stationary scripted-hold anchors; the 26,455 local chunks are all retained. An earlier `bc_chunks/` derivative retained 65,792 transport chunks and was used for the completed matched motion-guard ablation. Use `transport_chunks.jsonl` for TCP-body transport targets and `local_chunks.jsonl` for selected-port-frame local targets. The original `sc_llb_score_verified` episodes and full raw chunks remain unchanged. See the [cleanup record](experiments/2026-10-01-sc-llb-cleaned-bc.md).

## September 30 SC Low Lateral Bypass collection

The `outputs/experiments/2026-09-30_sc_llb_collection/` teacher collection completed at **200/200 attempted settings** with **104 official full insertions** and **198 valid scored bags**. Only successes with a bag and action trace are linked in `outputs/trajectory_datasets/sc_llb_score_verified/manifest.json`. This collection remains separate from canonical `expert_verified`; the October 1 cleaned derivative and BC runs are described above. [Exact collection results](experiments/2026-09-30-sc-llb-collection.md).

## September 30 transport continuation

**Current matched data:** v9 has two fully inserted Cartesian SC episodes (one per port in the exact same five-card world): 1,751 transport and 503 local training chunks. The older failed-insertion transport prefix remains archived in v8 and is excluded from v9. The paired teacher scored 75 at Tier 3. The equal-port shared transport BC run completed 1,500 updates, but failed on both autonomous development starts (103/603 mm planned-standoff error, 0/2 insertion). This is not an autonomous insertion result.
The teacher transport gate is accepted. The current two-actor continuation is documented in [the execution record](experiments/2026-09-30-transport-to-two-actor-continuation.md). This does not mark autonomous full insertion as successful. Control-matched SC dataset v8 has one fully inserted source and one reviewed transport-only prefix; failed insertion actions are excluded. Existing 268 SFP episodes retain their physical-reset split. The original successful SC local data remains separate. Ten learned candidate rollouts across the same two held-out starts failed; the controller-error mask improved v8 behavior but v9 paired-port training regressed. Routine feed-forward transport BC is stopped pending a bounded route-subgoal/progress diagnosis. No new RL or final-scene evaluation is started. Historical model results below refer to their own checkpoints and evaluation gates.


## Canonical expert collection (2026-09-18)

New ACT work uses `outputs/trajectory_datasets/expert_verified/`.
The original local collection has been renamed, without deleting data, to
`outputs/trajectory_datasets/clean_including_no_insert_trajs/`.
Historical experiment manifests retain their original paths; apply the prefix
mapping in `SOURCE_RELOCATION.json` at the archived root when reading them.
The S3 prefixes below were not renamed.

The current canonical collection contains **289 BC-eligible episodes**:
**268 SFP + 21 newly collected SC**, split into **250 training / 39 validation**
episodes and 169,710 cached frames. Cache v3 appended 17 SFP and 12 SC episodes
to v2's 260 episodes. Every previous record, array prefix, and split assignment
was preserved. Training and validation share no scene, and neither overlaps
the 59 reserved development/final evaluation scenes.

The initial filtered SFP collection contains **251 episodes**:
140 historical CheatCode episodes and 111 aligned expert/corrective episodes.
They comprise 141,556 cached frames, split into **223 training / 28 validation
episodes**. Exact scene/task configurations stay in the same split. No exact
scene/state/action/timestamp duplicates were found; image-level deduplication
was not performed. Seventeen additional aligned nominal expert episodes were
accepted after checking scores, labels, every image, and terminal frames. The
table below describes the **current 268-episode SFP portion**.

| NIC count | Eligible SFP episodes | Training | Validation |
| --- | ---: | ---: | ---: |
| 1 | 155 | 139 | 16 |
| 2 | 30 | 26 | 4 |
| 3 | 31 | 27 | 4 |
| 4 | 27 | 23 | 4 |
| 5 | 25 | 21 | 4 |

Target coverage is uneven: 258 episodes target card 0 / port 1. The ten
CheatCode smoke episodes target port 0, distributed across cards 0–4 as
2 / 2 / 2 / 1 / 3. Card-count coverage does not establish balanced target
coverage. The canonical task vector records family, target port, target card,
and card validity; it is now included explicitly in the ACT training cache.

### Newly verified SC coverage

The first aligned SC collection attempted 20 episodes and admitted nine; a
second collection admitted 12 of 16. The canonical SC total is therefore
**21 successes from 36 attempts**. Every admitted episode has official Tier 3
= 75, total score >=80,
no prohibited contact or force penalty, checked action/target reconstruction,
and inspected terminal camera frames. Fifteen failures remain outside the
canonical expert set. Five additional planned collection paths were never
attempted and are explicitly marked `unattempted_not_data`.

| Target SC port | SC receptacles | NIC cards | Training | Validation |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 4 | 2 |
| 0 | 1 | 2 | 4 | 2 |
| 0 | 2 | 3 | 2 | 2 |
| 1 | 1 | 1 | 4 | 1 |

These scenes use a fixed grasp, small board/rail variation, and fewer fixtures.
They do not cover SC port 1 with additional cards, or general difficult SC
layouts. The new append adds a held-out port-1 demonstration. New splits are
grouped by scene within task, target card/port, NIC-count and SC-count strata.
Sparse strata receive at least one validation scene when two independent
scenes exist, so SC alone is 14/7. Preserving old splits and small-stratum
holdouts gives the combined 250/39 split, rather than an exact 90/10 ratio.

The new cache is
`/var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3/`, also linked as
`outputs/experiments/2026-09-18_act_all_verified_6h50/cache_all_verified_sfp_sc_v3/`.
Read `expert_verified/manifest.json` → `training_cache` for the current version.
New SFP/SC source recordings are copied into the canonical folder; the NVMe
cache is derived data. Previous caches and canonical manifest versions remain
available. Append workflow: `scripts/append_verified_experts.py`; exact counts
and checks: `v3_append_result.json` and `v3_publication_audit.json` in the September 18
experiment. Initial SC evidence remains in `canonical_sfp_sc_membership_check.json`;
the second collection's terminal review is `sc_collection_gpu0_followup/visual_review.json`.

The [world visual follow-up](experiments/2026-09-18-world-followup.md) checked
all 289 episode ranges, 253 image shards, RGB/BGR provenance and scene
groups. Its matched tokenizer experiment keeps the older pilot's original
14 validation episodes sealed because 15 of the 74 pilot episodes have a
different assignment in this canonical split. The resampled image cache does
not establish how many of all 289 frames are geometrically near a port
opening; use synchronized plug/port transforms for that question.

The September 19 full tokenizer run uses all **250 canonical training
episodes** for optimization and all **39 canonical validation episodes** for
selection and reconstruction reporting. “All valid” therefore means the full
training side of the split, not training on validation episodes. Sampling gives
equal probability to historical SFP, aligned SFP, and SC, then samples an
episode uniformly within that category. Dynamics has a narrower eligibility
rule: an episode must provide native simulator timestamps, native observation
endpoints, and the actual four-command sequence for each 200 ms transition.
Visual eligibility alone does not make an episode dynamics-eligible. See the
[completed protocol and results](experiments/2026-09-19-full-world-training.md).

That strict audit retained **148/289 episodes** for dynamics (126 train, 22
validation): 127 SFP and 21 SC, with 68,884 native observations. The 140
historical video episodes have no native `frames.jsonl`; one aligned SFP episode
has frames but no actual command indices. They are excluded from dynamics
without synthesizing timestamps or commands, while their verified images remain
eligible for tokenizer training.
The prepared strict transition set contains 9,858 training and 1,486 validation
200 ms edges. The completed dynamics model used exactly this set and did not
outperform persistence at 200, 400, or 600 ms; visual eligibility must therefore
not be cited as evidence for planning-quality temporal coverage.

The **253** successful historical agent episodes (180 nominal SFP + 50 recovery
SFP + 23 SC) remain outside BC training under
`outputs/trajectory_datasets/successful_pending_label_repair/manifest.json`.
That manifest points to preserved archived data and records why each episode
is ineligible. The earlier conversational figure of 273 was an arithmetic
error. Sparse replay events and planned trajectories cannot recover exact
executed commands aligned to every recorded image; zeros occur during final
SC seating as well as initial joint motion. Actions were not invented from
observed motion. Separately, **212 scored non-insertions and 63 unresolved
episodes** are excluded from the verified folder.

`expert_verified/manifest.json` is the source of truth for membership, task,
score/source evidence, recording conventions, scene group, and split. The
folder contains complete verified CheatCode data/meta/videos and selected
aligned episode image/state/action directories. Unchanged source files use
hard links when on the same filesystem and byte-preserving copies otherwise;
consumers must treat them as immutable. The ACT cache reuses
audited image shards and converts their documented BGR/RGB conventions to RGB
when loading. Historical wall-clock timestamps remain a recorded limitation.
The small additional decoded cache is stored at
`/var/tmp/chmin_aic_20260918_act/cache_cheat10/` because `/data1` was nearly full;
it can be regenerated from the canonical recordings.

Creation command and distribution evidence:
`scripts/curate_verified_experts.py`,
`outputs/experiments/2026-09-18_act_all_verified_6h50/dataset_summary.json`,
and `source_relocation.json` in that experiment directory.

## Historical inventory and verification

Checked 2026-09-17. This page identifies collected demonstrations for the
ACT → offline SERL → online SERL workflow. The initial inventory fetched only
small S3 metadata files. The subsequent [eight-hour ACT experiment](experiments/2026-09-17-act-verified-8h.md)
audits scores, source arrays, and final camera frames and trains ACT locally.

## Main local collection

Use this root in the current checkout:

```text
/data1/chmin/yj/ws_aic/src/aic/outputs/trajectory_datasets/clean_including_no_insert_trajs/
```

There are **23 `accepted_dataset` directories, containing 668 stored episodes
and 539,480 frames**. Parquet row counts match each dataset's `meta/info.json`.
A first video frame from each of the three cameras in all 23 datasets decoded
successfully. This count is an availability inventory; training eligibility is
reported separately below. Cross-collection deduplication remains incomplete.

## Verified insertion episodes

**Historical camera/timing compatibility:** the 130-episode CheatCode source
below was recorded with the pre-fix collector, which converted camera RGB to
BGR before writing RGB video. Its cached pixels are preserved; models fitted
on those pixels require BGR live input, or a documented conversion to canonical
RGB during training. Checkpoint `image_channel_order` records that contract.
The collector also sampled on a wall-clock timer: parquet timestamps and video
PTS equal frame index / FPS, not original simulation time. Matching image PTS
does not establish a 20 Hz simulation trajectory. Do not assume these
conventions apply to later agent collections. See the [bounded ACT report](experiments/2026-09-17-act-verified-8h.md)
and its saved historical collector, image comparison, and clock audit.

The September 17 score/lineage audit found **393 episodes with official Tier 3
score 75 and accepted state/action arrays matching their raw source**. Another
212 episodes have scores indicating no insertion, and 63 have unresolved local
score/source mappings. Unresolved episodes are not counted as failures or used
in the initial ACT runs.

| Source family | Verified insertion episodes | Initial use |
| --- | ---: | --- |
| CheatCode SFP | 140 / 140 | Start with the 130-episode card 0 / port 1 collection. |
| Agent nominal SFP | 180 / 180 | Defer: many moving frames have zero recorded Cartesian actions. |
| Agent SFP recovery | 50 / 50 | Defer until action-label handling is addressed. |
| Agent SC | 23 / 298 | 212 near-gate episodes are not insertions; 63 mappings remain unresolved. |

For each source, first/middle/last episodes were sampled at the final frame and
one and two seconds before it, in all three cameras; CheatCode has additional
samples. Contact sheets were visually inspected. Some insertion views are
occluded, so images corroborate the official score rather than independently
proving microscopic seating. Raw/accepted end-frame checks for representative
CheatCode episodes agree within video-compression differences.

The 130-episode ACT cache also checks decoded frame timestamps across all three
cameras. It preserves the recorded full TCP-relative command labels and source
episode IDs, and canonicalizes quaternion signs to match the runtime. Of the
140 CheatCode episodes, only two frames combine a zero action with TCP speed
above 1 mm/s; the agent families have many such frames. Historical success and
raw-array equality still do not guarantee every action is an appropriate
imitation target: controller-state shortcuts and deployment timing are tested
in live rollouts.

Evidence: [episode audit](../outputs/experiments/2026-09-17_act_verified_8h/verification/episodes.json),
[source summaries](../outputs/experiments/2026-09-17_act_verified_8h/verification/sources.json),
and [experiment record](experiments/2026-09-17-act-verified-8h.md).

The paths below are relative to that root. Append `/accepted_dataset` to each
run path when loading LeRobot data. Wildcards describe multiple existing runs.

| Collection | Run path | Episodes |
| --- | --- | --- |
| CheatCode SFP smoke | `sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke` | 10 |
| CheatCode SFP card 0 / port 1 | `sfp_to_nic/cheatcode/nic_cards_mixed/n130__card0_port1_resume` | 130 |

| Agent/VLM nominal SFP, EC2 hard batch | `sfp_to_nic/agent/nic_cards_mixed/n400__ec2_02_sfp_hard` | 180, despite the `n400` target in its name |
| Agent/VLM SFP recovery, NIC counts 1–5 | `sfp_to_nic/agent/nic_cards_*/n10__sfp_nic*_start_near_gate_04mm_n10_recovery_gpt5_feedback` | 50 across five runs |
| Agent SC full-insertion collections, NIC counts 1–5 | `sc_to_sc/agent/sc_ports_2/n20__sc_ports2_nic*_full_insert_n20` | 71 across five runs: 7 / 11 / 18 / 20 / 15 |
| Agent SC explicitly stopped near gate | `sc_to_sc/agent/sc_ports_1/*stop_near_gate*` | 212 across four runs |
| Other agent SC collections | Remaining six SC `accepted_dataset` directories | 15 |

**October 3 task-coverage correction:** The later 268-episode verified SFP
waypoint/cache set used for transport and the arrival probe is highly
imbalanced: 258 episodes request **card 0 / port 1** (227/236 train,
31/32 validation). Only ten episodes total cover other requests, across five
other card/port combinations. Its complete-episode validation split has
disjoint scene hashes, but it is not evidence of reliable behavior on all
SFP target cards and ports. See the [machine-readable audit](../outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/audit.json)
and [arrival experiment](experiments/2026-10-03-sfp-arrival-probe-and-coverage.md).

**Older selected agent/VLM images recovered:** The separate 180 selected EC2
agent/VLM SFP replays score at least 90 and their numbered trial YAMLs recover
all ten target card/port pairs (8–30 episodes each). Their recorded Cartesian
actions include invalid zeros during motion, so the October 3 continuation
uses RGB and measured robot state for perception/arrival labels **only**. A
state/task arrival head trained from them failed an autonomous trace replay;
see [the balanced arrival experiment](experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md).

Exact paths/counts and inspection evidence:
[CSV inventory](../outputs/experiments/2026-09-17_trajectory_inventory/inventory.csv)
and [JSON inventory](../outputs/experiments/2026-09-17_trajectory_inventory/inventory.json).
These ignored artifacts are local, not supplied by a Git clone.

### What “agent/VLM” means here

The later `agent` collector uses GPT-5-mini for symbolic scene/approach strategy,
MoveIt for free-space motion, and ground-truth geometric alignment/insertion.
GPT-5 is used for failure critique in the recovery workflow. The SFP EC2 run's
`generation_config.json` records those strategy/analysis models. See the
[expert generator guide](expert_trajectory_generator.md) and the
[EC2 hard-batch request](../aic_utils/lerobot_robot_aic/config/data_generation_requests/ec2_02_sfp_hard.yaml).
This is not a VLM predicting all low-level actions directly.

A September 23 follow-up scanned 190 retained SC full-insertion
`agent_generation/replay_attempts` and replayed one low-score three-card
VLM/MoveIt joint trajectory. It also checked the one-port/five-card seed-51500
collection associated with an older cable-catch note. That collection's local
and checked S3 clean prefixes retain the accepted dataset, video, and
metadata, but the original
failed route/video from the note is unavailable there. The retained score-1
three-card attempt failed on a later port-handoff miss; its transient force
peak should not be labeled a cable snag. See the
[route and replay audit](experiments/2026-09-23-ordinary-development-cable-audit.md#agentvlm-route-follow-up).

A subsequent [fixed five-card route probe](experiments/2026-09-23-fixed-five-card-route-probe.md)
downloaded the accepted center-camera MP4 from the collection's S3 clean
prefix and verified that its SHA-256 matches the local file. The accepted
video covers attempts 27 and 28. The historical failed seed-51500 recording
was not found in that clean prefix, the inventoried `datasets/dev/` prefix,
or the inventoried `ec2_transfer/` prefix. The new route-probe videos are
fresh Gazebo runs, not recovered historical footage.

Earlier VLM experiments also exist under:

```text
outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/vlm_planner/
outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/vlm_planner_postprocessed/
```

These include individual `lerobot_dataset`/`raw_dataset` attempts. They are
separate from the 23 accepted collections counted above; their existence does
not establish success or justify adding them to ACT training.

## S3 location and current access

The original training orchestration script explicitly waited for a sync from:

```text
s3://aic-team-sprinkle/datasets/clean/
```

Evidence: [run_clean_act_serl_20260510.sh](../outputs/train/clean_sfp_sc/run_clean_act_serl_20260510.sh).
Collection manifests also retain the EC2 source prefix
`/home/ubuntu/ws_aic/src/aic/outputs/trajectory_datasets/`.

Read-only S3 access was tested. All **21 agent collections** have remotely
readable `accepted_dataset/meta/info.json` files under the same relative paths,
and those metadata files match the local bytes. Their roots are:

```text
s3://aic-team-sprinkle/datasets/clean/sfp_to_nic/agent/
s3://aic-team-sprinkle/datasets/clean/sc_to_sc/agent/
```

The older `vlm_planner/` and `vlm_planner_postprocessed/` prefixes also exist
remotely. Full data/video equality with S3 was not checked.

**The two CheatCode collections exist locally but their matching keys are
absent from the current S3 `datasets/clean/` prefix.** Do not treat their local
folder name as proof of a current remote backup. The attempted metadata reads
returned `NoSuchKey`. Preserve their local data, manifests, and score files.

The **10-episode CheatCode smoke collection was located under `dev/` instead**:

```text
s3://aic-team-sprinkle/datasets/dev/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset/
```

Its remotely read metadata matches the local bytes. The 130-episode collection
was not found at either matching `clean/` or `dev/trajectory_datasets/` key;
its remote backup location remains unconfirmed. No bulk S3 sync was performed.

## Correction to the earlier dataset audit

The [live-validation audit](experiments/2026-09-17-live-validation.md#demonstration-audit)
followed the old combined manifest's `outputs/s3_clean/` paths. That local mirror
is incomplete relative to the original `outputs/trajectory_datasets/clean/`
tree (now `clean_including_no_insert_trajs/`). Its two missing
CheatCode source paths were not evidence that the data were lost.

Both CheatCode sources were found in the broader local tree. Their selected
trial IDs map to **140 official scoring YAMLs, all with Tier 3 score 75**.
That initial lookup verified historical insertion outcomes. The subsequent
audit above additionally checks source arrays and image timestamps; it does not
rerun the expert. The previous raw audit
is retained as the result of its narrower lookup; this inventory corrects its
availability/provenance interpretation.

The earlier ACT/direct-actor dataset has **546 episodes / 389,907 frames** at:

```text
outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_h264
```

It is not the full collection. The broader source tree contains 122 additional
episodes: 71 SC full-insertion collection episodes, 50 SFP recovery episodes,
and one EC2 SC-easy episode. A later combined `*_raw32` directory also advertises
668 episodes, but its processed schema/media/training suitability were not
validated in this inventory.

## Before choosing ACT training data

- Start from explicit source runs and their accepted trial identities. Keep
  nominal, recovery, and stop-near-gate data identifiable. A directory named
  `clean`, an `n400` target, or an acceptance flag is not an insertion label.
- Counts in `generation_summary.json` can be stale. Three recovery summaries
  report fewer accepted episodes than their actual ten-episode datasets.
- The source-family audit above narrows the action-label concern: CheatCode is
  much cleaner than the agent collections, whose joint-space transport can
  produce zero Cartesian labels. Historical success alone does not validate
  every recorded action.
- The existing `prepare_clean_act_dataset.py` accepts explicit `--clean-root`
  arguments. Its default `outputs/s3_clean` would omit locally available data.
  Its generated `*_fake_scores` files are merge bookkeeping, not scoring evidence.
- Preserve original source IDs, use grouped/scene-balanced holdouts, and create
  a fresh derived dataset rather than replacing the historical one.


## New verified collections during the September ACT experiment

All paths below are under `outputs/experiments/2026-09-17_act_verified_8h/`.
These are additional local recordings, separate from the original 668-episode
inventory. No S3 upload has been performed.

| Source | Verified insertions | Eligibility and evidence |
| --- | --- | --- |
| `corrective_data_pilot`, `corrective_data_batch1`, `corrective_data_batch2` | 24 of 26 attempts | Bounded random expert-command perturbations; clean expert targets label observed states. Two partial episodes excluded. `cache_cheat130_corrective24` retains all lineage. |
| `nominal_absolute_data_pilot`, `nominal_absolute_data_batch1` | 19 of 21 attempts | Actual simulation timestamps, RGB, and recorded teacher target equal to executed absolute target. 15 training / 4 validation episodes in `cache_cheat130_nominal19`. |
| `student_corrective_data_pilot` | 1 of 1 attempt | Bounded ACT-influenced execution with privileged expert correction labels; total 92.55, Tier 3 75. This is a training collector, not a learned success. |
| `nominal_absolute20_batch1`–`batch4` | 20 of 20 attempts | Four independent five-scene batches; all accepted terminal frames inspected. Actual simulation timestamps and absolute targets. |
| `nominal_extra20_batch1`–`batch4` | 20 of 20 attempts | Additional independent clean scenes; full official insertion scores and all accepted terminal frames inspected. |
| `student_extra20_batch1`–`batch4` | 13 of 20 attempts | Bounded student interventions, clean expert targets; seven failed insertions excluded. All 13 accepted terminal sequences inspected. |
| `student_corrective20_batch1`–`batch4` | 14 of 20 attempts | Twelve episodes with recorded student interventions and two nominal episodes with no selected intervention. Six incomplete insertions excluded; every accepted episode's terminal frames inspected. |

Official scores, sampled terminal images and visual-review notes accompany the
collections. The clean and student pilot deliberately share one training scene
for control diagnostics; they are not independent reliability trials.

The latest complete combined cache, `cache_cheat130_aligned87`, contains 87 new
episodes: **70 train / 17 validation** (41,052 / 9,805 resampled frames). The
87-recording continuations used only these new training episodes for updates
and normalization. Their warm-start weights also inherit earlier training;
the 130 historical episodes remain in the cache as shared references. Their
overall validation logs additionally retain seven legacy NIC-1 episodes;
`corrective_holdout` reports recent validation alone. The final selected model
uses the preceding 74-recording stage (60 train / 14 recent validation), not
the newest checkpoint merely because it has more data. Incremental image shards use absolute paths in
`cache.json`: preserve referenced caches, or update shard roots after relocation.

These collection batches are complete. Consult the
[experiment report](experiments/2026-09-17-act-verified-8h.md) for training use. Successful outcomes and trustworthy action labels
are separate acceptance requirements.

### Strict 74-episode world-model pilot: event timing

The September 18 Dreamer pilot uses the original 60 training / 14 validation
aligned episodes, without the historical 130. Its scene audit found **73 unique
scenes in 74 episodes**: episodes 130 and 149 repeat one training scene, while
no scene crosses the training/validation split. The newly frozen paired
development and final scenes do not overlap these 73 scenes.

All 74 official bags contain a correct-port insertion event. The event occurs
**40–142 ms after the last saved observation** for the 60 training episodes and
**46–124 ms after** for the 14 validation episodes. Thus the episodes remain
verified successful command trajectories, but none contains an observed
post-insertion image/state. The aligned collector records an observation when
it sends a target on a fresh camera timestamp; the CheatCode policy then
continues that motion, checks the official insertion event, and exits after
success. The final saved image may show an approach or partial seating, but it
cannot by itself establish the exact insertion state. The later correct-port
event establishes that the expert **did** insert successfully; the missing
post-event frame is a recording-boundary problem for state/reward learning.
Do not label the last frame as a positive insertion state or train a
terminal-state reward from that proxy. The [strict-74 raw-bag audit](../outputs/experiments/2026-09-18_dreamer60_pilot/post_event_raw_bag_audit/report.json)
found short post-event controller-state tails in every bag but no camera or
video topics. The [all-verified metadata audit](../outputs/experiments/2026-09-18_world_followup/verified_bag_metadata_audit.json)
found no camera/image/video topics in any of the 149 linked aligned ROS bags;
the other 140 historical episodes link to LeRobot videos, with no event-linked
ROS bag in the canonical manifest. The additional aligned bags' state-tail
durations and historical videos' event timing remain unverified. Future
collection should save a short synchronized post-event observation tail and
explicit event timestamp. Evidence: [event audit](../outputs/experiments/2026-09-18_dreamer60_pilot/reward_gate_report.json)
and [paired scene manifest](../outputs/experiments/2026-09-18_dreamer60_pilot/paired_scenes/manifest.json).


### October 14 complete plug-tip target derivative (bounded comparison)

`/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/local_sc_tip_target_dataset_v5` preserves all 138,198 rows and the original episode split from the local mixed-family dataset. Only SC target translation changes: each recorded future TCP target is transformed to its corresponding plug-tip point. Target orientation remains TCP orientation. SFP labels are unchanged. Original recorded TCP targets are retained for decoded-command loss and scoring. True body-tip calibration is a label-building artifact and never a model input. Input grasp translation still comes from the camera-only predictor at a causal pre-local handoff. The source derivative remains unchanged. Numerical conversion parity is saved in `point_target_conversion_check.json`. This is an experimental action reference, not a promoted dataset or actor.


### October 14 qualified causal teacher continuations

The selected NAS derivative is `2026-10-14_transport_handoff_execution_v1/causal_local_continuation_dataset_v3`: 1,765 rows, with 1,184 rows from two complete training episodes and 581 from one complete validation episode. It supplements the 83 SC training / 21 validation source episodes; it does not replace them. All SFP data remains unchanged. Sources are qualified port-0 v3 and port-1 v2 controller successes, each after a frozen actor prefix. Their controller versions are recorded; do not claim 3/3 from one teacher version. The v3 command-cap feasibility screen itself passed 2/3.

Teacher ownership is reconstructed from exact serialized/logged target matrices. RGB triplets and all 32 measured state values match the recorded runtime observation. Ambiguous, delayed or inconsistent joins are excluded; no actor commands or post-physical-success horizons enter expert BC. True grasp position re-expresses labels only. Inputs use camera-estimated port and body-tip position. A v4 verification copy is byte-identical to v3, not another set of episodes. Twelve reviewed image windows and their provenance JSON are stored in the NAS execution root.

### October 16 continuation

Mandatory SC/SFP native refiner retraining is complete to plateau. The targeted SFP coarse repair also plateaued; on the same 1,120 held-out triplets it accepts 917 versus the old 913, with zero wrong ports. Native refinement rejects all 14 new coarse absent-target false positives. The ten-case live gate is still running; this small component change is **not** a live success claim. Its first failure reaches the correct image region but exceeds the native geometric residual gate. See the [execution record](experiments/2026-10-15-rule-perception-hierarchical-execution.md) for the inspected predicted/label overlays, full metrics and bounded diagnosis.

The new `RunRuleThenSharedRecovery` integration is implemented but unvalidated: SC/SFP rule transport -> fresh RGB port/grasp locks -> one shared actor owning the entire150mm approach -> measured-path local/hierarchical backtrack. Large retreats return to the measured acquisition checkpoint and require fresh perception. Pending chunks/history are cleared, contact never retares initial load, and supervisor commands are logged separately. **No new insertion BC or RL has been trained yet.** Wider training/held-out demonstrations and matched fresh versus visual-warm BC remain in progress. Final scenes remain sealed and the project limit is four GPUs including the untouched legacy allocation.

SC's targeted separate-GPU latency case passed (active p95 239ms, opening error0.414mm); the earlier seven-case single-GPU strict result remains6/7. Current status and exact artifacts are in [STATUS.md](STATUS.md) and the execution record.

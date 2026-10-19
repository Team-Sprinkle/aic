## October 19: SC start-side cable route diagnostic

The five-card SC development start that visually threaded the cable through the NIC row was generated from the pinned **predeadline public sample**, not a released production evaluation scene. Three named route variants are now selectable: `sc old` (back first), `sc skip back` (sideways from the current start), and `sc ascend` (lift before back). The initial two new variants failed with the original corridor. An opposite-side `sc skip back` route acquired the requested port once but failed on its seven-case repeat when cable/card contact stalled the tilt. Moving farther sideways or farther beyond the row did not fix that stall. Tilting **outside the row before sweeping toward the port** gave correct SC visual handoff in **7/7** sample-derived screen cases, with no scored harm or local recovery; **6/7** met the strict gate because one five-card run had active perception p95 390.8 ms. The exact same-case repeat passed at p95 189.5 ms. The difficult five-card port-1 start passed twice with early tilt. This is transport/handoff evidence only; the cable can still brush cards and no insertion was attempted. A frozen, independent 24-case sample-derived SC screen then achieved **22/24 strict correct-port handoffs (91.7%)**, including **8/8 four/five-card starts**, zero wrong ports and zero scored harm. One zero-card case missed the visual-acquisition budget; one two-card case blocked twice during the first sideways/down leg despite local recovery. Twenty-three of 24 had active inference p95 below 300 ms. This supports the dense-row route but does **not** establish a universal SC fix; it remains opt-in, and SFP is unchanged. [Routes, one-second images, video, commands and failed comparisons](experiments/2026-10-19-sc-start-side-route-comparison.md).

## October 19: local contact recovery now runs during transport

Both SC-to-SC and SFP-to-NIC transport now use a bounded observation-only contact gate: sustained force change **and** stalled measured waypoint progress. On contact, the controller retreats 8–12 mm against recent measured TCP travel, tries a horizontal sideways move perpendicular to that travel, then resumes the same route from the shifted side position. Visual timeout alone does not trigger recovery. This is a rule-controller change, with no training or privileged runtime geometry.

One five-card SC development start demonstrated the behavior: **8.4 mm measured retreat, 9.5 mm measured lateral move, and qualified correct-port visual handoff**. It was transport only. The matched released three-round rerun scored **98.749703/300, 1/3 strict insertions**, versus **98.856989/300, 1/3** before; **no transport contact occurred in that rerun**, so it cannot establish an insertion gain. The two failures were visual-acquisition timeouts. Reserved final scenes remain sealed. [Implementation, exact traces, video, commands and limitations](experiments/2026-10-19-transport-local-recovery.md).

## October 19: visual-timeout route retreat corrected and rerun

The SC transport code no longer treats a missed visual lock as physical blockage. A matched rerun of the exposed released `eval_config.yaml` changed only `RunSampleTransportRecovery.py` among 160 selected inputs: **98.856989/300, 1/3 strict insertions**, versus **98.826833/300, 1/3** before. The SC trial now stops at visual timeout with **zero** full-route backtracks instead of retracing 322 waypoints, but it still never acquires the selected port. Its 114 checks reject for pose geometry (47), crop margin (38), or required-view projection (29). SFP trial 2 still force-stops before acquisition. This fixes the unintended retreat, **not** the insertion score. No training or reserved-final access. [Code, exact command, videos, metrics and decision](experiments/2026-10-19-visual-timeout-route-retreat-fix.md).

## October18: frozen official qualification evaluation complete

The selected V56configuration was run once on the released three-trial `eval_config.yaml`: **98.826833/300, 1/3 strict insertions**. Trial1 SFP inserts (46.763mm depth,0.199mm lateral,0.232°); trial2 SFP stops on force during cable contact with intervening NIC card3 before port acquisition; trial3 SC triggers measured-path retreat but hits the controller's175s wall guard before port acquisition. A subsequent causal trace check corrected the SC diagnosis: `sample_visual_acquisition_budget`, not verified force blockage, triggered the 322-waypoint retreat. The event reason had incorrectly labeled this `force_or_measured_blockage`. Future source now excludes visual-acquisition timeout from full-route retreat; the frozen official result is unchanged. All actually used inference stages have p95below300ms. No model, route, scene or threshold was changed during evaluation.

This released qualification config differs from the default sample that scored290.19/300; those results must remain separate. The released scenes were previously exposed, so this is retrospective evaluation only. The reserved internal final split stays sealed. [Complete official score, causal diagnosis, source hashes and videos](experiments/2026-10-18-frozen-V56-official-evaluation.md).

## October 18: score target achieved; repair loop complete

**290.19/300, all three strict physical insertions pass, every used-stage inference p95 below300ms.** The five initial cycles ended281.05/300; the requested continuation reaches its target on cycle18. All18complete samples (54rollouts), failed implementations and the separate failed startup remain preserved. No new training/evaluation is queued.

| Public-sample round | Score /100 | Depth mm | Lateral mm | Orientation degrees |
| --- | --- | --- | --- | --- |
| SFP, NIC mount0 / port0 | 97.180 | 46.775846 | 0.101495 | 0.141225 |
| SFP, NIC mount1 / port0 | 96.556 | 46.748804 | 0.129038 | 0.385019 |
| SC, selected port1 | 96.457 | 15.528748 | 0.161249 | 0.427162 |

What helped: visually confirmed empty rows allow shorter SC transport; antiwindup prevents overshoot; persistent angular references remove slow correction. Firmer free-space/SFP lateral tracking, preserved public holding references at gain changes, exact parallel camera conversion, distinct force sampling and causal packet timing reduce measured losses. Shared grasp acquisition needs fewer frames, but SCmust retain three quiet far images for its existing near/far correction. Seating lasts1.5s, above the public1s contact requirement. Physical, force, harm, scene and scoring gates stay unchanged. Cycle17's290.09score is explicitly rejected because SCdepth15.499944mm fails15.5mm; the missing-context repair passes in cycle18.

**Runtime:** rule transport, requested-port/held-plug RGB perception, and explicit visual insertion. No BCactor actions or RL. Perception weights are unchanged; extra fits reach plateaus but fail qualification. SC local-backoff mode is enabled but not triggered; this is no evidence for hierarchical-backoff benefit. Maximum used-stage p95 is220.44ms; transportp99is323.23ms.

**Scope:** one selected public-sample pass, not broad/repeated qualification. SCdepth margin is0.028748mm; recent SFPgrasps near20.46° do not establish repair of the old26.52° out-of-support case. Final scenes remain sealed. Selected scoped candidate and raw artifacts: `/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/selected_public_sample_candidate.json` and`cycle_18/`. [Complete record, commands, failures and videos](experiments/2026-10-18-five-repair-cycles.md). Earlier sections below are historical.

## October 18: current explicit insertion continuation

**Re-estimating the held plug at 25 mm improves insertion, but reliability is still being checked.** Four completed SC trials in the original public sample now pass the physical and toolkit gates, reaching 15.501–15.522 mm depth. In one matched development pair, disabling the update fails at 9.30 mm depth and 2.27° orientation error; enabling it passes at 15.52 mm and 0.69°. These are results from an **explicit controller using RGB and public robot state**, not the learned BC actor or RL.

| Complete public sample | Score / 300 | Strict combined insertions | Remaining issue |
| --- | --- | --- | --- |
| V21 first | 257.36 | 3/3 | Small SC depth margin |
| V21 repeat | 170.20 | 2/3 | SFP seated, but a brief load event ended the policy before contact dwell finished |
| V26 first | 257.65 | 3/3 | Acquisition p95 295.7 ms; SC depth margin only 0.0011 mm |
| V26 repeat | 257.44 | 3/3 | Acquisition p95 315.7 ms fails the timing gate |
| V27 shared JPEG | 257.59 | 2/3 | Timing passes; SC depth 15.49057 mm still fails |
| V34 combined repair | 170.57 | 2/3 | SC passes; SFP trial 2 fails from a biased plug-orientation estimate |

**Latest result:** V34 passes **4/4 matched SC cases** (zero/five cards, both ports), then passes the SC round of the unchanged public sample: **15.510 mm / 0.141 mm / 0.469°**, toolkit 75, no harm. It combines the 25 mm visual plug update, tighter lateral centering, corrected public tracking-bias initialization and lateral stiffness 500 while keeping axial stiffness 90. All used-stage p95 timings pass; transport p99 is 620 ms. No local retreat is needed in those four SC cases, so this is not evidence of backtracking advantage.

**Remaining SFP failure:** the full sample is **170.57/300, 2/3 strict**, not a complete pipeline pass. SFP trial 2 stalls at 21.18 mm / 0.825 mm / 3.64°. Its initial plug-orientation estimate is already 3.53° wrong. Actual held-plug roll 26.52° is outside the model's original 17.81–23.12° training range. The same model remains about 3.57° wrong on five near images; repeated observations alone cannot fix stable bias. Oracle contact is plug/module against the NIC opening, not evidence of a cable snag.

**Next gate:** audit existing SFP grasp coverage, add a bounded perception-only extension only if needed, and train to original-plus-new-domain convergence before fresh live checks and three unchanged public samples. Automatic V34 repeats stop at the physical failure. No new capture/training job is running. Final scenes stay sealed. The SFP stiffness-only draft was not run and was removed after this diagnosis.

**The perception model is not generally qualified.** The fit with 80 near training grasps converges at 1.026° p95 on four original near validation episodes, above the 1° gate. Independent replay on four other frozen rollouts gives 0.717° orientation p95 and 0.129 mm position p95. Additional feature and augmentation fits do not improve the original result. Runtime experiments therefore keep `pose_gate_passed=false` and require an explicit, limited case scope. Final scenes remain sealed; no new RL, actor-Q or imagination training is enabled.

[Results, commands, videos and failures](experiments/2026-10-18-sc-explicit-controller-continuation.md). Current machine states: `current_controller_state.json`, `V27_three_sample_state.json`, and `V32_lateral_stiffness_failed_case/` / `V33_lateral_stiffness_failed_case/` under `/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/`.

> **October 18 SC follow-up:** the latest local-backoff pilot receives toolkit75 but reaches15.431mm depth/0.373mm lateral/1.124degrees; it **fails** the unchanged physical gate. Public discussions confirm other teams saw partial-insertion and repeated-trial collision problems. Both announced fixes are already in local source ancestry; the frozen eval binary provenance remains unverified. One-point returns in our full sample arise from an unsuccessful attempt, not proof of a scoring bug. [Evidence and bounded next diagnosis](experiments/2026-10-18-three-trial-insertion-repair-cycle.md#sc-follow-up-completed-and-public-discussion-checked).

> **October 18 insertion repair:** Control-only changes have not established better insertion. Initial expanded SC pose fits reached LR-floor validation plateaus but fail new-grasp accuracy; the public sample nominal grasp lies outside all three original SC initialization-angle ranges. Targeted perception-only SC captures and calibrated-pixel/balanced fits are in progress. Public controller context is a concrete actor hypothesis, but persistence already explains much of its offline gain. No new model is promoted. [Repair record](experiments/2026-10-17-insertion-control-and-grasp-repair.md).

## October 18 milestone: full SFP insertion and score timing repaired

The unchanged predeadline public three-trial sample was run twice from its original home. The explicit RGB/public-state insertion controller scored **152.61/300 before the seating-hold fix and 176.85/300 afterward**. The combined physical-and-toolkit gate improves **1/3 → 2/3**. Both SFP trials physically insert to about46.76mm; the remaining SC trial stops at about2mm with2.17degrees orientation error. The second result is not reliable all-task insertion.

The key SFP fix is holding publicly estimated stable seating for2seconds rather than0.5seconds. The toolkit needs1second of uninterrupted contact before publishing completion. The old callback sometimes returned first, despite correct terminal geometry. A matched five-scene development check improves1/5 → **5/5** combined successes; it includes the four previously partial-scored episodes. No model weights or physics were changed.

A separate latency repair shares an identical JPEG86copy between the grasp-position and grasp-orientation estimators. Actual frozen-estimator outputs are bit-exact in the equivalence check. Three live SFP repeats all insert, with fresh-acquisition p95 **219.6/222.4/227.7ms**, below300ms. These are three repeats of one development configuration, not broad latency certification.

These successes use **an explicit image-based insertion controller, not the learned insertion BC actor**. The completed masked/public-controller-feedback BC comparison is0/8; its matched no/local/hierarchical-backoff screen is0/12 with no retreat triggered, because those runs stop short with low force. No newRL was trained. SC now has a separate loaded-contact recovery diagnostic: the official one-second contact latch can freeze a shallow plug; public tracking compensation and fixed-goal lateral control are needed to execute an actual retreat. Its full-insertion gate remains open.

[Commands, failures, videos, controller versions and next gates](experiments/2026-10-18-three-trial-insertion-repair-cycle.md). Bulk artifacts: `/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1/`. Final scenes remain sealed.

**October 18 active continuation:** finish the frozen backoff screen, then unchanged three-trial sample score → targeted trace/video diagnosis and development repair → unchanged three-trial rescore. The first candidate uses an explicitly labeled RGB/public-state controller, not a learned actor. Seating dwell is the first bounded repair hypothesis; failed SC estimators remain unpromoted. [Plan, gates and running schedulers](experiments/2026-10-18-three-trial-insertion-repair-cycle.md).

**October 18, 02:59 UTC:** current training fits have completed their LR-floor validation plateaus; no SC grasp candidate passes its accuracy gate. The frozen masked/public-controller-context BC comparison finishes **0/8combined insertions**. Fresh SFP explicit image-controller evaluation finishes **6/10combined successes**(port0:5/5;port1:1/5), with all10terminal geometry checks passing but four toolkit partial-insertion results still requiring diagnosis. This is not BC learning. The matched no/local/hierarchical-backoff screen is active,2/12scored,0successes and no completed retreat yet; do not infer recovery effectiveness from an enabled but untriggered supervisor. Last complete original sample remains165.4180/300. [Detailed status](answers/2026-10-18-insertion-repairs-status-0259.md).

**October 18 follow-up:** finish the current insertion/grasp repairs, then compare no backoff, local backoff and hierarchical measured-path backoff on frozen development starts. No newRL or final-scene tuning. Full physical insertion and force/harm determine improvement; retreat alone does not. [Protocol and current repair details](experiments/2026-10-17-insertion-control-and-grasp-repair.md#october-18-continuation-backoff-comparison-after-the-current-repairs).

> **Completed control-only screen:** V1retains1/4strict insertions and introduces an unintended-retreat bug; V2fixes that bug and reduces handover jumps to0.216–0.800mm but achieves0/4insertions. Neither is promoted. Four expanded-source SC pose candidates also converged without passing new-grasp accuracy. Targeted coverage and matched actor-context fits continue. [Execution record](experiments/2026-10-17-insertion-control-and-grasp-repair.md).

## Major milestones — transport, handoff and insertion (October 17)

This summary is the current interpretation; dated running/queued updates below are historical. **Transport/handoff success and full insertion are different gates.** The four-rollout diagnostic has finished; the [unchanged-sample full rollout](experiments/2026-10-17-full-connector-sample-configuration.md) has completed at **165.42/300 points but 0/3 strict insertions**, with no retraining or final-scene tuning.

### 1. Rule-based transport became the working baseline

Learned transformer/diffusion transport did not reliably reproduce the desired route across starts. We selected explicit transport rules plus visual acquisition, rather than continuing to depend on diffusion for the route. **SC goes around the card row:** back away, move sideways while descending, travel beyond the row, cross over beyond its end, tilt, then return toward the selected port. SFP retains the successful lift/transfer/orient structure. Phases advance from measured robot motion, with bounded tracking compensation; early vision and viewpoint changes handle startup variation. This is observation-assisted rule transport, not a memorized exact port coordinate or a learned residual-action adapter.

The final development screen reached **7/7 SC and 22/22 SFP qualified handoffs**. Frozen evaluation reached **119/124 sample-derived cases**: SC **22/24**, SFP **97/100**, with zero wrong accepts or scored harm. SC covers 0–5 obstacle cards and both ports; SFP covers 1–5 cards, both ports and all five requested NIC identities, two repeats per setting. A separate exposed post-release test was **2/6**, giving **121/130 combined**. Keep these domains separate: sample success does not establish reliability on released evaluation scenes. This was a staged engineering improvement, not a matched experiment isolating rules versus diffusion. [Rules, revisions and evidence](experiments/2026-10-16-sample-conservative-rule-transport.md).

### 2. Handoff became a visually checked decision

Task identity selects the requested SC/SFP port and, for SFP, the requested NIC card. Coarse perception proposes its image region; the original-resolution crop refiner and calibrated multi-view geometry estimate its pose. Repeated consistent requested-port estimates and a measured stable hold are required before accepting. Bounded viewpoint reacquisition replaces blindly accepting a weak estimate. Refiner retraining used **72,857 RGB triplets from 497 source episodes**, followed by targeted SFP acquisition and SC rail-coverage repairs. A single-point crop-margin relaxation was rejected after accepting an absent target; the replacement uses five-landmark geometry. These changes and controller repairs produced the handoff results above; no one model receives sole causal credit.

Those counts concern the roughly **150 mm acquisition hold**, not readiness to insert directly. The latest pipeline adds a visual/proprioceptive approach to roughly **25 mm before the opening**, then starts the learned actor. The earlier plan for the actor to own the whole 150 mm has been superseded for this diagnostic. All four latest runs reached that near handoff, but insertion remains a separate challenge. [Acquisition repairs](experiments/2026-10-15-rule-perception-hierarchical-execution.md), [near-handoff audit](experiments/2026-10-17-near-handoff-and-insertion-continuity-audit.md).

### 3. Full connector pose produced a strict SFP insertion

The previous representation combined plug-tip position with gripper orientation. We trained RGB estimates of **the plug's rotation relative to the gripper**, retrained the actor with full plug pose relative to the port, and decoded its commands using the full estimated grasp transform. Port/grasp estimates remain fixed during an attempt. Simulator geometry supplies teacher/perception training labels and independent scoring, **not autonomous policy inputs**. Later privileged teachers demonstrate alignment before advancing; successful demonstrations do not imply that all observations were perfectly aligned, or prove learned jam recovery.

The new deterministic H4 actor uses **60,671 anchors from 499 usable episodes** (420 train/79 validation), current three-camera RGB and 1.2 seconds of robot history. Training stopped on a validation plateau at **13,400 updates**, selecting update 7,400. Four Gazebo rollouts, two repeats of one one-card setting per family, achieved **SFP 1/2 and SC 0/2 strict physical insertions**. The successful SFP reached **46.75 mm depth, 0.48 mm lateral error and 0.46° orientation error**. An older near-25 mm policy already reached 46.78 mm depth but failed orientation at 2.55°; do not call this the first full axial advance. Older controls also started near 25 mm, so proximity alone does not explain the new result. These are encouraging diagnostic findings, not broad insertion reliability.

SFP early command-boundary jump p95 decreased **15.4 → 4.4/3.2 mm**. No smoothing filter, longer history, RL or backtracking was added in these tests. The combined estimator/representation/decoding change is a plausible cause, not an isolated orientation ablation. SC smoothness did not consistently improve. Gazebo fixes the connectors and later the gripper after its touch-triggered completion event, including shallow SC events: frozen video tails are not learned stabilization. [Four-rollout results, videos and limitations](experiments/2026-10-17-full-connector-insertion-jerk-probe.md).

### Latest full sample check: acquisition passes, insertion still fails

One unchanged original-home `sample_config.yaml` run gives **165.42/300**: SFP48.63/35.48, SC81.31. All three reach the25mm actor handoff with opening-position errors0.08–0.14mm, but **0/3 meet strict physical insertion**. SFP stays at the mouth with3.50/2.40mm lateral error despite<0.5degree orientation error. SC gets the toolkit completion credit at only1.97mm depth/1.49mm lateral/2.43degrees and is then simulator-locked. Its near handoff already has3.36degree plug tilt; grasp estimation is2.04degrees off, and the approach checks nominal TCP posture. All three lose duration points. This limits the earlier1/2SFP success claim: it is a milestone, not established reliability. [Scores, causal geometry, contacts, images and repair priorities](experiments/2026-10-17-full-connector-sample-configuration.md).

### October 18: image-only insertion control is feasible on the SFP diagnostic

Keeping the same visual port/grasp estimates, an explicit alignment/advance controller achieves **2/2 strict SFP insertions** on two repeats of one reused one-card/card0/port0 start: **46.77 mm depth, 0.31–0.35 mm lateral error and 0.34–0.37° orientation error**, no scored harm, about45simulated seconds. Earlier BC was1/2on this setting; the corrected command-limiter BC screen was0/2. This is a small controller-feasibility result, **not a trained-actor improvement or broad reliability claim**. NoRL or backtracking is used. One cold acquisition narrowly exceeds300ms; a prewarmed repeat is running. The unchanged complete-sample result remains165.42/300with0/3strict insertions. [Controller comparison, videos and SC coverage repair](experiments/2026-10-17-insertion-control-and-grasp-repair.md).

### Remaining gate

**Step 5 remains open.** SC live grasp-angle errors around 2.5° expose a coverage gap. Controller-to-actor command jumps, forward references running ahead during blocked contact, and SFP early stopping remain. Hierarchical recovery exists in earlier runtimes but was **disabled in the latest four runs** and remains unintegrated with the near-25 mm diagnostic. Do not attribute its success to recovery or claim new SERL training. The sample assessment is complete. Next target command continuity, blocked forward demand, lateral teacher/control context and SC grasp coverage; reserved final scenes remain sealed.

> **Latest October17 four-rollout probe:** **1/4 strict physical insertions**: SFP1/2, SC0/2, on two reused one-card configurations. Official tier3=75 in three runs, but both shallow SC contacts triggered simulator completion/freeze at~2mm depth. Exclude frozen intervals from smoothness claims. SFP first15s boundary-jump p95 improves15.4→3.2–4.4mm; abrupt handoff steps remain(SC14–19mm,SFP4–8mm). SC live grasp-angle errors~2.5° expose coverage outside training. All rollouts/videos complete; no more rollout or training queued, noRL, step5/final promotion gate open. [Detailed results, videos and proposed fixes](experiments/2026-10-17-full-connector-insertion-jerk-probe.md).

> **Latest full-connector BC status:** Training finished at a validation plateau after **13,400 updates**, best checkpoint7,400. Relative-to-expert near-port orientation p95 improves SC2.28°→0.77°, SFP2.28°→0.28°. Command-position results are mixed: SFP improves, SC worsens. **No new autonomous rollout has run**, so insertion improvement remains unestablished. Runtime integration/stable orientation acquisition is the next gate; no training/evaluation process is active. [Record](experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md).

> **Latest October 17 result:** Image-trained connector orientation improves to angular p95 **0.89° SC (21 held-out episodes)** and **0.49° SFP (47)**. Retain the more accurate existing visual tip translation at acquisition. Saved SFP traces confirm oscillation with hardcoded recovery disabled; explicit teacher phase/integrator state creates an additional context gap. A single full-connector-pose H4 BC diagnostic is queued/running on GPU4 using60,671unchanged command anchors; no insertion improvement or runtime orientation approval is claimed. [Results, limitations and next gates](experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md).

> **October 17 connector-orientation continuation:** The 14-run history/horizon comparison finished with **0 strict full insertions** on two reused development configurations. SFP reversals also occur with recovery disabled; do not blame all back-and-forth motion on hardcoded backtracking. Explicit RGB-only connector-in-TCP pose fits are now running for SC and SFP on their existing episode/reset splits, with held-out keypoint and constant baselines. No new policy is promoted. [Plan, diagnosis and artifacts](experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md).

> **October17 active continuation:** Preserve measured transport/approach history into insertion; keep the accepted port frame fixed per attempt, with fresh3–5pose consensus at acquisition. Train matched **4/8/12-step** models plus an8-step cross-plan-consistency model from60,671common recorded-command anchors. GPUs3/4 train; GPU5 handles sequential runtime checks; legacyGPU0 remains untouched. No newRL or final-scene access. [Implementation, PoseInsert review and experiment plan](experiments/2026-10-17-history-horizon-poseinsert-continuation.md).

<!-- history-horizon-live:start -->
## Live history/chunk experiment

- Stage: `frozen_near_port_live_comparison`. Completed training variants: **4/4**.
- New trained-model live comparison: **14/14** complete; no insertion promotion inferred from training.
- Runtime preserves measured transport history; accepted port pose stays fixed during the actor attempt. New acquisition uses fresh pose-window consensus.
- Same60,671anchor samples for4/8/12step targets; recorded commands only. No newRL or reserved-final testing.
- Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1`.
<!-- history-horizon-live:end -->

> **October17 revised priority:** Transport now targets a visually/proprioceptively controlled **25mm stand-off**, superseding the earlier learned150mm-approach responsibility. The first SC pilot reaches24.20mm/0.547mm lateral, but the unchanged actor still fails full physical insertion. Audit confirms all four50ms insertion targets execute with zero insertion-stage stale drops; large replanning jumps, phase imbalance and unapproved plug-grasp orientation remain gaps. A separate four-run fresh/warm SC/SFP comparison is active on GPU5, recovery off to isolate the approach/actor; no new training or official tuning. [Audit and next decisions](experiments/2026-10-17-near-handoff-and-insertion-continuity-audit.md).

> **October17 insertion/recovery diagnosis:** On37actor-executing scenarios, port-pose errors are small but SC loses alignment and SFP often stays far from the opening. Tenepisodes enter backtrack; none completes it (six timeouts, two force-rise aborts, two task-budget endings). Repeated tiny retreat targets fail waypoint progress under load; this is a separately identified recovery execution gap, not evidence of cable snag. [Videos and analysis](answers/2026-10-17-evaluation-videos-and-insertion-failure-analysis.md).

> **October17 full-rollout status:** **0/40 full insertions** (SC0/12; SFP0/28). Coordinator stopped on the next SFP episode because independent terminal lateral geometry differs by1.437mm, above the unchanged1mm audit gate. No sample/eval complete-YAML scores yet; no process is continuing this batch. Resolve causal scoring before resuming. [Status details](answers/2026-10-17-full-rollout-status.md).

> **Full-rollout scope update:** One episode per scenario, **62 total**, supersedes the earlier 186-episode schedule. Separate complete sample/eval YAML tests remain three repetitions each. The current rollout finishes before the scheduler hands over; models, rules and scenes remain unchanged. See [evaluation record](experiments/2026-10-16-full-pipeline-three-per-scenario.md).

<!-- full-insertion-evaluation-live:start -->
## Full insertion evaluation continuation

- Stage: `scenario`. Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1`.
- Frozen rule transport → requested-port coarse/native refinement → shared 150 mm BC insertion actor.
- Planned: 62 scenario rollouts (62 groups × one), plus three complete sample YAML runs and three complete released-evaluation YAML runs.
- Hierarchical insertion recovery is rule-based; no new RL or training. Released evaluation is scoring only; reserved final scenes stay sealed.
- Counts and scores: see `progress.json`, `results_by_scenario.json` and `configuration_scores.json` in the artifact folder.
<!-- full-insertion-evaluation-live:end -->

<!-- sample-transport-live:start -->
## Live sample-derived transport results — 2026-10-16T20:25:57 UTC

- Adaptive SC development: **7/7 qualified**, seven planned.
- Unchanged SFP development: **22/22 qualified**.
- Frozen transport/handoff evaluation: **121/130 qualified**,130planned. This is not insertion.
- Rules/weights frozen: **True**. Stage: `frozen_transport_evaluation`.
- Decision: `frozen_evaluation_complete_requires_review`. No new training or reserved-final testing.
- Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3`. Sample-derived and exposed official results stay separate; no evaluation tuning.
<!-- sample-transport-live:end -->

<!-- sample-route-repair:2026-10-16 -->
### Current transport state: development gate passed; frozen evaluation running

Sample-derived development passes **7/7 SC and22/22 SFP handoffs**. The selected rule follows the original sample-based corridor, but can retrace measured TCP poses to an earlier checkpoint, switch sides and reacquire the port when blocked. A separate live trial exercised this recovery and handed off at155mmstand-off/2mmlateral. **Backtracking did not trigger in the seven-case SC screen**, so its7/7 result is not a measured causal benefit estimate; cable outcomes vary across repeats.

Rules and vision weights are now frozen. **130 transport/handoff evaluations** are running:124freshsample-derived cases with two episodes per complete SC/SFP/card-count/port/NIC-target setting, plus six separately labeled exposed official-scene repeats. No tuning follows those results. No insertion or model training is run; reserved internal final scenes remain sealed. Live artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3/`. [Full comparisons, actual backtracking bug, gates and video paths](experiments/2026-10-16-sample-conservative-rule-transport.md).
<!-- /sample-route-repair:2026-10-16 -->



<!-- evaluation-source-policy:2026-10-16 -->
## Development source correction — October 16

**Use the pre-deadline `sample_config.yaml` and task specification to develop routes, data and thresholds.** The separate `eval_config.yaml` was released after qualification and must not guide repairs. Its exact scenes **and derived stress scenes** are excluded from development. Existing frozen evaluation may finish as descriptive evidence; do not use it for training or tuning, and do not call the already inspected scenes unseen final tests. Audit earlier official-anchor data/model lineage before claiming this protocol is met. See [the evaluation protocol](EVALUATION_PROTOCOL.md), which supersedes earlier “work backwards from official settings” recommendations. Pinned public sources are under `configs/evaluation_protocol/qualification_public_predeadline/`. No new training is running.
<!-- /evaluation-source-policy:2026-10-16 -->

# Current experiment status

## October 16: official-setting transport recheck running

The recent strong acquisition screens use **rule-based transport plus learned requested-port perception**, not the newly trained insertion actor. Familiar-board success is not official-setting coverage: the earlier alternate-public-board screen was 0/10. A frozen **120-rollout** recheck is running: five repeats of each of the three unchanged published qualification scenes (15), five matched SC refiner-baseline repeats, then 100 separately labeled released-range card-count/port variations. GPU3 renders and GPU4 runs perception; no training or insertion actor is enabled. Reserved final scenes remain sealed. [Protocol, training inventory and limitations](experiments/2026-10-16-official-transport-recheck.md). Live artifacts: `/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1/{progress.json,progress.md,runner.log}`.

**Gate bookkeeping correction:** the prior corrected acquisition20 numerically meets its declared ≥90% per-family/no-wrong-accept criterion (9/10 each). Its summary incorrectly required a wrong-port flag of `False` even on the safely rejected SFP episode, whose flag was correctly `None` because nothing was accepted. The old failed artifact is preserved; `corrected_intended_perception_gate_bookkeeping_audit_v2.json` records the correction. The physical SFP stall and SC latency miss remain failures. This correction does not establish broad transport coverage or insertion success.

<!-- automatic-live-progress:start -->
## Latest recorded job state — 2026-10-16 07:24 UTC

This block supersedes earlier running/queued wording. Acquisition counts are **not insertion counts**. Completed update budgets are not automatically convergence.

- **BC fresh**: stop_reason=update_cap, updates_completed=30000, best_step=4600, stale_evaluations_at_end=254, summary_exists=True.
- **BC visual warm**: stop_reason=update_cap, updates_completed=30000, best_step=23100, stale_evaluations_at_end=69, summary_exists=True.
- **SC frozen acquisition new refiner effective**: completed=60, planned=60, qualified=44, wrong=0, harm=0, passed=False, episodes=60, wrong_accepts=0, summary_exists=True.
- **corrected intended acquisition20**: completed=20, planned=20, passed=False, family_groups={'sc': {'episodes': 10, 'qualified': 9}, 'sfp': {'episodes': 10, 'qualified': 9}}, episodes=20, qualified=18, wrong_accepts=0, summary_exists=True.
- **SC candidate acquisition20**: completed=20, planned=20, passed=True, family_groups={'sc': {'episodes': 20, 'qualified': 18}}, episodes=20, qualified=18, wrong_accepts=0, summary_exists=True.
- **autonomous insertion comparison**: status=perception_gate_failed, summary_exists=True.
- **conditional BC continuation**: extended=False, summary_exists=True.

Full-system promotion, new RL and reserved-final testing remain pending. Exact commands, weights, metrics and preserved failures are under `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1`.
<!-- automatic-live-progress:end -->


<!-- current-continuation:start -->
## Current approved continuation — October 16

Rule transport brings either SC or SFP to a visual acquisition hold about 150 mm before the opening. A fresh requested-port pose and RGB-derived plug offset then initialize **one shared learned actor for the entire 150 mm approach and insertion**. Large hierarchical retreats return along measured poses to that acquisition hold and require fresh perception. The old 30 mm controller is historical.

**Training is complete:** matched ImageNet versus visual-only transport initialization, same 13.725M full-target transformer, 30,000 updates per arm. Neither showed practical late validation improvement, so the conditional continuation started no extra training. Actor data contain **67,039 examples from 500 usable source episodes / 377 physical reset configurations**: SC 102/30 train/validation episodes, SFP 319/49. All wider-source admission and estimated-frame audits passed. Predictions are complete recorded command targets in the visually estimated port frame, with causal robot/pose history and no privileged geometry input. Completed-actor autonomous insertion remains unevaluated; its original queue stopped on the acquisition summary.

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


**October 15 execution update:** Broad SC/SFP native refinement now improves opening-position accuracy: conditional SC lateral p95 is 0.20 mm; the stronger SFP coarse pipeline plus the new refiner improves lateral p95 from 0.82 to 0.29 mm on 70 validation recordings, with zero observed wrong-port accepts in that comparison. Training continues because the first 12,000-update fit was still improving. Coarse identity repair remains experimental: rejecting known wrong handoffs also rejects some correct ones. Wider 150/180 mm SC teacher pilots now reach full seating after measured alignment and a corrected completion condition; broader data admission is still gated. New insertion BC, deployed hierarchy and RL have not passed their gates. [Evidence and active execution](experiments/2026-10-15-rule-perception-hierarchical-execution.md).


**October 15 approved continuation is active:** rule transport for both families, mandatory broad SC/SFP native refiner retraining, coarse identity/coverage audit, shared insertion actor owning the full 150 mm approach, and hierarchical retreat with reacquisition. Proceed through measured BC and recovery gates before RL. [Plan and execution state](experiments/2026-10-15-rule-perception-hierarchical-execution.md).

Runtime clarification: [exact execution order, frames, perception, handoff and test definitions](answers/2026-10-15-runtime-rollout-explained.md). The frozen SC branch used rule transport and an older verifier; SFP used learned transport without connected handoff/insertion.

**October 15: expanded frozen evaluation complete — no promotion.** Exactly 140 rollouts on 120 fresh development configurations, 28 groups with five episodes each. Primary SC rule transport + older camera check/refiner + visual approach + existing BC insertion actor: 23/50 visual approvals (three wrong ports), 3/50 actor starts, **0/50 full insertions**. Matched far-start control: **0/20**; alternate-board SC: **0/10**. SFP tests are learned transport only, with no integrated coarse/insertion: 0/50 primary and 0/10 stress visits to even the declared broad diagnostic region; full SFP insertion is N/A. Many correctly acquired SC scenes reach roughly 30 mm stand-off but fail the stricter approach-readiness test. Models were unchanged, no training/RL/backtracking started, final scenes stayed sealed. Small earlier acquisition screens and the one 11 mm pilot do not establish reliability. [Complete evidence and limitations](experiments/2026-10-14-frozen-pipeline-coverage.md) · [Easy explanation](answers/2026-10-15-frozen-coverage-explanation.md).

**October 14 current execution:** Supported normal-board SC rule transport and RGB handoff passed small screens. Matched corrective-data, decoder and 6-second robot-history repairs did not pass autonomous full insertion. A separately frozen observation-only approach reached **29.12 mm before the opening / 0.148 mm lateral**, then the unchanged trained actor achieved **11.07 mm insertion depth / 0.413 mm lateral / 0.94° orientation error** on one reused start. Official score 75, but the unchanged ≥14 mm physical gate failed. Logs confirm contact-triggered completion and end-effector locking; no cable snag explains this run. This is meaningful approach progress, not a promoted insertion policy. All new jobs finished; no routine collection, RL, world-model work or final-scene access. SFP local insertion, rotated SC and port-0 rail minimum remain unapproved. [Execution record](experiments/2026-10-14-transport-handoff-execution.md) · [Easy explanation](answers/2026-10-14-autonomous-execution-outcome.md).


**October 14 initial checkpoint (superseded by current execution above):** The acquisition replay accepted 6/6 recorded SC teacher transport endpoints, with 0/6 alternate-request accepts and 0.11–0.21 mm opening error. This is reused-image component evidence, not autonomous insertion. Timestamp reconstruction recovered 1,906 complete 12-step SC chunks and retained all 61 stop examples; all 104 episodes retain final-return action supervision. A conservative maximal corridor failed rigid reach/planning; a tighter common corridor passed six training-only rigid screens. Gazebo cable validation and the remaining direct/chunk executor pairs are in progress. First pair matched the original phase boundary within 2.0/3.3 mm, but both shifted 11–13 mm after an eight-second final-target hold. No new actor training, automatic insertion or RL promotion yet. [Execution record and durable state](experiments/2026-10-14-transport-handoff-execution.md).

**October 14 consolidated plan:** The current next-work entry point is [Transport and handoff improvement plan](TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md). It adds an early SC rule-based Low Lateral Bypass comparison: conservative corridor or observation-adapted waypoints, measured stage completion and visual handoff approval. This remains an untested autonomous candidate; the existing teacher retargets with privileged geometry. Terminal-label/executor repair and the learned-actor comparison remain in the plan. Analysis only; no new training or rollouts started.

**October 13 independent transport review (analysis only; supersedes the earlier immediate correction-collection recommendation):** The SC training/evaluation boundary was mismatched: **0/104 admitted transport prefixes** enters the old −100 to −70 mm TCP approach band; their median terminal TCP axial coordinate is −168 mm (plug tip −148 mm). Later teacher approach actions were excluded. The 12-step filter also leaves **29/83 training episodes without nominal final-return-stage supervision** and removes every cached handoff-positive row from its action subset. The diffusion branch has no stop head and sets handoff probability to zero. Rescoring existing runs at the actual taught endpoint gives fine70 **5/6** versus dense70 **3/6** transient visits within 20 mm/5°, followed by drift; these posthoc visits are **not autonomous handoffs or insertion successes**. Same-clock early separation also confounded speed and route error. Next: validate the component boundary, repair timestamp/terminal supervision and test the executor before another model/data sweep. No new training, simulation, or GPU inference; final scenes stay sealed. [Evidence, research and bounded plan](experiments/2026-10-13-transport-independent-root-cause-review.md).

**October 13 SC diffusion code, DINO and cadence audit (complete; no promotion):** The existing 100 ms dense-history actor remains the stronger six-scene transport baseline but has 0/6 approach arrivals. Saved-input replay of a newer model matched ten live proposals within 0.065 mm maximum, and target-frame reconstruction was essentially exact; no checked action-frame or preprocessing bug explains drift. A newly built 0.9 s decision cache initially exposed an auxiliary-label bug (zero valid labels); the corrected fit reached a validation plateau and, on three matched nonfinal Gazebo scenes, got closest to the port at 142/170/143 mm versus the frozen actor's 141/190/129 mm, with **0/3 approach bands, 0/3 insertions, no contact**, and first teacher-route departure near 6–7 s. Its inference p95 was 370–420 ms. An empty sampler-step environment variable also caused an invalid launch before commands; runtime fallback was fixed and all valid results come from the replacement launch. On 83 train/21 held-out SC episodes, a frozen DINOv2-S global or 4×4 spatial feature did not beat a state/task-only linear route probe, and it increased total actor parameters from 25.47M to 36.35M. DINO is **not promoted as the default**. Next: a three-training-reset safe SC LLB actor-state continuation feasibility pilot with a ≥2/3 no-contact route gate; then a matched corrective-data actor only if the pilot passes. No speed editing, handoff, insertion, RL or final-scene use. [Full code/vision/cadence audit, commands, artifacts and video](experiments/2026-10-13-sc-diffusion-code-vision-cadence-audit.md) · [plain-language conclusion](answers/2026-10-13-diffusion-history-code-vision-conclusion.md).

**October 13 SC actor drift follow-up (complete; no promotion):** The board/cable disappearance in previous videos was camera capture continuing through normal Gazebo trial teardown. Trimmed actor-only viewing copies preserve the board and cable. A 38-request fine-history model reached a held-out plateau but on six matched SC development scenes still had **0/6 approach-band arrivals and 0/6 insertions**; median closest TCP-to-port distance was **141 mm**, versus **135 mm** for the prior 27-request dense model. The extra requests yielded only 14 distinct training frames at late times because the cleaned cache is sparse around holds. A phase-balanced late-route fit and a faster six-step sampler each missed **0/3** approach bands. A deliberately memorized 38-history actor reached **0.49 mm training action p95** after three warm-start stages and a train plateau; its final matched three-scene rollout remained **0/3** approach bands and insertions. The second-stage model scored **0/6** on all six layouts. First actor/teacher divergence remains near 7–9 s, including on two training scenes. The old dense actor remains the stronger six-scene route baseline; it still fails the transport gate and 300 ms inference limit. Speed cleanup is deferred until route quality passes; the six successful matched teachers took 63–101 s and earned no duration bonus. [Full analysis, trimmed videos, commands and decision](experiments/2026-10-13-sc-history-drift-followup.md) · [machine summary](../outputs/experiments/2026-10-13_sc_pause_models_v1/history_drift_followup_summary.json). Step 5, automatic handoff, insertion, RL, and final scenes remain gated.

**October 13 SC pause and memorization check (complete; no promotion):** The active 83-episode SC training copy already removed 19,570 quiet hold rows from 89,251 raw transport rows; the 21 grouped validation episodes retain their holds. The old actor's 60 s history spans the scripted 8 s and 5 s pauses. We evaluated a SC-only 100 ms 12/6 actor with that history, a genuinely longer **70 s / 27-sample** variant, and a separate SC-only 50 ms actor fitted to **0.91 mm training executed-chunk p95**, against the frozen 12/6 actor on six identical nonfinal SC layouts. All four reached **0/6 approach bands and 0/6 insertions**, with no scored contacts. Median closest TCP-to-opening distances were **190 / 162 / 135 / 137 mm** (old / 100 ms / dense 70 s / intentional fit). The intentional-fit actor also failed 0/2 training-scene approach bands. Live inference p95 remains about 393–416 ms, above the 300 ms limit. This favors a closed-loop route/progress and cadence repair over more pause cleanup or teacher-state fitting. [Full experiment, videos and commands](experiments/2026-10-13-sc-pauses-context-and-overfit.md) · [machine comparison](../outputs/experiments/2026-10-13_sc_pause_models_v1/live_comparison_v2.json). **Step 5 remains open**; final scenes, automatic handoff, insertion and RL remain gated.

**October 13 SC 12/6 reevaluation (complete; no promotion):** The frozen 12-predict/6-execute actor was rerun on its three prior nonfinal SC scenes and tested on three additional development layouts covering one, two, and five cards and both ports. Across the six new runs: **0/6 axial approach-band entries, 0/6 safe buffers, 0/6 insertions, and no scored contacts**. The six successful matched teachers completed the route; actors departed from teacher motion within 6–9 simulation seconds and usually remained 160–266 mm TCP-to-opening 3D away at their closest point. Nearly all later issued references pointed axially away from the port. Exact saved-input replay matched early model outputs, while substituting only the teacher previous command sharply reduced early target gaps on two scenes. A 0.25–0.30 s between-chunk command gap also mismatches the 300 ms training decision cadence. This is a transport route/progress and feedback failure, not evidence of blocked insertion or cable snag in these rollouts. [Detailed analysis, plot, videos and limits](experiments/2026-10-13-sc-horizon12-reevaluation.md) · [machine summary](../outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/summary.json). Final scenes, automatic handoff, insertion and RL remain gated.

**October 13 longer transport action horizons (complete; no promotion):** The shared route-diffusion actor was trained on the full admitted 533 SFP / 83 SC training episodes to predict **8 targets and execute 4** at 50 ms each, and separately **12 targets and execute 6**. A new 300 ms feature cache matched the 12/6 decision cadence; both horizons retained valid long labels in every training and development episode. The 8/4 model reached a development plateau after a lower-rate continuation; 12/6 needed two capped phases and a third lower-rate phase to plateau. On matched held-out teacher rows, first-four-command translation p95 improved from **3.15→2.58 mm SFP, 2.75→2.64 mm SC** for 8/4 on 200 ms rows, and **3.63→2.37 mm SFP, 3.73→2.53 mm SC** for 12/6 on 300 ms rows. A runtime audit found the old scheduler discarded most chunk members when inference was slower than the chunk. A corrected simulation-clock comparison issued exactly 4/4/6 commands per decision on six identical nonfinal scenes. **All three variants reached 0/6 safe approach buffers and 0/6 insertions, with no scored contacts.** The 12/6 actor reduced SFP lateral miss on all three scenes (to 19.07/43.20/11.52 mm), but no scene met the 10 mm buffer and SC did not improve route progress. A bounded four-denoising-step check did not rescue the close SFP scene. No final scenes, handoff, insertion policy, or RL. [Full result and training curves](experiments/2026-10-13-route-action-horizon.md) · [machine summary](../outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json).

**October 12 transport feedback and cadence follow-up (complete; no promotion):** Replaying saved inputs reproduced live commands within 0.30 mm on four nonfinal SFP/SC traces. Swapping only the previous teacher command reduced second-decision gaps from 6–19 mm to 1–3 mm, implicating teacher-command feedback at deployment. Zero-prior and delayed-prior fine-tunes improved selected offline errors but failed the frozen actor-state gate; neither ran live. Four diffusion steps lowered matched live inference p95 to 241 ms SFP / 222 ms SC and reached one SFP geometric buffer, **with tool/card contact**; SC stayed far away. There was **no full insertion**. The initial supposed JPEG-86 improvement was an unchanged-policy repeat: the launcher had not forwarded that environment variable. True-codec pilots established no reliable gain. Next is a bounded causal route-progress or safe teacher-continuation repair, gated on ≥50% early action-gap reduction on all four frozen scenes before new autonomous starts. Handoff, insertion, RL, and final scenes remain gated. [Full follow-up](experiments/2026-10-12-route-feedback-and-inference-cadence.md) · [machine decision](../outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/feedback_cadence_decision.json).

**October 12 teacher-generator diagnosis and released-eval recheck (complete; no promotion):** Six held-out exact teacher scenes (three SFP CheatCode varied-start, three SC Low Lateral Bypass) had 6/6 successful, no-contact teacher insertions. The route-consistent autonomous diffusion actor had **0/6 insertions**: SFP missed the approach buffer in all three, and SC deviated from the teacher route and did not progress to insertion. One SFP run contacted card 4; none of the three SC runs scored contact, so the reviewed SC failures are not demonstrated cable snags. The first supposed SFP JPEG-86 ablation only set a launcher manifest field and was an unchanged-policy repeat; the corrected runs are reported above. The old SC 10 mm TCP buffer is not a valid universal gate: 10/21 fully inserted held-out LLB teachers miss it. [Full diagnosis, commands, videos, and scores](experiments/2026-10-12-teacher-distribution-transport-diagnosis.md).

**October 12 route-consistent diffusion retrain (complete; not promoted):** The shared transport actor was retrained for 6,000 updates on the active 533 SFP / 83 LLB-only SC training episodes, starting from a pre-mix checkpoint. The new 14.29M trainable-parameter checkpoint improved matched held-out first-action p95 from 4.21→2.64 mm on SFP public-anchor rows and 2.90→2.74 mm on LLB SC rows. In a frozen 12-rollout Gazebo comparison (three starts per family for each checkpoint), **both checkpoints reached 0/3 safe transport buffers in SFP and 0/3 in SC**. The candidate had one SFP gripper/card contact; one SC run had complete inference p95 354.7 ms, above the 300 ms limit. The new checkpoint is an offline diagnostic, not the deployed transport actor. No insertion handoff, RL, or final-scene evaluation followed. [Full fit, matched scores, and artifacts](experiments/2026-10-12-route-consistent-diffusion-retrain.md).

**October 12 data-provenance correction (complete for current data):** The 22 new SC CheatCode training prefixes were **removed from the active union** because their cable route was not verified as Low Lateral Bypass (LLB). The active `/home/nas/chmin/aic_archives/2026-10-12_route_consistent_union_v4/` contains **533 SFP and 83 LLB-only SC training episodes**, with 43/21 grouped development episodes; the old mixed union/cache and checkpoint are retired. Raw recordings remain as diagnostics. A new SFP first-trial-start generator changes the actual Gazebo xacro and requires scored-bag joint verification. Two bounded SFP collections yielded **40/40 valid starts, 38 full insertions, two contact-free partials, and 40/40 causally admitted safe transport prefixes** (36 train/four development). The partial insertion tails were excluded. The second set covers all ten requested card/port codes on both released board orientations; its predeclared 16/20 admission gate passed at 20/20. The route-consistent actor retrain above has now run. SFP multi-card combinations at the new orientations remain sparse, and the rotated-board SC LLB route is still missing, so this does not demonstrate full final-distribution coverage. [Correction record](experiments/2026-10-12-route-consistent-data-and-starts.md).

**October 12 public qualification-family route coverage (complete, not promoted):** The old SFP and SC teachers covered 1–5 cards but missed the released second SFP board side and SC board yaw. We collected 40 general plus eight exact-three-card SC rootless Gazebo starts, admitted 47 causally checked transport prefixes (20 SFP, 27 SC; one SC start never reached the safe buffer), and retained every full/partial/no-insertion score. A 14.29M-parameter route-diffusion actor was retrained on **602 episodes** for 6,000 updates from the same initialization as an unchanged-data control. New-SC held-out teacher-state near-action median error improved **4.8→1.0 mm**, but the autonomous frozen comparison reached the ≤10 mm lateral safe buffer on **0/3 SFP and 0/3 SC starts**. A later same-six-scene run of the older deterministic transformer also scored **0/3 in each family**. The coverage diffusion actor entered the axial band on all six and got closer laterally than that transformer on these scenes, but remained 16.7–68.4 mm off; SC complete inference p95 was 301.7–351.5 ms versus the transformer's 133–145 ms, so diffusion failed the 300 ms limit. No contact was scored for either of those two actors. The decisive gap is actor-visited states: target-directed actions looked good on teacher observations but the live policy drifted and often moved axially before aligning. **Do not promote; transport/handoff/insertion/RL gates remain open and final scenes sealed.** The originally proposed on-policy correction pilot is superseded until the route-provenance repair above is reflected in a new matched data experiment. [Full record, commands, traces, plot and artifacts](experiments/2026-10-11-public-evaluation-coverage-retrain.md).

**October 11 route-aware diffusion transport (latest):** The saved five-card wrong-route trace shows an early 5–10 mm departure that grows into the wrong card-side path; an older cache also severely underrepresented most SFP card/port requests. A 14.29M-trainable-parameter diffusion actor (plus 11.18M frozen three-view ResNet-18) was trained on a broader grouped cache with balanced requests and SC data. Velocity prediction fixed a failed epsilon sampler. Paired-request fine tuning improved offline route sensitivity, but autonomous transport still failed. A broad 10% corrective fit had no card1/port1 labels and worsened the reused development scene (70.14 mm lateral versus 22.10 mm control). An early card1/port1 corrective fit from ten safe `n130`-heavy teacher recoveries reduced *training-correction* action p95 9.65→3.88 mm but still missed the same live buffer at 52.31 mm. A separate `n400` hard five-card pilot then yielded **10/10 safe teacher recoveries** and 2,355 causal teacher chunks from training resets with higher lateral offsets. Its matched 3% corrective diffusion model achieved a genuine autonomous safe-buffer visit on one reused development start (4.55 mm lateral, no contact, 284 ms p95), versus 22.10 mm for the control. It failed the other two starts: 63.85 and 67.70 mm lateral versus control 14.89 and 40.66 mm, with two scored gripper–card contacts. **Only 1/3 safe-buffer arrivals, zero insertions: no checkpoint promotion.** All live p95 inference values were below 300 ms. The failure pattern points to narrow correction coverage across joint starting geometry and requested cards; model capacity and teacher-state fit alone did not solve closed-loop routing. Ground-truth TF flooding in privileged diagnostics was fixed separately. Automatic handoff, insertion and RL remain closed; final scenes sealed. [Full record, commands, plots and artifacts](experiments/2026-10-11-route-aware-diffusion-transport.md).

**October 10 complete-state route-restore preflight:** A rootless Gazebo Sim 9.5/Bullet Featherstone service probe found entity-component state read/write, but the AIC controller's target/interpolation state and cable-plugin attachment state are private to separate running components and have no verified snapshot/restore path. The predeclared controller-state prerequisite therefore failed before the two no-action AIC restores; **TCP/cable/force restore tolerances remain unmeasured**. No GPU, actor fit, AIC episode, RL, handoff or final scene was used. Exact-state route ranking is closed under current tooling. Next transport hypothesis is an on-policy, state-conditioned cable-risk predictor plus feedback controller, with whole-reset splits and a predeclared autonomous safety gate. [Record and proposed next gate](experiments/2026-10-10-sfp-state-restore-feasibility.md) · [machine decision](../outputs/experiments/2026-10-10_sfp_state_restore_feasibility_v1/decision.json).

**October 10 transport label and route-ranking gates:** A stage-aware read-only audit found short sustained movement segments in 16/16 saved SFP training and 6/6 grouped development takeovers; an independent later-direction check passed 5/6. Two bounded new four-card, contact-instrumented teacher takeovers both fully inserted, bringing the successful short-segment count to 20 train/six development resets. **No actor fit followed:** six cached takeovers moved >5 mm before the identified onset, so onset segments cannot be relabeled as actions at the actor boundary; a newly successful port-1 teacher continuation reached 55.52 N sampled cable-link/NIC-card point force despite zero official contact penalty. A separate same-config five-run route comparison had takeover TCP starts 11.83–31.07 mm apart and cannot train a matched-state route ranker. A five-run gallery also shows the diagnostic overhead camera sees more of the cable loop than the actor's three wrist views; this is an observability warning, not a measured model failure. The stage-label and route-safety training gates remain open; the subsequent complete-state restore preflight failed, so exact-state route ranking is closed under current tooling. Automatic handoff, insertion, RL and final scenes remain closed. [Audit, exact artifacts and decision](experiments/2026-10-10-sfp-stage-aware-transport-label-gate.md).

**October 9 bounded cable-side route repair (latest):** Two privileged five-card SFP route variants were planned and tested with Gazebo oracle cable contacts under a frozen 10 mm buffer / 6 N sustained wrist / 10 N cable–NIC-card point-force gate. The side-descent variant failed **0/2** starts and was stopped early; the high-side sweep passed **1/3**. One high sweep reached the buffer with low cable contact but sustained wrist-force change. Neither route is safe enough for actor labels. **No route actor was trained; automatic handoff, insertion and RL stay disabled.** Next is a bounded stage-aware, measured-successful SFP teacher continuation label test at actor-visited states, with cable-contact checking and a 20-train/6-held-out episode-grouped gate. [Full route results and videos](experiments/2026-10-09-sfp-cable-side-route-repair.md).

**October 9 SFP route continuation:** A causal selected-port position anchor was available at all **6/6** saved actor takeovers after replacing a two-second image-pair cap with a 150 mm measured-travel cap. On these reused scenes, that anchor plus a fixed board-normal prior placed an 85 mm transport buffer goal within **0.9–1.3 mm** laterally; this prior needs broader orientation stress testing. Three preselected *privileged* MoveIt routes were replayed with real Gazebo cable: 2/3 reached the buffer without scored contact, while the five-card route stalled during descent as force change rose to **12.64 N**. In a separate bounded contact-enabled diagnosis, one of three additional five-card starts reproduced high-force cable/NIC-card interference (about **98 N** sampled peak point normal force and **12.2 N** wrist-force change); its final buffer error was 10.44 mm. The original exact contact pair remains unknown, but this route is not reliably cable-safe. **Safe-route and actor-state-label gates remain open; no route actor was trained or promoted.** Automatic handoff, insertion and RL remain disabled; final scenes are sealed. [Record, videos and machine decision](experiments/2026-10-09-sfp-route-target-and-gazebo-gate.md).

**October 8 handoff-only shadow diagnosis:** Four newly frozen nonfinal SC two-port Gazebo starts (1, 4 and 5 cards; both port requests) made **4/4** requested-port shadow handoffs after appropriate posture checking, with 0.092–0.241 mm posthoc opening error and no scored contact. A separate one-port port-0 start reached within 0.931 mm of the desired standoff yet failed all three inspections because the close-view board yaw filter found too few usable rails. A bounded opt-in orientation-memory repair **did not prove a live benefit**: on independent one-port scene 910 both flag-off and flag-on runs passed using fresh board readings, while scene 919 never reached inspection in either arm (closest standoff 281/187 mm). A follow-up fixed replay of **lossless native camera frames** from the already used one-port scene likewise found **0/2 memory-only accepts**, **0/2 absent-port accepts**, and no naturally missing board rails. Memory remains **off by default**; stop routine reruns of that scene. On six saved SFP autonomous routes, only one visited the true stop corridor. Predicted selected-port position plus a consistent frozen predicted orientation proposed a stop at both sampled visits on that route with no false candidates on the other five, but one arriving route cannot establish reliable handoff. No insertion actor, RL, or automatic SFP handoff was enabled; final scenes remain sealed. [Handoff-only record and artifacts](experiments/2026-10-08-handoff-only-shadow-and-memory.md).

**October 8 actual Gazebo MoveIt recording (latest):** On one previously
used nonfinal five-card SFP scene, the frozen actor ran for five seconds,
then MoveIt planned from the **live measured** takeover state to the
requested card 0 / port 0 buffer. Gazebo executed all 57 waypoints. Final
TCP was 3.76 mm from the 85 mm buffered goal; measured port-frame lateral
offset was 3.75 mm. The official scorer reported no contact and tier 3 = 0:
the route stopped before insertion. Intermediate waypoint tracking lag was
27.66 mm p95. An earlier same-scene attempt correctly aborted because the
saved plan start differed by 23.96 mm. This is a **privileged one-scene
diagnostic**, not autonomous policy success; the ≥5/6 safe-route gate and
0/3 independent autonomous transport result remain unchanged. [Actual
five-camera Gazebo video and full metrics](experiments/2026-10-08-sfp-requested-port-corridor-audit.md#requested-gazebo-replay-actual-robot-and-cable-motion).

**October 8 requested-port SFP corridor (latest):** On six saved nonfinal
actor takeovers, a new planner selected the task's card/port and found rigid
MoveIt paths to an 85 mm TCP buffer on **6/6** groups without a future expert
joint goal. A conservative proxy carrying the takeover cable along the
planned path cleared all 20 links on only **3/6** groups (proximal five:
**4/6**), below the required ≥5/6 safe-route gate. The measured teacher
cable also later crosses this bound on all six runs, so the proxy cannot
certify contact or success. The offline screen led to no actor fit; the
single privileged Gazebo diagnostic above was requested afterward;
independent autonomous transport remains 0/3 safe arrivals. Next is one
bounded repair of the cable-aware route screen and outward corridors using
the existing bags, then at most three Gazebo replays only if ≥5/6 checked
routes pass. Automatic handoff, insertion and RL remain gated; final scenes
sealed. [Full audit, comparison image and decision](experiments/2026-10-08-sfp-requested-port-corridor-audit.md).

**October 8 actor-state MoveIt smoke test (latest):** Exact randomized
collision scenes and measured takeover joints produced **5/6 rigid MoveIt
plans** to future expert states (three development, three train; one train
state had no 10 mm net goal). But on **2/3 development scenes** the nearly
straight plan cut across a measured teacher detour of 22.13/27.65 mm to a
10 mm net endpoint. The cable was unmodeled and the planner was handed the
expert future joint state, so the full safe-route-label gate **failed**.
No Gazebo execution or actor fit followed. Next would be an SFP actor-state
path planner that chooses the requested port and verifies cable clearance
before at most three bounded Gazebo replays. Autonomous transport remains
0/3 safe arrivals; automatic handoff, insertion, RL and final scenes stay
gated. [Full path and artifact audit](experiments/2026-10-08-sfp-actor-state-moveit-feasibility.md).

**October 8 actor-state recovery-label pilot (latest):** A static shadow
CheatCode query is unsuitable as a route label because its first interpolated
command holds the current pose. Three actual five-second actor-prefix
takeovers on distinct train-split SFP resets yielded two privileged full
insertions, one partial insertion, and only **1/3** safe full insertion **and**
valid 20 mm-path/10 mm-net waypoint labels. The preset ≥2/3 pilot gate failed.
Saved measured TCP plots show path looping/controller settling; no new actor
or autonomous rollout followed. A first-10-mm-net label exists within four
seconds in all 16/6 old train/development boundary episodes, but only 11/16
train and 4/6 development paths were direct enough under a 1.5×
travel/net check; its safety and card-side choice remain unverified.
**Stop identical takeovers and keep the route-label gate open.** The next
distinct candidate is a queryable obstacle-aware route planner, with an
offline six-scene path-safety gate. Independent autonomous transport remains 0/3
safe arrivals; handoff, insertion, RL and final scenes remain gated.
[Pilot, plot, exact artifacts and next hypothesis](experiments/2026-10-08-sfp-actor-recovery-label-feasibility.md).

**October 8 short route-waypoint feasibility (latest):** A full-phase
one-second TCP-frame waypoint head improved moving teacher-route predictions
(13.51 mm p95 across 41 development episodes), but its capped translation
command aligned with subsequent teacher recovery on **0/6 actor-boundary
episodes**. Its numerical actor-boundary error (17.86 mm p95) was misleading
because early teacher takeovers barely moved during the first second. A
20 mm measured-path lookahead repair retained only 13 training/four
development actor boundaries and failed its error gate (27.37 mm boundary
p95 for the boundary-weighted fit). **No live run or deployment promotion.**
The prior independent autonomous transport gate remains **0/3 safe arrivals
for both actors**. Next test is bounded causal labeling of actor-visited
recovery states, with a direction gate before any new actor fit. Automatic
handoff, insertion and RL remain disabled; final scenes sealed.
[Full record](experiments/2026-10-08-sfp-short-route-waypoint-feasibility.md) ·
[machine decision](experiments/2026-10-08-sfp-short-route-waypoint-feasibility.decision.json).

**October 8 training-method diagnostic (latest):** The first post-takeover
teacher action causally joins to actor history in all nine grouped episodes,
but frozen pose is accepted at that decision in only 3/6 training and 1/3
development episodes. A 500-step small fit copied accepted-pose training
actions while barely using pose; resetting its action head raised development
error. In a matched 2,000-update full-data comparison with stronger corrective
sampling, no old-policy preservation and corrective checkpoint selection,
the warm head scored **2.833 mm** late-action p95 versus **3.694 mm** for the
reset head; pose ablation changed actions only 0.044/0.038 mm p95. Resetting
also worsened ordinary SFP and SC. The three development correction episodes
were used for selection, so these are diagnostics, not independent promotion
scores. **Do not train a full actor from scratch or open new live scenes under
this recipe.** Next design requires explicit route-level supervision and
better accepted-pose coverage at the first corrective decision. Automatic
handoff, insertion and RL remain disabled; final scenes sealed.
[Full record](experiments/2026-10-08-sfp-pose-actor-training-method.md) ·
[machine decision](../outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/decision.json).

**October 7 selected-port route-feedback decision (latest):** The frozen
three-camera board/crop pipeline correctly identified the requested SFP port
on **3/3 saved off-route actor prefixes**, with zero wrong accepts and 3.10 mm
accepted 3D p95. A 100-prediction grouped sample from the native-image
training/development cache also had zero wrong accepts and 2.39 mm p95.
However, two matched 2,000-update full-action actor comparisons failed:
without the 15-second recovery labels, held-out action p95 was **2.849 mm
zero-pose versus 2.906 mm predicted-pose**; with identical recovery labels in
both arms it was **2.95322 versus 2.95296 mm**. The latter gain is under
0.01% against a 10% gate, and zeroing accepted pose inputs changed its
actions only 0.032 mm p95. **Stop simple pose-projection BC**. No further
autonomous scenes were opened; the last independent transport gate remains
0/3 safe arrivals for both actors. A new route-error/progress control
objective and sufficient actor-visited correction labels are the next
design problem. Automatic handoff, insertion and RL remain disabled; final
scenes sealed. [Full record](experiments/2026-10-07-sfp-early-branch-support.md#route-feedback-ablation-result-the-pose-is-visible-but-this-actor-ignores-it) ·
[machine decision](../outputs/experiments/2026-10-07_sfp_route_branch_support_v1/route_feedback_decision.json).

**October 7 earlier SFP transport decision:** Requested-card visual route
signals and actor-prefix supervision passed bounded data/causal audits, but
neither produced reliable autonomous transport. A frozen early-route visual
probe improved next-1s motion p95 on four new successful hard-card teacher
pairs (18.57→9.28 mm versus pose-only), yet its matched full-action actor
missed the live gate. The first live comparison had an unnoticed overlap with
actor training/validation scenes and is retained only as a diagnostic. On a
replacement **truly unseen** 2/3/5-card development set, a matched actor
trained with five-second actor-prefix history improved held-out correction
action p95 2.897→2.338 mm, but both it and its control reached **0/3** safe
buffered arrivals and **0/3** insertions. It avoided contact penalties on
two of those starts, which does not establish route success.

The unchanged actor was recoverable by a privileged teacher after 15 s on
3/3 of those already-open scenes. Six separate balanced train-only 15-second
takeovers also fully inserted, and causal frame/context audits passed.
However, the final matched 2,000-update late-correction fit improved held-out
late-state first-action p95 only **3.086→2.978 mm (3.5%)**, below its fixed
10% gate; ordinary SFP/SC regression and direction gates passed. **Stop this
same-recipe corrective BC branch; do not run the prepared remaining live
starts.** Next test is a frozen, observation-only selected-port perception
shadow replay on three saved off-route actor traces, before any route-feedback
actor fit. The actor, pause switch, insertion and RL are not promoted; final
scenes stay sealed. [Complete experiment and commands](experiments/2026-10-07-sfp-early-branch-support.md) ·
[machine decision](../outputs/experiments/2026-10-07_sfp_route_branch_support_v1/late15_decision.json).

**October 7 earlier later-state actor repair:** Six of six privileged
CheatCode continuations after 6–8 s of actor motion fully inserted without
an official contact penalty. A matched 2,000-update full-action fit improved
withheld correction-state action p95 from 3.77 to 2.22 mm, but on three fresh
nonfinal autonomous Gazebo starts it reached the safe transport buffer only
**1/3** times versus **0/3** for its matched control, below the ≥2/3 gate.
Both actors incurred a −24 contact score on the two-card start. On the
five-card start the revised actor branched near card 0 instead of requested
card 1 and missed by about 69 mm laterally. Its p95 inference was 94 ms.
**Do not promote** the actor or resume this BC correction recipe. Next is a
bounded requested-card/early-route grounding test with an explicit offline
and live stopping rule; pause/coarse remains shadow-only, insertion and RL
remain disabled, and reserved final scenes are sealed. [Experiment, artifacts
and next test](experiments/2026-10-07-sfp-later-state-causal-repair.md) ·
[machine summary](experiments/2026-10-07-sfp-later-state-causal-repair.summary.json).

**October 7 SFP stall-label follow-up (latest):** The new scored SFP teacher
routes were not run through the earlier SC hold cleanup, but raw data show
almost no sustained low-command/low-motion holds. The legacy SFP source had
68 training episodes with ≥1 s of high commanded motion and almost no
measured TCP motion, mostly near the handoff area. An immutable loss mask
removed 689 affected training-cache rows. A matched 2,000-update CPU fit
with this mask was slightly worse than the unfiltered control on old/new
SFP and SC held-out action p95, same-scene request-pair error, and same-image
request sensitivity. **No live test or promotion** followed. The prior
transport gate remains 0/3; source data and final scenes are unchanged.
The larger unresolved gap is actor-visited early route states, and the old
two-second-prefix correction recipe already failed live. [Data audit,
plots, exact commands and results](experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md).

**October 7 fresh actor gate (latest):** A matched full-action trust-region
actor learned stronger requested-route sensitivity (2.28×) and passed grouped
offline action gates, but reached the safe transport buffer on **0/3** fresh
nonfinal Gazebo starts, equal to its control. Its three-card run hit a NIC
card mount with the gripper and incurred a −24 contact score. Read-only
measured-route comparison finds 13–22 mm median off-teacher distance by
8–12 s, followed by large commands with little measured TCP motion. It is
**not promoted**. A bounded correction experiment on actor-visited off-route
states is the next test; nominal data collection and repeated offline fits
are paused. The pause head stays shadow-only, automatic handoff/insertion/RL
stay disabled, and final scenes stay sealed. [Exact comparison, diagnosis and
next stopping rule](experiments/2026-10-07-actor-proposer-coarse-continuation.md).

**October 7 actor/proposer/coarse data and perception results:** A targeted
native-crop retraining on 91 additional verified SFP train reset groups
improved the selected-port crop from **15/18 to 18/18 early correct-port
anchors** on separate 3–5-card stress groups, with zero accepted wrong ports
and accepted 3D p95 **3.17→2.22 mm**. It reached 10/10 early on the older
sparse development set. Board → crop → three-view model-path p95 was 16.0 ms
on 60 preloaded-image trials; live ROS/capture timing remains open. Card
occupancy alone improved to 20/20 on a separate untouched group set. A new
frozen-action nonlinear pause head gave 27/27 timely teacher proposals and
2/3 timely saved autonomous-route proposals with zero false route proposals;
the third short valid window was missed, so it remains shadow-only. The first
new same-reset alternate-request collection added 16 full-insertion training
routes, but its strict cache gate yielded only 20 valid ≥5 mm paired rows and
no held-out pairs, below the 30-pair actor-fit gate. A completed second
collection added 20 fully inserted train routes and eight held-out routes,
yielding 34 training and nine held-out matched pairs. The live gate above
overrides the offline improvement. **No actor, automatic handoff, insertion
or RL checkpoint is promoted; reserved final scenes remain sealed.**
[Experiment record and exact artifacts](experiments/2026-10-07-actor-proposer-coarse-continuation.md).

**October 7 SFP diverse-trajectory and pause-proposal continuation:** Fresh
rootless Gazebo CheatCode on 137 varied nonfinal scenes scored 132 full
insertions; a five-scene targeted retry added four full insertions (one
no-insertion, one Gazebo spawn crash recovered by rerunning the identical
scene). All **136 successful new episodes / 35,383 transport chunks** passed
causal admission; the original expert data is unchanged. The new grouped
split contributes 119 SFP training and 17 development episodes, giving 445
SFP train / 27 SFP development total. A matched 3,000-update old-data versus
diverse-data actor comparison improved new-scene first-action p95 **1.856→
1.668 mm**, but same-image requested-code action diameter only **0.572→0.620
mm (1.08×)** against the 2× route-choice gate. No autonomous rollout or
promotion followed; prior SFP buffered-arrival evidence remains 0/3. A
frozen-action proposal head trained on the final data achieved **26/27 timely
teacher-route proposals**, matching the old head on the same validation, but
falsely proposed on **one of three saved off-route actor runs**. It remains
shadow-only. The next experiment needs causal same-observation divergent
request supervision or a better requested-goal representation, plus separate
off-route negatives for the proposer. Handoff, insertion and RL remain gated;
final scenes sealed. [Full record](experiments/2026-10-06-sfp-diverse-transport-and-proposal.md),
[machine decision](../outputs/experiments/2026-10-06_sfp_diverse_transport/decision_summary.json).

**October 7 matched spatial SFP transport test:** A frozen 4×5 feature grid
for each camera was added to the temporal actor through a requested-card/port
attention query. The cache passed sampled old/new RGB provenance and feature
checks. The no-spatial and spatial fits used the same checkpoint, grouped
split, balanced sampler, seed and 3,000 updates. Held-out teacher-action p95
improved slightly, **1.863→1.828 mm**, but same-image request-code action
diameter changed only **0.883→0.903 mm** (1.02×), far below the declared 2×
gate and the 12.44 mm paired teacher branch. This architecture did not fix
requested-route selection; no new live rollout or promotion followed. The
next distinct hypothesis is causally aligned same-reset alternate-request
branch supervision. A read-only alignment audit found only **27** usable
≥3 mm alternate-request pairs across 20 scenes, **eight** ≥5 mm and **none**
≥10 mm while observations still matched; the large teacher branch differences
mostly occur after observations diverge. Existing pairs are too weak for a
clean large-branch counterfactual fit. SFP transport safe
buffered arrival remains **0/3**; handoff, insertion and RL remain gated,
final scenes sealed. [Experiment and artifacts](experiments/2026-10-07-sfp-spatial-request-transport.md).

**October 7 SFP transport return:** Frozen same-reset teacher/actor traces
show route drift after startup rather than a camera timestamp or obvious
command-frame bug. The actor's measured path first passed 10 mm from the
teacher route at 4.9, 6.55 and 10.35 s in the three failed starts; material
commands and measured motion had median directional cosine 0.920/0.929/0.907.
One matched, one-GPU full-action actor added request-conditioned visual FiLM
and discrete future-route supervision. It learned the route labels (92.5%
held-out versus 50% majority baseline) and increased same-image request-code
action diameter 0.554→3.394 mm, but the action p95 worsened 2.153→2.811 mm
(+30.6%) and remained far short of the matched teacher's 12.44 mm branching.
Its predeclared offline gate failed, so no new live rollout or promotion
followed. SFP safe buffered arrival remains 0/3, with handoff, insertion and
RL gated; final scenes sealed. Next distinct transport test needs spatial
requested-card evidence while preserving full-action error. [Commands,
artifact map and decision](experiments/2026-10-07-sfp-transport-route-return.md).

**October 7 visible board-origin follow-up:** The old outline corner labels came from a collision envelope rather than the rendered board: 0/30 fixed single-reviewer panels showed an exact uniquely visible corner, and a deterministic silhouette-vertex probe clearly selected the intended physical corner in only 4/30. Actual visual-mesh M3 screw centers displaced the old projections by 19.7 px median and gave 21/30 usable marks in the first reviewer packet, but a frozen early 40-panel check had **0/10** groups with both required screws confidently visible in two cameras; screw 0 is often hidden behind cards. A separate privileged **true-one-screw** layout upper bound missed the requested opening by **22.62 mm median / 39.59 mm p95** with train-mean yaw; even true scene yaw left **15.02/22.18 mm** and two wrong-opening matches. Stop board origin as a stand-alone requested-port shortcut; local port vision and reviewed identity evidence remain necessary. No detector, actor, handoff, insertion or RL fit was promoted; final scenes sealed. [Images, commands and per-scene controls](experiments/2026-10-07-sfp-visible-board-origin-audit.md).

**October 7 data-size follow-up:** A matched 20/40/70-complete-group learning curve on the same original-image endpoint refiner gave development pixel p95 **34.58 / 34.67 / 32.17 px** and early accurate temporal board references **1/10 / 0/10 / 3/10**. The 20→70 p95 reduction was only **7%**, below the predeclared consistent 20% scale-up signal; absolute pixel and board gates remain failed. My fixed 30-panel single-reviewer check found no confidently unique numbered endpoint mark (23 ambiguous/repeated/obscured, seven uncertain). It is not an independent visibility certification. Do not collect more of the same CAD-projection labels by default; define a genuinely visible origin or an approved safe view first. Independent public-observation review remains pending; automatic handoff/insertion/RL stay off and final scenes sealed. [Learning curve, image packet and exact manifests](experiments/2026-10-07-sfp-rail-data-scale-and-visibility.md).

**October 7 temporal board-reference continuation:** A causal 3 s fit of the existing RGB board-endpoint predictions reached only **1/10** early accurate SFP development groups (gate 9/10). Privileged exact and 3 px-noise projection controls reached **10/10**, showing camera geometry can work when landmark pixels are accurate. A train-only constant pixel correction failed (0/10). One bounded 160×160 **original-image** crop refiner trained on 16,145 crops from 70 groups; its best grouped-development endpoint error was **9.92 px median / 32.17 px p95** against a 3/8 px gate, and the unchanged temporal fit reached only **3/10** early accurate groups. Representative crops show repeated structure and CAD endpoint projections without unique visible marks. Stop this projected-label board branch; the independent complete-public-observation review remains pending. No actor, automatic handoff, insertion or RL test followed. Transport remains 0/3 safe buffered arrivals; final scenes sealed. [Record, commands, image and summary](experiments/2026-10-07-sfp-temporal-board-reference.md).

**October 7 board/slot redesign diagnostic failed its gates:** A frozen all-port candidate set contained the true requested SFP opening within 20 mm early in only **4/10** hard development groups even when candidate names were ignored. A new head on frozen local board-slot features completed 2,000 updates but had only **1/10** early accepted groups due to cross-camera disagreement. A second head predicting one 3D cue completed 2,000 updates; its selected checkpoint accepted **five wrong neighboring ports**, had **39.75 mm** accepted p95 and overconfident uncertainty (only 48.7% of ≤20 mm-uncertainty accepts were actually within 20 mm). Both branches stopped before new-scene or actor tests. A matched privileged true-target-cue *offline* transport diagnostic improved early teacher-action p95 just **2.9%** (1.905→1.850 mm), below its 20% gate, and worsened overall p95 slightly; its weights cannot be deployed. The corrected 30-panel public-state review still has no independent annotations. Next obtain that review; if slot-zero cannot be identified, test one safe earlier inspection posture before a temporal board-coordinate model. Transport repair must address autonomous off-route behavior. Handoff, insertion and RL stay off; final scenes sealed. [Full record](experiments/2026-10-07-sfp-slot-feature-port-cue.md).

**October 7 SFP transport/coarse gate:** Both still block deployment. The transport actor reached **0/3** safe buffered arrivals on new sparse-card starts; selected-port acquisition was **3/10** early on the hard ten-request check. The first 40-panel RGB review omitted measured TCP pose, a public deployable observation, so its 11 abstentions cannot prove full-observation identity impossible. A disjoint 30-panel review packet now includes RGB/history, task request, measured TCP and fixed CAD slot order, but has **no independent reviewer result**. A fixed 320 px original-image crop around frozen RGB rail predictions retained the requested opening while excluding neighboring same-port openings in two cameras early in **6/10** development episodes, below the predeclared **9/10** gate (480 px: **3/10**). No crop or actor fit followed. The nominal base-frame port prior was already ruled out (44.6 mm median error). Handoff, insertion and RL remain off; final scenes sealed. [Evidence and artifacts](experiments/2026-10-07-sfp-observation-complete-review-and-tight-crop.md).

**October 6 board-reference tilt pilot stopped:** At the reused nonfinal 180 mm SFP standoff, a guarded local-x +10° wrist view reached 0.014° orientation error but drifted 6.38 mm in TCP position, beyond its unchanged 5 mm gate; it aborted before return. Original-RGB center/right/left comparison still did not identify the requested numbered card and port side in two cameras. No new contact or force rise occurred. Two earlier implementation-error attempts are preserved separately. No second scene or model fit followed. The independent 40-panel review remains incomplete (11 formal prereveal annotations); transport remains 0/3 safe arrivals and the hard selected-target identity prerequisite remains 3/10 early. Handoff, insertion and RL stay off; final scenes sealed. [Images, events and decision](experiments/2026-10-06-sfp-board-tilt-view.md).

**October 6 visual identity gate:** A frozen set of 40 original-RGB three-camera SFP panels includes all ten requests, route/approach/near phases, autonomous frames, and known neighbor swaps. The first 11 were blind-reviewed before any simulator projection: none supported a confident absolute numbered-card and requested-opening selection; the remaining 29 are unreviewed, so this is not a full-set accuracy estimate. A guarded opposite-side view on one reused nonfinal scene safely moved −77.8 mm along the row and returned within 0.3 mm, but showed *less* card-row context and did not reveal the numbered slot in two cameras. The predeclared view gate failed; no identity model, view policy, or handoff was promoted. Transport still has 0/3 safe buffered arrivals. Next distinct work needs an observation of a fixed board/slot reference and a matched target-conditioned transport diagnostic, with explicit stopping gates. [Full review, images and pilot](experiments/2026-10-06-sfp-requested-card-identifiability.md).

**October 6 explicit local-card repair: development gates failed.** A native 480 px crop model trained for 2,000 updates to predict the selected card rail anchor and both ports jointly, on 10,631 unique card/frame crops from 70 reset groups. Its strict geometry guard accepted only **5/119** development observations and gave an early pose in **2/10** groups; selected-port-only predictions remained zero-wrong but the extra pair check removed nearly all coverage. A second 2,000-update RGB-rail/TCP coordinate prior missed the held-out opening by **111 px p95** and separated it from neighboring same-port openings on only **92.96%** of views, below its 30 px/95% feasibility gate. Both branches are stopped. The earlier single-point crop detector remains an offline reference but had inconsistent 6/10 versus 10/10 early coverage across two ten-scene checks. Next audit actual card identifiability and occlusion on existing original-RGB approach views before another fit; no automatic handoff, PoseInsert or RL. Final scenes sealed. [Record and commands](experiments/2026-10-06-sfp-local-card-pair-repair.md) · [machine summary](experiments/2026-10-06-sfp-local-card-pair-repair.summary.json).

**October 6 joint camera candidate test: development pass did not generalize.** A frozen top-five heatmap matcher plus two-inspection/10 mm persistence scored zero wrong and 10/10 early on checkpoint-development scenes. Ten *new*, complete nonfinal SFP reset groups were captured with ordinary CheatCode (10/10 teacher insertions; 527 retained RGB triplets). On them, the joint rule accepted **150 frames, eight wrong ports**, 25.21 mm accepted 3D p95, 10/10 early. The same frozen single-point crop model on identical images scored **89 accepts, zero wrong, 2.71 mm p95, 10/10 early**, but had only 6/10 early on the older additional ten groups. Stable, cross-camera-consistent neighbor-card swaps defeat both geometric and temporal checks. Joint fusion is stopped; neither model is promoted. Transport remains 0/3 safe buffered arrivals; automatic handoff, insertion and RL remain off; final scenes sealed. Next bounded hypothesis: explicit visual requested-card identity and paired port prediction, with ambiguous-view abstention. [Full record, failure images and commands](experiments/2026-10-06-sfp-joint-camera-candidates.md) · [machine decision](experiments/2026-10-06-sfp-joint-camera-candidates.summary.json).

**October 6 selected-port crop continuation:** A frozen RGB-predicted board rail placed 480 px original-image crops around the requested SFP card; projected selected openings were inside at least two crops early in 10/10 development and 10/10 extra nonfinal groups. A request-conditioned ResNet18 crop model trained for 2,000 updates on 21,262 present-request examples from 70 complete train groups. Compared with its old checkpoint on the same ten extra groups, accepted poses improved from 38 to 62 frames, wrong-port accepts from one to zero, and accepted 3D p95 from 4.46 to 1.80 mm. Early reliable three-view poses remained **6/10**, below the predeclared 9/10 gate. Confident cameras often selected different neighboring ports; this is a correspondence failure. Two-view and fixed-horizontal 3D rail checks also failed early-coverage gates. No handoff, insertion or RL promotion; transport still has 0/3 safe buffered arrivals and final scenes remain sealed. Next test joint multi-camera heatmap candidate matching on new nonfinal reset groups. [Experiment, commands, overlays and limits](experiments/2026-10-06-sfp-slot-conditioned-crop-feasibility.md) · [two-view audit](experiments/2026-10-05-sfp-robust-two-view-rail-check.md).

**October 5 SFP slot-local perception improvement: occupied-slot subtest passed; handoff remains open.** A small head that reads ResNet features around five **RGB-predicted** rail positions replaced the whole-image occupancy head. With the same 70 train reset groups and 2,000-update cap, the selected checkpoint improved exact five-slot maps from **5/10 to 10/10** on grouped development and from **5/10 to 10/10** on ten additional disjoint nonfinal teacher-route groups, with zero confident false occupied rails; it also mapped **3/3** saved sparse autonomous actor routes. The extra scenes were not used for this fit or checkpoint choice. Standalone three-camera JPEG/resize/inference p95 was **83.8 ms**, excluding the rest of deployment. A separate causal 3D rail check removed wrong port-candidate accepts in a saved replay but found a usable reference in only **2/10** groups and gave an early correct candidate in **1/10**. The model knows which rails appear occupied, but does **not** yet verify the requested port or a safe pause. Keep transport-to-insertion handoff and RL disabled; final scenes sealed. [Full matched comparison, images and commands](experiments/2026-10-05-sfp-slot-local-occupancy-ablation.md) · [machine decision](experiments/2026-10-05-sfp-slot-local-occupancy-ablation.summary.json).

**October 5 SFP fixed-rail perception probe: diagnostic gate failed.** A single-GPU, 2,000-update RGB model learned two fixed board-rail endpoints and a five-bit occupied-slot map from three native wrist cameras, trained on 70 complete nonfinal reset groups. The selected checkpoint improved exact occupied-slot maps over a constant all-occupied baseline from **1/10 to 5/10** on ten separate development groups, but missed the predeclared **8/10** gate and made **three confident empty-rail errors**. It mapped **3/3** saved sparse-card autonomous actor groups, a promising but small diagnostic. Development endpoint error was 12.0 px median / 26.4 px p90 against projected CAD labels; actor-route error was 17.5/95.0 px. Original-RGB overlays were checked, and the model did not establish requested-port identity, true visibility or safe stopping. This branch is stopped at its gate; handoff, insertion and RL stay disabled, and final scenes are sealed. [Complete record, overlays and commands](experiments/2026-10-05-sfp-board-rail-reference-probe.md) · [machine decision](experiments/2026-10-05-sfp-board-rail-reference-probe.summary.json).

**October 5 sparse-card camera data continuation:** The camera-check training source lacked one- and two-card resets. A predeclared four-case train-only CheatCode pilot stopped at its gate (2/4 full insertions); both failed cases were requests redirected to the second card. Two same-scene original-card controls fully inserted. Only these controls and the two successful one-card pilots were added to a separate strict v3 manifest: 2,026 new synchronized three-camera moments, 3,690 train moments / 70 reset groups total. All 20,260 new all-request presence/task labels matched saved scene YAML; 120 sampled RGB triplets decoded and reset/timestamp audits passed. This fixes a source-coverage gap, but four new episodes and geometry projection do not establish visual slot identity, visibility or safe pause. No camera-check training or handoff promotion followed. A new four-case, three-camera native-RGB identity review marked all four requested numbered slots uncertain before opening geometry; this small single-reviewer result agrees with the earlier 4/10 board-slot and 0/4 wider-view gates. The 166-case review is deferred until a distinct rail-origin/occupied-slot observation hypothesis is tested; autonomous transport arrival remains open and final scenes sealed. [Evidence and commands](experiments/2026-10-05-sfp-camera-check-data-quality.md#sparse-card-coverage-follow-up-october-5).

**October 5 initial SFP camera-check data pilot (v2, before sparse addition):** A request-expanded native-RGB manifest pairs 1,664 train triplets (66 reset groups) with all ten card/port codes: 16,640 training request-image rows, including 1,460 requests for absent ports. A data-quality audit removed 92 stale or mixed-timestamp old-source triplets; 120 sampled retained triplets / 360 RGB files passed structure, resolution and ≤10 ms synchronization checks. After unmarked RGB inspection, projected-point crops showed an obvious foreground obstruction at the selected point in 10/18 near-port triplets. Thus true visibility, selected-slot identifiability and safe-pause labels remain **unavailable**; no new camera-check head was trained. The 166-item blind review pilot remains open before any bounded fit. [Record, exact artifacts and commands](experiments/2026-10-05-sfp-camera-check-data-quality.md).

**October 5 spatial transport prerequisite:** A new frozen RGB replay on the three saved one/two-card autonomous candidate routes found the requested port early on **3/3** routes with each of two existing locators, with zero accepted wrong identities (38 selected-head and 64 structured-head accepted triplets). The transport actor nevertheless reached **0/3** safe buffered arrivals. This narrow result does not clear the full identity gate: the earlier ten-request harder replay acquired only 3/10 early selected-head anchors, and the old five-card autonomous crop path accepted an anchor on only 1/3 routes. A spatial actor was **not trained or launched** under the preregistered prerequisite. Next distinguish visible wrong-slot selections from absent/occluded targets on original RGB, then test one bounded slot-aware repair across all requests before actor training. Handoff, insertion and RL remain disabled; final scenes sealed. [Record and commands](experiments/2026-10-05-spatial-transport-prerequisite.md).

**October 5 latest transport decision:** Added **29 fully inserting SFP teacher episodes and 7,584 TCP-body transport chunks** across one to five cards; one partial insertion was excluded. The corrected 326-episode train set has 35 same-board paired request groups, but remains uneven: 235 episodes request card 0/port 1. A matched action-only continuation improved held-out SFP first-target p95 **2.717→2.153 mm** and SC **2.240→1.987 mm**. It still reached the requested safe buffered corridor on **0/3 new autonomous one/two-card starts**, matching the old actor's 0/3; the new actor came closer on two starts and had less scored contact on one, with p95 inference near 100 ms. More SFP handoff labels left the frozen stop head at **2/10 timely teacher events**, so learned stopping remains disabled. Two bounded task-selected action-head repairs also failed their preregistered held-out action and task-sensitivity gates. Next test a requested-port spatial visual representation before more generic data collection or RL. No model was promoted; final scenes stay sealed. [Full experiment and commands](experiments/2026-10-05-transport-route-handoff-continuation.md) · [machine summary](experiments/2026-10-05-transport-route-handoff-continuation.summary.json).

**Earlier October 5 route/stop comparison:** A route-conditioned temporal transport model worsened held-out first-target p95 versus an action-only control: SFP **3.435 vs 2.717 mm**, SC **3.418 vs 2.240 mm**. The initial frozen handoff head was timely on only **2/10** grouped teacher episodes at the safe threshold and three-decision rule. SC positive labels span only 1–2 decisions. This failure motivated the completed targeted paired-request continuation above. [Details](experiments/2026-10-05-transport-route-handoff-continuation.md).

**October 5 upstream transport and identity follow-up: autonomous gates still open.** On the same held-out SFP image/state/history, changing only the requested card/port code moved the frozen transport actor's first target by a median maximum of just **0.489 mm** across all ten requests. A bounded complete-action task-gated head on 50 balanced successful train episodes improved held-out first-target p95 **3.236→3.013 mm (6.9%)**, below its 20% gate. On 111 original-image rows from three saved autonomous SFP routes, the selected-port crop head accepted five proposals, zero wrong, all from one episode. The first paired-request pilot preflight mistakenly read the instantiated-key field as a quota; the actual UID key quota is **1024**. After correcting it, six train-only same-board alternate-request CheatCode runs completed: **5/6 full insertions and 5/6 measured buffered arrivals**. The matched fit on those successful pairs improved held-out p95 **3.236→2.937 mm (9.3%)**, still below its 20% gate, and was stopped without live rollout. Keep handoff and RL off, and final scenes sealed. [Detailed commands, images, artifacts and next gates](experiments/2026-10-05-transport-identity-task-code-audit.md) · [machine summary](experiments/2026-10-05-transport-identity-task-code-audit.summary.json).


**October 5 global-coordinate proxy:** Official observations expose measured TCP position in `base_link`, but exact selected-port world pose is simulator ground truth and cannot be an evaluation input. A nominal port position fitted on 66 SFP training resets varied from actual positions by 47.6 mm median on ten separate nonfinal reset configurations. On ten CheatCode teacher rollouts, a 100 mm nominal-position threshold coincided with a true safe standoff in 9/10; on three saved **autonomous** policy rollouts it yielded 0/3 safe standoffs and two false stop candidates, one 31 mm and one 99 mm laterally off. This can only be a weak cue for when to inspect the cameras. Rootless Docker's kernel key quota was 117/117, so this was a saved-rollout replay, not a fresh launch. Final scenes remain sealed. [Audit, commands and per-episode evidence](experiments/2026-10-05-global-coordinate-arrival-proxy.md).

**October 5 coarse model gate:** Original-resolution direct row detection plus rail fallback and a native crop refit improved proposal coverage but still made **one accepted wrong-port prediction among 299 accepts** on the ten already inspected SFP diagnostic scenes, with only 7/10 early episodes. Source-offset metadata, a rail-band mask, more counterfactual requests and frozen distance-history fusion all failed the zero-wrong plus 9/10 early gate. The needed next supervised repair is independent visual board/slot identity and reviewed safe-stop/visibility labels on autonomous views, with new development starts for the eventual gate. No automatic handoff, insertion policy or RL was enabled. [Full comparison](experiments/2026-10-04-coarse-shared-history-execution.md).

**October 5 coarse-perception update:** The October 4 original-resolution rail-crop baseline remains the best safe candidate on grouped development, but still fails the full handoff gate. Matched hard-neighbor, source-offset, same-image alternate-request, and frozen 30 mm history-fusion repairs all failed to remove wrong-port accepts while reaching at least 9/10 early requests on the already inspected diagnostic scenes. The history-fusion arm scored 248/370 accepted with two wrong and 7/10 early; the alternate-request continuation had two wrong among 54/112 accepted on grouped development. A root-cause audit found independent lateral shifts of adjacent cards from roughly −19 to +24 mm across the central 80% of source-training pairs, so a crop inferred by shifting from another card can miss by centimeters. The next bounded model will localize the individual card/port at higher original-image resolution with full-row context. True visual-visibility and safe-stop supervision, autonomous transport arrival, and live latency remain open; handoff and RL stay disabled. [Full record](experiments/2026-10-04-coarse-shared-history-execution.md).

**October 4 shared coarse-handoff continuation:** The pipeline is transport → requested-port coarse crop/stop → original-image pose refiner → port-frame insertion actor. The stop/handoff is **still disabled**. A 30 mm measured-distance, 12-anchor, three-camera coarse history fit did not pass wrong-port identity. Six targeted five-card source-training episodes all fully inserted and supplied 327 causal labeled rows (66 train episodes total). Original-resolution observation-only rail crops, retrained on corrected all-channel proposals, improved the previously inspected fresh diagnostic to 244/370 accepted with one wrong and 6/10 early episode anchors, versus an older crop checkpoint's 107 accepted and 12 wrong. This remains below the zero-wrong and 9/10 early development gate. True visibility and safe-stop labels remain missing; no actor handoff or RL is authorized from these offline numbers. A bounded hard-neighbor refit is underway, selected only on grouped development episodes. [Updated full plan and artifacts](experiments/2026-10-04-coarse-shared-history-execution.md).

**October 4 SFP board-slot reference audit: gate failed.** On ten existing nonfinal teacher camera pairs anonymized before review, original center/right RGB at 11 s supported an exact five-slot occupancy map in **4/10**, all with five cards installed. I abstained in six, including **all five sparse-card scenes**; there were zero confident wrong maps. Board ends or cards were clipped, and saved 2/6 s views did not provide a dependable slot-zero reference in representative sparse cases. This fails the predeclared ≥8/10 and ≥2 sparse gate. Teacher positioning used privileged target geometry, so centered teacher frames do not establish autonomous transport or target identity. No new model, Gazebo rollout, handoff, insertion or RL was run. Next inspect actual autonomous transport pauses for observation-only local requested-port identity, while separately addressing route misses. [Blind protocol, scored mappings and next gate](experiments/2026-10-04-sfp-board-slot-reference.md), [machine summary](experiments/2026-10-04-sfp-board-slot-reference.summary.json).

**October 4 wider SFP observation gate: failed; automatic handoff stays off.** A guarded privileged teacher photographed four reused nonfinal card/port requests at a 180 mm plug-tip standoff before and after a +80 mm row-direction/+40 mm height view detour. Three returned safely and then scored 75 with the ordinary teacher; the fourth stopped at the unchanged 5 mm measured-pose gate before insertion. Unmarked original RGB review found that the wider view helped show the complete row for one four-card scene, but a visible card count alone did not establish the requested numbered slot or port convention; one three-card case occupies slots 1, 2 and 4. Strict visual identity certification was 0/4, so the predeclared ≥3/4 gate failed. Posthoc simulator projections put the requested opening in frame at all baseline and wide holds, confirming that field of view alone is not the issue. No new perception fit, autonomous insertion or RL ran. Next test an explicit board-slot reference and true selected-opening visibility on saved nonfinal frames before another model fit. Final scenes remain sealed. [Images, commands and decision](experiments/2026-10-04-sfp-wide-observation-gate.md), [machine summary](experiments/2026-10-04-sfp-wide-observation-gate.summary.json).

**October 4 requested-port history crop: bounded branch failed; handoff remains off.** A new coarse head used the requested SFP card/port code, synchronized three-camera images, causal frames up to 6 s old, and measured TCP motion to predict only the requested crop center in each camera. It trained on 50 episode-grouped teacher runs; a targeted continuation added ten existing five-card training runs to fill missing request combinations. Pooled-history, shared-3D, spatial-history, and coordinate-aware variants all failed safe early identity. The best raw coordinate-aware stress replay still placed **173/572 proposed crops** closer to another port and centered the target within a 60 px radius in all views on **303/572**. A fixed three-view geometry gate rejected wrong centers but left only **3/10 early episodes**, equal to the old selected-only model and below the 9/10 gate. The targeted five-card spatial variant reached 4/10 early, also failing. History preserved some later estimates but did not reliably establish identity from the initial view. The projection label is not a true visibility label, and some first images omit part of the card row. No refiner connection, actor, insertion handoff or RL was run; final scenes stay sealed. Next establish an informative wider observation and independent visual identity/occlusion labels before fitting another coarse model. [Experiment, commands and limits](experiments/2026-10-04-requested-history-crop.md), [machine comparison](experiments/2026-10-04-requested-history-crop.summary.json).

**October 4 replay correction:** The first SFP crop candidate list was filtered to ports marked physically present by simulator labels. That made the offline proposal audit privileged. The corrected replay enumerates all ten SFP output channels before choosing crops from observations: 362/370 crops contain the requested opening in all three views, the crop head makes 12 wrong accepts, and the four-head union makes 47. Its identity gate still fails. The older machine summary and tables in the experiment record are retained for provenance; use the [corrected replay and artifacts](experiments/2026-10-04-sfp-fresh-anchor-shadow.md#corrected-all-channel-crop-replay-october-4) for current counts. Training proposals also used the old filtered list, so any future refit must rebuild those proposals.

**October 4 SFP context audit: history preserves identity but does not acquire it.** On ten existing nonfinal teacher camera episodes, a causal tracker anchored only by the frozen selected-port head accepted 154 frames versus 101 current-frame accepts, with zero wrong accepts in both. The extra 53 frames did **not** add any acquired episode: both methods reached 5/10 eventually and 3/10 within 3 s, below the prior 9/10 early gate. Raw first views show partial card rows on some scenes, while the five-card card-2 scene shows the full row yet still defeats the model; card 4/port 1 has only three installed cards, so visible-card rank does not equal slot code. Projection is not a verified visibility label. The tracker is diagnostic only; automatic handoff, insertion and RL remain disabled. Next review actual identifiability and a board/slot reference before one bounded slot-aware full-view plus crop repair on new nonfinal episode groups. Final scenes remain sealed. [Full context audit and replay command](experiments/2026-10-04-sfp-context-sufficiency-audit.md).

**October 4 SFP side-view check: safe motion, no identity gain.** Four reused nonfinal development configurations were photographed at an 80 mm plug-tip standoff and measured lateral TCP offsets near +15/−15 mm. The final sweeps completed with ≤0.97 N extra wrist load and only light cable/environment contacts; earlier overly strict safety aborts are preserved. At the three matched view holds, the frozen native-crop head accepted 3/4, 1/4 and 3/4 requests. It already accepted those three at baseline; the hard card-2/port-0 request was rejected at every view by every frozen head. No new visual model, autonomous insertion or RL was run. Side-view motion is **not promoted**. Next diagnose true opening visibility and occlusion on existing hard images before further training. Final scenes remain sealed. [Full record and original RGB grids](experiments/2026-10-04-sfp-side-view-diagnostic.md), [machine decision](experiments/2026-10-04-sfp-side-view-diagnostic.summary.json).

**October 4 SFP identity continuation: bounded repair failed.** A one-GPU, 2,000-update model jointly predicted five rail slots and each card's two ports from original RGB. It accepted only **1/136** observations and acquired **0/10** early development anchors; later checkpoints accepted none. The best checkpoint's port-keypoint p95 errors were 127/117 px, and cross-camera pair geometry usually failed. A read-only sparse-history check on the ten frozen fresh scenes found that two consistent frames reduced structured-head early anchors from 5/10 to 3/10 while retaining one wrong first anchor; requiring 10 mm measured tool motion removed that wrong first anchor but left only 3/10 coverage. No new scenes were opened for the failed model, and no handoff, PoseInsert or RL followed. Next bounded hypothesis: a collision-checked side-on observation may make neighboring card/port identity distinguishable. [Experiment and commands](experiments/2026-10-04-sfp-slot-pair-identity.md), [machine decision](experiments/2026-10-04-sfp-slot-pair-identity.summary.json).

**October 4 fresh SFP shadow check: ten complete; identity gate failed.** The kernel key limit was raised to 1024, and all ten frozen, non-final Gazebo scenes were captured sequentially without interrupting Isaac. Nine ordinary CheatCode teachers fully inserted; card 2/port 1 partially inserted. Across 581 synchronized original-RGB triplets, an observation-selected crop contained the requested opening in all three cameras on 362/370 proposals and did so by 3 s in all ten scenes. The four frozen learned heads still made coherent neighbor swaps: all-port 27 wrong accepts, structured 8, crop 12; selected-only made zero wrong accepts but reached only 3/10 early episodes. Their OR union reached 7/10 early with **47 wrong accepted model predictions**. One structural same-card pair check removed wrong accepts on these same scenes but reached only 5/10 early. It is a posthoc diagnostic, not a promotion. Offline image-stage latency was 190.8 ms p95 on the first scene; complete live latency is unmeasured because accuracy failed. No handoff, autonomous insertion or RL was started. Final scenes remain sealed. [Full failure analysis, commands and artifacts](experiments/2026-10-04-sfp-fresh-anchor-shadow.md), [machine summary](experiments/2026-10-04-sfp-fresh-anchor-shadow.summary.json).

**Rootless Docker cleanup, October 4:** Twenty-five stopped September AIC containers were removed after auditing their writable layers and bind mounts. The active Isaac job and older uncertain containers remain. Key usage stayed 199/200 immediately after cleanup, and a later full rollout temporarily hit 200/200. The limit was subsequently raised to **1024**; all nine remaining frozen SFP Gazebo captures completed sequentially, ending at 193/1024. [Cleanup summary](experiments/2026-10-04-rootless-container-cleanup.summary.json), [rollout record](experiments/2026-10-04-sfp-fresh-anchor-shadow.md).


**October 4 SFP rail-crop development result, superseded by fresh check above.** An observation-selected crop around the requested rail slot was audited on 50 training and ten checkpoint-selection development resets. With a 12 px source-pair proposal tolerance, the true requested opening lay inside all three original-image crops in **95/102 development proposals**; the frozen crop refiner accepted **59/102**, with zero accepted wrong ports and **3.98 mm** accepted 3D p95. A read-only union of four frozen predictors acquired an anchor in **9/10 episodes by 3 s and 10/10 eventually** on those saved development starts. An offline four-network image-stage profile measured **201.8 ms p95**, excluding ROS, actor and controller. The subsequent ten-scene independent check above found 47 wrong accepted model predictions and failed promotion. [Original development record](experiments/2026-10-04-sfp-observed-rail-crop.md), [fresh check](experiments/2026-10-04-sfp-fresh-anchor-shadow.md).

**October 4 SFP middle-card identity repair: still gated.** Original-resolution three-camera review of the six previously missed middle-card episodes found five marked port faces apparently exposed from the start and one offscreen initially but exposed later. This is provisional review with geometry markers, not certified visibility. A single 2,000-update, one-GPU all-slot detector with explicit neighboring-opening supervision improved accepted observations **51→66/136** versus the previous selected-only head and held **0 accepted wrong ports / 2.14 mm accepted 3D p95** at its selected checkpoint. Yet it acquired an anchor by 3 s in only **7/10** complete development episodes (same as the frozen older all-port locator), below the 9/10 gate; card-2/port-0 and card-3/port-0 never acquired one. Per-camera audits show persistent wrong-neighbor swaps that the three-view rule safely rejected. **No actor, live Gazebo run, insertion or RL followed.** Next repair must identify the requested rail slot from observation or obtain a safe clearer view, then test on fresh grouped starts. [Full experiment](experiments/2026-10-04-sfp-middle-card-identity-repair.md), [machine decision](experiments/2026-10-04-sfp-middle-card-identity-repair.summary.json).

**October 3 SFP selected-port anchor: numeric gate passed, operational coverage failed.** A bounded 2,000-update, one-GPU task-conditioned locator trained on original wide RGB from 50 balanced successful SFP teacher resets; ten disjoint development resets covered all ten card/port requests. It triangulated the requested opening with **0 accepted wrong ports** and **2.65 mm p95** on 51 accepted observations, but **only 4/10 episodes** ever acquired an anchor: both requests for outer cards 0 and 4, none for cards 1–3. The frozen older multiport locator reached 8/10 episodes on the same observations, while still missing most card-2/3 opportunities. A no-training card-row association check did not rescue another episode; a constant offset misses the 20 mm criterion even on training layouts because card rails shift laterally. Projection-inside-image labels are not certified visibility. **No actor continuation or live run followed**; SFP transport, handoff, PoseInsert and RL remain gated. Next audit middle-card visual identifiability and fit one explicit card-slot/port-face association model or test safe active views, then evaluate on fresh grouped starts. [Experiment, commands and artifacts](experiments/2026-10-03-sfp-selected-anchor.md), [machine decision](experiments/2026-10-03-sfp-selected-anchor.summary.json).

**October 3 SFP explicit route probe: failed waypoint gate.** On existing 50 successful train and ten disjoint development episodes, a 2,000-update temporal probe predicted CheatCode's approach/descent stage at **98.7%**, but its next coarse TCP-body waypoint error was **48.2 mm p95** versus the predeclared **20 mm** limit. The state/task-only ridge baseline was 50.2 mm p95; frozen RGB history did little for the high-error tail. Approach alone was 67.6 mm p95, descent 19.8 mm. No route-conditioned actor continuation or live rollout was started. The next proposed bounded repair is a task-conditioned spatial anchor for the selected nearby card/port from original-resolution wide views, with simulator geometry used for labels only. Transport, visual handoff, PoseInsert, RL and final scenes remain gated. [Full result, plot and code](experiments/2026-10-03-sfp-route-stage-waypoint-probe.md), [machine summary](experiments/2026-10-03-sfp-route-stage-waypoint-probe.summary.json).

**October 3 SFP corrective continuation: failed live gate.** Stock CheatCode recovered safely from three actor-created early deviations, and ten balanced training recoveries supplied 2,302 causally audited **post-takeover full-action** chunks. A single warm-started temporal actor improved grouped teacher-action p95 from 3.13→2.47 mm for SFP, but autonomous requested-port buffered arrival stayed **0/3**, with **2/3** scored contact and actor-only p95 ≤103.17 ms. Two routes came closer; none crossed the 10 mm handoff buffer. Measured route drift still began around 3.85–3.95 s. The corrective BC recipe is stopped; explicit route-stage/subgoal representation is the next bounded hypothesis. Visual handoff, PoseInsert, RL, and reserved final scenes stay gated. Rootless Docker key quota stopped nine planned second-tranche launches before simulation; only the ten audited episodes trained the candidate. [Full record and decision](experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md#october-3-bounded-corrective-full-action-continuation-failed-live-gate), [machine summary](experiments/2026-10-03-sfp-corrective-full-action.summary.json).

**October 3 SFP cross-target transport repair:** Ordinary CheatCode inserted on **10/10** source-trial target identities. The old 268 SFP BC episodes were heavily card-0/port-1 skewed. A bounded scored-command collection produced **50/50 training** and **10/10 disjoint development** full teacher insertions, 12,940 new verified TCP-body transport chunks, and a 39,161-sample frozen three-camera feature cache. Two matched shared BC actors reached **3.13 mm** SFP first-action p95 with ordinary previous-command history and **9.27 mm** with that input zeroed on grouped teacher observations. A **nine-run autonomous comparison** of the frozen baseline and both new actors on three new development scenes gave **0/3 safe requested-port buffered visits for every actor**. Both new actors had scored contact in 3/3 starts; actor-only inference p95 stayed <105 ms. Measured paths split from the matched teacher by >10 mm within 3.7–4.2 simulation seconds. A bounded longer training check plateaued at 2.42 mm SFP teacher-action p95 and failed its predeclared live-test gate. An explicitly privileged five-second teacher-prefix diagnostic improved one route's closest arrival-band lateral offset **92.24 → 16.95 / 28.65 mm** in a pair of repeats, but neither reached the 10 mm buffer; the actor drifted away after takeover. The new weights, visual stop, PoseInsert and RL are **not promoted**; old agent action arrays are excluded and reserved final scenes stay sealed. Next test whether ordinary CheatCode can safely correct actor-created early deviations on at most three development resets before collecting any corrective training data. [Experiment, scores, plot and artifacts](experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md).

**October 3 balanced SFP continuation:** The older selected agent/VLM recordings *do* retain exact target identity in numbered trial YAML, correcting my earlier provenance mistake. All 180 scored-successful episodes cover ten card/port requests; a frozen 5 Hz RGB/state cache and 108/36/36 complete-reset split were built without using their invalid action arrays. A state/task-only stop proposal reached 31/36 teacher audit episodes with no early proposals, but a frozen replay on three existing **autonomous** transport traces produced two unsafe early triggers **35–37 mm laterally from the requested port** and one miss. RGB-only and RGB+state heads were worse on the grouped audit. Automatic SFP handoff stays disabled; stop proposals require a native-resolution selected-port verifier and on-policy negatives. [Full result and commands](experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md).

## October 3 SFP arrival diagnostic

The frozen-feature SFP stop probe found the buffered teacher arrival region in **28/32** validation episodes, with **2 early proposals and 2 misses**; no visual identity or insertion gate was connected. Its 268-episode source is severely skewed: **258 episodes request card 0 / port 1**, leaving almost no coverage of other targets. Smoothing did not repair the errors. The validation set selected the checkpoint, so this is exploratory evidence. SC sparse history remains shadow-only, and SFP stopping stays disabled pending balanced scene-grouped data, selected-port identity tests, and full handoff latency. [Experiment and artifacts](experiments/2026-10-03-sfp-arrival-probe-and-coverage.md).

**October 3 sparse-history SC handoff diagnostic:** A request-specific observation-derived opening anchor now has an off-by-default shadow implementation. On seven saved one-port development routes, the frozen current-image identity accepted 235/367 projected triplets; combining it with a causal, original-RGB-refined three-view history match accepted 363/367, with 0/444 absent-request single-frame accepts. Projection is not reviewed visibility. Four two-port routes kept the other opening at least 68.4 mm from selected near-standoff views. In one live repeat, original identity rejected two real port-0 inspections after posture correction, while history matched the fresh opening by 0.975/0.889 mm and logged two **would-accept** decisions; it never changed the actual rejection. A first history run missed the standoff by 48.25 mm, but its route already differed from the matched baseline before history inference began. Live policy plus periodic-history p95 was 196 ms; complete attached-actor latency is unmeasured. The automatic switch, PoseInsert attachment, RL, and final scenes remain gated. Next test independently reviewed clear-to-occluded and coherent wrong-port history sequences, plus SFP-specific stopping. [Full evidence and commands](experiments/2026-10-03-sc-sparse-history-handoff.md), [machine summary](experiments/2026-10-03-sc-sparse-history-handoff.summary.json).

**October 3 SC orientation/identity repair:** The known port-0 transport stop's commanded orientation was about **6.9°** from the RGB-derived board target, while measured TCP tracked the command within about 0.3°. An opt-in observation-only held-state correction reduced fresh-inspection posture error **6.73° → 0.13° → 0.08°**; on one newly frozen five-card port-0 reset it reduced **8.93° → 0.03° → 0.13°**. Neither run handed off because the original selected-port head rejected the real port after rotation. A same-image TCP-quaternion swap reproduced the score collapse as a diagnostic; it is not a physically valid training pair because the cameras move with the TCP. A bounded head refit that masked the quaternion recovered the corrected images offline but accepted a physically absent port in **11/100** triplets on the new reset, versus zero for the original. The matched new-reset baseline never reached the standoff (82.84 mm), while the correction run reached 1.04 mm; route divergence prevents a causal transport claim. Keep both options off, insertion/RL gated, final scenes sealed. Next use physically matched before/after-rotation RGB on training resets. [Trace, controls, image panels and next gate](experiments/2026-10-03-sc-orientation-identity-repair.md), [machine summary](experiments/2026-10-03-sc-orientation-identity-repair.summary.json).

**October 3 original-RGB SC crop identity decision:** Four new complete one-port development resets were frozen before fitting. A selected-port 160-pixel crop head trained on older groups failed the independent check: it recalled **102/135** projected in-frame three-camera triplets and accepted the physically absent port on **25/144**, versus **134/135** and **0/144** for the frozen spatial identity head. It raised the earlier missed port-0 identity scores, but the full verifier still rejected that handoff for **6.71° TCP posture error** against the 3° limit. In a failing port-1 reset, the observation locator placed real/absent crop centers just 2–10 px apart, so the crops showed the same opening. The crop checkpoint was not promoted, no actor or RL ran, and complete live latency was not measured. Keep the spatial identity head and the automatic handoff disabled. Next diagnose the port-0 orientation command/measurement mismatch under a bounded matched gate. Final scenes sealed. [Experiment and review images](experiments/2026-10-03-sc-original-rgb-crop-identity.md), [machine summary](experiments/2026-10-03-sc-original-rgb-crop-identity.summary.json).

**October 3 SC one-port stress continuation:** The frozen verifier kept 4/4 known good saved pauses and rejected 4/4 same-image wrong-port requests, 4/4 early frames, and 20/20 provisionally hidden requests. On fresh native RGB from two held-out one-port development resets, it accepted port 1 and rejected an absent-port counterfactual, but **missed a real port-0 arrival**: the center-view identity score was 0.247 despite the TCP reaching within 0.94 mm of the true transport standoff. A longer matched port-0 repeat again reached within 0.85 mm but made no pause proposal after losing fresh visual consistency. A near-view-weighted head worsened the missed center score to 0.078; a larger full-image pass retained only 3/5 prior good controls. A further opt-in stale-pause plus fresh-frame retry made three port-0 inspections and 24 extra checks but accepted none, despite true standoff error reaching 0.542 mm. All repairs remain off by default. The SC handoff gate remains **open on valid-arrival recall**, so no insertion actor was attached and full ROS-to-command latency remains unmeasured. SFP's cache has 236/32 train/validation episodes but no SFP arrival labels; its SC-only stop remains disabled. Final scenes sealed. [Experiment, commands, artifacts and decision](experiments/2026-10-03-sc-shadow-oneport-stress.md), [machine summary](experiments/2026-10-03-sc-shadow-oneport-stress.summary.json).

**October 3 live SC shadow handoff:** The frozen transport/perception stack now holds the measured TCP, inspects fresh RGB, rejects inconsistent SC board views or wrong posture, and resumes. A known wrong-posture pilot was correctly rejected at about 4.5° true error; an observation-derived board-angle correction produced an accepted repeat at 0.24° true posture error. Four **new** SC development starts across both requested ports and one/four/five cards yielded **4/4 accepted transport handoffs**, with post-run opening error **0.11–0.30 mm** and port-orientation error **0.04–0.18°**; all had no scored contact. The first JPEG verifier took 332–364 ms and failed the 300 ms budget. After batching crop inference, parallel camera searches and direct original RGB for the verifier, matched repeats on the **same four groups** again gave 4/4 accepted handoffs, opening error **0.10–0.25 mm**, and verifier p95 **176 ms** across six inspections. They did **not** insert. An opt-in SFP transport-head diagnostic exposed a false pause at 16.95 s, **162 mm** from the selected opening; the frozen head was trained on SC arrival labels only. SFP stopping remains disabled. Hard SC occlusion/coherent wrong-port cases and complete attached-actor ROS-to-command timing remain open. The switch, insertion actor, RL, and final split remain gated. [Experiment and commands](experiments/2026-10-03-live-sc-shadow-handoff.md), [machine summary](experiments/2026-10-03-live-sc-shadow-handoff.summary.json).

**October 3 geometric handoff repair:** An existing frozen SC board-rail estimator was replayed on the newer one-port development resets. Its full port-orientation p95 fell from **4.434° to 0.056°** on 87 sampled near-port triplets across all four resets, combining sparse and existing exact-timestamp labels without double-counting a reset. On six frozen transport stops, its 3° posture check kept **5/5 valid** and rejected **1/1 known wrong** stop. The complete offline SC perception path measured **100.27 ms p95**, excluding ROS and actors. An SFP two-camera-vote plus three-view geometry rule, fixed from 19 fitting episodes, kept **85/87** and **159/159** correct constructed candidate triplets with **zero wrong accepts** in both five-episode validation halves. This repairs the measured offline orientation/candidate failure modes, but **does not authorize automatic handoff**: true visual identifiability, coherent wrong-port cases, live latency, safe pause/resume and insertion remain untested. No new weights, actor/RL training or final scenes. [Experiment, commands and next gate](experiments/2026-10-03-geometric-handoff-repair.md), [machine summary](experiments/2026-10-03-geometric-handoff-repair.summary.json).

**October 3 native-crop handoff decision:** The requested-port original-image ablation improved SC orientation on a matched four-reset sample from **4.62° to 3.70° p95**, but missed the fixed **3°** gate. Oracle crop centering was worse (3.97°); the one bounded spatial-feature repair also failed (3.95°). A 320-pixel SFP same-card crop verifier improved constructed three-view correspondence over frozen low-resolution V5 but still accepted **1/423** wrong selection triplets and **1/366** wrong further-development triplets. Requiring both verifiers eliminated those false accepts but retained only **39/87** and **62/159** correct triplets. The crop test and targeted repair are complete; the decision tree stops before integrated handoff, insertion-actor attachment and full rollouts. No new Gazebo episodes, actor training, RL, or final-scene evaluation. Original RGB visibility remains uncertified and full live p95 latency was not measured because the accuracy/identity gates failed. [Full experiment, images, commands and next revision](experiments/2026-10-03-native-crop-handoff-gate.md), [tracked machine decision](experiments/2026-10-03-native-crop-handoff-gate.summary.json).

**October 3 frozen handoff gate and SC orientation repair:** The fixed saved-scene handoff review **failed**: selected-port identity retained all **5/5 valid SC stops** but also the **1/1 known wrong-posture stop**. SFP V5 rejected 4/4 saved wrong-port proposals and retained 69/69 correct ones, while its three original grouped point false accepts all concern adjacent ports on the same card in one reset. A bounded one-port SC orientation-head continuation used **196 triplets from seven usable new training resets**, with original SC/SFP replay, and improved independent synchronized near-port p95 **7.80° → 4.65°** on 149 triplets from four new resets. It preserved original SC/SFP validation and would reject the bad frozen stop while retaining 5/5 good stops at 3°, but **4.65° misses the 3° independent gate**. Correcting per-camera training poses did not change this result. No automatic switch, insertion actor, RL, or final-scene promotion. [Full comparison and next bounded repair](experiments/2026-10-03-frozen-handoff-gate-and-orientation-repair.md), [machine decision](../outputs/experiments/2026-10-02_two_family_perception/selected_port_orientation_repair_decision_20261003.json).

**October 3 handoff continuation:** Frozen SC identity accepted **10/10 proposals around a known wrong-route stop** with 6.71° scored TCP orientation error. A fixed-orientation guard rejected that false stop but also **2/5 valid stops**, so it was not adopted. A bounded SFP request-to-candidate verifier trained with matched onscreen and offscreen examples rejected **4/4 saved wrong-port proposals** and kept **69/69 saved correct proposals**. On two new held-out SFP resets it kept 21/21 accepted correct cascade proposals; no accepted wrong proposal occurred there. Original grouped point validation still had 3/382 neighboring-port false accepts and 1/382 counterfactual-request false accept. SFP orientation p95 was 5.78° on new accepted views, and complete live latency is unmeasured. **No automatic handoff, actor, RL, or final-scene promotion.** [Full record](experiments/2026-10-03-selected-port-correspondence-continuation.md), [machine decision](../outputs/experiments/2026-10-02_two_family_perception/selected_port_correspondence_decision_20261003.json).

**October 3 spatial SC identity continuation:** Eight new one-port SC fitting resets and four new development resets (both requested ports, varied board/rail/cards) supported a matched 800-update spatial-versus-pooled identity comparison. The heads use the same frozen RGB backbone, explicit requested-port code, camera identity and measured TCP; no simulator geometry enters inference. On all saved development frames, the spatial head with a fixed **all-three-camera** 0.5 rule accepted **0/109 physically absent-port requests** and **96/100 real requests projected into all views**; the matched pooled head scored 32/109 and 34/100. Per-camera spatial errors remain (9/304 absent false accepts on the presampled development set), and a two-of-three rule accepted 2/109 absent triplets. Projection is not a true visibility label: some misses occur behind the gripper. On the older 20-example provisional hidden review, the identity head alone accepted 4; the old native keypoint/reprojection rule and their conjunction accepted 0. An in-memory coarse-locator + identity + native-refiner benchmark was 11.19 ms p95, excluding ROS, actors and controller. **Only the SC one-port identity subtest passes provisionally.** Certified hard occlusion, frozen wrong-route replay, SFP wrong-port, complete live p95 <300 ms, safe handoff and autonomous insertion remain open. No actor or RL was promoted; final scenes remain sealed. [Full grouped result, commands, checkpoints and next gate](experiments/2026-10-02-sc-absent-port-identity.md#spatial-selected-port-continuation-on-new-reset-groups).

**October 2 selected-port identity gate failed:** On two SC development resets with only port 0 physically present, the frozen three-camera verifier accepted **16/16 counterfactual requests for absent port 1** while retaining **24/24 matched clear real-port requests** across three resets. It consistently localized the visible port 0 under the wrong identity, with high scores and subpixel camera agreement. The original 104 SC projection episodes all had both ports, leaving this configuration out of training. An older absent-loss checkpoint accepted 14/16 negatives and only 19/24 positives. Four targeted one-port Gazebo resets and two bounded frozen-encoder presence probes did not fix held-out identity (best image+TCP probe: 149/404 absent-slot false positives on four development groups). These are development stress tests, not real task failure rates; the task request itself names a present port. **No verifier, handoff, or actor was promoted.** New spatial, task-conditioned identity/visibility supervision is required before insertion; SFP wrong-port and hard-occlusion gates remain open. [Evidence, videos/images, commands and machine summary](experiments/2026-10-02-sc-absent-port-identity.md).

**October 2 SC multi-camera check:** A bounded diagnostic on the existing selected-port crop cascade found ≤0.777 px three-view disagreement on all **49 reviewed clear SC approach frames** from three new development resets. An exploratory rule requiring three views, frozen in-frame scores ≥0.5 and ≤2 px reprojection disagreement retained **49/49 reviewed clear** examples and accepted **0/20 provisionally hidden** examples from an older, three-scene review batch. The older negatives are mostly easy offscreen cases. At the closest saved observations on the new resets, the rule accepted **2/2 clear** port-1 triplets and **1/6 uncertain plug-overlapped** port-0 triplets; uncertain is neither positive nor negative for gate scoring. Two views also agreed on one hidden wrong prediction. The rule is **not promoted**; certified hard-occlusion labels, wrong-port identity, SFP, orientation, full live latency, safe handoff and autonomous insertion remain open. [Commands and exact results](experiments/2026-10-02-selected-port-handoff-execution.md#bounded-three-camera-disagreement-diagnostic).

**October 2 SC clear-view crop continuation:** The frozen original-resolution 160×160 SC crop refiner was combined with the repaired RGB coarse proposal and tested on **three new, complete development resets**: two requested port-0 layouts and one requested port-1 five-card layout. In 15/16/18 reviewed synchronized clear-view frames, port-frame lateral p95 was **0.230/0.175/0.149 mm** and axial p95 **0.331/0.304/0.296 mm**. Every 45/48/54 per-camera aperture projection fell inside the RGB-selected crop. These are independent scene-group IDs relative to the 104 SC source episodes, but correlated frames and no occluded-view reliability claim. A component-only in-memory three-camera chain took **23.57/31.55 ms p95** on five-card port 0/1; full live latency remains unmeasured. A new 20-scene-group SC RGB review added 20 clear visible and 20 offscreen examples, while 20 plug-covered contact views remain uncertain. The earlier hard occlusion false accepts and all SFP visibility labels remain unresolved. Thus **only the SC clear-view position subtest passes**; true visibility, orientation on these resets, safe transport handoff, autonomous insertion, RL and final scenes remain gated. [Full commands, overlays, and metrics](experiments/2026-10-02-selected-port-handoff-execution.md#continuation-native-sc-crop-second-target-and-grouped-visibility-review).

**October 2 independent SC perception check:** A decoder-only repair of the frozen coarse locator was evaluated on **two new, complete SC port-0 development resets**, including a five-card layout. In visually reviewed clear-opening windows, the left-view >50 px wrong-object misses fell from 6/15 and 15/16 to **0/15 and 0/16**. Three-view TCP-relative port-position p95 fell from **42.76 to 2.86 mm** and **139.07 to 3.78 mm**. This is a useful coarse-proposal repair, **not** insertion-grade pose: measured port-frame lateral p95 is 1.43/1.26 mm, above the 0.5 mm insertion corridor. The old apparent holdout shared a full scene configuration with training and is excluded from generalization evidence. A corrected 100-image original-RGB review batch exists. A separate single-reviewer SC triage found four high-score false-visible calls among 20 clear hidden cases, but covers only three scene groups; all 50 SFP views remain uncertain. No certified training labels exist yet. No classifier, handoff, or insertion promotion followed; final scenes remain sealed. [Exact resets, hashes, commands, images and metrics](experiments/2026-10-02-selected-port-handoff-execution.md#independent-five-card-replication-and-3d-consequence).

**October 2 selected-port handoff execution update:** Frozen checkpoints and
development examples were hashed before testing. A bounded repair of the
transport arrival head used 9,960 cached actor-route decisions from 12 SC
episode groups, with the five frozen comparison groups held out from fitting.
It scored 1 safe / 6 false / 0 missed pauses on the seven frozen paths, versus
5 / 2 / 0 for the original head at the same threshold; it was **not deployed**.
The two-stage gate now distinguishes a proposal to pause from acceptance into
insertion: false proposals require a tested visual rejection and safe resume.
A fresh non-final SFP RGBD label-source repeat also failed: attached and
world-follower auxiliary cameras went flat gray with invalid depth throughout
simulated 5–40 s, while ordinary wrist RGB remained useful. No SFP true-
visibility labels or selected-port handoff can be promoted from these captures.
The next bounded repair is a visibility label source checked against original
wrist RGB and a verifier that rejects known wrong-route/wrong-port examples.
No autonomous full insertion or RL promotion; final scenes remain sealed.
[Execution record and commands](experiments/2026-10-02-selected-port-handoff-execution.md).

**October 2 active handoff plan:** Use transport to reach the **requested**
port's neighborhood, then let its observation-only arrival head propose a
pause. Hold the measured TCP and run a new **task-conditioned, one-selected-port**
perception model. It returns the requested opening's pose and per-camera
identifiable visibility, or **invalid**; another visible port is never a
substitute. A separate learned confidence head is deferred. Only a verified
selected-port pose, stable views, safe waiting pose/orientation, and acceptable
route/contact permit the insertion actor to take over. This is a plan, **not a
validated deployment switch**. The current locator predicts all candidate
image positions and uses projected-in-frame rather than true visibility; the
current transport stop returns without starting insertion. Next work is a
matched arrival-head repair, checked visibility labels, a selected-port versus
all-candidate comparison, a two-stage handoff test, then repair of the local
actor's initial retreat. Autonomous full insertion and RL remain gated; final
scenes stay sealed. [Plain-language design, exact steps and stop rules](SELECTED_PORT_HANDOFF_PLAN.md).

**October 2 stopping decision:** On seven frozen SC development paths
covering both ports and one/three/four/five cards, the visual standoff stop
gave 5 valid, 1 false, and 0 missed valid arrivals; the frozen transport-only
head gave 2 valid, 2 false, and 3 missed arrivals. The visual false handoff
was a wrong route with 1.02 mm TCP position error but **6.71° orientation
error**. A small frozen live pair on one four-card scene diverged in route:
the head-run timed out 101 mm from standoff, the visual-run stopped near
1 mm. This cannot isolate a stop-rule effect. A separate SFP replay found
two accepted **wrong-port** estimates in the card-0-absent, card-1-selected
scene; its five-pose lock did not fire in 100 sampled frames. SC visual stop
remains development-only and SFP visual stop disabled. No insertion or RL
promotion; final scenes remain sealed. [Full comparison, plot and videos](experiments/2026-10-02-transport-stop-comparison.md),
[machine decision](../outputs/experiments/2026-10-02_two_family_perception/sc_sfp_stop_decision_v1.json).

**October 2 SFP corrective BC decision:** A bounded stage-preserving
teacher-takeover collection produced 10 full **teacher-assisted** insertions
from 12 attempts across all ten five-card card/port identities. It yielded
1,392 admitted paired-RGB post-takeover BC rows from ten episodes, including
261 early corrective rows. Two matched 1,000-update actor continuations used
the same checkpoint and five held-out episode groups. Correction reduced
near-port lateral target p95 only **1.82 → 1.72 mm** (5.5%), versus the
predeclared ≥20% and ≤1.5 mm gate. On nine frozen aligned live RGB states,
both actors still commanded retreat in **9/9**. The corrective rows primarily
cover later z≈−58 mm recovery states, whereas the failed initial decision is
near z≈−94 mm. No new autonomous rollout was justified; full insertion,
SFP/SC generality, visibility confidence and RL remain open. Final scenes
remain sealed. [Detailed result and next bounded hypothesis](experiments/2026-10-02-two-family-perception-local-bc.md#bounded-on-policy-correction-comparison-declared-before-training),
[machine decision](../outputs/experiments/2026-10-02_two_family_perception/sfp_dagger_matched_decision.json).
The subsequent five-decision takeover probe also failed its declared BC
gate: 5/5 **teacher-assisted** full insertions yielded 709 rows, yet grouped
near-port p95 was 1.75 mm and the actor retreated on all nine frozen aligned
live states. It advanced on 94/110 aligned training states, exposing a
teacher-to-actor state generalization gap. No live insertion was started
from that checkpoint. [Early-probe decision](../outputs/experiments/2026-10-02_two_family_perception/sfp_dagger_early5_decision.json).

**October 2 pre-opening actor gate:** On a non-final five-card SFP reset,
the scored tip began 35.9 mm before the opening and 1.45 mm lateral. RGB
perception was accurate to about 0.11 mm lateral / 0.086° orientation and
complete inference p95 was 235 ms. With an immediate one-frame handoff, the
frozen local actor still drove the tip to about **20.4 mm lateral** at its
maximum axial progress; official full insertion was **0**. This rules out
the previous model-loading and five-frame wait artifacts as the main cause
on this start. Relabeling the same 2,356 recorded native expert rows into the
new paired-opening RGB frame and running a matched 600-update comparison
reduced held-out near-port lateral target p95 only **2.20 → 2.07 mm** and
worsened first-command p95 **9.05 → 9.81 mm**. Neither checkpoint passed an
insertion gate. The narrower aligned-approach continuation reached 7/9 correct
held-out forward signs but missed its companion lateral gate (2.17 mm p95),
and a replay of nine frozen on-policy states still advanced in only 1/9.
It was not deployed. This earlier diagnostic preceded the later teacher
takeover data and stop comparison above; no RL, autonomous insertion, or
final-scene opening is claimed.
A privileged CheatCode restart after the same pre-opening handoff reached
only tier 3 = 50 (partial insertion); the uninterrupted saved-reset teacher
scored 75. Restarting rewinds the route stage; the later stage-preserving
teacher-takeover comparison and its failed BC gate are reported above.
[Detailed evidence and videos](experiments/2026-10-02-two-family-perception-local-bc.md#october-2-immediate-lock-failure-and-matched-actor-frame-repair).

**October 2 latest checkpoint:** Ten additional independent five-card SFP-to-NIC
stock-CheatCode starts all fully inserted (tier 3 = 75). Their commands were
recorded as TCP-frame deltas, unlike the earlier absolute-command native runs;
the corrected causal conversion yielded 1,914 near-port source rows and 1,478
RGB-pose-filtered BC rows. The added scenes did **not** make the full-image
selected-port locator reliable on the existing grouped holdout; both bounded
continuations are retained as failed comparisons. True target visibility and
calibrated confidence are still missing. A read-only contact repeat of the
prior warm SFP insertion actor again scored 0: the plug tip contacted the
selected card and axial motion stalled, while the cable also contacted a
different card. The 39-scene warm insertion actor improved grouped near-target
p95 from 2.52 to 2.13 mm and advanced 37.4 mm rather than 8.5 mm in a matched
scored handoff, but both scored 0. It stopped roughly 16.5 mm short of the
successful teacher's final TCP, with 2.46 mm lateral error; complete inference
p95 was 233 ms. Two privileged ±3 mm / 4° near-port perturbation pilots fully
inserted and support a bounded corrective-data experiment. No autonomous full
insertion, RL promotion, or final-scene opening is claimed. [Detailed metrics,
commands, and videos](experiments/2026-10-02-two-family-perception-local-bc.md#october-2-follow-up-new-sfp-starts-body-frame-targets-and-live-contacts).

**October 2 latest supervised insertion status:** The selected SFP opening now uses an RGB-only coarse-to-480px-to-160px native crop cascade, three-view triangulation, and a runtime-matched orientation head. Across ten non-final development scenes (722 replay frames), 82.5% of observations passed the three-view consistency check; accepted lateral position error was 0.35 mm p95, with perception-only p95 about 20 ms. A single-frame false agreement on a wrong feature was found, so live control now holds until five consistent RGB pose estimates form a static-port lock. This is operational pose confidence, **not certified target visibility**. The new 29 full-success SFP expert scenes cover all ten selected card/port identities; causal native local action rows now augment the older verified data. The corrected SC on-policy BC run fitted its one-scene labels but still scored 0 live insertion, reaching the port face about 1.44 mm lateral. Separate SFP-native and SC advancing-phase insertion BC runs are active; neither has a demonstrated autonomous full insertion yet. [Current evidence, videos and open gates](experiments/2026-10-02-two-family-perception-local-bc.md#october-2-continuation-complete-sfp-cascade-and-insertion-diagnosis).

**October 2 earlier SC insertion update:** A serial SC near-port handoff to the trained warm local actor ran without privileged geometry input and failed to insert. The scored plug tip ended 6.85 mm short and 7.81 mm lateral in the first thread-limited run; full inference p95 was 387 ms. An RGB-only five-observation static-port lock removed a measured occlusion-induced orientation drift (17° in the previous run), but the matched repeat still failed with approximately 12.6 mm lateral error; p95 was 389 ms. The initially locked port estimate was only 0.09 mm / 0.74° from the scoring label. The later general and SC near-weighted continuations both reached validation plateaus and failed matched live insertion; the latter's p95 complete inference was 320 ms. [Trace, force, videos and decision](experiments/2026-10-02-two-family-perception-local-bc.md#live-sc-near-port-diagnosis-and-targeted-continuation).

**October 2 selected-target perception and local insertion BC:** A shared RGB locator was trained on 268 verified SFP and 104 successful SC episodes, with all physically present alternate ports as explicit candidates. Near-range coarse triangulation has lateral p95 1.05 mm SC / 1.26 mm SFP. An observation-selected 160-pixel native SC crop improved lateral p95 to 0.44 mm on 11 refiner-selection holdout episodes, but the coarse locator had seen those groups during its own checkpoint selection. A visual orientation head has validation p95 2.56° SC / 3.70° SFP. The projected-in-frame score is **not true visibility**: it falsely accepted 78.7% of in-frame occlusions in a four-scene RGBD audit. A visibility pilot improved that audit but has too few scene groups, especially for SC port 1; SFP visibility labels and calibrated confidence remain open. Matched predicted-port-frame local BC runs (scratch and transport-vision warm start) completed on 234 SFP + 83 SC training episodes. Warm start improved offline first-command p95 to 4.60 mm SC / 8.40 mm SFP, still far from a reliable insertion gate. Live insertion has **not** been demonstrated; a first near-port run was invalidated by a concurrent Gazebo transport conflict and is being repeated serially. Final scenes remain sealed; no RL was run. [Methods, videos, artifacts, and limits](experiments/2026-10-02-two-family-perception-local-bc.md).

**October 2 transport update:** A causal visual-stop repair now returns the five-card transport actor at its measured pre-insertion pose (0.58 mm / 2.08° terminal error, 178 ms complete inference p95), but route p95 remained 27.6 mm. Reducing the previsual command cap from 60 to 30 mm helped one five-card start (27.6→19.5 mm route p95) and hurt a different four-card start (19.3→42.1 mm); the cap is rejected. Teacher and actor previsual Cartesian gains match, so that simple mismatch does not explain the drift. A base-frame phase-waypoint probe improved over a constant baseline but missed the held-out 20 mm gate (best predicted-stage p95 22.36 mm); it was not deployed. Reliable route following, autonomous insertion and RL remain open; final scenes remain sealed. [Full evidence and next waypoint experiment](experiments/2026-10-01-transport-reliability-continuation.md).

**October 1 current transport continuation:** A frozen long-history SFP/SC action transformer plus an explicit two-port SC visual locator and late visual standoff controller reached the measured **10 mm / 3° transport gate** on two held-out five-card starts, a held-out three-card start, and a fresh one-card start in selected development runs. The final optimized one/three-card repeats had complete inference p95 **176/184 ms** and route p95 **18.7/17.5 mm**. Reliability is still open: one fast five-card repeat deviated **93.6 mm p95** from the teacher route and ended at **6.7°**, despite reaching 1 mm position error. A training-route-derived optional 180 mm TCP height guard fired 103 times in one repeat, kept TCP above 176.8 mm, and that run reached **0.71 mm / 2.77°** with 23.9 mm route p95; an earlier successful repeat never exercised the guard. The locator, orientation probe, and guard use RGB, ordinary state and public task code only at runtime. These are **transport-only** trials: no autonomous full insertion, local actor handoff, RL, or final-scene evaluation. The guard is disabled by default. [Full record, videos and limitations](experiments/2026-10-01-transport-reliability-continuation.md), [machine comparison](../outputs/experiments/2026-10-01_transport_reliability/visual_transport_development_comparison.json).

**October 1 latest supervised transport comparison:** Five requested directions completed on the same verified 268 SFP / 104 SC data and grouped held-out split: up to 60 s causal history, explicit route heads, a conditional diffusion transformer, a live SFP check, and separate task-specific heads. The long-history action head improved a **single SFP** live development start to **8.0 mm from the teacher pre-insertion TCP standoff**, versus 315 mm for its matched no-history head and 130 mm for the older frozen actor. It still had 6.9° orientation error, about 8 mm lateral error, and no insertion. On a matched **SC five-card** development start, long history was worse: 158 mm final standoff error versus 85 mm with no history. The route head missed its next subgoal by 44.5 mm p95 against a 20 mm gate; diffusion was 12–13 mm p95 on first actions versus about 2–3 mm for deterministic heads; task-specific heads did not beat the shared head. All live p95 latency values stayed below 300 ms. **No autonomous full insertion; step 5, local handoff, RL, and final scenes remain gated.** [Experiment record, commands, videos, and metrics](experiments/2026-10-01-transport-five-way-comparison.md), [plain-language answer](answers/2026-10-01-why-history-port-code-and-five-experiments.md).

**October 1 transport code/frame audit (before the new comparison):** The first 16 failed live starts were SC only; a matched SFP check was subsequently performed above. Across sampled SFP/SC teacher rows and all 18,979 logged SC commands, TCP-body label and runtime base-link conversion roundtrips agree to numerical precision; 16/16 live SC task codes match their scene. The clean model responds weakly to switching the selected SC port code (about 0.4–0.5 mm median first-target change on held-out teacher observations). Its action-token transformer is deterministic and has no temporal route memory; the earlier 16 s temporal model failed offline and was not rolled out. A transport-only timeout success return and two diagnostic-script frame/conditioning mismatches were fixed; no model was retrained **in that audit**. See the [full code/frame audit](answers/2026-10-01-transport-failure-code-and-frame-audit.md). Step 5 stays open.

**October 1 temporal transport continuation:** The supervised plan is now explicit: one shared SFP/SC transport actor, then a separate task-conditioned selected-port perception model, then a separate port-relative insertion actor and a stable two-actor handoff. Hierarchical backtracking and RL are later matched/gated experiments. The first transport-label audit passed for 104 SC episodes and 268 SFP episodes; all six SC Low Lateral Bypass stages are present in order and the 83/21 SC split remains sealed by scene/reset. The SFP split is 236/32 but heavily targets card 0/port 1. The one-GPU frozen-vision history/no-history comparison and direct-state repair finished; none passed the 20 mm full-held-out subgoal gate (best 39.50 mm), so no new autonomous rollout was started. History reduced false teacher-frame handoff signals from 43 to 7, but the switch is not deployable. Step 5 remains open. [Plan, audit, commands, and limits](experiments/2026-10-01-temporal-three-model-continuation.md).

**October 1 final bounded SC transport BC result:** Four frozen supervised actors (clean, old mixed-route warm start, motion-safe cleanup, causal elapsed time) ran on the same two non-final Gazebo development layouts at both 20 and 60 mm action caps. **0/16 inserted and 0/16 reached a stable 10 mm handoff**. Best final handoff error was 75.4 mm; best transient error 27.9 mm, followed by drift. The 60 mm cap did not fix the mismatch despite 35.8% of SC training targets exceeding 20 mm in at least one component; it worsened final error in five of eight matched pairs. Complete inference p95 stayed at or below 182.6 ms. Videos and traces show route choice/progress errors without a sustained cable snag in reviewed examples. Stop routine single-frame full-target BC; the bounded route-progress/subgoal follow-up is reported above. Step 5, local handoff, RL, and final scenes stay gated. [Full record and next gate](experiments/2026-10-01-sc-llb-cleaned-bc.md#final-bounded-supervised-result-and-step-5-decision), [machine cap comparison](../outputs/experiments/2026-10-01_sc_llb_cleaned_bc/cap_comparison.json).

**October 1 earlier 20 mm cap comparison:** Three cleaned-data arms plus the causal elapsed-time ablation failed both fixed development layouts at a 20 mm per-component cap. This first matched test exposed route drift, but a cap audit found that the prior accepted transport experiment used 60 mm. Both caps were subsequently tested as summarized above. [20 mm comparison JSON](../outputs/experiments/2026-10-01_sc_llb_cleaned_bc/autonomous_comparison_cap20.json).

**October 1 dataset gate passed:** All 104 officially full-inserted SC Low Lateral Bypass bags were decoded. The grouped 83/21 preferred motion-safe derivative has 69,681 cleaned transport chunks (19,570 quiet nearly stationary hold anchors removed from 89,251 raw) and 26,455 unchanged local chunks. Its independent audit passed exact recorded target preservation, untouched validation, complete episode boundaries, both named holds in all 83 training episodes, no removed excess-force or measured-motion rows, and no reset-group leakage. The original scored source remains unchanged. The three initial matched BC arms have finished and failed the autonomous gate as summarized above. [Experiment record](experiments/2026-10-01-sc-llb-cleaned-bc.md), [machine audit](../outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/audit_motion_safe_v2.json).

**October 1 access and cleanup history:** An earlier workspace-write Codex session blocked the rootless socket with `EPERM` and hid NVIDIA devices; this was a [session access issue](LOCAL_WORKFLOW.md#when-a-codex-session-cannot-reach-the-working-rootless-daemon), resolved by switching the session profile. Extraction, chunk filtering, and the independent audit have since completed; current status is above.

**September 30 SC Low Lateral Bypass collection:** Completed after **200/200 attempted settings**, with **104 official full insertions** and **198 valid scored replay bags**. The pilot passed 17/30; varied robot starts were checked against scored bags. Only full insertions with a bag and action trace are linked in `outputs/trajectory_datasets/sc_llb_score_verified/`. No BC training or final-scene evaluation was started. [Evidence](experiments/2026-09-30-sc-llb-collection.md).

**September 30 latest:** teacher transport gate accepted. Two same-world Cartesian SC teachers fully inserted, providing v9 with 1,751 transport and 503 local chunks, plus the verified SFP source. The balanced-port BC retry finished 1,500 updates and failed both autonomous development starts: final planned-standoff errors 103 and 603 mm versus 76 and 74 mm for the prior masked actor; neither reached handoff or inserted. A bounded trace/video audit found early wrong-route commands without sustained cable snag. Stop routine feed-forward BC. A checkpoint-provenance audit found that this run reused the full action head from an older mixed-SC-route model and had only two new SC episodes; it was not a clean test of the new Low Lateral Bypass route. Audit clean route data and initialization before deciding on a route-subgoal model. No new training was started for this audit. True visibility, autonomous two-actor insertion and RL remain gated; reserved final scenes stay sealed. [Evidence and exact artifacts](experiments/2026-09-30-transport-to-two-actor-continuation.md).

**September 30 continuation:** transport gate accepted. Four changed-card-layout MoveIt replays reached handoff/standoff; one fully inserted, one failed local insertion and is excluded from expert BC, two were transport-only validation. Four full-success SC teacher episodes now provide 3,001 transport and 1,000 local training chunks, with 1,751 held-out transport chunks. True-visibility recorder checks caught invalid auxiliary depth during tilt; labels are being repaired before training. Autonomous two-actor insertion and RL remain pending. [Execution record](experiments/2026-09-30-transport-to-two-actor-continuation.md).

**September 30 user-approved transport continuation.** The transport gate is **passed**: the near-port misalignment is acceptable for transport supervision. Prefer fully inserted route-valid demonstrations; omit failed insertion segments. The [continuation](experiments/2026-09-30-transport-to-two-actor-continuation.md) advances to bounded changed-layout checks, command-derived transport/local data admission, task-conditioned perception, and two-actor BC/handoff. Autonomous insertion and RL gates remain unpassed.

**September 30, oracle force diagnosis.** Frozen older-route replay reproduced a plug/selected-SC-port catch, not evidence of a transport cable snag: wrist baseline delta 4.217 N and sampled peak normal point force 8.828 N. Actual retreat was only 0.00623 mm despite 24 mm commanded. Gazebo 8.3.0 contact reporting repeats impulse as XYZ and reverses depth sign; reporting is corrected using upstream #890, preserving raw data and physics. The newer tilt-before-return route passed **2/2** oracle insertion repeats, with sampled peak plug/port point forces below 0.8 N; final counts are in the [oracle record](experiments/2026-09-30-force-oracle.md) and [summary](experiments/2026-09-30-force-oracle-results.json). Sensor-frame and commanded-versus-measured trace labels are repaired. This is privileged fixed-scene diagnosis; step 4/5 remain open and final scenes sealed.

**September 30, SC tilt-before-return pilot.** An explicit MoveIt subgoal now
tilts the plug beyond the card row **before** returning toward SC port 1.
Scored sim-time tracking measured 1.59 mm TCP translation but 8.78 mm plug-tip
motion during tilt, confirming why pivot location matters. With updated
privileged teacher alignment and guarded insertion settings, the latest fixed
five-card port-1 route inserted in **2/2** exact repeats; an earlier setting
also inserted once. A matched older route with the same new handoff settings
inserted in **1/2**; the small, plan-level comparison does not establish that
tilt timing caused the improvement. Other targeted attempts stalled at a
near-opening contact or failed tracking gates. The allowed criterion is
continued progress despite brief cable/card touch, not zero touch. This is
**teacher-only fixed-start progress**, not autonomous BC/RL success. Step 3's
fixed-scene route gate passes, while step 4 new-start generality and step 5
robust insertion data stay open; final scenes remain sealed. Saved five-card
port-0 and shifted three-card port-0 route-only plans each reached standoff
again with about 5 mm TCP endpoint error and no scored off-limit contact or
excess force. They did not attempt insertion and repeat saved resets, so Step 4
remains open. See the
[full record and video](experiments/2026-09-30-sc-tilt-before-return.md),
[machine results](experiments/2026-09-30-sc-tilt-before-return-results.json),
and [verified archive](experiments/2026-09-30-sc-tilt-before-return-archive.json).

**September 29, low-cross MoveIt continuation.** After measuring a 1.331 m
NIC-card top, the revised privileged MoveIt route moved away, left, and down
together. The measured plug tip passed below the card tops before the beside-row
traverse. Scored route-only Gazebo runs reached standoff on five-card SC port 0,
five-card SC port 1, and a shifted three-card port-0 layout, with no Cartesian
fallback or observed sustained snag. Cable touching a card is acceptable under
the user's clarified progress criterion; no zero-contact claim is made. A
privileged port-0 insertion teacher succeeded once and failed at near-opening
contact on an exact repeat (**1/2**). Port-1 prealignment improved the handoff
from a 122 mm axial abort to 3.9 mm from the opening, but contact still stopped
insertion. Two later teacher-control tests with a lateral servo and a larger
preinsert bias limit both stopped at the initial alignment gate, so they do
not establish an insertion repair. This is promising route progress, **not a reliable autonomous full
insertion**. Step 3 passes only the revised snag/progress gate on tested starts;
step 4 generality and step 5 full-insertion data remain open. Perception labels,
two-actor BC, RL, and final evaluation remain gated. See the
[continuation, measurements, commands, and videos](experiments/2026-09-29-moveit-low-cross-continuation.md)
and [machine results](experiments/2026-09-29-moveit-low-cross-results.json).

**September 29, MoveIt fixed-scene route gate.** An actual MoveIt 2 planning
scene now contains 240 rigid shapes from the Gazebo board, five NIC cards, and
fixtures. Model origins and yaw agree with scored Gazebo TF within 0.00013 mm
and 0.000016°. Three rootless Gazebo runs executed the saved joint plans with
zero Cartesian fallback and reached the same pre-insertion standoff. The
sparse plan, 20 mm wider outside lane, and ordered farther return all had a
negative cable-to-NIC surface-clearance proxy near card 4 late in the route;
the videos do not prove that the full cable stays beside the cards. Scored
off-limit contact was absent, but no named cable/card contact trace was
available. **The fixed-scene rigid-planning gate passed; cable route quality
did not under its original zero-clearance criterion.** The low-cross continuation
above supersedes this earlier route decision. None of these three original
route-only runs attempted insertion or entered BC.
The [report and videos](experiments/2026-09-29-moveit-sc-route-pilot.md),
[machine summary](experiments/2026-09-29-moveit-sc-pilot-results.json), and
[verified raw archive](experiments/2026-09-29-moveit-sc-pilot-archive.json)
preserve the failed attempts and repairs. The independent
[perception-label audit](experiments/2026-09-29-perception-label-readiness.json)
found 1,485 native SC frames in 21 scene groups, but only projected in-frame
flags, no actual occlusion/identifiability labels, wrong-port negatives, or
SFP rows in that source. Two-family visibility-gate training therefore remains
closed. Generality, BC, handoff, RL, and reserved final evaluation remain
closed; the next bounded repair is the low-cross continuation's near-opening
port-1 alignment diagnosis, while preserving the original contact caveat.

**September 29, targeted SC ±20 mm route comparison.** The exact
five-card, SC port-0 low-cross partial insertion was repeated with its
scene and policy bytes unchanged. It is not deterministic: the archived
run plus six repeats gave **4/7 full insertions**. Holding its short
forward traverse fixed, moving the outside lane 20 mm farther left gave
**4/5 full insertions** and a larger median sampled cable-center gap
(19.24 versus 16.15 mm in new runs), but one lane run still stopped at
about 3.2 mm beyond the port entrance with about 0.55 mm lateral offset
and no insertion event. Farther-behind and combined variants each inserted
in 2/2; moving forward 20 mm from the failed setting inserted in 1/2.
These small fixed-scene privileged teacher counts do not prove reliability
or a clean cable path; wide videos still show the cable near card tops.
The [full comparison](experiments/2026-09-29-sc-low-cross-2cm-rescue.md),
[machine summary](experiments/2026-09-29-sc-low-cross-2cm-rescue.json),
and [review videos](../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/)
preserve successes and failures. Raw MCAPs are hash-verified on NAS.
At that point no MoveIt trajectory had been executed; no actor was trained,
step 5 remains open,
and final scenes remain sealed.

**September 26, storage cleanup.** `/data1` filled. Old May smoke artifacts
and non-selected sweep checkpoints were removed; sweep configs and metrics
were verified on NAS first. The older May 27/28 policy-run trees passed
full-content checks (26,164 and 3,372 files) before local release and now
resolve through NAS symlinks. `/data1` has about **244 GB free**, up from zero.
Canonical verified expert episodes, original failed trajectories, September results,
and selected checkpoints remain protected. See the
[cleanup ledger](experiments/2026-09-26-output-space-cleanup.md).

**September 26, SC route sensitivity and proposed two-actor handoff.** A
five-card, all-left-rail Gazebo teacher route now crosses farther behind the
row, lowers while behind it, travels in an outside lane, passes the last card,
then returns to the selected SC port. On one development scene, its fixed
baseline and five of six one-axis ±10 mm variants fully inserted; the shorter
forward variant inserted only about 10 mm. A larger forward-margin check also
failed its preregistered all-insertion gate. The candidate therefore is **not
yet a robust cable-safe route**. Overhead/side video and sampled cable-link
geometry support a better outside path but cannot prove zero cable/card
contact or generality. A separate corrective teacher has inserted on this
scene; no autonomous actor was trained or promoted. Exact variants, failures,
videos, and limitations are in the [route-sensitivity record](experiments/2026-09-26-sc-far-return-route-sensitivity.md).
The [perception and handoff design](POSE_VISIBILITY_HANDOFF.md) specifies two
task-conditioned actors, selected-port visibility/confidence labels, and a
proposed temporal gate. It is a proposal, not a trained system. Step 5 stays
open, final scenes sealed, and RL deferred.

**September 25, SC teacher route correction.** The old teacher lift still
crossed the card region and is no longer a route-quality example for crowded
SC scenes. An opt-in privileged Gazebo teacher now moves behind the five-card
row, travels beside it in an outside-left lane, and approaches the port after
the last card. It scored full insertion on a five-card port-0 egress scene
and on a fixed five-card wide-camera diagnostic, with sampled cable-center
to main-PCB gaps **11.50 and 13.81 mm** respectively. The first port-0
attempt was deliberately rejected before motion by its clearance gate. A
five-card port-1 scene inserted with an 85 mm lane but passed within
**2.11 mm** of a main PCB, so it is not route-quality BC data. Widening that
lane to 110 mm only improved the gap to **3.25 mm** and lost insertion,
ending 40 mm from the port. Stop the simple lane-width sweep. These are
privileged teacher runs, not autonomous policy insertions; causal/grasp
admission and broader scene coverage remain open. The
[experiment and videos](experiments/2026-09-25-sc-lateral-bypass-and-policy-handoff.md)
include a synchronized 20 Hz overhead/side full-cable view, failures, exact
commands, geometry, scores, and NAS bag archives. Step 5 and RL remain gated.

**September 25, current gate.** The shared SFP/SC BC actor still has no
autonomous full-start SC insertion on the two opened, preregistered
development scenes. Adding ten new verified SC successes improved one
offline starting-command metric but scored Tier 3 = 0/2 in Gazebo. A further
SC-heavy continuation improved held-out SC initial-command p95 to 1.995 mm
and near-port lateral p95 to 0.409 mm, yet also scored Tier 3 = 0/2 and
contacted the enclosure. Its [raw bags](experiments/2026-09-25-shared-bc-sc99-dev2-mcap-archive.json)
are archived. A fresh 60-scene SC teacher collection has now scored **31/60**
full insertions and admitted **17,796** causal chunks. A matched six-scene
five-card teacher replay scored 0/6 without a route lift and 3/6 with an
80 mm pre-route lift; the three full lift trajectories added **2,287**
admitted chunks. The [source scores](experiments/2026-09-25-sc-diversity-batch3-scored.json)
and [paired lift scores](experiments/2026-09-25-sc-fivecard-lift-scored.json)
preserve successes and failures. A same-parent, same-budget BC control versus
34-new-episode treatment then scored **Tier 3 = 0/2 in both arms**, with
wrist/enclosure contact and final plug-port distances about 0.9 m. The
treatment's offline selection error also worsened, 2.772→3.182 mm.
The [matched gate](experiments/2026-09-25-shared-bc-batch3-matched-gate.json)
keeps this negative result. The posthoc teacher trace on the same two opened
development scenes is complete and excluded from BC. At the successful
teacher start, the frozen actor pose was wrong by about 42–44 mm while its
first command was within 0.5–1.6 mm; substituting true pose in the frozen
action head made that command much worse. A corrected offscreen locator
still missed held-out offscreen openings by 22.46 px median / 118.78 px p95.
The expanded three-camera visible-opening locator reached 14.44 px p95 on
held-out far views, but its only geometrically visible view in one difficult
development start missed by 104.69 px. It is not a usable full-start pose
cue. See the [diagnosis](experiments/2026-09-24-shared-actor-sc-serl-execution.md#september-25-expanded-bc-full-start-diagnosis).
An explicit direct TCP-body full-target BC ablation (same shared actor,
pose head auxiliary only) changed the same two autonomous SC starts from
runaway/contact at about 0.9 m to no scored contact at **0.04/0.14 m**.
Neither inserted; its SC near-port lateral held-out p95 **0.644 mm** was
worse than the port-frame treatment's **0.399 mm**. The training budgets
differ, so this is a promising diagnostic rather than a promoted policy.
See the [body-target gate](experiments/2026-09-25-shared-body-bc-batch3-gate.json).
Its lower-rate continuation also failed insertion on the same two scenes.
A training-only two-second body-actor takeover followed by teacher rescue
fully inserted on **3/4** scenes and produced **2,135** verified correction
chunks, including 123 actor-owned frames. Twenty additional frozen training
scenes completed; 15 inserted under teacher rescue. Across pilot and
expansion, 18 distinct successful scenes yielded 9,524 causal chunks,
including 467 actor-owned anchors. A matched same-parent body-target BC
comparison trained both arms for 1,500 updates. The treatment's offline
selection error was 2.633 mm versus 2.645 mm for control, but both remained
**0/2 autonomous insertions** on the same opened SC development scenes.
Overall official scores were 41.50 treatment and 43.80 control; neither
had scored contact. See the [matched gate](experiments/2026-09-25-shared-body-bc-rescue18-matched-gate.json).
This is supervised teacher rescue, not autonomous success or RL.
Continuous near-port body-actor teacher-rescue pilots scored only **1/4**
full insertion each at two and one seconds; one short rerun had off-limit
contact and 18.17 mm grasp drift. A separate interleaved training-only
pilot used 0.2 s actor pulses every 4 s and scored **4/4** teacher-rescued
insertions. Strict force and grasp checks admitted only **2/4** episodes
as correction labels. The 20-scene expansion then scored 14/20 full
teacher-rescued insertions; the combined 24-scene strict audit admitted 16
episodes with 10,042 causal chunks and 347 actor-owned anchors. Two whole
episode BC replay weights worsened held-out SC initial and near-port
metrics. A focused 0.5 s post-pulse correction selection kept 1,458 rows;
its matched treatment improved combined offline selection 2.645→2.621 mm
but worsened SC initial p95 1.760→2.315 mm and near-port lateral p95
0.312→0.439 mm. The frozen two-scene autonomous comparison finished
**0/2 full insertions in both arms**. The treatment's higher partial score
(45.64 versus 28.17) is not reliable evidence of improvement: identical
control weights scored 43.80 in an earlier repeat. See the
[collection gate](experiments/2026-09-25-sc-body-interleaved-rescue-batch20-gate.json)
and [matched live gate](experiments/2026-09-25-shared-body-bc-window0p5s-matched-live-gate.json).
The frozen four-target cadence comparison also finished **0/2 insertions
for both cadences**. Its scored-trace diagnosis found the actor commanding
away from the opening at ~20–30 s, with measured plug motion following
that direction and no scored contact. The bags lack synchronized RGB, so
visibility and target-recognition versus action-mapping remain unresolved.
See the [diagnosis and plot](experiments/2026-09-25-step5-body-cadence-scored-diagnosis.json).
Routine BC/data expansion has stopped. The [revised, bounded next
experiment](experiments/2026-09-25-step5-next-steps-recommendation.md#executed-update-and-revised-next-experiment-september-25)
records synchronized frozen-actor observations on at most four training-only
starts and permits one targeted command-generation repair only if a common
cause is demonstrated; its budget and continuation gate are predeclared.
The other four development scenes and four reserved final IK scenes remain
sealed. Step 5 is open; supervisor and SERL experiments have not begun. See
the [execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md).

**September 25, full-start root-cause update.** A fresh SC rollout of the
latest corrected shared BC actor still scored Tier 3 = 0 (total −35) with
enclosure contact. Posthoc scored geometry shows its auxiliary pose already
wrong by 28.21 mm at frame 0; action error grows from 5.81 mm at frame 0 to
141.1 mm by frame 35. A separate frozen native RGB locator misses the
opening by roughly 350–390 px in initial side views, despite subpixel median
error near the port. A bounded far-view RGB retrain brought initial side-view
opening error to 3–4 px median. Fixed left/right calibrated triangulation
then reduced initial lateral port error from 125.7/140.5 mm to **1.2/2.3 mm
median/p95**; including the center view gave 103.8 mm median because the
opening was outside that view. Near-port p95 after actor runaway remains
6.1 mm, so this does not meet the insertion corridor. A nondeployable
exact-pose substitution **worsened** frame-0 action error to 24.36 mm, so
perception and action mapping both need work. See the
[execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md),
[action audit](experiments/2026-09-25-shared-corrected-fullstart-pose-action-audit.json),
and [matched triangulation](experiments/2026-09-25-sc-global-far-triangulation-side2.json).
The shared BC gate remains closed; no RL or final scene evaluation.

An eight-scene yaw-edge development check then found that the opening is
outside **both side cameras** at frame 0 in six of eight starts. The causal
side-view filter's near-port lateral p95 was 1.624 mm across seven near
episodes, so the single-scene 0.236 mm p95 did not generalize. A bounded
offscreen visual extrapolation ablation was tested with training-only
labels. See the [eight-scene report](experiments/2026-09-25-sc-global-far-eight-causal40.json).

That offscreen regression failed: eight-scene offscreen opening p95 was
482 px. A from-scratch true-pose actor upper bound also failed its matched
offline gate (selection score 5.852 mm versus 4.425 mm for the
observation-only control). A 2 mm action cap slowed the fresh actor's drift
but still scored Tier 3 = 0, total −35 with enclosure contact. The data
inventory showed only **38 SC training episodes** in the 274-episode shared
BC set. The subsequent 20-scene training-only SC teacher batch and lift
retries yielded ten causally audited successes. The
[execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
contains the commands and negative comparisons. No RL or final scenes.

The 20-scene SC teacher batch yielded eight audited full insertions; an
8 cm lift retry of its 12 failures yielded two more. The matched BC
treatment added 10 SC successes and 5,700 chunks. Its grouped initial SC
command p95 improved **2.791→2.565 mm**, but near-port lateral p95 worsened
**0.319→0.544 mm**. On two new, pre-registered development starts, both
control and treatment scored **Tier 3 = 0/2 and total −23 on each trial**
with contact penalties. The four remaining development starts and four
reserved final IK configurations remain sealed. Step 5 is still failed;
no SERL training has begun.

**September 25, latest shared SFP/SC actor gate.** The 14-step plan remains at
step 5: the single shared BC actor has **no autonomous full-start SC insertion**
on the new development start, so recovery RL and final evaluation remain
closed. Nine additional scored, successful 3 mm teacher-bounded SC episodes
(4,859 causal chunks) did not change that outcome in a matched 1,500-update
BC comparison. Correcting an SC/SFP clock mismatch improved grouped offline
SC initial-command p95 from 2.764 to 2.156 mm, but autonomous Tier 3 stayed
zero. Teacher-prefix handoff showed the actor could finish the last ~1 mm
when the privileged teacher had already placed the plug ~13.8 mm inside the
opening; when handed control ~20.8 mm before the opening, it drifted laterally,
hit the wrist/card, and failed. A frozen high-resolution SC estimator was
accurate at that handoff (~0.14 mm lateral error) but lost the target after
large actor-driven drift. Four training-side direct-takeover failures supplied
only 25 safe local teacher-correction chunks; their matched BC continuation
still scored Tier 3 = 0. A separate, nondeployable true physical-tip input
continuation was trained as an information upper bound and has not fixed the
failed-handoff action error in offline replay. A bounded 15× local-row
sampling ablation still scored Tier 3 = 0. A direct two-nominal-second actor
takeover followed by teacher rescue produced three scored full SC successes
in four training-side scenes and 2,046 causal chunks. Adding them in a
matched BC continuation still scored Tier 3 = 0 autonomously, and grouped
held-out near-port lateral p95 worsened from 0.470 to 0.568 mm. A runtime
ablation that replanned after each command also scored zero. Stronger
sampling of actor-owned rescue rows and an earlier takeover window were
also tested. The 15× near-port actor-row sampling still scored Tier 3 = 0
autonomously with wrist/enclosure contact. Direct actor control during the
first 2.0 or 0.5 nominal seconds followed by teacher rescue produced only
one full success per four-scene batch. Their two admitted episodes and 1,037
chunks did not pass the offline BC checkpoint gate when actor-owned rows
were sampled 20×. A nondeployable all-phase true-SC-pose upper bound is
training to test whether early geometry alone can repair action learning.
That privileged all-phase geometry upper bound also failed its offline
gate: held-out SC initial-command p95 was 4.817 mm versus 2.156 mm for the
observation-only parent. A provenance re-audit found nine historical
matrix/lift successes already present in the original SC supplemental
source; a later arm that duplicated them is invalid as a diversity test.
The trainer now rejects duplicate episode/frame rows. A genuinely new
eight-scene clean-teacher batch scored **5/8 full SC insertions** and
admitted **2,463 unique causal chunks** after all-source overlap checks.
A lift retry rescued two more starts, giving **seven unique full-success
episodes and 3,685 causal chunks**. In a matched BC continuation,
held-out SC initial-command p95 improved from 3.124 to 2.169 mm with
seven new scenes, but near-port lateral p95 worsened from 0.312 to
0.446 mm. Control, +5, and +7 actors all scored **Tier 3 = 0** on the
same fresh autonomous SC start with enclosure contact. The supervised
gate remains closed.
Two more full-success SC teacher recoveries were collected after the +7
actor directly controlled the first two nominal seconds with a 2 mm
per-command cap. Their 1,387 causal chunks improved matched grouped BC
metrics, but control and corrected actors both still scored Tier 3 = 0
on the fresh autonomous SC start; the corrected arm contacted the
forearm/enclosure wall. A longer from-scratch +7 BC fit also failed the
offline gate. The next diagnosis is the earliest autonomous command and
pose trace, before changing the architecture or opening RL.
See the [execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
and [artifact map](../outputs_README.md). No offline or online SERL began;
the four reserved final configurations remain sealed.

**September 25 latest shared BC gate.** Three clean 3 mm early teacher-assisted
SC episodes were admitted as causal labels and compared in a matched
1,500-update continuation against the same BC run without them. Both actors
scored **Tier 3 = 0** on the same new autonomous SC start with wrist/enclosure
contact; the added episodes did not solve full-start transport. On the same
training-side teacher scene, 8 mm and 6 mm bounded early actor pulses followed
by teacher control scored **16.88** (no insertion) and **46.05** (partial),
respectively. Their nominal 1.0/0.5 s intervention windows were actually
5.6/3.2 simulation seconds; neither episode was admitted as successful BC.
A 3 simulation-second observation-only startup hold followed by the same
actor still scored Tier 3 = 0 with wrist/enclosure collision on that fresh
scene. An SC-weighted 1,500-update continuation improved held-out first-command
p95 from 3.47 to 2.46 mm but also scored Tier 3 = 0 with the same contact.
On 242 held-out teacher-state SC transport frames, its direction agreed with
the teacher (median cosine 0.998); the fresh closed-loop route still fails.
A separate four-scene training-only bounded student collection is running.
The supervised autonomous gate remains closed, so offline/online SERL,
Gazebo transfer, and reserved final configurations have not started. See the
[execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md).

**September 24, shared-actor correction gate.** The guarded shared BC
actor still has no autonomous full-start insertion and stopped at 42 N in
its SC development rollout. A DAgger-style collector now executes short,
bounded actions from that same observation-only actor while the privileged
teacher supplies correction targets. Isolated repeats of a previously
successful SC teacher scene ended at partial insertion even with zero
student input. Replaying the original two-scene reset order restored a
Tier 3 = 75 control. A second-only student intervention in that order
also scored 75; its admitted episode has 400 student-active frames and
910 causal supervised chunks, 317 anchored during student control.
The warm-up scene and all failed/zero-intervention probes remain outside
corrective BC. The matched 2,750-update continuation finished:
SC held-out 0–5 mm lateral command p95 was **0.380 mm without** and
**0.629 mm with** fivefold-sampled corrective data. Both autonomous
guarded SC rollouts scored Tier 3 = 0 and hit a scored wrist/enclosure
contact; the corrective arm also incurred a sustained-force penalty.
This one-scene correction did not solve transport. The unit-weight
ablation improved offline near-port SC error, and three more scored
student-correction successes were admitted from an eight-scene collection.
A start-aware shared model trained on those additions, with newly generated
actor development scenes kept separate from training. No RL,
reserved final scene, or autonomous promotion has followed from this
teacher-assisted result. See the [execution record](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
and [artifact map](../outputs_README.md).

**September 25 supervised gate:** a start-aware shared actor still scored zero on
four new SC development starts, all with wrist/enclosure contacts.
Training used 288×256 JPEG images followed by PIL 224×224 bilinear resize;
live inference had used direct native-to-224 OpenCV area resize. On 64
paired saved frames, this changed predicted targets by **0.90/3.25 mm
median/p95**. Matching the training JPEG path reduced the disagreement to
**0.075/0.256 mm**. The converter and one-hot task encoding passed
posthoc checks. The same-checkpoint, same-scene corrected-image rerun
also scored **0/4**. A second model with three extra successful
teacher-correction episodes scored **0/4** on those scenes. The runtime
then proved to have a 5 mm per-coordinate target cap despite teacher
setpoints commonly being 10–40 mm away; raising the cap to 20 mm scored
**0/4** and worsened off-limit contact. A scored teacher-only reference
on the first fresh scene inserted fully, while the actor moved in the
opposite board-direction within seconds. The actor predicts 3.35 mm
median first movement on SC training starts where the teacher almost
holds still. Corrective collection previously began after 3 s, missing
this failure. A bounded 3 mm teacher-assisted early-intervention probe
fully inserted on one development scene without scored force/contact
penalty, but it is not an autonomous actor result. A stronger first-command
loss improved teacher-state predictions yet the actor still went the wrong
way and hit its 44 N live force stop. The first early corrective
training-suite run filled scratch before its final score file and admitted
no episodes; its seven bag byte streams are archived. A replay with more
space and the same scene order is active. These findings
are in the
[experiment record](experiments/2026-09-24-shared-actor-sc-serl-execution.md);
there is still **no autonomous shared-actor full-start insertion** and
offline or online RL remains closed.

**September 24, latest gate.** The fourth SC perception suite was frozen
before eight new scenes were generated. Four scenes reached near-port views
(154 frames); the observation-only physical plug-to-opening lateral error was
**0.156/0.294 mm median/p95**, axial **0.201/0.309 mm**, and the frozen
board-line yaw error **0.048/0.084°**. Both position histories and yaw history
initialized on every near-port frame. These numbers are independent
confirmation of SC perception, not autonomous insertion. Eight raw MCAPs were
losslessly archived and SHA256 verified on NAS; images and scored metrics
remain under `sc_yaw_boundary_refreeze_eval_20260924/` in the existing
`/var/tmp/chmin_aic_20260918_act/` root. Only four of eight scenes reached
near-port, so broader coverage and complete live latency are still open.

The shared BC actor still fails its supervised insertion gate. A causal
elapsed-time variant improved the matched initial SC command error to
**5.29 mm**, but grouped near-port SC body-command error remained
**1.158/3.338 mm median/p95**. A bounded **training-only oracle pose**
diagnostic then supplied noisy ground-truth **robot TCP** pose near the opening and
selected update 2,000 of 3,000. Its grouped SC near-port body-command error
was **1.167/3.270 mm**: accurate TCP localization alone did not close the
action gap.
This diagnostic checkpoint explicitly refuses deployment. It does **not**
test the physical plug-tip pose estimated by the high-resolution SC cameras.
A separate noisy **ground-truth physical-tip** input test reached
**0.289/0.714 mm** lateral command error in the 0–5 mm SC band, a modest
improvement over the elapsed-time actor's **0.369/0.820 mm**. Giving both
oracle TCP and plug-tip inputs worsened this to **1.351/1.868 mm** in its
selected checkpoint; that branch failed and is not evidence that RGB pose
cannot help. All oracle checkpoints are nondeployable and selected by an
all-phase validation median. A near-port lateral-weighted
nonprivileged BC arm lowered SC 0–30 mm lateral p95 from **1.204 to 0.916
mm**, but worsened its median from **0.366 to 0.595 mm**; the 0–5 mm band
was **0.586/0.752 mm median/p95**. SFP near-port lateral p95 was **2.736
mm**. A new eight-start qualification-like SC teacher batch yielded **2/8
full insertions**, both admitted as 1,579 causal BC chunks. Four failures
ended with less than 0.4 mm TCP motion over the final five seconds while
32–76 mm from the issued setpoint; a specific cable/card contact is still
unproven. Matched 8 cm teacher pre-route lift rescued **0/3** starts.
A 5 mm near-port cap rescued **0/2** closer failures. The
near-lateral-selected BC rerun with the two new successes selected update
1,750/2,750 and reached **0.198/0.337 mm** lateral command error in the
0–5 mm SC TCP-axial band over 65 frames from five held-out episodes.
SFP 10–30 mm lateral p95 remains **2.650 mm**, and the matched first SC
command still misses by **7.17 mm**. The guarded absolute-base Gazebo
development rollout **failed**: the actor moved about **0.416 m** along a
different transport route from the scored teacher and stopped itself at
42.0 N near an enclosure-looking structure. Scored Tier 3 was zero.
Its exact contact pair is unverified. Near-port offline precision therefore
does not establish autonomous transport or insertion. Next is bounded
student-perturbed teacher correction data, then the same guarded start.
No recovery RL or final
reserved scenes have been opened.

**September 24, current continuation.** A third, independently generated
eight-scene SC perception suite completed with the revised frozen
observation-only tracker. Six scenes reached near-port observations, giving
228 frames. Combined physical plug-to-opening lateral median/p95 was
**0.213/0.443 mm**; 227/228 frames initialized. One scene's own p95 was
**1.004 mm**, and yaw p95 was **0.672°** overall with a **0.981°** scene
outlier. Complete offline native-image perception took **147.36 ms p95** on
one GPU, excluding ROS, actor, and command conversion. This supports the
pooled 0.5 mm perception gate but does not prove every scene or autonomous
insertion. A principled rail-yaw search-boundary rejection improved the prior
development suite from 0.132° to **0.083° p95** and the now-inspected third
suite from 0.672° to **0.085° p95**. It was frozen before a fourth independent
scene suite (`sc_yaw_boundary_refreeze_eval_20260924/`) was generated; its
completed result is summarized above. The [audit](experiments/2026-09-24-sc-supervision-audit.md#third-independent-frozen-check-september-24)
has exact metrics and artifacts; all eight raw MCAPs were losslessly archived
with SHA256 verification. Shared BC labels use complete recorded TCP targets
in port-opening frame. A first **13.60M-parameter shared supervised actor**
was trained from 268 SFP plus 32 successful SC episodes on one GPU in rootless
Docker. It stopped at 6,500 updates after a 2,000-update plateau, selecting
update 4,500. Held-out first-target translation median/p95 was **2.48/11.81
mm SFP** and **2.63/5.55 mm SC**. A no-privilege SC development rollout
failed badly: no insertion, final plug 0.51 m from the port, peak force
785.95 N and an off-limit wrist contact. On the matched scored teacher
scene, the model's first command was 16.01 mm from the near-zero teacher
command; its current TCP-in-port pose estimate was 6.04 mm and 8.06° wrong.
No further live rollout of that checkpoint is planned. A warm-started
command-loss/initial-phase-balanced supervised run completed 4,000 updates.
It reduced held-out SC near-port command error from **1.57/4.98** to
**1.46/3.84 mm median/p95**, but SFP near-port p95 remained **9.16 mm** and
the matched initial SC command still missed by **10.66 mm**. The guarded
absolute-setpoint runtime has not been live tested because these offline
gates failed. **No SC recovery RL checkpoint exists.**

**September 24, earlier continuation.** The second frozen eight-scene SC
perception check exposed a genuine port-tracker failure: the original
three-view causal tracker seeded a false position before the target was
visible and held it after correct detections appeared. Its physical
plug-to-opening lateral p95 was **45.5 mm**, so that frozen candidate failed.
The apparent **1.44° yaw p95** was a *scoring bug*: the metric combined a
scene-specific physical-tip label with the old fixed TCP-to-tip orientation.
Scoring against the selected port's bag TF directly gives **0.132° yaw p95**;
the image estimator and its inputs were unchanged. The separate plug crop
gave **0.197 mm physical-tip lateral p95**. An observation-only three-view/
two-view fallback tracker, with proximity acquisition and stable-target
history, gives **0.395 mm combined lateral p95** on those now-inspected
scenes and **0.164 mm port lateral p95** on the earlier eight-scene suite.
These are development results. Exact scorer/tracker code and thresholds were
frozen before generating a **third, separate eight-scene confirmation**;
that completed run is under
`/var/tmp/chmin_aic_20260918_act/sc_port_tracker_refreeze_eval_20260924/`.

The SC teacher search now has at least one scored, causally audited
full-insertion candidate in **all 12** `(0–5 NIC cards, selected SC port)`
cells. A 5 mm near-port integrator cap turned the previously missing
two-card/port-1 start from partial to **Tier 3 = 75**, but reduced the matched
five-card/port-0 control from 75 to **16.0**. This cap is a teacher data probe,
not a deployable universal rule. The 21 canonical SC experts plus 11
supplemental successful candidates have physically corrected port-frame
waypoint labels. A separate [admission audit](experiments/2026-09-24-sc-bc-admission.json)
passed all 11 new episodes and 6,220 causal chunks after scene-split,
image/command, physical-grasp, and force/stall checks; the original candidate
manifest remains unchanged for provenance. See the
[matrix](experiments/2026-09-24-sc-near-bias-and-success-matrix.json) and
[execution log](experiments/2026-09-24-shared-actor-sc-serl-execution.md).
That earlier audit preceded the shared BC checkpoints summarized above;
no SC recovery RL checkpoint exists.

The active [shared SFP/SC actor and recovery execution plan](experiments/2026-09-24-shared-actor-sc-serl-execution.md)
starts by freezing scene/data inventory and validating Isaac distal-cable
fidelity, SC action targets, and observation-only SC pose. It then fills
missing successful Gazebo SC demonstrations before shared BC and gated Isaac
RL. One actor will own normal motion and post-retreat choices; the measured
backoff supervisor is deterministic. Shared BC has since trained and failed
its autonomous gate; SC recovery RL remains unopened.

The bounded Isaac SC mechanics gate now passes with robot self-collision off:
0/3/5-card scripted scenes insert, and external plug/port and gripper/card
contacts remain active. The [fidelity record](experiments/2026-09-24-isaac-sc-fidelity-fix.md#cable-self-collision-diagnosis-and-corrected-moving-view)
preserves the failed kinked cable and corrected 20 Hz video. The next SC
supervision audit found a quaternion-order bug in the old port-frame TF
conversion. It is corrected in the [v2 audit](experiments/2026-09-24-sc-port-target-audit.json);
the old label orientations were wrong by 153.7° in a checked terminal sample.
Strict and held-target connector chunks are now built from scored Gazebo
episodes. The older low-resolution [SC pose probe](experiments/2026-09-24-sc-pose-probe-results.json)
reached 1.70 mm near-port lateral p95 on seven held-out episodes. The
[native-image continuation](experiments/2026-09-24-sc-native-pose-ablation.md)
recollected 1152×1024 RGB from all 21 verified SC reset groups and generated
new, episode-grouped training and development scenes. It exposed wrong-port
detections after occlusion, so raw per-frame predictions are not treated as
ready for control. A three-camera landmark model and causal fixed-port filter
use RGB, measured TCP, and fixed calibration only. After adding 12 yaw-edge
training scenes, the position head reached **0.141 mm lateral and 0.106 mm
axial p95** over 208 near-port frames in four separate yaw-edge episodes;
its history initialized in all four. The yaw-constrained orientation head
reached **0.804° p95** on the same suite, but one episode had no trusted
orientation history. A position-anchored fallback for only the first two
orientation updates reached **0.806° p95 with 100% initialized coverage**
there; on the older 12 development scenes it gave **0.596° p95**. This
fallback was designed after inspecting the yaw-edge suite, so these numbers
are **development evidence**, not an untouched confirmation. The frozen
eight-scene follow-up and its failure are described below. Offline CNN
perception p95 was 37.5 ms after retraining, excluding the rail stage;
live ROS+actor latency and autonomous insertion are unmeasured.
The SC perception gate and shared BC/RL remain open. The
[supervision audit](experiments/2026-09-24-sc-supervision-audit.md) records
the quaternion fix, timings, camera provenance, and label limits.

That eight-scene check confirmed the position filter (**0.214 mm lateral
p95**, seven near-port episodes) but exposed **1.608° orientation p95** in
one partly occluded scene. A new [board-rail yaw estimator](experiments/2026-09-24-sc-native-pose-ablation.md#frozen-suite-failure-and-board-rail-yaw-ablation)
uses long visible rail lines in the original RGB views, measured camera pose,
and a coarse learned visual yaw prior. It reached **0.064° p95** on the 12
grouped development scenes and **0.078° p95** on the already inspected
eight-scene suite, with every near-port episode initialized. The rail stage
measured **88.8 ms p95 offline**, separately from the 37.5 ms CNN stage.
Because the failed scene motivated the rail method, a different eight-scene
development suite was generated only after checkpoint/code/threshold hashes
were frozen. The [fresh evaluation](experiments/2026-09-24-sc-board-lines-fresh-evaluation.json)
measured **0.165 mm lateral / 0.183 mm axial / 0.109° board yaw p95**
over 236 near-port frames in seven of eight new scenes, with both causal
histories initialized on all near-port frames. The eighth scene never reached
the near-port region. This establishes development accuracy for the **port**
and board yaw. The [physical-tip audit](experiments/2026-09-24-sc-supervision-audit.md#fresh-native-image-confirmation-and-remaining-physical-tip-question)
then found that the fixed TCP-to-tip proxy is wrong by 1.15–10.58 mm across
16 scored pilot scenes, while the episode-specific physical grasp is nearly
rigid. The frozen RGB tip head failed on the new eight-scene suite at
**6.83 mm lateral p95** against the actual tip. A first retrained shared-crop
head reached **0.356 mm lateral p95** on 12 grouped development episodes,
but **0.555 mm** on the inspected eight-scene check, with a cold-start
failure. The revised [separate plug-crop candidate](experiments/2026-09-24-sc-physical-tip-model-comparison.json)
uses scored physical-tip labels for training and a frozen RGB locator for
crop selection. It reached **0.290 mm tip lateral p95** on 12 held-out
development episodes. On the earlier eight-scene check, its combined
physical plug-to-opening error was **0.450 mm lateral / 0.174 mm axial p95**
with both histories initialized throughout near-port motion. Because that
check motivated the new crop, a second eight-scene suite was generated only
after freezing the candidate. Offline four-model perception measured
**128.3 ms p95** including native image decode and rail extraction; full
live ROS+actor latency and autonomous insertion remain unmeasured. Shared BC
stays gated on the new confirmation and SC teacher-data admission.

## September 24 corrected Isaac SC route contacts

After the [mechanics repair](experiments/2026-09-24-isaac-sc-fidelity-fix.md),
controlled five-card routes reproduced named **gripper/card** blockage twice
and **plug/card** blockage on a different route. In the instrumented across-card
run, the gripper moved only 6.74 mm over 75 commanded steps while gripper/card
contact exceeded 100 N; route reversal cleared it. Cable links 1/2 also touched
a card, but only during the gripper-led jam. A nearby clear lane had no sampled
scene contact and reached the transport waypoint. See the [route record, plot,
and raw trace map](experiments/2026-09-24-isaac-sc-contact-routes.md).
This is one constructed scene with a privileged scripted controller. An
**independent cable snag is still unproven**, and the high-force routes are
unsafe as training demonstrations. Gazebo's retained bridge logs only
off-limit contact pairs, so the historical Gazebo cable-trap candidate remains
causally ambiguous. The newer Isaac container now passes a three-camera
[RGB smoke test](experiments/2026-09-24-isaac-sc-fidelity-fix.md#camera-follow-up-september-24)
after the same local driver-check override used by the earlier working
container. A diagnostic view visibly shows orange cable and the card row.
A 61-frame moving clear-lane clip also renders, but its distal cable looks
sharply kinked/looped relative to the retained Gazebo wide view. Distal cable
fidelity and cross-simulator appearance checks remain before visual SC
recovery training. No SC BC or RL training was started.

## September 24 Isaac SC mechanics repair

The [SC fidelity repair](experiments/2026-09-24-isaac-sc-fidelity-fix.md)
calibrated the SC grasp and first cable links from scored Gazebo TF, restored
the source SC-port SDF colliders in the correct port frame, and removed two
measured internal grasp-contact artifacts without disabling scene collisions.
In one constructed five-card development scene, a privileged scripted
full-start Isaac probe reached the scored Gazebo port-base target with
0.073 mm final axial and 0.054 mm lateral error; peak plug/port contact was
1.04 N. Two deliberate near-port lateral commands yielded plug/port
blockage at 54 and 92 N, with gripper/first-link contact zero. The old
gripper/card item-2 collision did not recur on the centered route with the
calibrated grasp. Deliberate card-crossing routes above reproduce a separate,
named gripper/card incident.
These are [physics-only diagnostic traces and exact hashes](experiments/2026-09-24-isaac-sc-fidelity-summary.json),
not autonomous insertions or RL replay. The original camera attempt failed
under the rootless Isaac Sim 5.1 driver check; the later container-local
override passed a short repaired-scene RGB smoke test. Distal cable and cable/card snag
fidelity also remain unverified. No SC BC or SERL training was started.

## September 24 hierarchical recovery continuation

The fixed five-card Gazebo route probe now has seven 20 Hz wide-view reruns:
six partial insertions and one no-insertion stop with 52.9 N peak raw wrist
force norm (72.16 N vector-tared official force maximum),
38.7 mm plug axial distance before the opening, 4.45 mm lateral offset, and
111.6 mm final TCP command error. The cable visibly crosses the cards, but
the contact pair is unknown and the new run differs from the older nearly
stationary-cable candidate. Official post-friction evaluation image and
single-trial reset evidence make the historical ghost-collider bug a poor
explanation. Preserve both incidents as **ambiguous obstruction**, not proven
cable snags. Videos, commands, geometry, and toolkit cross-check are in the
[fixed-route record](experiments/2026-09-23-fixed-five-card-route-probe.md).

The approved branch is [hierarchical recovery](HIERARCHICAL_RECOVERY.md):
learned policy for normal motion and retry choices, deterministic measured-pose
retreat on a validated force-plus-stall trigger, and escalation to an earlier
safe route checkpoint after repeated same-corridor blockage. The existing
Isaac controller only performs local position backtracking; neither the
hierarchy nor SC-to-SC learned recovery has been validated. The
[execution plan](experiments/2026-09-24-hierarchical-recovery-continuation.md)
ranks failures, specifies the SC BC gate, matched no/local/hierarchical
ablations, optional offline critic warm-up, online SERL, Gazebo transfer, and
unchanged official-evaluation gate. Hierarchical assembly and cable recovery
have prior art, but the two closest papers reviewed in the
[core design](HIERARCHICAL_RECOVERY.md#related-work-and-claim-boundary) do not
describe our proposed local-to-route measured-path checkpoint escalation.
Any publication claim still depends on a measured gain and a wider novelty
review.

Execution update: an [indexed incident audit](experiments/2026-09-24-hierarchical-recovery-continuation.md#execution-log)
contains 24 retained Gazebo SC route/targeted episodes (8 full, 11 partial,
5 none); fixed-seed repeats are one scene group. Seven 20 Hz bags now have
measured TCP, command, and force-vector traces. The initial two-sample
force-plus-stall trigger, when correctly tared as a vector, marked **7/7**,
including six partial-insertion runs; a longer-window variant also marked
**7/7**. Subtracting force norms had misleadingly marked zero and two runs,
respectively. Neither is approved for robot control. A simulator-independent hierarchical
checkpoint prototype passes four isolated tests but has not driven an
autonomous rollout. Canonical SC supervision remains 21 verified Gazebo
episodes, all NIC1–3. The later corrected Isaac SC grasp/port now passes one
scripted five-card mechanics scene, but image collection and broader scene
validation remain open. **No SC BC, offline SERL, online SERL, Gazebo transfer, or
official score is claimed for this continuation.** The precise gate and
commands are in the execution record.

A broader read-only audit of 17 Gazebo traces across five scene/config groups
found peak TCP target-tracking error of 89–127 mm in the three no-insertion
runs versus 56–73 mm in fourteen full/partial runs. An 80 mm line separated
these saved cases, but it was selected after seeing them, and neither force
nor tracking error reveals the contact pair. This remains a candidate
diagnostic for new-scene validation, not a deployed backoff trigger.
Three fresh normal-collision five-card Gazebo variations then yielded one full
and two partial insertions, using either the privileged route plus stock
CheatCode or stock CheatCode alone. None exceeded 80 mm tracking error; there
is still no fresh failed-scene sensitivity check. The stock run's layout was
only 7 mm in X and 4 mm in Y from an earlier no-insertion layout, yet became
partial. Their [scores, config hashes, traces, and videos](experiments/2026-09-24-hierarchical-recovery-fresh-scenes.json)
are saved. These are diagnostic teacher runs, not learned recovery.

The SC supervision audit then reconstructed 13,349 recorded teacher TCP
targets in the selected port base frame from all 21 canonical scored bags.
It found 84 missing command indices and nine image-time gaps over 250 ms;
four adjacent recorded targets can span 150–450 ms. The labels are useful,
but they are not yet exact fixed-50-ms connector-waypoint chunks, and there
is no validated observation-only SC port estimator. See the
[port-frame audit](experiments/2026-09-24-sc-port-target-audit.json) and
[execution notes](experiments/2026-09-24-hierarchical-recovery-continuation.md#sc-teacher-target-frame-audit-details).

## Fixed five-card route probe and S3 audit (September 23)

The SC cable's opposite SFP-module end is free. A free end can still drape
over or catch on the NIC cards. The five-card collection's accepted video
**was downloaded from S3** and matches the local accepted MP4 byte for byte.
The original failed seed-51500 VLM/MoveIt video was not found under the
checked clean, development, or EC2-transfer prefixes; the clean collection
contains accepted attempts 27 and 28, not that old failed attempt.

A privileged diagnostic route used one fixed generated five-card Gazebo scene,
normal collisions, and stock CheatCode insertion after a long transport path.
Across-card runs were **one full, one partial, one none**; outside-left runs
were **one full, two partial, zero none**. The no-insertion across-card run
brought a cable segment center to 0.9 mm from a main PCB collider and later
held it nearly fixed while commanded TCP motion advanced and the plug stayed
46.7 mm before the port. Successful/partial repeats of the same planned
across-card route had at least 8.6 mm measured center clearance. A lower
route failed with the cable far from the cards and is a separate robot-path
clearance confound. This is a **credible route-sensitive cable-trap candidate,
not a named cable/card contact or proven causal snag**. No RL or model training
was run. Review the [fixed-scene comparison, overhead/side plots, and videos](experiments/2026-09-23-fixed-five-card-route-probe.md).

## Ordinary Gazebo and VLM-route cable follow-up (September 23)

The broad development audit now has **19 valid, clean stock-CheatCode Gazebo
scenes** across SFP-to-NIC counts 1--5, SC-to-SC counts 0--5 on both rails,
and two saved randomized SC layouts: **13 full, five partial, one no
insertion**. The original long batch became invalid after Gazebo
physics-entity errors; each affected scene was rerun alone. Three SC rail-1
partial runs with one, three, and four NIC cards stopped at almost exactly
10.5 mm past the port entrance despite 0.09--0.38 mm lateral error. Two
historical SC layouts failed with larger lateral approach error. The
two-card SFP scene was partial. These are insertion/approach failures, not
verified cable-to-card snags. See the [audited scenes and videos](experiments/2026-09-23-ordinary-development-cable-audit.md).

An older [expert-generator note](expert_matrix_template_fixes.md#sc-to-sc-nic-bypass-full-insertion-template)
does record a five-card VLM/MoveIt route where the center-camera reviewer saw
the cable catch on the NIC stack; a later outside-left bypass succeeded with
score 89.17. Its failed raw trajectory/video is absent from the checked local
and S3 clean archives. Of 190 retained SC full-insertion agent/VLM replay
attempts, 150 scored at most 1. A saved three-card score-1 trajectory with a
50.5 N transient force peak was replayed against its exact scene and again
scored 1, but both runs failed later at a roughly 14--16 mm lateral port
handoff miss. The TCP kept moving through the old force spike. Regenerating
the first five-card seed-51500 scene and running stock CheatCode produced a
partial axial insertion without a visible cable snag. **The historical
route-dependent snag is credible but is still missing a retained causal
incident suitable for RL labels.** The outside-left bypass remains a route
prior; cable-specific RL stays gated on a recorded contact/tension incident.
No model training was run in this audit.

## Ordinary-evaluation cable coverage correction (September 23)

The recent Gazebo audit did not cover every normal development setting. The
current three-trial `sample_config.yaml` has one-card SFP-to-NIC scenes and a
zero-card SC-to-SC scene, so it cannot expose SC cable snag among several NIC
cards. The selected production-family stress runs were also not an exhaustive
`training_broad` sweep.

The unfiltered historical archive contains 275 excluded SC-to-SC episodes from
randomized one-through-five-card Gazebo collections. Of these, 212 have official
noninsertion scores from collections deliberately stopped near the gate. The
remaining 63 lack the raw score or trajectory lineage needed for verification;
the exclusion count is not a cable-failure count. These used planned initial
motion followed by CheatCode. Initial timeline review shows cable/card
proximity and mid-trajectory force spikes, but the retained data cannot name
the contacting body or prove that cable contact caused a stall. Cable snag is
therefore reopened as a high-priority causal audit rather than treated as
disproved by the recent bounded runs. See the updated
[ranked failure record](experiments/2026-09-23-ranked-failure-scenarios.md).

## Ranked failure scenarios and targeted reproduction (September 23)

A new evidence-based ranking separates learned-policy alignment drift, local
port-lip blockage, large approach blockage, and unverified cable snag. Ten
additional post-fix Gazebo SC trials used stock CheatCode. Three exact Trial 3
repeats and all three bounded grasp-offset repeats inserted fully. Both
five-card runs to SC rail 1 stopped at nearly aligned partial insertion. Both
five-card runs to rail 0 failed with 89--126 mm terminal TCP tracking error and
brief scorer force peaks of 32--47 N.

Camera review showed the cable clear of the NIC card field in these failures.
They increase the priority of local axial recovery and larger approach-routing
recovery, but they do not establish cable snag. The ordered continuation is:
repair the collision-faithful Isaac SC scene, collect local alignment/contact
incidents first, train matched correlated-exploration and measured-backtrack
RL arms, then add large obstacle routing. Cable-specific RL remains gated on a
causal snag incident. See the
[ranked failure record](experiments/2026-09-23-ranked-failure-scenarios.md).

## Official Gazebo CheatCode failure audit (September 23)

The previous failure taxonomy mixed Isaac development evidence, broad Gazebo
data-collection scenes, and hypotheses. It was not a production failure census.
The corrected audit ran the stock official CheatCode in rootless Docker against
the exact released qualification YAML five times. SFP inserted in 10/10 trials.
SC inserted in 4/5; the remaining run was a partial axial block with about
0.06 mm lateral error and 0.19 degrees orientation error, but the plug stopped
3.21 mm outside the entrance and emitted no insertion event. CheatCode still
returned `True`, so engine task completion is not an insertion label.

A separate, labeled production-family stress suite covered SFP and SC, card
counts 0/1/3/5, both SC rails, a rail-limit position, released board poses, and
norm-bounded grasp changes. Card count alone did not cause a snag: every
0/1/3/5-card SC trial inserted. The SC/Trial-1-board-pose cross-combination had
an 8.05 mm lateral miss and 72.6 mm TCP tracking error. Of three five-card
grasp tests at 2 mm translation norm and 0.04 rad rotation norm, two inserted
and the negative-y/negative-pitch case stopped axially after partial insertion.
No current production run established a cable snag or named gripper-card
collision. The later targeted suite above shows that five-card outcomes can be
worse and intermittent even though the visible cable remains clear of cards.

Further recovery work starts from failures actually recorded in post-fix
Gazebo: intermittent near-aligned SC axial blocking and larger approach
tracking failures. See the
[official audit](experiments/2026-09-23-official-cheatcode-failure-audit.md).

## SC-to-SC mechanics and routing bring-up (September 23)

The approved multi-card SC continuation reached its simulator-mechanics gate.
The audit corrected the SC port mount, reversed cable endpoint topology and
fixed-joint transforms, reset ordering, multi-card scene support, and the SC
contact target. With normal collisions and no intervening cards, a scripted
14 mm retreat plus positive 4 mm lateral retry reached 0.099 mm final lateral
and 0.008 mm axial error. The matched negative retry entered the corridor only
briefly and diverged, confirming that recovery direction matters.

The apparent five-card cable snag was instead dominated by the gripper housing
striking a NIC card. Removing plug and cable collisions did not change the
failure; direct contact sensing measured about 128 N at the gripper/card
contact. The source reversed Gazebo grasp was not physically reachable in the
current Isaac base/board arrangement. Therefore no SC BC or RL was trained and
the failed rollout is not labeled as cable-snag data.

A follow-up source audit found that Gazebo intentionally removes two endpoint
collision groups near the palm and shortens/shifts the first rope collider. The
Isaac builder now reproduces that contract and uses the equivalent wrapped
wrist solution. This did not fix the full scene: a hold still produced a
36.98 kN peak. Removing the board, port, and self collision made the motion
low-force, but physical interpolation stopped 44.27 mm from the requested tip
pose. The remaining fault is narrower but unresolved: grasp transform, scene
placement, or articulation actuation. The SC learning gate remains closed.
The corrected asset is stable at its native initial posture, which localizes
the failure to near-port placement/reset rather than an always-unstable cable
topology. Gazebo welds to `ati/tool_link`; the prepared USD welds to the right
gripper finger. That frame contract and the source board/port placement are the
next items to reconcile.

A declared gripper-collision-disabled proxy tested routing while retaining
plug, cable, card, board, and port collisions. A direct high approach held the
final component gate in 1/3 seeds; a privileged 100 mm route around the card
edge held it in 3/3. This supports hierarchical routed transport as a future
candidate, but it is neither autonomous nor valid policy data. The active gate
is a reachable collision-faithful SC grasp/scene contract, followed by scripted
card-count 0--5 validation. See the
[SC mechanics record](experiments/2026-09-23-sc-mechanics-and-routing.md) and
[machine summary](../outputs/experiments/2026-09-23_sc_cable_bringup/summary.json).

## Probabilistic SERL recovery continuation (September 23)

The deterministic local RPDP controller was converted into a four-component
mixture over complete four-waypoint port-frame trajectories. It is a full
policy rather than a residual correction. Frozen perception plus the actor has
2,586,195 parameters, and complete live inference measured
20.43/22.67/23.78 ms p50/p95/p99. Measured-path backtracking, force plus stall
detection, lateral post-clearance exploration, replay ownership masks, balanced
prior/online replay, twin critics, and mixture SAC updates are implemented.

Frozen stochastic collection produced 3/7 local 8 mm seating successes. The
first backtracking implementation was invalid because force alone triggered on
normal insertion and abort persisted. After repair, no blockage trigger fired
in five episodes; its failures were lateral drift/timeouts, so backtracking had
no opportunity to help. A 12-episode prior contains seven successes and five
failures. A ten-episode critic warm-up ranked its one held-out success above its
one held-out failure, but this two-episode check is weak.

On eight new episode-grouped development starts, frozen BC scored **3/8**.
Initial correlated and backtracking SAC actors both scored **0/8** because the
actor collapsed deterministic probability onto unvalidated alternative modes.
Adding full-mixture behavior regularization stopped that collapse, but the
requested lower actor learning rate was silently overwritten when optimizer
state was restored and the result remained 0/8. Optimizer restoration now
retains momentum while reapplying explicit learning rates. The final tighter
trust-region run scored the same **3/8 on the same three episodes** as frozen
BC. It preserved the warm start but did not improve it.

A separate three-camera recording rerun on the same eight configurations scored
**1/8 BC, 1/8 tight-trust RL, and 3/8 tight-trust RL plus backtracking**. All 72
per-episode clips are on the [video review page](../outputs/experiments/2026-09-22_serl_recovery/videos/index.html).
Because BC and plain RL differed from their earlier 3/8 metrics-only outcomes,
the recording rerun is preserved as separate evidence and does not establish an
RL or backtracking improvement.

Do not promote this online checkpoint as better than BC and do not open the
reserved final split. Before another actor update, collect substantially more
episode-grouped recovery and blockage outcomes and validate critic ranking on
more than two held-out episodes. The selected checkpoint is preserved because
it verifies a mechanically safe continuation path. See the
[experiment record](experiments/2026-09-22-serl-mixture-recovery.md),
[actor explanation](SERL_PROBABILISTIC_ACTOR.md), and
[recovery strategy](SERL_RECOVERY_STRATEGY.md).

### Video diagnosis and current boundary

The three-camera rerun was reviewed configuration by configuration using the
recorded axial depth, lateral offset, orientation, and cable motion. The dominant
failure is lateral divergence: several actors move from a nearly centered reset
to 6--18 mm off-axis, sometimes crossing the entrance plane beside the port.
One configuration stalls at the port lip with roughly 0.04--0.05 rad angular
error. No episode provides conclusive evidence of a cable snag. Cable motion is
a possible contributor in configurations 12 and 14.

The recovery controller does more than retreat. It retraces measured TCP
positions in 0.25 mm commands until at least 1 mm of clearance and low force,
then constrains 12 control steps toward near-lateral motion. One millimeter is
twice the 0.5 mm local success corridor, so this can release a small lip contact.
The recording did not retain per-step recovery-mode traces, however, and the
visible backoff may also be a learned action. The two additional successes do
not prove that hard-coded recovery caused the improvement. See the
[committed failure analysis](experiments/2026-09-23-serl-video-failure-analysis.md).

### Approved SC continuation now in execution

The active work targets SC-to-SC with multiple intervening cards, where
the cable can require a larger retreat and a routed detour rather than local
realignment. Current selected pose/RPDP/SERL evidence is SFP-to-NIC only. The
ordered plan first audits Isaac SC support, defines cable snag causally, records
a bounded full-episode discovery set with compact force/motion/recovery
telemetry, and creates reproducible short incident resets. It then gates SC pose
estimation, port-relative BC, fixed multi-scale recovery, a learned recovery
option, offline critic validation, short-incident online SERL, and finally
full-episode evaluation. See [SC cable-snag recovery plan](SC_CABLE_SNAG_RECOVERY_PLAN.md).

## RPDP and DPPO continuation (September 22)

**Scope correction:** the RPDP evaluations below use
`scene.target.seated_depth_m: 0.008` and privileged near-opening resets. The
physical SFP entrance is about 45.8 mm from the seated port reference and the
cage is about 48.72 mm deep. Therefore these successes are **local 8 mm seating
events**, not official full-depth or full-episode insertions. The controller has
not yet been evaluated from the normal initial distribution or with the
official Gazebo insertion event. Earlier wording saying simply “strict
insertion” was too broad.

The selected supervised controller is now a PoseInsert-inspired, deterministic
AIC-RPDP policy. It predicts the connector pose after each of four recorded
50 ms teacher target commands, always relative to the fixed port opening. A
verified rigid transform converts those four waypoints back to four
TCP-body-frame commands. This is a complete target trajectory, not an additive
correction to ACT and not a base-frame absolute action.

The follow-up audit fixed four material BC problems: recovery data had used the
75% blended executed action instead of the recorded teacher target; four
different commands had been compressed into one 200 ms endpoint; a numerically
oversized checkpoint-improvement threshold kept undertrained update-200
models; and live insertion commands were sometimes clipped from 0.5 to 0.2 mm.
The original 322-row diagnostic also lacked insertion, but the current data
does not: the 823-row successful fit split contains 424 insertion decisions.
A DAgger extension keeps safe teacher queries from failed blended rollouts
while rejecting stale reset images and force at or above 5 N, producing 980 fit
decisions.

The repaired diffusion arm improved to 2/6. A matched deterministic trajectory
regression arm then achieved **6/6 autonomous local seating events** on the fixed
development starts and **8/8** on an additional set unused by RPDP fitting or
selection. Across all 14 episodes, final lateral error was 0.024--0.402 mm,
orientation error was 0.002--0.066 degrees, post-reset maximum force was
1.69--3.69 N, and complete p95 inference was 19.86 ms. Evaluation used no
guide, guard, exploration, privileged geometry, or RL.

A later frozen five-start video check achieved **4/5 local seating events** in the same Isaac
SFP-to-NIC scene and held-out cable-template family. The failure timed out at
5.987 mm depth despite 0.329 mm lateral and 0.032 degree orientation error;
the four successes reached 7.562--7.640 mm. Complete inference was 20.12 ms
p95. The cumulative post-repair development observation is therefore 18/19 and
shows that slow or stalled axial completion remains possible. Three per-episode
camera videos, a review page, and exact metrics are linked from the
[RPDP experiment record](experiments/2026-09-22-rpdp-dppo.md#post-selection-five-rollout-video-check).

The selected supervised arm has 2,004,329 trainable policy parameters plus 536,631 frozen
perception parameters, or 2,540,960 total. It passes the supervised development
gate. The older DPPO round remains a valid but unsuccessful 1/6 result and was
not continued. The four reserved final configurations remain sealed; no new
RL from the original RPDP/DPPO stage ran. The later probabilistic SERL
continuation is reported above. No world-model, SEER, or Gazebo-adaptation
training ran. See the
[RPDP/DPPO and BC repair record](experiments/2026-09-22-rpdp-dppo.md).

The staged
[perception, supervised control, and model-free RL experiment](experiments/2026-09-20-perception-supervised-rl.md)
has now completed its bounded supervised-controller diagnostic. Corrected
plug/opening supervision and oracle triangulation passed earlier, and the later
natural-cable observation-only estimator was frozen despite its remaining p95
tail. A matched action-only versus pose-conditioned GRU comparison then reached
**0/8 strict insertions in both arms** on new development starts. The policy
learned to make almost the same commands when pose inputs were zeroed or
shuffled. The predeclared stop rule therefore keeps RL, SEER adaptation, and
predictive world-model control parked.

The approved explicit-correction continuation is also complete. Force-safe
collection retained 322 fit rows from 25 episodes with all eight lateral
directions paired, plus 124 calibration rows from seven unseen-template
episodes on two paired axes. A 66,596-parameter correction was forced to pass
through the predicted plug-to-opening translation on top of the frozen
546,686-parameter nominal GRU. On calibration it reduced translation command
MAE from 0.04786 to 0.04638 mm, only 3.10%, and produced a 0.02909 mm median
correction; the fixed gates required 5% and 0.030 mm. A second checkpoint
selected directly on translation MAE still failed, shuffled pose slightly beat
correct pose on calibration, and zero pose beat correct pose on the untouched
offline development split. The six live starts were therefore not opened. No
autonomous rollout, final evaluation, or RL ran. See the
[explicit-correction record](experiments/2026-09-22-explicit-pose-correction.md).

The latest explicit-opening continuation used 20 train configurations, six new
development configurations, original-resolution 160x160 crops, six landmark
heatmaps, and per-observation three-camera calibration. Calibration rejected
the landmark and temporal variants and selected the coarse RGB locator. Across
240 near-port development decisions it reached **1.085/1.740 mm median/p95
lateral error**, 91.5% correction-direction accuracy, **1.204/2.928 mm
axial error**, and 0.211/0.363 degree orientation error. Complete inference was
1.406/1.982/58.099 ms p50/p95/p99. A world-coordinate temporal filter improved
lateral error to 0.862/1.505 mm only as a post hoc, promotion-ineligible
diagnostic. The accuracy gate remains closed.

A subsequent calibration-only ImageNet MobileNetV3-small landmark model was
close enough to earn six further fresh development starts. It is only 233,662
parameters and uses a 160x160 native crop. On 173 near-port decisions it reached
the current best observation-only result: **0.278/0.670 mm median/p95 lateral
error**, 94.1% direction accuracy, **0.664/1.363 mm axial error**, and
0.215/0.506 degree orientation error. Complete inference was 5.450/5.949/71.825
ms p50/p95/p99. It still fails the fixed 0.25/0.5 mm gate, so policy and RL
remain stopped.

A post hoc causal world-space opening filter reached 0.250/0.616 mm lateral
error and 97.8% direction accuracy, while axial error regressed to
2.085/4.407 mm. It was tested after that development split was opened and is
diagnostic only. Since its p95 still failed, no additional split was collected.
Heatmap-confidence camera rejection also failed to improve calibration, which
selected all three views; no further development collection was justified.

The approved temporal/multiview continuation is now complete. A replay audit
found strong cable occlusion in the sparse center-camera mask sample but clear
side views, and no independent cable-shape reset variable. It also fixed a
null episode-index bug so causal history resets at actual simulator resets.
Learned current-frame multiview and six-step temporal heads passed calibration,
then were evaluated once on six new reset positions. The current-frame head was
best on fresh data at **0.307/0.573 mm** near-port lateral median/p95, 99.4%
direction accuracy, and **0.304/0.796 mm** axial median/p95. The temporal head
reached 0.289/0.667 mm and did not reproduce its calibration p95. Full temporal
perception plus the existing trunk measured 6.82 ms p95. Accuracy still fails
the 0.25/0.5 mm gate, so no policy, final split, or RL was opened. The next data
collection must independently vary cable shape/occlusion and retain visibility
or depth supervision before another temporal attempt. The articulation has 40
cable joints in addition to the six arm joints, while the current reset event
selects only the arm. A safe follow-up needs seeded cable-shape templates,
physics settling, collision/force rejection, and a compensating arm solve that
restores the requested plug pose; unconstrained cable-joint noise would confound
the experiment.

The cable-shape/visibility continuation and its force-safe follow-up are
complete. Exact 40-joint resets initially caused a 35 N transient and plug
drift. A closed-loop collection controller now waits for settling, rejects
force above 5 N, restores the measured plug pose, and releases to the fixed
CheatCode teacher only after two safe observations. This is privileged data
generation and is never available to the autonomous actor.

The full natural-trajectory run saved 1,500 causal decisions across 50
episodes. Thirty-five episodes handed off successfully; filtering retained 460
safe near-port observations from all five included cable templates. The frozen
complete-template split used 205 fit, 79 calibration, and 175 development rows.
The calibration-selected current-frame model reached **0.322/0.626 mm**
near-port lateral median/p95 and 100% direction accuracy on 153 samples from ten
held-out episodes. A fixed temporal variant reached **0.247/0.617 mm** but still
failed the p95 requirement. Combining static and natural data was worse at
0.331/0.753 mm. Complete perception plus the existing trunk measured 7.18 ms
p95. The 0.25/0.5 mm gate still fails, so no policy, RL, or reserved-final
evaluation was opened. The subsequent frozen-model audit found that the right
camera was strongest and the center camera weakest, but every two-camera subset
had worse lateral p95 than the three-camera model. Exact labelled pixels gave a
sub-0.0011 mm two-camera triangulation ceiling, confirming consistent camera
geometry. A fixed per-camera bias correction worsened p95 to about 0.79 mm.
The tail was concentrated in one cable episode and small-motion rows; clear
views were worse than occluded bins, so occlusion is confounded with trajectory
rather than established as the cause. More independent safe cable layouts and
camera-specific landmark supervision are required before another frozen
representation test.
See the [execution record](experiments/2026-09-22-cable-visibility-perception.md).

A label-free qualitative audit now overlays only model predictions and their
three-camera reprojections; it never reads simulator plug/port pose labels from
the reviewed episodes. The predicted opening aligns visually with the SFP cage,
including cable-obstructed examples, and cross-view reprojection RMS is
0.807/1.309 pixels median/p95. The predicted stationary opening nevertheless
jitters by 0.369/1.316 mm and its normal by 2.98/7.97 degrees. This confirms the
model is attending to the intended region while independently motivating causal
temporal fusion; it is not a substitute for the labelled accuracy metrics.

The frozen-pose supervised control diagnostic is complete. Two matched
546,686-parameter policies used six causal decisions and emitted four 6D
TCP-body-frame delta commands. One received the frozen visual features, state,
measured state change, force, previous action, and visibility; the other also
received predicted relative translation, uncertainty, and observation-derived
phase. On offline held-out episodes, action-only was marginally better
(0.0367/0.0894 mm translation MAE/p95 versus 0.0369/0.0908 mm). On eight new
autonomous starts each, both scored **0/8 strict insertions**. Best lateral
p50/p95 was 1.479/1.678 mm for action-only and 1.417/1.871 mm for
pose-conditioned; terminal medians grew to 6.382 and 6.284 mm. A counterfactual
audit showed that zeroing pose changed development commands by just
0.0095/0.0264 mm p50/p95, so the generic BC head largely ignored the added
signal. Full inference was 7.135/7.388/8.249 ms p50/p95/p99. The final split
stayed sealed and no RL ran. See the
[execution record](experiments/2026-09-22-pose-conditioned-gru-policy.md).

The selected full-data world policy remains the strongest preserved Gazebo
controller: 2/20 full insertions, 9 partials, mean official score 45.14. Its
learned dynamics is worse than persistence, so predictive imagination remains
disabled.

The September 20 Isaac continuation resolved the earlier joint/reset diagnosis.
The expert joint convention is usable; the original supported YAML had placed
the port about 552 mm from the expert pose. A second silent issue then showed
that the first generated scene variations retained a joint-state reset and
ignored their requested Cartesian starts. The corrected reset uses 6D damped
IK and reproduces requested 4--10 mm axial and approximately 0.7 mm lateral
starts within the declared tolerances.

On those corrected starts, the privileged adaptive guide achieved 9/9 strict
insertions in its first pass and 7/8 in its second. The world-policy replay path
stores exact 24D four-command transitions, 384D frozen causal features, guide
labels, and commands actually executed. The trainable 333,024-parameter head
receives nonzero gradients; its frozen perception/world trunk has 22,236,320
parameters.

The autonomous gate still fails. The round-one aggregate, round-two aggregate,
and varied-only heads each achieved **0/4 strict insertions** on heldout IK
starts. The varied-only head was the most consistent: all four trajectories
reached within 0.10 mm before the opening plane, but lateral offset at closest
approach remained 3.25--3.89 mm against the 0.5 mm success threshold, and force
reached the configured 35 N clip. The four reserved final starts remain
untouched. No Gazebo transfer or actor-Q update was started because the
supervised anchor does not insert. The diagnostic checkpoints, videos, and
machine-readable summaries are under
[`outputs/experiments/2026-09-20_isaac_world_rl/`](../outputs/experiments/2026-09-20_isaac_world_rl/).
See the [execution record](experiments/2026-09-20-isaac-world-policy-rl.md).

The privileged guide is an **Isaac CheatCode policy**, not a new Gazebo
implementation. It uses the same ground-truth plug-to-port rigid-transform idea
as the existing Gazebo CheatCode, but is currently a local near-port feedback
servo rather than the Gazebo policy's full approach, settle, minimum-jerk
insertion, and completion-event sequence. The 552 mm finding is a spatial scene
placement mismatch, not a purely lateral port-frame error or proof that every
cross-simulator asset transform agrees.

DAgger fractions apply to every command rather than to a fraction of timesteps:
50% means `0.5 * model + 0.5 * guide`, while 25% means
`0.75 * model + 0.25 * guide`. Replay keeps the unblended Isaac CheatCode chunk
as the BC label and the blended command as the action that produced the next
state. The detailed record also specifies the Isaac shaping reward and makes
clear that it is not an official Gazebo score.

The causal replay audit is complete. Earlier replay joins were unsafe at every
terminal because Isaac had already reset before returning `next_obs`; new
collection retains a separate pre-reset terminal snapshot and all requested
causal decision fields. The frozen 384D pose probe was evaluated on four unseen
episode-grouped configurations. Near the port it reached 0.730 mm median and
2.533 mm p95 lateral vector error with 93.5% correction-direction accuracy.
This fails the predeclared 0.25/0.5 mm resolution gate, so no conditioned policy
was trained. The bounded native 576×512 crop ablation is also complete. Its
learned RGB locator uses simulator masks only as training labels and selects
evaluation crops from RGB. On the matched 225-decision heldout set, near-port
lateral error improved only from 0.962/2.630 mm median/p95 to 0.851/2.154 mm,
while correction-direction accuracy fell from 84.8% to 80.4%. It therefore
failed the material-improvement and absolute resolution gates. Complete live
inference measured 2.96/3.20/182.50 ms p50/p95/p99, satisfying the 300 ms p95
limit. The policy comparison was correctly skipped. The reserved final starts,
actor-Q, Gazebo transfer, reward-model training, and imagination remain closed.
See the
[full execution record](experiments/2026-09-20-isaac-world-policy-rl.md#causal-replay-audit-and-frozen-feature-probe).
The record's [plain-language interpretation](experiments/2026-09-20-isaac-world-policy-rl.md#plain-language-interpretation-of-the-crop-experiment)
explains the crop stages, median/p95 errors, label limitation, recent robotics
alternatives, and why pose prediction alone does not perform insertion.

The crop failure was subsequently traced more precisely. The old labels were the
SFP module centroid and nearest NIC-card mask pixel, not the plug tip and port
opening. An oracle-coordinate ceiling using those labels still failed at 1.132
mm near-port lateral median and 3.141 mm p95, so further fitting to them was
stopped. The Isaac camera projection path was repaired and visually checked on
all three views. The replacement 1,057-decision train and 225-decision held-out
collections are complete. Exact-keypoint triangulation reached 0.124 mm median
and 0.235 mm p95 near-port lateral error, proving the geometric ceiling is
sufficient. The deployable calibration-selected RGB locator reached only 2.380
mm median and 6.947 mm p95, with 82.6% correction-direction accuracy. It failed
the 0.25/0.5 mm and 90% gate. Complete inference p95 was 3.311 ms. Accordingly,
that stage did not train a pose-conditioned policy or RL. A later frozen-pose
diagnostic is reported above; the four final starts remain sealed. See the
[decision record](experiments/2026-09-20-perception-supervised-rl.md#executed-result).

## TCP-delta commands and tracking: current decision

**Train on the clean expert's commanded TCP-frame delta pose**, relative to the
TCP pose in the same observation. This delta is a desired *controller pose
reference*, not the displacement the TCP must achieve before the next image.
For corrective recordings, use `teacher_target_pose` for policy supervision and
`executed_target_pose` for action-conditioned dynamics; they can differ. Do not
replace the command label with the difference between two measured TCP poses.
The latter includes controller lag and contact and is useful for dynamics or
path evaluation, not as a drop-in `MotionUpdate` target. See
[`CollectCorrectiveCheatCode.py`](../aic_example_policies/aic_example_policies/ros/CollectCorrectiveCheatCode.py)
and [dataset eligibility](DATASETS.md).

An audit of **23,252 consecutive 50 ms observation pairs** in the 74 verified
one-NIC, no-SC aligned/corrective SFP episodes found these median translation
magnitudes: **9.83 mm** from the observed TCP to the *executed command target*,
**0.64 mm** actual TCP displacement during the next 50 ms, and **9.35 mm**
remaining from the next measured TCP pose to that same target. These are
separate medians, so they need not subtract exactly. In the last three recorded
seconds, the corresponding medians were 5.35 / 0.04 / 5.36 mm. The clean
teacher delta was also 9.83 mm median across all 32,183 native observations.
This is a target-tracking gap, **not** a 9.35 mm expert-label or model-prediction
error. The consecutive-pair audit used the saved `states.npy`,
`executed_physical.npy`, `teacher_physical.npy`, and each episode's native times
and command indices under the [Dreamer pilot data](../outputs/experiments/2026-09-18_dreamer60_pilot/artifacts/data), retaining only 50 ms intervals with
consecutive command indices. Position-delta magnitude is unchanged when a TCP
delta is composed into an absolute `base_link` target.

These episodes all have official full-insertion Tier 3 scores. They represent a
restricted aligned scene without SC distractors, so they do not establish
general performance in cluttered settings. The wider historical CheatCode SFP
collection has **140/140 verified full insertions**; its action labels are much
cleaner than the successful agent/VLM recordings with missing Cartesian labels.
The CheatCode policy itself issues a smoothly changing target about every
50 ms, including slow insertion and a settling phase. Its success despite the
gap above argues **against** waiting for every commanded pose to be reached or
shrinking the learned delta to the next observed displacement.

The existing [AIC controller](aic_controller.md) already runs at **500 Hz**,
interpolates position and orientation references, and applies Cartesian
impedance control. Its tracking-error reset is a coarse stuck-target safeguard
(`min_translation_error: 0.2 m`, timeout 2 s in
[`aic_ros2_controllers.yaml`](../aic_bringup/config/aic_ros2_controllers.yaml));
it is not a millimetre-level waypoint gate. The synchronized policy
[`Observation`](policy.md) arrives at up to **20 Hz**. In `insert_cable()`, use a
fresh observation as the inference event, publish the bounded Cartesian
`MODE_POSITION` target at the 20 Hz cadence, and let the controller run its own
fast loop. A chunked model may infer once per four commands, but every delta
must retain its trained observation-reference convention. For an
observation-relative delta, compose it with that observation's TCP pose and
publish the resulting absolute target in `base_link`; sending the delta in
`gripper/tcp` lets the controller apply it from the *later* TCP pose at receipt.
The runtime already has this observation-relative transport path in
[`RunACTTorchScript.py`](../aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py).

**Do not add general tracking-aware action-chunk pacing by default.** A second
fast loop would duplicate the controller's interpolation, and a reach-before-
advance rule would distort these successful expert commands, especially in
contact. Keep the nominal command cadence and log measured TCP, reference
pose, force, command, and simulated time. Consider a guarded
alignment-to-insertion transition only if the learned policy shows excess
off-path motion or harmful contact relative to experts, using observable
signals rather than privileged geometry. Compare the same frozen policy with
and without that guard on matched simulator scenes before adopting it. The
expert tracking-gap statistics alone do not demonstrate that extra pacing
will improve insertion.

## September 18 work

At 18:14 UTC the action contract was corrected: the verified expert label is
a **TCP-frame delta relative to the recorded observation**, rather than an
absolute base-link target for the model to predict. The completed all-data ACT
and fresh ACT60 results below used absolute predictions and remain historical
baselines. The new strict 60/14 ACT run completed 6,000 updates at 18:55 UTC
on the 32,183 native observation-command pairs, predicting one TCP delta per
observation. Its fixed held-out rule selected update 6,000 (2.96 mm first
command and 2.63 mm final-three-second translation error); four fresh paired
development scenes ran from 18:56 to 19:09 UTC: all four valid, **0/4 full
insertions**, mean official score **27.73**. Three ended 5 cm from the port;
one drifted to 33 cm. These imitation errors are not live insertion scores.
The corrected world dynamics completed 3,191 updates and stopped early after
its fixed held-out prediction gates failed; the selected update-1,000 model's
one-step TCP error was 10.33 mm versus 1.85 mm for persistence (14 episodes).
Here, one step means **200 ms and four actually executed 20 Hz commands**.
Starting from a recorded observation, the dynamics model predicts the TCP's
*measured position after those commands*. The 10.33 mm is the mean distance
between that predicted future position and the recorded future position in
`base_link`; it is not the error between a controller command target and the
robot. The 1.85 mm persistence baseline instead predicts that the future TCP
will remain at its initial measured position, then compares that unchanged
position with the same recorded future position. It works well over this short
interval because the TCP usually moves little, even when the commanded target
is farther away. See the [held-out dynamics evaluator](../outputs/experiments/2026-09-18_dreamer60_pilot/world_tcp_delta_final_runtime_archive/sources/dreamer_source/dreamer4/aic/evaluate_world.py).
A fresh supervised policy using corrected delta labels ran from 18:46 to
19:20 UTC on GPUs 2–3, stopping at update 4,509 after a documented held-out
plateau; fixed selection chose saved update 4,000 (2.16 mm combined
first-command error). The first live world startup found a camera-size
mismatch before any command. After matching the collector's resize to
288×256, trained raw-camera inference measured **32.76 ms p95** over 1,000
calls and the four-command callback loop **33.79 ms p95** in isolation.
The same-GPU live diagnostic measured **464.67 ms p95** across 168 published
commands, above the 300 ms requirement. Separating renderer and policy GPUs
alone did not fix this. A private inference worker preserved byte-identical
commands on verified native samples and passed a full live-scene diagnostic.
The fresh world development set was **4/4 valid, 1/4 full insertions**, mean
official score **52.14**, with pooled **81.00 ms p95** live decision latency.
The frozen paired final assessment completed all **20/20 eligible scenes per
policy**: ACT **0/20 full insertions**, one official partial, mean total
**22.69**; world **0/20 full insertions**, two official partials, mean total
**32.32**. The world policy's 8,908 live decisions measured **77.01 ms p95**,
under the 300 ms requirement. A failed ACT startup before trial 17's scored
rollout was preserved, then that same scene completed as an unchanged-model
retry. The previous absolute-action dynamics and BC runs are superseded.
The [follow-up](experiments/2026-09-18-world-followup.md) found the world
policy closer to the opening in more scenes but with persistent lateral error
and a large requested/measured TCP gap. Across all strict held-out 200 ms
dynamics windows, future measured TCP error was **12.06 mm** versus **2.29 mm**
for persistence; near the actual opening it was **13.94 mm** versus **0.57 mm**.
The 289-episode visual audit and matched tokenizer comparison improved SC
gross reconstruction with expanded data, but fine connector/port features
remained blurred. Reward and imagination training remain disabled. See the
[delta correction record](experiments/2026-09-18-tcp-delta-correction.md) and
[contract audit](../outputs/experiments/2026-09-18_dreamer60_pilot/act60_delta_contract_audit.json).

The [bounded supervised initialization comparison](experiments/2026-09-18-world-supervised-init-ablation.md)
trained the same six-view control architecture for 2,500 BC updates in each
arm and evaluated four fresh paired scenes disjoint from all 289 verified
episodes and the final set. Held-out first-command error was 2.285 mm with
selected world weights versus 2.578 mm from a fresh world trunk. Live mean
official score was 32.85 versus 26.24, with **zero full insertions in either
arm** and one partial for the selected-world arm. Both met the 300 ms limit:
pooled p95 command latency was 76.26/79.69 ms. The fresh arm's world weights
were a separate random draw rather than the exact pretraining ancestor, so
one seed cannot establish a causal pretraining advantage. The reward and
imagination gates remain closed.

The [September 19 full verified-data run](experiments/2026-09-19-full-world-training.md)
is complete. A fresh six-view tokenizer optimized all 250 canonical training
episodes, held out the 39 scene-disjoint validation episodes, and stopped at
78,000 updates under the declared validation plateau rule; update 75,000 was
selected. On 156 held-out frames it reduced whole-image/contact-field MSE by
91.25%/93.29% and contact edge L1 by 53.09% relative to the earlier bounded
reference. Corrected dynamics also converged by rule, but future TCP error was
11.46/19.17/20.12 mm at 200/400/600 ms versus 2.07/4.42/7.24 mm for persistence.
Every dynamics gate failed, so reward and imagination remain disabled.

The supervised full-data world policy selected update 1,000 and then completed
all **20/20 eligible sealed final scenes: 2 full insertions, 9 official partials,
mean total 45.14**. It beat the corrected60 ACT score on 13/20 paired scenes;
ACT remained 0 full / 1 partial with mean 22.69. Across 8,895 live decisions,
the new policy measured **80.77 ms p95**, 98.29 ms p99, and 146.74 ms maximum,
with no 300 ms misses. This is the strongest learned result in that paired set,
but 2/20 is not reliable and the complete-pipeline comparison does not isolate
which training change caused it. Next, run the full-data same-architecture
supervised comparison from the exact preserved random ancestor versus selected
world weights on new development scenes and multiple seeds. Keep the sealed
final scenes out of selection.

For size context, ACT has **16,354,566 inference parameters** (65,418,264
FP32 bytes), while the selected world policy has **22,569,344 acting
parameters** (90,277,376 FP32 bytes) and 27,130,942 parameters in the complete
training model. Thus the acting world policy is 1.38 times ACT's size. On their
matched corrected 60/14 data and frozen 20 scenes, ACT achieved 0 full / 1
partial insertion with a 22.69 mean score; the world policy achieved 0 full / 2
partial with a 32.32 mean. Neither met the reliability goal.

The user approved a new **6h50m ACT window, 11:51:28–18:41:28 UTC**, on physical
GPUs **0–1**. The canonical `expert_verified` collection now contains **289**
eligible episodes: 268 SFP across all five NIC counts and 21 newly collected
SC, with a grouped **250 train / 39 validation** split. The time-input
comparison used immutable v2 (260 episodes; 229/31). A matched visual-grid
comparison started on all of v3 at **15:18:10 UTC**, including its 29 additions.
The 253 successful agent recordings remain pending action-label repair;
212 non-insertions and 63 unresolved historical episodes are excluded.
Task conditioning and DDP passed GPU checks. The first SFP stage completed
6,000 updates in 53m33s at its requested cap. Mixed SFP/SC training started
13:16:39 UTC with batch 128 per GPU and balanced task/card-count sampling,
then stopped deliberately at 6,001 updates after 48m33s. This freed time for
a matched comparison of retaining versus masking elapsed time; it was not
stopped by the overall deadline or claimed to have fully converged.
SC collection accepted 21 of 36 attempts across two batches, all full insertions with
verified labels and terminal images, within a restricted scene distribution.
Historical SC command labels cannot be faithfully repaired. The initial
2,000-update ACT checkpoint completed five fresh development scenes with
**0/5 insertions**, mean score 31.71. A measured entrance stall was 7.91 mm
off axis. Step-4,000 and step-6,000 checks on two development scenes also had
zero insertions. The first mixed-task checkpoint completed **0/5 SFP and 0/4
SC insertions**; SFP NIC1 regressed while other SFP counts improved approach
scores. The time-input pair started at 14:06:35 UTC and completed 2,000 updates per
branch from the same mixed-task step-6,000 parent and immutable dataset.
Elapsed time is strongly associated with task identity in that dataset;
the first nine-scene evaluations gave **1/9 insertions for retained time**
(SFP 0/5, SC 1/4; mean 41.48) versus **0/9 for masked time** (mean 27.22).
The clean SC port-1 success scored 86.71. Those runs used a wall-time cap;
two masked SFP cases were still moving at the end. At 90 simulated seconds,
retained time completed **0/9**, mean **35.57**; its earlier SC success did not
repeat. Masked time had zero insertions and one shortened SC trial with contact
and force penalties. A separate full-duration retry also failed. Its supplementary
eight-valid-plus-retry comparison scored **23.22**, but does not erase the
original adverse outcome. Subsequent review found grossly displaced pre-command
robot starts in two masked SC trials, including the shortened one; the composite
is therefore not a clean policy comparison. The engine readiness source now
requires finite named arm joints near the configured home pose; it passed an
isolated rootless build/test but has not been installed into the pinned
evaluation image. Current trials restart the simulator for each scene and audit
the pre-command arm joints and first camera frames independently.
Retained time remains the parent for matched stride-32/stride-16 continuations.
Both completed 3,000 updates on v3, with held-out command errors **5.60 mm**
and **5.32 mm**; their fresh-simulator development trials completed with
physical start checks. No reliable learned insertion is established.
Their completed nine-scene fresh-simulator development results were
**0/9 insertions** each, mean score **33.27** for stride 32 and **24.44**
for stride 16. The earlier parent was also **0/9**, mean **23.03**.
The prespecified development rule selected stride 32; its frozen 32-scene
final assessment ran on GPUs 0 and 1 from 17:03 to 18:03 UTC. All **32/32**
trials passed the fresh-simulator initial-state and duration checks and were
officially scored. The policy achieved **0/32 full insertions** (SFP **0/20**,
SC **0/12**), mean official total **25.70** (SFP 21.04, SC 33.48). Two SC
trials were scored as partial insertions; nine trials incurred a prohibited
contact penalty. All 32 one-frame-per-second videos and start/end contact
sheets are archived with the selected model. The target was not met; no
offline or online SERL was trained in this window. See the
[ACT record](experiments/2026-09-18-act-all-verified.md) and
[frozen final report](../outputs/experiments/2026-09-18_act_all_verified_6h50/selected_act_final_single/final_single_results.md).

The parallel [Dreamer-v4 pilot](experiments/2026-09-18-dreamer-proposal.md)
was approved and started at 15:52:56 UTC, with a separate 22:42:56 UTC deadline.
Its strict first comparison uses the original 60 aligned SFP training episodes
and 14 held-out episodes. A fresh ImageNet ACT baseline completed 6,000 updates
on GPU 4; its four paired development scenes scored **0/4 insertions**.
Dreamer's first tokenizer stage stopped at 5,981 updates at its 50-minute cap;
two bounded detail refinements completed 2,500 updates each on GPUs 2–3, but
fine contact details remained blurred. A selected six-view tokenizer was frozen
for a supervised diagnostic. Its action-conditioned dynamics stage stopped
deliberately after 6,344 updates at 18:11:50 UTC; it beat persistence at
four/eight steps on average but failed the one-step and near-contact gates.
The held-out selected checkpoint was step 1,500. The superseded supervised BC
started at 18:12:54 UTC and stopped at 18:14:32 UTC when the action contract
was corrected. An untrained Dreamer inference path measured
28.49 ms p95 over 1,000 decisions including image
preprocessing and command conversion. Early tokenizer reconstructions blur
connector/port details. An audit of all 74 expert bags found the official
insertion event after the final saved observation in every episode, so the
strict pilot's insertion-state reward/imagination gate fails. The corrected
TCP-delta pilot's current training and live results are summarized at the top
of this page.

## Completed September 17–18 experiment

Updated: **2026-09-18**, branch `feat/hybrid-train`, base `7534090` plus
working-tree repairs and the direct visual actor. Evidence now includes CPU
regressions, the earlier 40-update BC smoke, **live Gazebo policy/control trials,
Isaac reset/terminal probes, a dataset audit, and new supervised ACT training**.
The earlier live-validation pass used at most two GPUs and did no training.
The eight-hour ACT experiment completed training and final evaluation using
physical GPUs 0–3, within its **2026-09-18 05:12:28 UTC** deadline.
[ACT experiment results](experiments/2026-09-17-act-verified-8h.md) ·
[Earlier results and videos](experiments/2026-09-17-live-validation.md) ·
[Initial audit](experiments/2026-09-17-reentry-audit.md) ·
[Accounting repairs](experiments/2026-09-17-evaluation-curriculum-fixes.md) ·
[Actor implementation](experiments/2026-09-17-direct-visual-policy.md).

## Where we stand

There is **no verified reliable general learned insertion policy in the evidence reviewed**.
The repository has usable training and evaluation machinery, ACT checkpoints,
many failed or partial RL experiments, and privileged local insertion diagnostics.
These have different evidential value; see the [experiment ledger](EXPERIMENTS.md).

The verified-data ACT experiment finished with **1/20 full insertions** on
fresh SFP NIC-1 / card-0 / port-1 / rail-0 scenes (mean total **39.16**).
All 20 evaluations completed normally with ground truth disabled. The **18/20
reliability target was not met**. The selected model had 2/10 development
insertions; its checkpoint, normalizer, runtime, lineage and exact scene set are
preserved in `outputs/experiments/2026-09-17_act_verified_8h/selected_act_final/`.
See the [report and videos](experiments/2026-09-17-act-verified-8h.md).

Historical RGB/BGR and wall/simulation-clock differences were identified; new
collections record aligned RGB observations and teacher commands. Quaternion,
command transport and warm-start normalization contracts were checked/repaired.
At the end of that earlier run, its newest cache held **87 recent episodes
(70 train / 17 validation)**;
the selected continuation used 60 train / 14 recent validation episodes from the
74-recording predecessor. There were 47 bounded ACT jobs, including small fitting
diagnostics and continuations. **No offline or online SERL updates ran.**

Training-command errors fell below 1 mm while recent held-out errors remained
around 4–5 mm. On one final failure, the tip was at the port entrance but
12.6 mm off axis, while the policy continued requesting descent. A separately
labeled privileged expert control on that same scene scored 94.68
(Tier 3 75); it is never counted as learned-policy success.

The latest located long run started June 14 and ended June 19. Its recovered
events show **200 evaluation cycles, zero successful evaluations, and zero
promotions**, ending at level 0. The recorded `episodes_used=2000` is a wrapper
budget counter: the old wrapper added 10 after every timed training segment, without
counting completed simulator episodes. Do not report this as 2,000 measured
episodes. The July handoff note remains useful historical context.

The ACT 175k CPU/CUDA exports, associated pretrained checkpoint/normalizer, and
historically named “clean” dataset are present locally. The new audit found
action-label and split-quality concerns in that dataset. Their presence establishes availability.
The inspected May 13 Gazebo evaluation has `policy_ready=false` and no score.
The May submission bundle's referenced model/evaluation directories are missing
from this checkout; its reported scores are historical, not reverified.

## Architecture direction

The current requested sequence is **ACT → offline SERL → online SERL**. The
completed experiment trained and evaluated ACT only, starting with verified
CheatCode demonstrations, then aligned expert and corrective collections.
Episode holdouts and a frozen 20-scene assessment were used. No offline or
online SERL updates have been run in this pass.

The [June 17 tutorial notes](https://github.com/yoonjung0705/tutorials/blob/master/python/libraries/tutorial_aic.md#617-2026-hybrid-train-branch-model-training-especially-for-isaac)
call for dropping the residual adapter, giving the actor direct visual features,
considering DINOv2, and reconsidering whether ACT warm starting is needed. They
also report an expert rollout withdrawing when initialized already inserted.
The two CheatCode controllers were found to request a 20 cm approach pose even
from such starts; they now preserve depth when an aligned final-descent gate is
met. The change has CPU coverage and still needs live verification.

The source still contains several distinct actors:

| Path | What the code does |
| --- | --- |
| **New `actor_mode=direct_visual`** | Shared image/state actor for offline learning, Isaac loading/training, and Gazebo inference/adaptation. Predicts the full command from visual features; no ACT action proposal or residual. Default for the offline CLI. |
| Direct actor backbones | Small conv, ResNet18, ImageNet ResNet18, and DINOv2 ViT-S/14. Optional ACT initialization copies visual weights only. Backbone parameters train by default. |
| Offline `ACTAdapterSERLActor` | Predicts a residual from encoded state plus ACT's proposed action chunk; the residual head does not receive image features directly. |
| Offline `actor_mode=act_direct` | Uses `ACTChunkActor`; this is a different implementation from Isaac's identically named mode. |
| Isaac `IsaacACTAdapterActor`, `act_direct` | Predicts the executed action without adding ACT's action, but still feeds state plus ACT's proposed actions to the trainable head. This retains the visual information bottleneck described in the notes. |
| Vision critics | The offline encoder supports small convolutional networks, ResNet18 variants, and ConvNeXt Tiny variants. DINOv2 is not implemented in this encoder. A visual critic does not give the actor direct visual features. |

The legacy actor modes remain for old checkpoints and recipes. Use the
[direct visual workflow](DIRECT_VISUAL_POLICY.md) for new architecture work.
The new actor completed 40 BC updates on saved demonstrations; a pretrained
DINOv2 forward/backward check also passed. These are implementation/learning
checks. The subsequent direct actor and ACT Gazebo trials both completed with
no insertion. Fix reset/control and dataset problems before comparing encoders.

## Repairs completed on September 17

| Area | Current behavior |
| --- | --- |
| Gazebo score interpretation | Correct insertion requires each trial's official Tier 3 score of 75. Tier 1 remains model validity. The parser exposes per-trial outcomes and counts; transfer validation requires complete scoring and all observed trials to succeed. |
| ACT runtime evaluation | Explicit command mode is required. Empty checkpoint selections and incomplete runs fail. Matching complete evaluations may be reused; failed, legacy, or changed-setting evaluations get fresh attempt directories. Completion requires readiness, clean engine exit, and all configured trials scored. |
| Episode accounting | Isaac writes one outcome per terminated/truncated environment episode, using pre-reset identity and terminal success flags. The wrapper counts these completed episodes. A checkpoint left by a failed process does not permit continuation. |
| Promotion and demotion | The wrapper enforces the configured success rate and consecutive-failure threshold. Evaluation must complete the fixed per-level episode set without updates, exploration, guide, or guard overrides. Empty, partial, or legacy sampled logs cannot authorize promotion. |
| Evaluation settings | SFP module consistency is required; defaults are 0.5 mm axial/lateral error, 0.03 rad orientation, and 1.0/1.5 mm module axial/lateral error. Modified collision geometry is an explicit diagnostic option. Settings are saved with the result. |
| Stops and schema checks | `--updates 0` no longer ends evaluation after its first step. The no-promotion limit stops the wrapper with exit 3. Actor state width mismatch raises an error instead of silently padding/truncating input. |
| Artifacts | Stateful/axial launchers default to mounted `outputs/experiments/`, preserve cycle directories, and reject accidental reuse. Missing checkpoints fail explicitly. Other historical scripts still require path inspection. |

The [repair record](experiments/2026-09-17-evaluation-curriculum-fixes.md) lists
tests and compatibility changes. These repairs do not reclassify historical
results or establish that a policy inserts successfully. Old metrics lack the
terminal evidence needed to reconstruct reliable episode success rates.

## Latest live findings

Subsequent [source inventory](DATASETS.md) found the broader 668-episode local
collection and recovered both CheatCode sources previously absent at their old
manifest paths. All 140 accepted CheatCode trials have historical official
Tier 3 success records. The subsequent audit verified raw state/action equality,
sampled final images, and checked the 130-episode cache's video timestamps.
Across all sources, 393 episodes have verified score/array lineage, 212 are
scored non-insertions, and 63 mappings remain unresolved. See [data eligibility](DATASETS.md#verified-insertion-episodes).

The ACT result above supersedes the early pilot status. The table below records
the preceding live-validation pass; it is not the final ACT assessment.

| Check | Result |
| --- | --- |
| ACT 175k / direct visual actor | One official SFP trial each, both complete, neither inserted. Total scores 36.58 / 23.28; these are unequal training budgets and execution settings, not an architecture ranking. Tier 2 sampling anomalies also limit aggregate-score comparisons. |
| Gazebo control | Final bridge probe completed 260 commands with three-camera recordings. Initial 20 zero commands drifted 7.13 mm over 2.35 simulation seconds. Signed translation effects remain confounded by drift. |
| Isaac resets | Requested −40/−2/+43 mm aligned starts drifted and never met strict insertion criteria in a five-second zero-action probe. It completed zero episodes. |
| Isaac terminal accounting | Separate one-second probe completed exactly three episodes, recorded three timeouts, and stopped at the requested count. Zero successes. |
| Demonstrations | 51.74% zero actions; 189,885 zero-action frames have recorded TCP speed above 1 mm/s. 6.11% of all labels exceed the direct actor's translation limits. Current holdout is entirely SFP card 0 / port 1. |
| Runtime | Rootless Unix-socket bridge, image-sized IPC messages, local model imports, visible policy exceptions, and one-second recording now work in live checks. Isaac cameras require the documented container-local workaround on this old driver. |

See the [full report and videos](experiments/2026-09-17-live-validation.md) for
settings, measured geometry, failure records, and interpretation limits.

## Still unresolved

- Calibrated Isaac placement across task families and scene variations. One SFP
  port-1/card-0 near-opening scene now has stable reset, correct action direction,
  guide insertion, and terminal validation; broader coverage is unproven.
- Demonstration command validity/time alignment, verified insertion labels,
  source-trial grouping, and balanced held-out scenes. Do not treat every zero
  command as corrupt or every historical scalar score as insertion success.
- Earlier policy trials had missing scorer world-frame connectivity. The new
  evaluator provides the fixed world transform while keeping object ground
  truth disabled for learned policies; old scores remain historical evidence.
- No verified reliable learned policy or controlled backbone performance comparison.
  Legacy Isaac `act_direct` still means the ACT-action-context head.
- Some historical model dependencies and container-local artifacts remain
  unbacked-up. See the [artifact map](../outputs_README.md).

## Next experiments, in order

The SFP pose-refitting and Isaac snag-reproduction branches remain parked. The
active continuation begins from the failure modes actually observed in the
post-fix Gazebo evaluator:

1. **Collect compact repeated exact replays.** Repeat the released three-trial
   YAML with event-driven telemetry rather than multi-gigabyte full-rate TF
   bags. Retain plug/port pose, TCP command/state, force, insertion event, named
   contact pairs, and low-rate cameras. Estimate the intermittent SC axial-block
   rate without tuning on a reserved subset.
2. **Name the blocking contact.** Add diagnostic Gazebo contact logging for the
   SC plug, port rim/latch, gripper, cards, board, and cable links. Reproduce the
   near-aligned partial and determine which collision prevents the final 3--5 mm.
3. **Create short recovery starts from real failures.** Snapshot states before
   the observed axial block and larger approach-tracking failure. Validate
   reset identity and cable state; do not synthesize a cable snag label that the
   production run did not exhibit.
4. **Compare two recovery scales.** For local axial blocks, test a 10--15 mm
   measured-path retreat followed by coherent submillimeter/millimeter lateral
   search and reapproach. For the 8 mm approach miss, test a larger retreat or
   collision-safe transport replan. Keep unmodified CheatCode as the matched
   baseline.
5. **Train only after causal labels exist.** Add port-relative BC/recovery data,
   then critics and bounded online SERL, using observation-only RGB/state/force
   inputs and episode-grouped splits. Preserve the supervised anchor and require
   autonomous insertion before promotion.
6. **Use Isaac as a matched accelerator later.** Reopen Isaac only after its
   grasp, board, port, and collision contract reproduces one of the named Gazebo
   failures. A collision-disabled proxy is not training or promotion evidence.

World-model dynamics, SEER, reward-model training, imagination, and the reserved
final split remain parked. If explicit corrective control fails, return to the
perception restart checklist in the
[cable experiment](experiments/2026-09-22-cable-visibility-perception.md#parking-decision-and-restart-checklist).

Preserve actor-only evaluation for the June experiment lineage. Any privileged
guide or action override belongs to a separately labeled diagnostic experiment.
Report metrics for the same timestep/episode together: independent best depth,
lateral error, and orientation values cannot establish insertion.


### September 30 continuation: deployment input audit

The transport gate is accepted; full insertion remains a separate gate. SC dataset v5 fixes canonical TCP quaternions and the raw ROS joint-prefix order: all 4,752 transport rows exactly match the live state assembler. Earlier v3 BC is retained as an invalid-input diagnostic. Corrected rootless GPU7 BC is bounded at 1,500 updates; auxiliary depth visibility remains rejected. See [the continuation record](experiments/2026-09-30-transport-to-two-actor-continuation.md). Autonomous handoff and RL have not passed.
# October 4 current direction: shared coarse crop and stop

The next supervised experiment is a **single requested-port coarse perception model** with a shared three-camera spatial/history backbone, a current-image crop head, and a proposal-conditioned pause/accept head. The late camera-check role moves into that head. History is triggered by measured TCP travel; ten saved nonfinal SFP teacher routes favor a provisional **30 mm / up to twelve triplets** over fixed-second sampling, subject to autonomous replay and full p95 latency. Current images decide the crop and pause; old views provide identity/route context. The output goes to a separate precise refiner, then the insertion actor only after handoff checks. Projection-in-frame is not actual visibility. SFP autonomous transport remains **0/3** safe buffered arrivals in its frozen comparison, so this model cannot yet be connected to insertion. Final scenes remain sealed; RL stays gated. [Full updated pipeline and execution gates](experiments/2026-10-04-coarse-shared-history-execution.md), [history audit](experiments/2026-10-04-distance-history-design.md).

### October 16 continuation: completed SFP repair fits and shared recovery implementation

The bounded SFP acquisition-state fits completed at validation plateaus: current best/last 3,000/5,000 updates; history 1,750/3,750. Both retain the full original corpus plus ten new training reset configurations (350/63 source train/validation episodes, 283/55 resets). Current-only all-three projected crop capture is 95.6% on old held-out data and 96.3% on added held-out scenes, versus history 88.8%/89.6%. These flags are projection labels, not certified visibility or confidence.

On the identical 1,120 native held-out triplets, old/current-repair/history-repair cascades accept 913/917/827 frames, with **zero wrong-port accepts**. Current repair position/lateral/orientation p95 is 0.396 mm / 0.204 mm / 0.917 degrees. All 14 structurally absent queries passing the new current coarse flags are rejected by native refinement. This is only a small recorded-image change; it does not establish improved live acquisition. Ten fresh matched validation rollouts are running under the existing 9/10, zero wrong/harm gate.

The first remaining live failure has good coarse crops (about 5/5/22 px centre error), but native five-reference errors reach 4.25 px and the geometric fit exceeds its unchanged 2 px threshold. A native predicted-crop audit was visually inspected: the requested port is present and the markers identify its opening; this is not evidence of a cable snag or a gross wrong-port proposal. The fixed-normal calibration differs from the labelled normal by only 0.013 degrees in this frame. Preserve the failure and audit a bounded decoder/refiner repair; do not simply loosen the gate without held-out wrong/absent-target checks. `SFP_gap_first_failure_diagnosis_v1/predicted_crop_reference_audit.jpg` contains red predictions and green posthoc labels; labels never choose the crop.

SC's same-scene separate-GPU latency diagnosis passed: 0.414 mm opening error, active inference p50/p95/p99 82/239/297 ms. The earlier uniform single-GPU seven-case gate remains 6/7 including latency; do not rewrite it as seven strict passes.

`RunRuleThenSharedRecovery` is a new **unvalidated integration candidate**, not a trained policy result. It dispatches the SC low lateral bypass or SFP rule route, freezes fresh RGB port/grasp estimates, and uses one two-family actor for the full 150 mm approach. Matched modes are none/local/hierarchical. Hierarchical route retreats follow measured position AND orientation back to the measured acquisition checkpoint, cancel pending chunks, invalidate port/grasp estimates, and require fresh RGB locks. Initial load correction is frozen once during a free hold; contact never retares it. Hardcoded recovery commands are logged separately and are not actor log-probability actions. The same actor resumes after recovery. Autonomous insertion and recovery/RL gates remain OPEN; no new insertion actor or RL has been trained yet.

The wider command builder exposed a 32 ms trace/publication offset against its old 30 ms matching window. The repaired derivative searches at most 50 ms, still requires exact monotonic recorded-pose equality, reports unmatched trace events, and excludes any unmatched command from labels. It never substitutes future measured motion or supervisor actions. Originals and the failed partial derivative are preserved. The 23 existing causal frame, anchor and hierarchy tests pass.

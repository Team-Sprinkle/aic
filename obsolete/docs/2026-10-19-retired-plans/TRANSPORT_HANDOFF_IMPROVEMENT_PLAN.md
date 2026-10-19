> Retired plan: preserved for provenance. The [current experiment summary](../../../docs/EXPERIMENTS.md) and [single future-work plan](../../../docs/FUTURE_WORK.md) supersede its active/queued instructions.

<!-- sample-route-repair:2026-10-16 -->
### Current transport state: development gate passed; frozen evaluation running

Sample-derived development passes **7/7 SC and22/22 SFP handoffs**. The selected rule follows the original sample-based corridor, but can retrace measured TCP poses to an earlier checkpoint, switch sides and reacquire the port when blocked. A separate live trial exercised this recovery and handed off at155mmstand-off/2mmlateral. **Backtracking did not trigger in the seven-case SC screen**, so its7/7 result is not a measured causal benefit estimate; cable outcomes vary across repeats.

Rules and vision weights are now frozen. **130 transport/handoff evaluations** are running:124freshsample-derived cases with two episodes per complete SC/SFP/card-count/port/NIC-target setting, plus six separately labeled exposed official-scene repeats. No tuning follows those results. No insertion or model training is run; reserved internal final scenes remain sealed. Live artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3/`. [Full comparisons, actual backtracking bug, gates and video paths](../../../docs/experiments/2026-10-16-sample-conservative-rule-transport.md).
<!-- /sample-route-repair:2026-10-16 -->



<!-- evaluation-source-policy:2026-10-16 -->
## Development source correction — October 16

**Use the pre-deadline `sample_config.yaml` and task specification to develop routes, data and thresholds.** The separate `eval_config.yaml` was released after qualification and must not guide repairs. Its exact scenes **and derived stress scenes** are excluded from development. Existing frozen evaluation may finish as descriptive evidence; do not use it for training or tuning, and do not call the already inspected scenes unseen final tests. Audit earlier official-anchor data/model lineage before claiming this protocol is met. See [the evaluation protocol](../../../docs/answers/archive/2026-10-19-documentation/evaluation_protocol_full_history.md), which supersedes earlier “work backwards from official settings” recommendations. Pinned public sources are under `configs/evaluation_protocol/qualification_public_predeadline/`. No new training is running.
<!-- /evaluation-source-policy:2026-10-16 -->

# Transport and handoff improvement plan

<!-- current-continuation:start -->
## Current approved continuation — October 16

Rule transport brings either SC or SFP to a visual acquisition hold about 150 mm before the opening. A fresh requested-port pose and RGB-derived plug offset then initialize **one shared learned actor for the entire 150 mm approach and insertion**. Large hierarchical retreats return along measured poses to that acquisition hold and require fresh perception. The old 30 mm controller is historical.

**Training is running:** matched ImageNet versus visual-only transport initialization, same 13.725M full-target transformer, 30,000-update budget per arm. The fresh arm is active; the warm arm follows automatically. Actor data contain **67,039 examples from 500 usable source episodes / 377 physical reset configurations**: SC 102/30 train/validation episodes, SFP 319/49. All wider-source admission and estimated-frame audits passed. Predictions are complete recorded command targets in the visually estimated port frame, with causal robot/pose history and no privileged geometry input. Training error is not autonomous insertion success.

**Mandatory refiner retraining is complete**, including SC and SFP: 72,857 native RGB triplets from 497 original source episodes, followed by a targeted 20-episode SFP acquisition extension. Both base and repair reached validation plateaus. The strict repaired SFP offline component gate passes with the 0.95 projection threshold; this projection flag is not calibrated confidence or certified physical visibility.

**Launcher audit correction:** several newer host environment settings were not forwarded into Docker. The recent SFP live10 result remains 10/10 correct acquisitions, but actually used projection threshold **0.90**, not the intended 0.95. The active SC broad60 uses the **new repaired native refiner**, not the separately specified older SC checkpoint. Preserve these results under their actual configuration; they do not certify the intended family bundle. Let the frozen SC job finish. A separate corrected launcher records the effective container environment and forwards these settings for subsequent matched tests. The original launcher/source hashes remain intact during active evaluation.

The hierarchical runtime is implemented, including retained pre-return RGB/state, coherent lateral supervisor retries, measured route retreat and reacquisition. **28 focused contract/recovery checks pass; live actor/recovery validation is still pending.** Supervisor exploration is not RL. No new SAC/SERL, reward model, imagination or reserved-final evaluation has run. Keep final scenes sealed and at most four project GPUs including untouched legacy GPU0.

Next: finish matched BC fits and current frozen SC acquisition evaluation; verify corrected runtime settings; run fresh autonomous full-depth development rollouts, diagnose failures, then compare no/local/hierarchical recovery with identical actors and scenes. Advance to RL only after the supervised insertion gate.

See [execution record](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md) and artifact `state.json` for exact commands, failures, comparisons and paths.

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

Rootless runs use private bridge networks and unique Gazebo partitions. Cooperative per-GPU locks serialize future rollout/training workers without interrupting active jobs. Legacy GPU0 is untouched. Current work uses GPUs3/4/5 plus that legacy allocation. Final scenes remain sealed; reward/imagination and new actor-Q training remain gated. See the [execution record](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md) for commands, failed comparisons and artifact paths.

### Latest: rail coverage, clean simulation isolation and rule pilots

**Current gates remain open.** Mandatory native refiner retraining is complete to plateau on both families, but live acquisition still needs repair before the new 150 mm insertion actor can be trained and promoted. No new insertion BC or RL actor has been trained in this continuation.

The SC coarse failure has a concrete data gap: old SC0 rail translations cover roughly −2 to +18 mm and SC1 −60 to −43 mm, whereas the released range spans −60 to +55 mm. Even the 95 failed original collections do not fill that gap. An old-layout current/history locator fit cannot establish generalization to these missing port arrangements. The bounded repair collects 29 new physical reset configurations across the rail range, both requests, 1–5 cards and absent-other-port negatives. These are perception-only recordings, not successful insertion experts. Native overlays from the first two runs were inspected and passed causal-label QA. Current/history fits use grouped splits and balance original versus expanded-layout sources; live correct-port acquisition remains the promotion gate.

The first six SFP rule-transport cases yielded **1/6 fully qualified acquisitions, zero wrong-port handoffs**. The one pass stopped about 148 mm before the port, 5.7 mm laterally off-axis, with 0.324 mm visual opening error. Several failures waited at the elevated acquisition plane without a usable visual lock; another reached the region but could not sustain the stability gate. A bounded matched pilot tests acquisition at the nominal 150 mm plane. It does not relax the visual handoff gate. This is rule transport only; no insertion actor ran.

Three repaired SC teacher starts now pass independent raw-bag terminal checks, including 150/180 mm lateral/orientation offsets. Depth is 15.57–15.59 mm, lateral error 0.15–0.46 mm, with no scored harm. A smaller reset integral gain fixes measured oscillation, and 6 mm/s axial insertion reaches full depth before the simulator completion event freezes motion. These are privileged teacher demonstrations. Restart only the bounded 30-attempt teacher gate with this controller; require at least ten independently qualified successes before expansion.

BC preparation now has **53,921 command rows from 454 qualified source episodes**, including SC and SFP. All sources have matching completion events; 4,822 later commands are excluded. Episode IDs include source-path hashes to avoid basename collisions. Four labels are actual active complete TCP targets at +50/+100/+150/+200 ms. Original data are preserved. Body-command conversion uses preceding header-time TCP and is explicitly an approximation to RT-consumption state; a four-episode audit measures p95 discrepancy of 0.07–0.12 mm. Causal RGB-derived port locks are still needed before this becomes an actor-ready dataset.

A confirmed infrastructure bug affected two later runs: concurrent host-network containers shared Zenoh port 7447 and Gazebo discovery despite separate ROS domains. Both launchers now use private bridge networks and unique Gazebo partitions. Audits found no conflict warnings in the 454 admitted original sources or the frozen 12-run SC comparison. Failed infrastructure runs remain preserved and excluded from policy evidence.

See the [execution record](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md) and artifact-root `state.json` for commands, exact counts, failures and subsequent updates. Earlier October 15 paragraphs below are intermediate history where they describe running jobs or older derivatives.

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

Artifacts and exact commands: `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/`. Detailed chronology: [execution record](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md).


**October 15 execution update:** Broad SC/SFP native refinement now improves opening-position accuracy: conditional SC lateral p95 is 0.20 mm; the stronger SFP coarse pipeline plus the new refiner improves lateral p95 from 0.82 to 0.29 mm on 70 validation recordings, with zero observed wrong-port accepts in that comparison. Training continues because the first 12,000-update fit was still improving. Coarse identity repair remains experimental: rejecting known wrong handoffs also rejects some correct ones. Wider 150/180 mm SC teacher pilots now reach full seating after measured alignment and a corrected completion condition; broader data admission is still gated. New insertion BC, deployed hierarchy and RL have not passed their gates. [Evidence and active execution](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md).


**October 15 approved continuation is active:** rule transport for both families, mandatory broad SC/SFP native refiner retraining, coarse identity/coverage audit, shared insertion actor owning the full 150 mm approach, and hierarchical retreat with reacquisition. Proceed through measured BC and recovery gates before RL. [Plan and execution state](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md).

**October 15: expanded frozen evaluation complete — no promotion.** Exactly 140 rollouts on 120 fresh development configurations, 28 groups with five episodes each. Primary SC rule transport + older camera check/refiner + visual approach + existing BC insertion actor: 23/50 visual approvals (three wrong ports), 3/50 actor starts, **0/50 full insertions**. Matched far-start control: **0/20**; alternate-board SC: **0/10**. SFP tests are learned transport only, with no integrated coarse/insertion: 0/50 primary and 0/10 stress visits to even the declared broad diagnostic region; full SFP insertion is N/A. Many correctly acquired SC scenes reach roughly 30 mm stand-off but fail the stricter approach-readiness test. Models were unchanged, no training/RL/backtracking started, final scenes stayed sealed. Small earlier acquisition screens and the one 11 mm pilot do not establish reliability. [Complete evidence and limitations](../../../docs/experiments/2026-10-14-frozen-pipeline-coverage.md) · [Easy explanation](../../../docs/answers/2026-10-15-frozen-coverage-explanation.md).


**October 14 current execution:** Supported normal-board SC rule transport and RGB handoff passed small screens. Matched corrective-data, decoder and 6-second robot-history repairs did not pass autonomous full insertion. A separately frozen observation-only approach reached **29.12 mm before the opening / 0.148 mm lateral**, then the unchanged trained actor achieved **11.07 mm insertion depth / 0.413 mm lateral / 0.94° orientation error** on one reused start. Official score 75, but the unchanged ≥14 mm physical gate failed. Logs confirm contact-triggered completion and end-effector locking; no cable snag explains this run. This is meaningful approach progress, not a promoted insertion policy. All new jobs finished; no routine collection, RL, world-model work or final-scene access. SFP local insertion, rotated SC and port-0 rail minimum remain unapproved. [Execution record](../../../docs/experiments/2026-10-14-transport-handoff-execution.md) · [Easy explanation](../../../docs/answers/2026-10-14-autonomous-execution-outcome.md).

**Execution is now authorized and running:** See the [October 14 execution record](../../../docs/experiments/2026-10-14-transport-handoff-execution.md) and its machine state for actual completed gates and active runs. The planning-only notice below describes this file’s creation, not the current session. Final scenes remain sealed.

Updated: **2026-10-14**. This is the consolidated high-level plan for subsequent work. It incorporates the [October 13 root-cause review](../../../docs/experiments/2026-10-13-transport-independent-root-cause-review.md) and the rule-based transport review below. **Planning only:** no new training, Gazebo/Isaac rollout, or dataset collection was started for this update. Step 5 of the existing execution plan remains open; final scenes remain sealed.

For a future execution session, use the [autonomous execution prompt](../../../docs/answers/2026-10-14-autonomous-transport-handoff-execution-prompt.md). It expands this plan into ordered diagnostics, comparisons, resource limits, failure handling and continuation instructions. It was invoked on October 14; see the execution ledger for actual runs.

## Objective and deployment picture

Get the robot reliably around the cards, stop where the requested port can be identified and localized, then hand control to the insertion actor. Measure these separately:

1. Following a suitable route without sustained blockage.
2. Reaching and holding an observation pose.
3. Confirming the requested port and obtaining an adequate pose estimate.
4. Entering the insertion actor's validated starting region.
5. Full insertion, eventually with hierarchical backoff and RL recovery.

```text
Public task + current robot state + camera observations/history
                             |
             +---------------+---------------+
             |                               |
      Learned transport              Rule-based transport
      + stop/proposal head           + measured waypoint completion
             |                               |
             +---------------+---------------+
                             |
                  Hold and inspect candidate view
                             |
           Requested-port coarse check and crop selection
                             |
                    Precise pose refinement
                             |
           Pose inside validated insertion starting region?
                       /             \
                     yes              no
                      |                |
              Insertion actor    Bounded reacquisition / approach
                      |          or reject; no blind insertion
             Later: hierarchical backoff + learned recovery
```

The two transport branches are alternatives to compare. They share downstream perception and insertion. A rule-based winner could replace the learned transport actor for a supported task family; keeping both in deployment is not a requirement. Neither branch gets the true port/board pose at runtime.

## Established findings that constrain the plan

- SC training uses 83 train / 21 validation successful Low Lateral Bypass (LLB) episodes. The source collection attempted 200 settings and obtained 104 full insertions. That is **not** a transport success rate; some other episodes completed transport and failed during insertion. Preserve and inspect failures as well as selected successes.
- The taught transport prefixes stop at median TCP axial −168 mm / plug-tip axial −148 mm. None of the 104 prefixes enters the old −100 to −70 mm TCP approach band. Subsequent teacher approach actions were excluded from transport labels.
- The latest diffusion branch has no trained stop head and sets its handoff probability to zero. Fine70 reaches within 20 mm and 5° of the taught transport endpoint on 5/6 reused development scenes, then often leaves it. This is retrospective proximity, not validated handoff.
- Requiring a complete 12-slot future chunk removes nominal final-return-stage labels from 29/83 train episodes and excludes every cached handoff-positive row from that valid-action subset. Repeated command indices can represent legitimate holds; filtering them as corrupt removes usable examples.
- Equal-clock actor/teacher differences confound speed with route error. Future analysis must report geometric route error, ordered progress, endpoint error, orientation, stopping and contact separately.
- Broader SC board orientation coverage remains missing. Existing LLB training board yaw is approximately 2.96–3.16 rad; the released rotated SC example near −1.8 rad is not covered. Direct privileged route transformation there previously failed planning.

## Why revisit rule-based transport?

**It is a valid candidate to test, especially for SC. It is not yet demonstrated as an autonomous replacement.**

The [earlier coordinate audit](../../../docs/experiments/2026-10-05-global-coordinate-arrival-proxy.md) asked whether measured TCP near a nominal port position could identify a safe handoff. On three autonomous SFP traces it could not: the rule either did not trigger or triggered at the wrong geometry. It did not execute a conservative SC waypoint route. It also did not test a rule that uses coordinates only to reach a broad visual acquisition region, then checks the requested port visually.

Transport does not require knowing the opening to submillimeter precision. It needs enough clearance, the right route topology, and a reachable observation pose. Precise identity and alignment can remain downstream. This makes transport a better candidate for geometric rules than insertion.

### What the existing teacher really does

`scripts/prepare_sc_llb_collection.py` reuses port-specific LLB templates. It transforms them using the **true randomized board transform**, adds the **true selected-rail translation** to later waypoints, and adjusts wrist orientation. `scripts/plan_moveit_sc_fixed_route.py` then plans using exact simulator obstacle geometry and the requested/measured robot start. Gazebo validates actual execution and cable behavior.

Consequently, successful teacher routes demonstrate a useful route structure. They do not establish that a single frozen list of base-frame coordinates works, or that the privileged planner can be deployed unchanged.

A lightweight audit of all **83 successful training plans**, without fitting to validation scenes, found:

- Teacher board translation spans about **50 mm in x and 99 mm in y**, with roughly **11.3° yaw span**.
- Within each requested port, waypoint positions deviate from their stage median by roughly **47–64 mm at p95**, depending on stage. Maximum deviation reaches about 86 mm. This pools different randomized settings **separately for each port and stage** (35 port-0 / 48 port-1 training episodes); it is not repeatability under a fixed setting, measured tracking error, or a demonstrated safe tolerance. Next decompose scene-explained variation before sizing corridor margins.
- Low crossing/traversal goals use 180 mm base-frame TCP height in these plans. That is evidence of a shared template, **not a universally safe height** for other board heights, wrist postures, or layouts.
- Both port templates have similar early bypass stages; their final sweep/return differs. This supports a common obstacle-avoidance prefix followed by a requested-port-specific approach.

[Exact training-only waypoint statistics](</home/nas/chmin/aic_archives/2026-10-14_rule_transport_review/training_waypoint_spread.json>). These statistics describe successful selected teacher plans, not a certified collision-free corridor. A mean waypoint can lie in an obstacle even if each individual teacher was safe.

### Candidate designs, in priority order

| Candidate | How it handles variation | What it can and cannot establish |
| --- | --- | --- |
| **Conservative common corridor** | Build an outer route around the supported obstacle envelope, connect from measured initial TCP/joints, treat missing cards conservatively as potentially present. | Simplest bounded baseline. Must check reachability, table clearance and cable motion. More clearance is not always better: it can stretch the cable or exceed reach. |
| **Observation-adapted LLB template — recommended general candidate** | Use rough camera/history estimates of board orientation and outer row extent to place the corridor; use requested-port perception for the last approach. | Does not require precision port pose throughout transport. Rough scene estimation still needs validation; previous failures to recover exact board origin cannot be assumed solved. |
| **Small route-template library** | Choose among left/right or orientation-specific route patterns using observed scene features, then adapt from measured robot start. | Useful if one corridor cannot cover all layouts. Start with the demonstrated left bypass; a right bypass and rotated-board branch require separate validation. No lookup by evaluation scene ID. |
| **Hybrid prefix** | Execute robust rule-based early stages, then use the learned actor or a visually guided approach where geometry becomes uncertain. | Useful fallback if the final return is the only weak part. Adds a switch, so keep it only if it outperforms the simpler alternatives. |

A trajectory expressed only relative to initial TCP fixes robot-start variation but **not independently randomized board or rail position**. A board-relative template handles those shifts only if the board estimate is trustworthy. A useful division is: measured start for the first connection, rough obstacle envelope for the bypass, precise requested-port perception near the end.

### Intended rule controller

Preserve the successful order:

```text
leave the cards on the starting side
  -> move behind the row while descending in the clear corridor
  -> travel beside and beyond the entire row
  -> cross toward the target side while still beyond the cards
  -> tilt with adequate tool/gripper clearance
  -> return toward a visual acquisition region
  -> hold, identify requested port, refine pose
  -> approach further only if the next actor's starting gate allows it
```

Use measured position/orientation error and progress to advance stages, not fixed-duration blind playback. Use smooth bounded segments, joint limits, force/stall monitoring, and explicit rejection when a waypoint is not reached. The swept gripper/plug volume matters when tilting around TCP. Cable/card touch alone is acceptable under the user's criterion; sustained snag, forceful blockage and prohibited contacts are not.

Do not hardcode insertion based on a nominal endpoint. If the port is absent or ambiguous, use a bounded observation-based reacquisition motion within checked clearance; otherwise stop/reject. Coordinate arrival is a request for visual inspection, not permission to insert.

MoveIt is an optional segment planner, not a cable-dynamics solution. A deployable planner would use permitted robot state and observation-derived obstacle estimates. The existing exact-YAML collision scene remains an oracle diagnostic only.

### Official interface boundary

Measured TCP in `base_link`, camera images, robot state and task identity are available; exact target geometry from evaluation configuration or simulator TF is not a runtime policy input. The official [rules](https://github.com/intrinsic-dev/aic/blob/main/docs/challenge_rules.md) prohibit evaluation state leakage and exploitative configuration hardcoding. The proposed deployment candidate therefore uses general route rules and observation-driven adjustment, not a table of published evaluation configurations. This is a design constraint, not a claim of organizer certification.

The public [configuration](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/eval_config.yaml) includes SC rail translations from −60 to +55 mm; the old SC collector varied rails only around its references by roughly ±10 mm. Do not mistake existing selected successes for full published-range coverage. [Qualification documentation](https://github.com/intrinsic-dev/aic/blob/main/docs/qualification_phase.md) describes additional scene variation. Published examples guide development coverage, not runtime ground-truth lookup.

## Ordered execution plan for later work

### 1. Freeze the component contract and measurements

Inspect existing teacher/actor endpoint windows with coarse perception and refinement. Establish the acquisition region from correct-port identity, visibility and pose accuracy, then establish the insertion actor's supported offset/orientation range. Assign the missing approach segment explicitly. Prefer extending transport with existing safe approach observations if perception cannot operate at the old endpoint.

**Budget:** one offline analysis session, about 2–4 hours. **Gate:** labels and evaluation refer to the same endpoint; report TCP and plug-tip geometry separately. An oracle endpoint visit is not a deployed stop. Keep final scenes sealed.

### 2. Screen rule-based transport before another broad model sweep

Use training scenes only to define a common corridor and uncertainty margins. Inspect failed collection attempts to distinguish transport failure from later insertion failure. Test the conservative corridor and, only if necessary, one observation-adapted template. Check both SC ports, one/five-card extremes, actual robot-start offsets, board translations/yaws and rail extremes. Include a rotated-board development case as an explicit unsupported-case test, not a silently omitted cell.

**Budget:** one offline session, about 2–4 hours; rigid planning only where useful. **Gate:** each supported route is reachable with checked tool/obstacle clearance, and all runtime route-selection inputs have an observation-only source. If no common corridor exists, drop the fixed-corridor candidate and continue with observed adaptation. Never infer cable safety from rigid planning alone.

### 3. Bounded Gazebo comparison of the rule candidate

Later, run **six nonfinal SC development settings** with the frozen candidate, covering both ports and card-count/start extremes within the declared support region. Compare with frozen learned-actor traces or matched reruns as necessary. A privileged retargeted teacher can be a separately labeled diagnostic control. Record all wrist views plus overhead/side views, commanded/measured TCP, stage transitions, contact/force, stop decisions and visual approvals.

**Budget:** six candidate runs initially, one GPU/rootless instance; repeat at most three promising edge cases with ±10/20 mm corridor perturbations. **Continuation gate:** at least 5/6 reach and hold the validated visual acquisition region, no wrong-port accepted handoff, and no sustained snag/blockage or prohibited contact. This is a feasibility screen, not a reliability certificate. Treat failed visual acquisition separately from blocked motion. If a wider margin worsens reach/cable load, do not keep widening it blindly.

If the rule candidate passes, prioritize it for transport integration and retain the learned actor as a comparator. If it fails, identify whether the cause is an incorrect scene estimate, infeasible route, cable behavior or final approach. Test at most one distinct repair per identified cause before revising the design. Do not require the same transport method for SFP and SC.

### 4. Repair learned-actor data and terminal supervision

Independently of which branch wins, repair the known training-data defects using original recordings: timestamp-based future commands, legitimate held targets, preserved observation histories, pause loss weights and per-slot terminal validity. Stop/proposal labels must not depend on having a complete future action chunk. Audit at least 12 transition windows and all episode-level final-stage counts. Do not fabricate post-reset labels or unobserved holding actions.

**Gate:** verified final return/terminal observations survive or have explicit exclusion reasons; grouped splits and original datasets are preserved. No large new collection is needed for this repair.

### 5. Separate controller execution from model prediction

Replay recorded teacher targets through the learned actor's exact chunk executor on at most three nonfinal starts, with original teacher controls for comparison. Verify anchor conversion, command age, gain/mode consistency, measured tracking and terminal stopping. This is a labeled oracle diagnostic, not policy performance.

**Budget:** at most six runs. **Gate:** reproduce the taught endpoint within a proposed 10 mm/3° diagnostic tolerance without harmful blockage, or fix execution before fitting a new actor. Timing gaps and missing labels are different hypotheses.

### 6. One matched learned-transport comparison

Keep fine70 as the frozen reference. Compare old and repaired data under identical architecture, split and update budget, with appropriately supervised stop heads in both arms. Report phase-specific translation/orientation, route deviation, stable endpoint capture, wrong stops and perception handoff. Train long enough to assess a declared plateau; explicitly label budget-limited runs.

**Budget:** one GPU initially, up to four GPU-hours per arm; maximum four project GPUs concurrently if later needed. First use reused diagnostic scenes, then six new grouped SC starts after freezing selection. **Continuation gate:** at least 5/6 stable acquisition arrivals and no observed wrong-port approvals/blocking, with improvement over the matched control. Compare the winner with rule-based transport under the same downstream checks. Do not select by offline action p95 alone.

### 7. Conditional representation and recovery work

Only for a remaining measured gap:

- Poor visual localization/route discrimination: matched ResNet spatial features versus DINOv2-S patch features; same data/head/budget. A single-frame linear probe is not decisive.
- Poor stage disambiguation: dense recent context plus distance/rotation-spaced older memory; explicit observed progress or route-stage completion. Do not infer progress from elapsed time alone.
- Timing-induced error: a buffered/asynchronous executor with correct reanchoring of overlapping targets. First show the timing change affects measured behavior.
- Failures off the demonstrated route: at most three safe causal teacher-continuation pilots before larger collection; require at least two valid pilots. Earlier failed SFP waypoint/continuation experiments are not successful evidence.

**Stop rule:** no architecture/data scale-up without both held-out task-relevant improvement and a bounded live benefit. Keep complete action prediction; do not revive a residual adapter.

### 8. Expand coverage and connect insertion

Validate the selected transport method on new grouped settings, both task families, broad starts/rails, both ports, all card counts and board orientations. Repair the rotated SC route before declaring full SC support. Reuse SFP teachers consistent with its intended route; do not mix rejected over-card SC teachers back in.

Then connect observation-only coarse approval, precise refinement and insertion using the verified starting envelope. Track transport arrival, correct visual handoff and full insertion as separate denominators. Later restore full-path p50/p95/p99 latency checks and the 300 ms p95 goal; the temporary latency relaxation does not remove the task's wall-time limit.

### 9. Recovery/RL, transfer and final evaluation

Once ordinary transport/handoff/insertion works, compare no backoff, local backoff and hierarchical backoff with shared learned recovery as previously planned. Use reproducible failure scenarios, explicit supervisor ownership and task coverage. Isaac can support efficient RL; Gazebo remains the final transfer/evaluation environment. World models/SEER, reward imagination and final-scene tuning remain outside this plan. Open reserved final scenes only for the separately approved frozen final evaluation.

## Decision in plain words

**Yes: add rule-based transport as a serious alternative, and test it early.** The most promising version hardcodes the *sequence of safe maneuvers*, adjusts their placement from ordinary observations when necessary, and lets vision authorize the handoff. A single exact trajectory or nominal-distance threshold alone has insufficient evidence. If the simple controller works, there is no need to learn the entire SC bypass with diffusion; learning can focus on perception, insertion and recovery.

This file is the current planning entry point. Detailed historical results remain in their original records; the [October 13 audit](../../../docs/experiments/2026-10-13-transport-independent-root-cause-review.md) remains the source for label and endpoint findings. Current gate outcomes are listed below; the earlier planning language is historical.


## Current execution gates, October 14

| Stage | Actual result | Remaining condition |
| --- | --- | --- |
| 1. Component contract | Teacher endpoints support visual acquisition; local approach owns the remaining roughly 150 mm | Local autonomous starting envelope remains unapproved |
| 2–3. Rule corridor and Gazebo screen | Normal-board SC rule: 6/6 reused and 6/6 fresh acquisitions; stricter yaw screen later 5/6 | Rotated board and port-0 rail minimum remain unsupported; this is acquisition, not insertion |
| 4. Labels and stopping | Timestamp repair retains final return examples and all 61 stop positives | Preserve raw elapsed time and separate stop supervision from action-horizon completeness |
| 5. Executor parity | Bounded recorded-target transport parity passed | The older project step 5 full-autonomy gate is a different gate and remains open |
| 6. Matched learned transport | 20,000-update comparison: 1/6 acquisition in each arm | Not promoted; separate visual docking repair passed supported small screens |
| 7. Targeted branches | Point-target and corrective-data fits improve offline errors; autonomous approach still fails. Decoder repair fails offline gate | Masked-versus-causal robot-history comparison finished and failed; no routine collection |
| 8. Full insertion and broadening | Not passed; paired actors stop about 141 mm before the opening | Physical insertion requires depth ≥14 mm, lateral ≤0.5 mm, explicit orientation and acceptable contact |
| 9. Recovery/RL/transfer | Disabled by failed ordinary supervised insertion prerequisite | No world-model work and no reserved final scenes |

The current bottleneck is near-port insertion after supported SC transport, handoff and the newly tested observation-only approach. Accurate RGB port/grasp estimates and low teacher-forced action errors have not made this actor progress to insertion. A privileged feedback controller can insert, but that is a diagnostic upper bound, not a deployable result. The bounded history experiment addresses measured phase-dependent action errors. If it also fails, summarize the distinct failed repairs and revise the local control design explicitly rather than scaling collection or claiming RL can conceal the gap.

### October 16 continuation

Mandatory SC/SFP native refiner retraining is complete to plateau. The targeted SFP coarse repair also plateaued; on the same 1,120 held-out triplets it accepts 917 versus the old 913, with zero wrong ports. Native refinement rejects all 14 new coarse absent-target false positives. The ten-case live gate is still running; this small component change is **not** a live success claim. Its first failure reaches the correct image region but exceeds the native geometric residual gate. See the [execution record](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md) for the inspected predicted/label overlays, full metrics and bounded diagnosis.

The new `RunRuleThenSharedRecovery` integration is implemented but unvalidated: SC/SFP rule transport -> fresh RGB port/grasp locks -> one shared actor owning the entire150mm approach -> measured-path local/hierarchical backtrack. Large retreats return to the measured acquisition checkpoint and require fresh perception. Pending chunks/history are cleared, contact never retares initial load, and supervisor commands are logged separately. **No new insertion BC or RL has been trained yet.** Wider training/held-out demonstrations and matched fresh versus visual-warm BC remain in progress. Final scenes remain sealed and the project limit is four GPUs including the untouched legacy allocation.

SC's targeted separate-GPU latency case passed (active p95 239ms, opening error0.414mm); the earlier seven-case single-GPU strict result remains6/7. Current status and exact artifacts are in [STATUS.md](../../../docs/answers/archive/2026-10-19-documentation/status_full_history.md) and the execution record.

> Retired plan: preserved for provenance. The [current experiment summary](../../../docs/EXPERIMENTS.md) and [single future-work plan](../../../docs/FUTURE_WORK.md) supersede its active/queued instructions.

## October 19: short local recovery added to transport

The deployed rule transport for both connector families now has the same *local* recovery concept previously available during insertion. Its trigger requires a sustained wrench load increase from a quiet free-space reference plus stalled measured progress toward the current route waypoint. The recent measured TCP travel vector defines the axial direction for this maneuver; a horizontal perpendicular vector defines the sideways retry. It retreats 8 mm (up to 12 mm if load persists), verifies unloading, moves sideways with measured axial drift limited to 3 mm, and resumes the same intermediate route stage with shifted remaining waypoints. A failed clearance aborts; visual-acquisition timeout is not a blockage signal. This does not by itself establish the larger hierarchical checkpoint retreat or an RL recovery policy.

The first live five-card SC diagnostic reached the correct visual handoff after 8.4 mm retreat and 9.5 mm lateral movement, with no insertion phase. The matched released three-round run had no recovery trigger and remained 1/3 strict insertions. The technique is an implemented, bounded recovery path, not a demonstrated full-insertion improvement. [Trace, failure variants, code and video](../../../docs/experiments/2026-10-19-transport-local-recovery.md).

## October 18: gated close-up pose update and SC insertion result

V34 passes **4/4 matched SC development cases** (zero/five obstacle cards, both ports), followed by a passing SC round in the unchanged public sample: **15.510 mm depth, 0.141 mm lateral error, 0.469° orientation**, toolkit completion, no harm. Five fresh agreeing images support one bounded plug-pose update at 25 mm; the port frame stays fixed. Tighter lateral centering and stronger lateral tracking are also needed. All used-stage p95 timings pass; these are explicit RGB/public-state control results, not BC or RL.

The complete sample is **170.57/300, 2/3 strict** because SFP trial 2 has a 3.53° biased grasp estimate before contact. Its actual roll exceeds the original SFP training range; close-up predictions remain wrong. Repeats stop, and the next targeted repair is SFP orientation coverage rather than a stiffness-only change. No new training or collection is running. The SC model's original 1.026° qualification failure remains explicit; four successful development cases do not establish general reliability. Final scenes remain sealed.

[Detailed evidence, commands, images and next repair](../../../docs/experiments/2026-10-18-sc-explicit-controller-continuation.md).

> **October18 live milestone:** the discrete25mmRGBheld-plug pose update passes one failedSCdevelopment start and then **3/3strict insertions** in the unchanged public three-trial sample(257.36/300). A matched controller without the update failsSC. One local recovery occurs in the development success; this is **not evidence of hierarchical superiority or newRL**. Public-sample timing/guard repairs continue with failures preserved; model original subgroup1.026°stillfails1°. [Evidence and limits](../../../docs/experiments/2026-10-18-sc-explicit-controller-continuation.md#first-live25mm-update-full-sc-physical-insertion).

## October 18: explicit-controller recovery and near-view pose repair

SC recovery has not yet passed the strict full-insertion gate. Two complete public samples pass both SFP trials, while SC receives toolkit credit at about15.48mm, just below15.5mm. Current improvements do not establish hierarchical-backoff superiority. The next repair targets visual grasp bias: a discrete, gated re-estimate at25mm, with the port pose fixed and actual issued reference retained to avoid a jump. Repeated frames or persistent large disagreement must not authorize forward motion. [Controller versions, failures and tests](../../../docs/experiments/2026-10-18-sc-explicit-controller-continuation.md).

# Hierarchical recovery for cable insertion

## October 18: Gazebo recovery requires correct timing and reference semantics

The official SC TouchPlugin can latch the cable/gripper after1second of shallow contact. A recovery requested afterward cannot undo that simulator lock. Runtime must use public force/progress cues to release contact early; the policy never reads completion or oracle topics.

A measured physical waypoint differs from the API command reference under impedance control. Public free-space tracking error was about6mm while the first retreat reference moved only2–3mm; retaining that offset enables measured retreat. Constant compensation still left0.91mmerror against a0.75mm lateral arrival tolerance. Integrating moving clipped steps then produced windup. The corrected retry ramps a **fixed measured lateral goal**, resets its integral at the phase boundary, and uses the actual issued reference to restart insertion.

The sensitive low-force detector must be phase-specific: it was interrupting free-space angular alignment8–17mm before the port. The latest test uses it only within5mm of the estimated opening; the generic8Nguard remains global. Candidate angular retries are bounded, observation-only exploration. None of this is learnedRL, and completed retreats are not full insertions. Hierarchical superiority over local recovery has not been established. [Full experiment record](../../../docs/experiments/2026-10-18-three-trial-insertion-repair-cycle.md).

## October 18: insertion-repair follow-up queued

After the current converged-training and frozen rollout comparisons finish, run identical development starts with **none, local, and hierarchical** recovery. This is a trained-BC actor comparison, not newRL. Keep image-only explicit-controller successes separate. Preserve the full measured approach path beyond the short actor context; cancel pending chunks and reseed the command governor after lateral retry. Repeated blockage escalates to an earlier checkpoint and fresh RGB acquisition. Only improved independent full insertion without harm supports promotion. [Executable protocol and artifacts](../../../docs/experiments/2026-10-17-insertion-control-and-grasp-repair.md#queued-executable-backoff-screen).

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


**October 1 scope:** Hierarchical backtracking remains a core recovery proposal but is not enabled in the current supervised transport → perception → insertion baseline. Once that baseline reaches autonomous official full-depth Gazebo insertion, compare the same frozen actors and scenes under no backtracking, local backtracking, and hierarchical backtracking. RL is also gated until the supervised insertion result. [Current three-model sequence](../../../docs/experiments/2026-10-01-temporal-three-model-continuation.md).

**September 30 latest:** teacher transport gate accepted. Two same-world Cartesian SC teachers fully inserted, providing v9 with 1,751 transport and 503 local chunks, plus the verified SFP source. The balanced-port BC retry finished 1,500 updates and failed both autonomous development starts: final planned-standoff errors 103 and 603 mm versus 76 and 74 mm for the prior masked actor; neither reached handoff or inserted. A bounded trace/video audit found early wrong-route commands without sustained cable snag. Stop routine feed-forward BC. A checkpoint-provenance audit found that this run reused the full action head from an older mixed-SC-route model and had only two new SC episodes; it was not a clean test of the new Low Lateral Bypass route. Audit clean route data and initialization before deciding on a route-subgoal model. No new training was started for this audit. True visibility, autonomous two-actor insertion and RL remain gated; reserved final scenes stay sealed. [Evidence and exact artifacts](../../../docs/experiments/2026-09-30-transport-to-two-actor-continuation.md).

**September 30 continuation:** transport gate accepted. Four changed-card-layout MoveIt replays reached handoff/standoff; one fully inserted, one failed local insertion and is excluded from expert BC, two were transport-only validation. Four full-success SC teacher episodes now provide 3,001 transport and 1,000 local training chunks, with 1,751 held-out transport chunks. True-visibility recorder checks caught invalid auxiliary depth during tilt; labels are being repaired before training. Autonomous two-actor insertion and RL remain pending. [Execution record](../../../docs/experiments/2026-09-30-transport-to-two-actor-continuation.md).

## September 30: measured retreat and force-reporting audit

The [oracle SC insertion diagnosis](../../../docs/experiments/2026-09-30-force-oracle.md) found a reproduced plug/port catch. A commanded 24 mm retreat moved the TCP only 0.00623 mm. Count a backoff as completed only from measured movement and release evidence; commanded distance is insufficient. The trace now names commanded and measured distances separately. A confirmed legacy Gazebo contact-reporting bug is corrected for diagnostics while preserving the raw wrist sensor and collision physics. The newer tilt-before-return route inserted in 2/2 oracle repeats, but recovery authority and cross-start generality have not been established by those successes. No new RL training or promotion occurred.

Status: proposed core controller; not yet validated as an improvement (2026-09-24)

The [September 24 Isaac SC repair](../../../docs/experiments/2026-09-24-isaac-sc-fidelity-fix.md)
now reaches the Gazebo-derived SC target in one normal-collision five-card
scripted scene and reproduces two local plug/port blockages. It also shows
that the earlier gripper/card item-2 collision vanished on the centered route
after calibrating the grasp. [Deliberate corrected-asset route probes](../../../docs/experiments/2026-09-24-isaac-sc-contact-routes.md)
then reproduced named gripper/card and plug/card jams; cable/card contact
occurred only alongside a gripper jam. These traces support specific contact
mechanics, not an isolated route-level cable snag or autonomous recovery. A
container-local RTX driver-check override later passed a short three-wrist-camera
SC smoke test, with an orange cable visible in a separate diagnostic view.
Longer moving-cable, observation-contract, and policy gates remain open.

The [first read-only trace replay](../../../docs/experiments/2026-09-24-hierarchical-recovery-continuation.md#trace-diagnostic-command-and-interpretation)
found that a correctly vector-tared two-sample force-plus-stall trigger fires
on all seven retained 20 Hz repeats, including six partial insertions. A longer
force-history and motion-progress rule also fires on all seven. Subtracting
force norms instead misleadingly misses most, because it discards direction.
The controller code is a tested prototype, **not** a deployable recovery
policy. Named contact and controller-limit diagnosis plus independent Gazebo
scene calibration are required before a trigger can own motion.
In a broader scored set of 17 retained traces, the three no-insertion runs
reached at least 89 mm peak TCP command-tracking error, while fourteen
full/partial runs stayed below 73 mm. Three new scene variations added a full
and two partial controls that also stayed below 80 mm. Because that line was
chosen after seeing the saved outcomes and no new failure validated it, it is
only an incident-search hypothesis. Tracking error also cannot tell whether
the cable, gripper, port, IK, or controller caused the stop.

## Why this is a project direction

The robot can command motion toward a port while its measured TCP and plug
hardly move and wrist force rises. The failure may be a local port contact, a
gripper/card collision, cable tension, or a controller/workspace limit. A
single small retreat can clear a port lip; it cannot be assumed to undo a
cable route across several cards. We therefore propose **hierarchical
backoff**: a force-and-progress supervisor that retraces measured motion to a
verified safe checkpoint, then escalates to an earlier route checkpoint and a
new corridor when a local retry repeats the same obstruction.

The [five-card Gazebo route probe](../../../docs/experiments/2026-09-23-fixed-five-card-route-probe.md)
is the motivating example. In one across-card run the cable lay close to a
card, a cable link became nearly stationary, and the TCP stopped despite
advancing commands. A later 20 Hz rerun visibly draped the cable over cards
and stopped without insertion, but its larger lateral error and moving cable
link mean **its exact contact pair and root cause are unproven**. The official
[friction and physics notice](https://discourse.openrobotics.org/t/note-to-participants-gazebo-bug-fix-changes-to-friction-properties-of-the-cable/54098)
describes related simulator issues; our fresh single-trial runs use the
post-friction official image and do not match the old between-trial ghost
collision symptom. Do not label either run as a proven cable snag.

## Controller contract

```text
RGB + robot state + wrist force + measured motion
                    |
             learned BC/RL actor
                    |
             normal action proposal
                    |
          force/progress supervisor
             |              |
           clear         blocked
             |              |
        execute actor   stop forward push
                            |
                    retrace measured poses
                            |
                    safe local checkpoint B
                            |
                  actor chooses new retry
                            |
                    repeated blockage?
                         |       |
                        no      yes
                         |       |
                    continue   earlier route checkpoint D
                               + different observed corridor
```

The supervisor owns immediate unloading and safe reversal. The learned actor
owns normal transport, alignment, lateral/rotational retry, and choice among
observation-supported route options after clearance. No simulator ground-truth
plug, port, cable, or card pose may enter the deployed policy or route selector.
Simulator geometry and named contacts can be used for training labels and
post-run diagnosis.

### Detecting a real blockage

Do not use force alone: correct insertion is contact rich. Trigger on
persistent force plus meaningful command with little measured TCP/plug progress,
or an independently validated tension signal. Use a lower force threshold to
declare release. Record the blocked direction and the first stall time. During
reversal, stop if force grows, the cable tightens visibly, or motion diverges
from the stored path.

### Checkpoints, escalation, and memory

- Save measured **position and orientation**, force, image/visibility summary,
  timestamp, and route phase at verified low-force checkpoints. Index by
  traveled distance and phase, not only by decision count.
- Level 1: retreat to a free near-port checkpoint, then attempt a different
  lateral/rotation choice. This addresses port-lip or alignment contacts.
- Level 2: after repeated blockage in the same spatial corridor or insufficient
  net progress, retrace to a pre-obstacle checkpoint and choose a different
  card/cable route. The route must differ meaningfully from the failed one.
- Level 3: if neither route releases load within a bounded force, distance,
  time, and retry budget, stop safely and mark the episode unresolved. An
  endless cycle of local backoffs is not recovery.
- Track local attempts, route resets, tried corridors, maximum and integrated
  force, measured clearance, progress since the last checkpoint, and whether
  the same obstacle was revisited. Expose compact recovery state to the actor.

If a new contact at C follows substantial safe progress from B, a local retry
may be appropriate. If C repeats A's position/blocked direction or cable
tension remains after reaching B, escalate to earlier checkpoint D. The
thresholds, number of retries, and route candidates must be selected on
development episodes, then frozen before held-out comparison.

The [current Isaac recovery code](../../../aic_utils/aic_isaac/aic_isaaclab/scripts/serl/measured_path_recovery.py)
stores only recent TCP positions and implements a local 1 mm minimum/10 mm
maximum retreat with a 12-decision near-lateral retry. It counts backtrack
events but has no orientation history, safe-checkpoint stack, route-level
escalation, or Gazebo deployment parity. The [matched SFP result](../../../docs/experiments/2026-09-22-serl-mixture-recovery.md)
is inconclusive for recovery and has no proven cable snag.

## Learning and evidence boundary

Keep every complete failure and recovery transition, including executed
commands, measurements, recovery mode, actor ownership, force, terminal
observation, and scorer outcome. Hard-coded retreat actions can update a critic
through their outcomes but must be masked from actor gradients. Successful
scripted recoveries can supervise the learned post-clearance decision; failed
traces alone do not reveal the successful alternative. Compare the same actor
with no recovery, local recovery, and hierarchical recovery on identical
episode-grouped starts and budgets. Measure **full insertion**, not just force
release, plus peak/integrated force, recovery count, time, and p95 inference
under 300 ms. Preserve the sealed final configurations until development
gates pass.

## Related work and claim boundary

- [ARCH](https://arxiv.org/html/2409.16451v2) uses a high-level policy to
  select motion-planning and RL assembly primitives. Its recovery trials teach
  retries after failed skills; the paper gives a failed grasp followed by a
  new grasp attempt as an example. Its *hierarchy is in skill selection*. It
  does not describe retracing measured motion, local and earlier route
  checkpoints, or escalation based on repeated blockage.
- [Hierarchical DLO Routing](https://arxiv.org/html/2510.19268v2) uses a VLM
  to choose Insert, Pull, or Flatten skills for routing a flexible object
  through clips. It counts consecutive failed insertions; when the cable head
  is twisted or poorly aligned, Flatten reorients it into an insertable state
  before another attempt. Pull and Flatten can move the cable away from a clip.
  This *is* cable-specific recovery within a hierarchical skill system, but
  the described method does not reverse the measured gripper path to a local
  checkpoint and then escalate to an earlier route checkpoint or new corridor
  after another blockage. We should not cite it as prior implementation of
  that particular two-level backtracking rule.
- [Multi-Stage Cable Routing](https://arxiv.org/abs/2307.08927) learns both
  sequencing and motor skills for routing through clips. Its routing setting
  differs from a gripped connector inserting beside five NIC cards.
- [SERL](https://arxiv.org/abs/2401.16013) demonstrates sample-efficient
  off-policy robotic RL, including cable routing and insertion, with prior
  data and recovery behavior. It does not validate our exact checkpoint
  escalation scheme.
- [Recovery RL](https://arxiv.org/abs/2010.15920) learns safe recovery zones
  for contact-rich tasks. It motivates a recovery boundary, not a novelty
  claim for simple retreat.

A defensible **candidate** paper contribution is the *specific, measured-path
checkpoint escalation coupled to cable-aware route changes and learned
post-clearance insertion*, if matched ablations show better full insertion
than both no recovery and local-only recovery. The bounded literature search
above did not find this exact combination for the AIC wire task. Neither ARCH
nor Hierarchical DLO Routing establishes that particular mechanism, though
both are relevant context. This is not an exhaustive novelty review, and no
performance claim is warranted yet.

## Execution

The [September 24 continuation plan](../../../docs/experiments/2026-09-24-hierarchical-recovery-continuation.md)
contains the ranked failure matrix, data requirements, BC/RL sequence,
simulator gates, matched ablations, and Gazebo transfer plan. Current failure
artifacts are mapped in [outputs_README.md](../../../outputs_README.md).

### October 16 continuation

Mandatory SC/SFP native refiner retraining is complete to plateau. The targeted SFP coarse repair also plateaued; on the same 1,120 held-out triplets it accepts 917 versus the old 913, with zero wrong ports. Native refinement rejects all 14 new coarse absent-target false positives. The ten-case live gate is still running; this small component change is **not** a live success claim. Its first failure reaches the correct image region but exceeds the native geometric residual gate. See the [execution record](../../../docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md) for the inspected predicted/label overlays, full metrics and bounded diagnosis.

The new `RunRuleThenSharedRecovery` integration is implemented but unvalidated: SC/SFP rule transport -> fresh RGB port/grasp locks -> one shared actor owning the entire150mm approach -> measured-path local/hierarchical backtrack. Large retreats return to the measured acquisition checkpoint and require fresh perception. Pending chunks/history are cleared, contact never retares initial load, and supervisor commands are logged separately. **No new insertion BC or RL has been trained yet.** Wider training/held-out demonstrations and matched fresh versus visual-warm BC remain in progress. Final scenes remain sealed and the project limit is four GPUs including the untouched legacy allocation.

SC's targeted separate-GPU latency case passed (active p95 239ms, opening error0.414mm); the earlier seven-case single-GPU strict result remains6/7. Current status and exact artifacts are in [STATUS.md](../../../docs/answers/archive/2026-10-19-documentation/status_full_history.md) and the execution record.
## October16: live phase-level transport recovery

A new **sample-derived Gazebo transport** diagnostic now demonstrates measured route retreat and an alternate-side retry, with no RL or privileged deployment geometry. In one exercised SC case, a static corridor force-stopped; after fixing the progress index for tiny waypoints, the robot reached its earlier pre-lateral checkpoint, discarded the old visual lock, switched sides, and handed off at155.23mmstand-off/2.00mmlateral. Opening estimate error0.109mm and active perceptionp95153.2ms; no scored harmful contact.

This is one successful **phase-level transport recovery**, not validation of the full multi-level insertion hierarchy. The original fixed corridor passes6/7SCdevelopmentcases and a fixed opposite corridor4/7; adaptiveSC7-casevalidation is active before the frozen two-per-setting evaluation. No insertion actor or RL is evaluated here. [Complete variants, force/contact diagnosis, actual progress bug and artifacts](../../../docs/experiments/2026-10-16-sample-conservative-rule-transport.md). Development uses pre-release sample/specification only; exposed post-release settings do not guide the repair.

# Explicit SC insertion continuation — October18

## Current summary

The pipeline uses rule-based transport, requested-port perception and an explicit insertion controller. At 25 mm it now re-estimates the held plug from five fresh, agreeing camera triplets. It keeps the port fixed and updates the command smoothly. **No learned insertion actor or new RL is responsible for these successes.**

| Public sample | Score / 300 | Strict physical + toolkit passes | Decision |
| --- | --- | --- | --- |
| V17 baseline, two repeats | 257.76 / 257.65 | 2/3 each | SC stops just short of 15.5 mm |
| V21 with close-up update | 257.36 | 3/3 | First full sample pass |
| V21 repeat | 170.20 | 2/3 | SFP force guard returns before contact dwell |
| V26, two samples | 257.65 / 257.44 | 3/3 each | Second acquisition timing p95 exceeds 300 ms |
| V27, shared JPEG acquisition | 257.59 | 2/3 | Timing passes; SC depth still short; repeats stop |

The SC update also improves a matched zero/five-card, both-port development grid from **1/4 to 3/4**. Five-card / port 1 still fails at 15.49053 mm. One matched update-on/off pair supports the update: 15.52 mm / 0.69° with it, versus 9.30 mm / 2.27° without it. These small tests do not establish all-settings reliability.

Training with 80 independent near grasps improves orientation p95 from 1.587° to **1.026°**, at a complete learning-rate-floor plateau. Additional feature and augmentation fits do not improve it. The original four-episode validation subset still fails the 1° gate. An independent four-rollout replay gives 0.717° near orientation p95 and 0.129 mm position p95. The model is therefore tested only under explicit experimental scopes; its general qualification flag remains false.

The latest V34 candidate passes all **4/4 matched SC development cases**, then passes the SC round of the unchanged public sample at **15.510 mm / 0.141 mm / 0.469°**. Its full sample is **170.57/300, 2/3 strict insertions** because the second SFP trial fails. Automatic identical repeats stop on that failure. The SFP failure has a confirmed orientation-coverage gap: actual held-plug roll 26.52° exceeds the training range 17.81–23.12°. Its initial orientation error is 3.53° and the same model's close-up error remains about 3.57°. Further lateral stiffness alone is not the selected repair. No new training or collection is running.

The close-up update requires quiet, synchronized observations, agreement within 0.2 mm / 0.5°, and a change from the initial estimate within 1 mm / 3°. Persistent large disagreement aborts safely; that branch does not yet automatically reacquire. The pose stays fixed within the attempt. Gazebo fixes this plug to the gripper, so these results do not demonstrate real gripper-slip recovery. All scoring uses retained pre-reset terminal observations. Final scenes remain sealed.

Artifacts: `/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/`. Live machine states include `V27_three_sample_state.json` and the current development run folders. The chronological sections below preserve negative results and plan revisions.

## Goal and constraints

Autonomously repair SC explicit image-based insertion, retain the working SFP controller, and repeatedly run the unchanged original three-trial predeadline sample after meaningful candidate improvements. No true simulator geometry or scoring events may steer deployment. Keep final scenes sealed. Do not change physical gates, official contact behavior, task timeout or collision settings to obtain success. Rootless Docker, independent domain/network/GPU locks, at most four project GPUs including untouched legacyGPU0. All new artifacts go to NAS; `/data1` is nearly full.

## Starting evidence

Previous full sample152.6066→176.8479/300; bothSFP physically seat, SCstops2mm/R2.18degrees. A separate V17local-recovery pilot gets toolkit75 but physical15.431mm/0.373mm/1.124degrees still fails. The masked/public-feedback BC comparison0/8and backoff0/12are not explicit-controller success evidence. No newRL.

## Execution order and decision rules

1. Freeze V17and run the unchanged three-trial public sample. Analyze causal pre-reset geometry, contacts, force, actual retreats and videos. This is a baseline for the selected explicit SC candidate, not promotion from its single pilot.
2. Check runtime physics provenance against announced upstream ghost-collision fixes. Fixed source ancestry and a3.8.0/8.3.0version string do not prove binary patch identity. Fresh standalone SCfailures show ghost collisions cannot explain everything. Only an order-dependent matched force/blockage difference justifies a simulator repair. Do not alter physics during a frozen run.
3. Repair the measured acquisition-view data gap: collect36bounded SCperception-only holds at actual visually accepted roughly150mm acquisition, bothports,0/1/5cards, public-sample board/rail/grasp variation.24training/12validation complete resets; firstsix require raw causal labels, stable motion, no force violation and visual image/projection QA before continuing. No BCexpert admission. Keep source datasets unchanged.
4. Merge with2039triplets/174existingSCepisodes. Train a spatial full-pose model through LR-floor validation plateau. Predeclared deployment gate: translationp95<=0.4mm andorientationp95<=1degree on every actual-acquisition held-out domain; report home domain separately because deployment does not use this model at home. Save best, last, manifests, commands, convergence and prediction-only QA. If accuracy fails, diagnose domain/crop/keypoint errors and try one distinct targeted model repair rather than repeat arbitrary training.
5. Deploy only a passing pose model in a new explicit-controller version. Port and grasp estimates stay fixed per attempt; invalidate old estimator-specific correction priors. Preserve physical-goal tracking and actual measured-path recovery. If model precision remains insufficient, investigate bounded observation-only visual refinement or contact-based search, without importing evaluation geometry.
6. Run freshSCdevelopment episodes covering bothports and0/5cards; inspect force and real plug movement and require full geometry+toolkit pass. Compare no/local/hierarchical recovery on matched cases once a working controller exists. No claim of hierarchical advantage from one successful local retry.
7. Run the unchanged full three-trial sample after a validated improvement, analyze every failed trial, iterate targeted fixes and repeat. Report officialscore separately from physicaldepth>=15.5mmSC/44mmSFP, lateral<=0.5mm, orientation<=1degree, noharm, and live latency. A sample score alone never closes the physical gate.

## Jobs started

`sample_V17/`: `scripts/run_selected_recovery_sample_configuration.py --policy RunVisualConnectorServoRecoveryV17 --recovery local --output <root>/sample_V17`; GPU5/domain100.

`acquisition_captures/`: `scripts/collect_sc_grasp_variations.py --output <root>/acquisition_captures --viewpoint acquisition --gpu3 --domain116 --seed2026101802` (CLI arguments use spaces: `--gpu 3 --domain 116 --seed 2026101802`). The new `RunAcquisitionGraspCapture` runs the existing observation-only transport then holds for five seconds. Capture does not insert. GPU3and5pluslegacy0remain belowfour.

Bulkroot:`/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/`. Jobs use host-owned sessions; inspect state before resuming, never duplicate them.

## First complete continuation sample:257.763943/300

V17local recovery, frozen original three-trial sample, no training or physics changes during evaluation. SFP1:46.7686mm/0.2645mm/0.2490degrees; SFP2:46.7515mm/0.1552mm/0.6005degrees; both strict passes. SC:15.4789mm/0.2771mm/0.5389degrees, toolkit75, **strict depth failure by0.0211mm**. Do not round15.479to15.5and call it successful. The public terminal estimated tip depth15.541mm; its calibrated reference depth15.3905mm. The standard15.6mm target plus training-only−0.2095mm calibration and0.2mm seating tolerance can permit a below-gate endpoint.

V18tests only SC nominal depth15.6→16.0mm (before that same calibration), preserving force limits, alignment, recovery and physical thresholds. One reused pilot onGPU5/domain119; if it passes, test the unchanged full sample again. This is a bounded command-margin hypothesis, not permission to push through excess force or alter scoring. V17fresh development4cases runs independently onGPU4/domain118, bothports×0/5cards, seed2026101813.

The first acquisition capture reaches the visual acquisition region but fails repeated-port-pose consensus and retreats; no training label is admitted. Preserve original failure. A capture-only revision accepts a native visual estimate within15mm/3degrees of the safe acquisition TCP viewpoint, then records a five-second quiet hold. It does **not** assert a deployment handoff or insertion success; held-plug supervision does not require requested-port identity consensus. Its first retained three-camera projected-label image was actually inspected: white plug housing points agree, orange cable is present. Six-case visual/causal QA remains mandatory before completing36.

A second capture worker waits for that first-six QA, then may prefill future cases in reverse onGPU5/domain117, keeping at least eight manifest positions ahead of the primary worker and never writing primary progress/summary. The primary collector admits/rechecks completed holds and remains the sole owner of the36-case summary. It uses the same scene manifest and split; no extra episodes are added. GPU locks serialize it behind the active pilot if necessary.

`continue_SC_acquisition_pose_repair.py` waits for all36verified holds, merges with immutable existingSCsupervision and fits one spatial full-pose candidate onGPU4to LR-floor validation plateau. Selection uses predeclared original/acquisition domains; all/home metrics still saved. No automatic model swap occurs without prediction QA and held-out accuracy.

Runtime image created2026-04-16T02:29:12Z, after the source ghost-collision pin commit2026-04-15T16:09:31Zand friction commitApril8. This is consistent with inclusion but is not proof of loaded binary identity. The fullSCcase now reaches15.479mm, and fresh pilots also show shallowlatch behavior, so unverified stale collision provenance is not a reason to halt direct pose/control repair.

## Bounded SC command and first-contact tests

V18's0.4mm seating-margin pilot fails at10.402mm/0.355mm/0.879degrees. No full sample is launched for a failed pilot. Actual selected-port contact is between `sc_port_link::cube_collider_box.006` and `sc_plug_link::housing ...box.004` at67.36s, when public estimated insertion is about3mm. It is housing/port contact, not cable/card snag. The recorded oracle wrench repeats scalar components, so do not use it as a trustworthy vector force decomposition; public F/T and contact identities/times remain separate evidence.

V19applies the same frozen training-only acquisition calibration before the first contact, not only after a recovery. The exposed fresh zero-cardport0case progresses to15.478mm/0.247mm/0.686degrees, but public seating is not reached and the task returnsFalse(tier3=0). This improves actual depth versus that exact V17fresh case's8.726mm/0.221mm/2.281degrees, but does not pass the full gate. Its first actual port contact is58.73s, public estimated depth about3.5mm; later motion freezes around15.54mm estimated. No evaluation geometry selects the calibration.

V20tests one additional variable on the same exposed fresh case: calibrated SCnominal advance6→10mm/s, with SFPunchanged6mm/s, the same16mm commandtarget,25mm/s reference cap, force guards and physical gates. The training-expert moving-entry p95is11.29mm/s. The hypothesis is that early side contact plus the official1scontinuous-contact latch can stop an otherwise aligned slower approach before seating. Do not change simulator latch rules. A strict pilot pass queues another unchanged three-trial sample; failure requires trace/contact review, not blind speed escalation. GPU5/domain121.

## Near-view re-estimation branch (user-approved)

Do not assume a stable far estimate remains correct at25mm, and do not assume a stable close estimate is accurate. `audit_SC_close_reestimation.py` evaluates the unchanged RGB models on five causal triplets before150mm lock and five at25mm in the same exposed development episode. First attempt misses a required metadata field before producing results; preserved empty output. CorrectedV2report: far translation/orientation p950.246mm/2.453degrees; close0.319mm/2.006degrees. True held-plug drift is only0.00314mm/0.000212degree, while predicted orientation changes0.542degree. This episode supports estimator bias/domain error, not physical slip. It does not rule out slip elsewhere.

The old SCexpanded supervision explicitly filters estimated gap>50mm, so it does not adequately train a25mm re-estimator.12new bounded perception-only25mmholds are queued/running onGPU4/domain122, bothports×0/5cards×3repeats;8train/4validation bycomplete episode/reset, seed2026101817. Firstfour require image-time/visual QA. `RunNearGraspObservationCapture` uses existing RGB/public-state transport and25mmapproach then holds5seconds, with no learned actor inference or insertion attempt. These are not BCexperts or autonomous insertion successes.

Planned deployment gates: first validate the close estimator on independent near-view episodes; then require several distinct consistent observations while held safely before any update. Keep port pose fixed during this experiment, bound translation/rotation disagreement, and reseed the actual TCP command/reference so changing the grasp transform cannot cause a jump. Large disagreement triggers a second observation window; persistent disagreement requires measured-path retreat/reacquisition or safe abort, not blind averaging or forward motion. Calibration learned for the far estimator must not be blindly reused at25mm.

V20speed10mm/s fails15.471mm/0.395mm/0.606degrees, so no additional full sample is launched for it. Stop speed escalation.12successful training-only SCexperts independently checked through causal cached raw TF seat15.565–15.571mm, median15.569mm, orientation median0.0147degree. The chain agrees with the current independent raw-bag audit to<0.001mm. The15.5mm gate is achievable and remains unchanged. This makes sub-degree residual tilt/pose precision a stronger hypothesis than simply raising depth demand. See`expert_SC_seating_reference_audit.json`.

## Collection reliability repair

Two far-view collection routes fail before retaining a grasp hold; no data from them are admitted. Held-plug perception supervision does not require solving transport for each new reset. Keep the eight admitted route-based holds and the completed prefill hold. For remaining cases, reuse a recorded PUBLIC acquisition robot configuration as a training-only reset; rotate cable WORLD RPY and WORLD gripper offset with the changed wrist orientation to preserve each planned relative grasp (`aic_engine.cpp`1223–1260). New scene directories and original manifests are preserved. No per-scene true port location, physics change, BCexpert admission or autonomous success claim. Pause after the first reset-mode hold for new visual/causal QA. This also avoids wasting full transport runs for pose-only labels.

A development scoring helper also compared a missing/None simulation timestamp against a float. It now requires a numeric timestamp before selecting the causal observation. Preserve the failed scoring log and recompute from the same retained bag; do not rerun the physical episode or join post-reset geometry.

## Second frozen full sample and negative control ablations

A second unchanged V17full sample scores **257.654574/300**. BothSFPpass again. SC15.4756mm/0.3021mm/0.6179degrees fails the unchanged depth gate, despite toolkit75. First/second SCaxial tilt is0.5046/0.4964degree; center seating depth is consistently short. Do not count this as reliable strict SCsuccess.

The bounded V22near-contact compliance ablation ramps rotational stiffness50→5and damping20→2over10ticks, leaving translation gains,6mm/s advance, pose calibration and force guards unchanged. It fails15.467mm/0.322mm/0.657degrees and is not expanded. This is TCPcompliance, **not** a remote-center-compliance implementation; [MIT's manipulation notes](https://manipulation.csail.mit.edu/force.html) explain why rotation about the contact versus gripper matters. Do not keep sweeping gains without evidence.

The ordinary 25mm close-estimator audit across four fresh cases initially stops on a stale same-time TCPcandidate. Those frames must be skipped and counted explicitly, never relabeled using post-reset/future geometry. The one completed paired episode remains sufficient to reject blindly adopting the old close estimator; broader accounting follows.

## Pose-training geometry repair before the next fits

The legacy spatial joint trainer orthonormalizes four housing-edge references for its rotation loss, while deployment and validation rigidly fit all five **noncoplanar** CAD references. On510saved held-out predictions, these rotations differ median0.105degree,p950.404degree,max0.569degree; translation differences are small(p950.0127mm). This inconsistency affects the spatial candidate branch, not the current direct-RGB rotation head, and is not claimed as the sole cause of current SCfailures.

`differentiable_rigid_fit.py` uses exactly the deployed rigid-SVDforward and an implicitSO3stationarity backward. Finite-difference checks pass for noisyCADand repeated singular values; forward rotation matches numpySVDto7.8e-16. Next far-only and far+near fits share this repair, dataset episode/reset split, architecture, initialization and LR-floor plateau stopping. Save helper/trainer hashes and snapshots.

A codec audit finds another small mismatch: `_fresh_report` already converts toJPEG86and the old grasp helpers normally convert again. On the first paired episode, single/doubleJPEGrotation difference p95is0.007/0.009degree at150/25mm; this does not explain a2degreebias. Preserve frozen old results; the new close-loader consumes exactly one training-matched conversion.

`close_grasp_consensus.py` implements five distinct fresh three-camera observations, maximum0.2mm positional span and0.5degree angular span. It rejects repeated camera stamps, invalid rotations and outliers. A change>1mm or>3degrees is not applied blindly; obtain a second window and retreat/reacquire or abort if disagreement persists. Model absolute accuracy must pass first. No live close update is claimed yet.

## Capture label admission and exceptions

12near25mmholds complete,8train/4validation, after actual four-case visual/causal QA. A single far synthetic reset does not settle under repeated longer holds/alternate public reset/nominal-grasp trials; all failed retries remain. Its original five-second recording still has four **quiet**, causally labeled triplets. These were actually viewed: zero image/label age, relative-grasp translation span0.05199mm androtation span0.01156degree, raw terminal force<40N. They are useful perception labels despite the world robot failing the extra long quiet-window test. Admit only those four under an explicit sample-count exception; do not admit other unsteady/unlabeled frames or call this an expert trajectory/success.

All other captures retain the>=6image rule. The original rejected4-frame derivative and subsequent failed physical attempts remain; sampled-window revisions/scene plans are versioned. Both frame-weighted and **episode-balanced** held-out p95enter the unchanged0.4mm/1degree component gate, so the lower frame count cannot hide this validation episode. No physical insertion threshold or image-time/force limit is relaxed. The collector resume bug that skipped invalid episodes is fixed; they remain rejected until independently reviewed or validly recollected.

## Gated close-view controller implementation (not yet deployed)

`RunVisualConnectorServoRecoveryV21` now implements the requested discrete SCnear-view re-estimation. It requires an exact-checkpoint held-out approval, holds the last issued TCPreference, waits for five distinct synchronized fresh camera triplets while stationary and unloaded, and bounds their position/orientation spread. A change within1mm/3degrees is accepted once; large disagreement requests another independent window and aborts safely if it persists. The fixed port frame is unchanged. Pending actions/governor state are cleared, the unchanged physical APIreference is reseeded, and the ordinary bounded-speed/orientation controller performs subsequent alignment. No old-estimator SE3calibration is reused. SFPuses its existing controller. This code is compiled and its consensus/reseeding numerical contracts checked; **no live deployment or success is claimed** before the new model passes QA.

The corrected four-episode old-estimator audit is `close_reestimation_audit_fresh4_v2/`: recorded native JPEG86frames are consumed without a second conversion, matching live raw RGBplus one helperJPEG86conversion. Far16samples/4episodes: orientationp952.401degrees, translationp950.300mm. Near14samples/3episodes: orientationp952.398degrees, translationp950.323mm. Fourfar/fivenear candidate frames lack valid causal labels and are explicitly counted/excluded, not joined to future/resetgeometry. V3mistakenly adds a second JPEGconversion; capture-source audit confirms filenames and `IMWRITE_JPEG_QUALITY=86`, so preserve V3only as the double-encoding diagnostic. The small codec change does not alter the failed near-orientation conclusion.

## Causal wall-window label repair

A prefilled hold initially retains only3/12labels because the label helper searches only250ms of past MCAPwalltime. Direct raw inspection finds exact same-simulation-time plugTFrecorded~400–480ms before some JPEGcapturewalltimestamps. Expand only the past log-time search to2seconds; retain the original end-at-image-wall cutoff, <=10ms image/TFage, <=2ms chain skew and <=4ms same-timeTCPgate. The same original hold now supplies12/12validlabels and its native projected-label montage was actually reviewed. No simulation rerun, futureTF or resetjoin is needed. Preserve the rejected short-window derivative and explicit repair QA. This fixes label lookup rather than relaxing physical/causal gates. Another five-cardtraininghold uses an expanded within-hold4.5s sampling window with unchanged quiet-image gates; its reviewed initialrecording contains8validimages.

The four fresh developmentportframes independently compared with raw pre-reset stationary-fixture geometry have0.037–0.055mm translation and0.007–0.104degree orientation errors, predominantly twist rather than axis tilt. Oldheld-plug estimates show axis-tiltp95~1.10degree at acquisition and~1.04degree at25mm. These limited developmentmeasurements support repairing held-plugpose while leavingportrefinementfixed; no privileged pose enters runtime. See`port_frame_error_decomposition.json`and`old_grasp_error_tilt_twist.json`.

The completed post-lookup-fix four-episode audit is`close_reestimation_audit_fresh4_v4/`and supersedesV2for causal coverage: acquisition20samples/4episodes,0rejections, translationp950.303mm/orientationp952.408degrees; near18samples/4episodes,1genuinely stale frame excluded, translationp950.323mm/orientationp952.396degrees. No error improvement is claimed from a label lookup repair. The oldcloseorientation remains unsuitable for blind deployment.

Live-test preparation is complete:`run_visual_servo_close_grasp_probe.py`and`run_SC_close_grasp_full_sample.py`freeze exact models/approvals/code and use`run_close_grasp_insertion_eval.sh`to forward and verify both new Dockerenvironmentpaths. The full runner preserves the exact original three-trial sample, originalhome, physicalgates and SFPcontrols. These are prepared commands, **not executed results**.

## Matched pose fits started

Collection is complete:36acquisitionholds(24train/12validation),12nearholds(8train/4validation), all completeepisode/reset splits. Actual valid-frame counts yield2,441triplets/210episodes in`acquisition_pose_dataset`and2,582triplets/222episodes in`far_and_near_pose_dataset`; train/validationepisodes154/56and162/60respectively. Eachtriplet has three nativecamera crops. Full merged nativecrop QAwas actually viewed; fixed training-derived crops cover projected references, including occluded cases. Fits run concurrently onGPU4andGPU3. Minimum5,000updates, reduceLRto1e-6and stop only on validationplateau; caps are explicitly incomplete. Do not interpret intermediate scores as final.

## Conditional compact-crop fallback prepared, not yet trained

If both matched pose fits finish plateau without passing, one predeclared visual-resolution ablation uses the same far+near rows, split, model/init, optimizer and plateau rule. TRAIN-only projected-reference extrema plus12nativepixels padding, rounded upward to16pixels, give fixed center/left/right nativecrop sizes **128/160/176**, then endpoint-aligned affine sampling to192encoderpixels. Held-out references remain inside these TRAIN-chosen ROIs; held-outgeometry did not select/enlarge them. Earlier speculative112-all-camera ideas are invalid and are not used. Metadata:`compact_crop_training_only_plan/crop_geometry.json`.

`prepare_sc_compact_native_grasp_crops.py`prepares this derivative without changing sources; `train_sc_compact_spatial_grasp_to_plateau.py`uses correct per-camera nativepixel spans for both supervision and pose solving. The inference loader supports exactly the same affine crop mapping. No arbitrary pixel-resize/half-pixel mismatch is allowed. This branch is **conditional**; preparation alone is not a model run or improvement claim.

Before any new fit is complete, clarify deployment scope:V21keeps the old far-acquisition model and uses the newmodel **only during the held25mmobservation**. Consequently exact-checkpoint runtime approval requires the unchanged0.4mm/1degree frame/episode-balanced gates on the four held-outnear episodes, runtimeparity and actualvisualinspection. The original worst-domain training gate remains reported separately; near-only approval is not a far/home model replacement. Currentnearerrors~2degrees stillfail, so this scopeclarification does not authorize a current model. Newcapture/traininggroups/finalscenes remainunchanged.

The closeupdate is restricted to estimated20–30mmstand-off, once per freshacquisition. Localrecovery retains that lockedgrasp instead of extrapolating a25mmestimator to a37mmbackoffview. Route-level freshacquisition re-enables the discrete25mmcheck. Command/governorcontinuity and forceguards remain unchanged.

## Far-only pose result: completed, not promoted

The far-only fit reaches validationplateau at17,000updates/LR1e-6, best1,200. Original-domain translation/orientationp950.188mm/1.229degrees; acquisition0.470mm/2.141degrees(14held-outepisodes). The unchanged componentgate fails. NativeRGBinference loader matches saved predictions in the runtimeparity check. Predicted-only virtual-landmark/axis overlays for two acquisitionepisodes were actually viewed; they track the white housing roughly, which is insufficient evidence of sub-degree accuracy. No deployment or insertionimprovement is claimed. Far+near fit and the conditionalcompactbranch remain independent.

## Far+near pose result and matched fitting diagnosis

The matched far+near fit reaches LR-floor validationplateau at20,800updates, best5,800. Nearheld-out4episodes/48triplets: translationp950.341mm, orientationp952.137degrees, axial-tiltp951.683degrees. It fails the original and near-only deployment gates; no model swap.

On the **same** nearrows, the far-onlymodel has orientationp952.754degrees; addingneartraining improves that to2.137degrees but not enough. The far+nearmodel fits its eightneartrainingepisodes at orientationp950.252degree while validation remains2.137degrees. This is a concrete generalizationgap, not a reason to keep running the sameoptimizer indefinitely. The compact-nativecrop ablation now trains onGPU4with the identicalsplit/budget and no newcollection. Preserve both failed baselinefits.

One inferenceparity intermediate-pixel threshold initially fails by0.0008pixel: CPUvsGPUmax0.0108px. Directcachevsnativeimagecrop comparison is **bit-exact**; resultingpose difference is0.00050mm/0.00182degree. Preserve the strict0.01pxdiagnosticfailure, report the numericalissue, and use0.05pxintermediatepixel tolerance while keeping strict poseparity0.01mm/0.01degree. This changes no estimatoraccuracy or insertiongate. The candidate stillfails its absoluteaccuracygate.

## Conditional independent-near-grasp coverage repair

If the compact-native model also fails near-only qualification after plateau, the measured0.252degree trainingvs2.137degree validationgap justifies **one24episode training-only extension**, not routine collection. Existingneartraininghas only8independentgrasps; held-outnearposes are~0.82–1.49degrees from their nearesttraininggrasp. Use newseed2026101829, bothSCports,0/5cards, randomgrasps/boardcontexts, uniqueepisodeprefix`coverage24`, and no addedvalidationepisodes.

Use recorded PUBLIC near-robotposture resets with cable WORLDrotation/offset compensation, not trueportgeometry. These are near-view held-plug perceptionimages; do **not** claim the fixed reset is25mmfrom every randomizedport, or autonomous transport/insertion. First6need actualvisual/causalQA; stop on invalidcapture; maximum24admittedepisodes, noBCexpertadmission. Keep physicalfailures and samplingQA. Preserve the current60validationepisodes byte-for-byte.

Then one original192nativecrop spatialpose fit uses the sameoriginalinitialization/optimizer/plateau rule, selecting the nearvalidationdomain because onlythatviewpoint willdeploy. Allotherdomains remainreported. This explicitly changes bothdata and checkpointselection versus the matched worst-domain baseline, so anygain cannot be attributed solely todata. It is a targeted deploymentrepair, not an isolatedcausalablation. `continue_SC_targeted_near_coverage.py`waits for the compactresult and skips allcollection if nearqualificationpasses. No automatic modeldeployment.

The newcontroller's actual hold/update method passes a ROS-free controlcontract test (`test_close_grasp_update_contract.py`): accept2degrees afterfive distincttriplets; reject5degrees aftertwoindependentwindows; reject8Nload immediately; preserve exact APIreference and portmatrix. This is not liveGazebovalidation. Newpose retries preserve alignedorientation rather than applying legacy2.5degreeangularsearch offsets, which would confound testing a correctedgrasp estimate.

## Compact-crop outcome: completed negative comparison

The compact-native fit reaches LR-floor validationplateau at17,200updates(best2,200). Nearheldouttranslation/orientationp95 **0.414mm/2.277degrees**, versus the ordinarynativefit0.341mm/2.137degrees on the sameepisodes. Runtimecrop/poseparitypasses, nativeinputandpredicted-onlyoverlaywereactuallyviewed. It is not an improvement and is notpromoted. Do not keep changing crop sizes.

The conditional24training-onlyneargrasp extension has consequently started onGPU3/domain123. It preserves all currentvalidationepisodes. A concrete remainingarchitecturalhypothesis, if thisdatacoverage fitfails, is that the spatialhead predicts eachcamera's virtualpoints independently and onlyfuses them through triangulation; it cannot use another camera's image to disambiguate an occludedpoint. One jointthree-view directorientationhead on the same data would test this, while retaining the already-estimatedvisualtranslation. This is a proposed boundedfallback, not an executed model or proof that moredata cannothelp.

## Additional bounded architecture check while coverage collection runs

The code audit confirms the landmarkmodel cannot share imagefeatures acrosscameras before guessing virtualpoints. In parallel with the already-started24resetextension, use one existing **jointthree-view directorientation** architecture on the already frozen2,582row dataset. Its sharedCNNfeatures from allthreecameras enter one orientationhead; add the existing crop/colouraugmentation to reduce the measuredtrainingmemorization. This is one model, GPU4, minimum5,000updates then LRfloorplateau, no addeddata/finalscenes. Near-only checkpointselection and half-near episode-balanced training match its deploymentrole. Allotherdomains and constantbaselines remainreported. Architecture, augmentation and selection change together: this is an engineeringcandidate, not an isolatedcausalclaim. Retain the independent existing visualtranslation; this candidate estimatesrotationonly. If it qualifies, test its gated25mmorientationupdate before waiting on another largefit.

The completed four-episode paired rawlabelaudit finds actualgrasp orientationdrift150→25mm only0.000041–0.000127degree andtranslation0.0019–0.0111mm. In these exposedepisodes, wrongvisualorientation is supported and physicalslip is not; do not generalize absenceofslip to all futureepisodes. `paired_actual_grasp_drift_4episodes.json`preserves the per-episode values. This further supports testing a rotation-only nearupdate while retaining independentvisualtranslation.

## Current Gazebo grasp is fixed, not a physical slip model

`aic_gazebo/src/CablePlugin.cc`228–234attaches cableconnection0to the end-effectorwith a **fixed** detachablejoint; `aic_assets/models/sc_cable/model.sdf`fixes theSCplug to thatconnection. This explains the negligible measured150→25mmrelativegraspdrift in the fourdevelopmentepisodes. Episodeinitialization can stillrandomize thegrasp, and visualestimates can bebiased. Do not describe this Gazebobranch as evidence of handling real gripperslip.

Because the camerasare wrist-mounted, held-plug appearance inTCPcoordinates should largely persist while background/occlusions change. A prediction changingwhile truegrasp staysfixed is evidence of view/background sensitivity. The24trainingresetextension randomizesboardcontexts independently ofgrasp to address this. Its firstsix allretain12causaltriplets with0mslabelage; the18nativecameracrops withGTlabelQA wereactuallyviewed and passed before continuation.

A matched fused-orientation8→32neartraininggrasp comparison is queued onGPU4once theextensiondataset isready, with identicalarchitecture/init/augmentation/near-selection/budget. This separates its dataeffect more cleanly than the expandedspatialfit, whose checkpointselection alsochanges. Both preserve the fournearvalidationepisodes and originalall-domain reporting. No furthercollection or modelpromotion is automatic.

## Remaining SCseat-depth error matches housing-corner tilt

For the two frozenfullsampleSCterminals, rotation vectors have~0.50/0.48degree rotation about the housing's wide-dimension tiltaxis. The front-reference halfwidth/height are10.8/3.65mm. Their tiltedfrontcorner axialexcess isapproximately `10.8*abs(R20)+3.65*abs(R21)+2.25*(1-R22)`mm:0.0957/0.0983mm. Observedshortfalls from the trainingexpertmedian15.56895mm are0.0900/0.0934mm. The simple frontbound predicts terminaldepth within0.006mm. This supports corner-limited seating rather than a need forlargerforce; it is a posthoc geometricproxy, not an exact contactsolver or runtimegeometryinput. A0.5degree wide-axis tilt is already significant; the remaining0.069mmseatmargin corresponds to about0.37degree if thataxisdominates.

V21also separates estimatedseat acceptance from commandeddepthmargin: SCnominalseat15.6mm and existing0.2mmtolerance remain, while command target is16.0mm. This avoids V18's bug of demanding an impossible15.8mmestimatedseat at a physical~15.57mmstop. SFPnominal/commanddepthremain46.5mm; forceguards, dwell, scoringgates unchanged. The newcontroller is still **notlive** until its estimator qualifies; this is an explicit preparatory change, not a measuredsuccess.

## Withdrawn pixel-loss concern — source check prevented a false repair

An offline audit initially assumed the pixel term used normalized0..1 residuals. This was **incorrect**: the actual trainer already uses `smooth_l1((prediction-target)*191,0,beta=1)`. A one-pixel error contributes0.1after its0.2weight, not2.74e-6. The earlier audit formula and associated diagnosis are withdrawn and preserved as invalidated artifacts. The attempted repair script generation failed its source assertion before creating a trainer; the following launch failed because no file existed. **No new training ran, and no loss-scaling bug was found.**

Native landmark error measurements from that audit remain real(neartrainp950.425px/validation3.933px), but loss contributions are recomputed from the actual formula. Do not use the withdrawn claim to justify more training. Existing32-grasp fits continue unchanged.

A separate fixed, observation-only all-landmark reprojectionfit on completedspatialpredictions improves nearorientationp952.137→1.779degrees; compactresults worsen2.277→2.525degrees. Neitherpasses. Optimization uses onlypredictedpixels/CAD/calibration; GTonlyscores it. No runtimechange is made.

## Plan revision: systematic near-grasp coverage, after the32-grasp fits fail

The fullRGB-head warm start finishes at LRfloorplateau(17,000updates), nearorientationp50/p951.368/1.587degrees. It improves thetail versus freshhead32-grasp1.802degrees but is not qualified. No model is deployed. Source snapshots also confirm pixelresiduals were already multipliedby191 in both completedspatialfits; the withdrawnloss concern is not reinstated.

The next experiment is **48systematically spaced training-only grasps**, followed by **one fullRGB-head warm fit** on80neartraininggrasps. Earlier radius-uniformsampling favours thecentre of the rotationball; despite24additionalresets, threeheldoutgrasps still have unchangednearesttrainingrotationdistance~1.1–1.5degrees. The new inputrotations use48Fibonacci-sphere directions: half at the existing0.06radian bound, half volume-uniform radii. Independenttranslation offsets are volume-uniform within the existing1.5mmrange. This fills the samepredeclaredrange rather than selectingposes fromvalidationlabels. Include0/1/3/5cards andbothSCports, sixresets per setting; board/railrandomization remains sample-derived. Unique` shell48 `episodeprefix; allnewrowsTRAINonly.

Budget: threeisolated rootless workers onGPUs3/4/5andROSdomains125/126/127,16episodes each; together with untouchedlegacyGPU0this isfourprojectGPUs. Wait for existingjobs to finish; neverinterrupt them. First3episodes perworker pauseforactualnativeimage/causalQA. Stop oninvalidcapture. Maximum48admittedepisodes, then onefit atGPU4to the sameLRfloorplateau; no automaticcollection beyondthat. Keep currentvalidationepisodes byte-for-byte and finalssealed.

Continue to the live25mmupdate only after exact-checkpoint nearqualification(Rp95<=1degree, episode-balancedalso<=1, runtimeparity, actualprediction-overlayinspection). Then test the previouslyfailedSCpilot and the unchangedfullthree-trial sample, with independentphysicalscoring. Success stillrequires15.5mmdepth/<=0.5mmlateral/<=1degreeorientation/toolkit75/noharm. If80-graspRp95doesnotmateriallyimprove(15%relative to1.587degrees) and stillfails, stop routine scaling and revise the representation/geometry supervision. If itimproves butfailsqualification, make a separate evidence-basedplanrevision. Do not call componentqualification an insertion success.


## Completed 80-grasp fit and bounded fine-feature revision

The 80-grasp warm fit reaches an LR-floor plateau after 18,400 updates (best 15,400), rather than stopping for time. On the unchanged four near validation episodes / 48 camera triplets, orientation p50/p95 is **0.909 / 1.026°**. Runtime parity passes. Five fixed one-pixel crop shifts reduce p95 to **1.008°**, still failing; no deployment follows. These are development episodes, not evidence of broad reliability.

The exact five-fresh-observation consensus was scored separately with the qualified position component. All four first accepted updates are below 1° (0.635 / 0.960 / 0.861 / 0.997°), but all accepted overlapping windows have p95 **1.004°**. Per-episode error-vector standard deviations are only about 0.02–0.06°, versus systematic errors around 0.6–1°. Averaging cannot remove most of this bias. Do not promote by rounding or selecting the favourable first-window statistic; the original component qualification remains failed.

**One targeted architecture experiment:** the current fused model pools layer-3 features to 6×6 before predicting orientation. Add a layer-2 feature path pooled to 12×12, retaining finer spatial detail from all cameras. The additional 451,214 parameters bring the model to 4,597,500 total parameters. The new path starts at zero: a preflight confirms bit-exact physical position and orientation predictions versus the completed 80-grasp model before training. This is a perception architecture ablation, not an action residual actor. Same 3,446 samples / 294 episodes (234 train, 60 validation), augmentation, split, selection and LR-floor stopping; no new data or final scenes. GPU4 only, with legacy GPU0 untouched.

Gate: unchanged near orientation p95 and episode-balanced p95 ≤1°, runtime parity and actual prediction-only image review. If it qualifies, combine only its orientation with the independently qualified 0.325mm position component, then run one failed SC development pilot with the gated 25mm update. Require independent physical insertion before broader evaluation. If it fails to improve, preserve it and revise the representation hypothesis; no routine data expansion is queued.

Artifacts: `fine_feature_preflight.json`, `final80_orientation_review/`, `five_shift_final80_v1/`, `final80_five_fresh_consensus/`, and `fine_feature_near_orientation/` beneath the continuation root. The 25mm controller remains untested live at this point.


### Fine-feature comparison: retain the warm candidate and protect its features

The first fine-feature fit continues unchanged. A code audit finds its candidate selection starts after training, so it can discard the better initial warm model if early updates worsen validation. This did not negate the completed80-grasp improvement over its much worse original initialization, but matters for this small incremental ablation. The next snapshot saves step0 as a candidate.

A second bounded fit onGPU3 freezes the entire existing model and trains only the new fine-feature path. Its purpose is to distinguish missing fine detail from destructive fine-tuning of the already useful coarse model. Same images/split/augmentation/selection/LR-floor stopping; no added data. Both runs finish before further decisions. No live approval is implied.

Standalone combined preprocessing + position + fine-orientation inference onGPU3 has warmed p50/p95/p99 **36.4/41.0/44.2ms** over30 repeats. First cold call337ms; V21 now warms network kernels before transport, without storing any synthetic-pixel pose or bypassing real observation gates. These numbers exclude ROS assembly and control; complete live timing remains required. `fine_feature_combined_latency_prototype/summary.json` preserves the exact prototype hash and limitation.


### Completed fine-path result and rejected CAD-edge screen

The jointly adapted fine-path model reaches an LR-floor plateau at17,000updates: near orientationp95 **1.083°**, worse than the1.026° warm baseline. Runtime parity passes; no deployment. The frozen-base comparison is still finishing.

One fixed four-episode midpoint-triplet CAD-edge screen uses known static plug mesh, RGB-derived initial pose and RGB Canny edges; true pose is used only for scoring. It improves two estimates and worsens two (orientation0.662→0.858,1.022→0.977,0.869→1.366,0.968→0.615degrees), taking1.65–1.67seconds per triplet. A prediction-only wireframe image was actually viewed: many rendered edges are hidden in the real view. The screen lacks adequate external-occlusion handling and fails both consistency and latency; **reject, no tuning or deployment**. Artifacts: `mesh_edge_registration_screen/`.

### Bounded calibrated-coordinate augmentation check

Source audit confirms direct-orientation training randomly translates each camera crop independently by up to4pixels, while the direct model receives neither shift nor updated crop origins. This encourages invariance, but can remove fine calibrated inter-camera position cues. It is not established as the main cause. An ideal-projection training-only audit finds that these shifts create geometric translationp95 **1.61mm** and orientationp95 **0.177°**, versus numerical-zero unshifted error. The orientation perturbation is smaller than the full observed failure; do not present it as explaining all1degree bias.

One fixed-data/base-architecture warm fit removes spatial jitter while **retaining colour augmentation**; the old implementation bundles the two, so the snapshot explicitly separates them. It retains the initial80-grasp model as a candidate and always saves a qualification-threshold crossing even below the usual0.5percent checkpoint-improvement tolerance. Same episode split, near selection and LR-floor plateau stopping. GPU4 after the first fine fit finishes; the frozen-base fine fit onGPU3 continues untouched. No new data, final scenes or actor/RL jobs. Proceed to the prepared25mm pilot only on actual qualification and visual review; otherwise report the remaining perception gap rather than promote a rounded near-pass.


## Explicit plan revision: one unpromoted live diagnostic after independent replay

The same80-grasp model was replayed on the already-recorded causal images from four *different frozen development rollouts*, not the four model-validation episodes. Episode and reset-scene identities are verified disjoint from training. At25mm, its orientationp95 is **0.717°** (episode-balanced0.719°), positionp95 **0.129mm**; oldRGB orientationp95 was2.396°. At150mm its orientationp95 is0.850°. These independent development results support feasibility, but **do not erase the original validation subgroup failure1.026°**.

Revise the earlier absolute prohibition on any live test into **one explicitly experimental diagnostic** on the already selected failed `fresh_SC_explicit_n0_p0_seed1813` start. This is not model qualification or deployment promotion. The runtime certificate keeps `pose_gate_passed=false`, `original_validation_gate_failed=true`, and `experimental_trial_only=true`. The diagnostic runner requires an explicit flag, one exact case and one variant; normal qualified paths continue to reject this bundle. It retains all25mm observation-consensus/change/load gates, physical scoring and force limits. Images/weights/code/certificates are frozen. No true geometry enters control. No broad evaluation follows a failure without targeted diagnosis.

This controlled test checks the actual requested hypothesis: does a stable close-up pose update improve autonomous insertion? Physical success, not crossing a proxy model metric, decides whether a further public-sample experiment is warranted. Final scenes stay sealed. No extra collection is queued.

The narrower long-housing-line alternative to dense CAD edges also fails its fixed four-triplet screen:0.662→0.664,1.022→2.025,0.869→0.201,0.968→2.063degrees. Runtime60–72ms is feasible, but associations are unreliable; do not deploy or tune it on these labels. `housing_line_registration_screen/` preserves failures.


## First live25mm update: full SC physical insertion

On the exact previously failed `fresh_SC_explicit_n0_p0_seed1813` development start, V21 applies the five-observation RGB update at52.65simulated seconds: **2.574°** orientation change and **0.339mm** position change. The port remains fixed and the issued reference is retained. One real loaded-stall local retreat/lateral retry occurs; no legacy2.5°angular search is applied.

Independent pre-return raw-MCAP scoring confirms **15.523913mm depth,0.113655mm lateral,0.688331° orientation,toolkit75,no scored harm**, TFages0/0ms. This passes the unchanged physical SC gate. It is **one development success**, not broad model qualification or proof of real slip recovery. V21 combines close re-estimation, separated16.0mm command/15.6mm acceptance depth, and aligned retries; the causal contribution of each is not isolated yet. No learned BC actor or RL is used.

Complete close-update processing p50/p95/p99 is87.2/102.1/104.3ms (five calls); insertion control8.69/11.84/18.61ms (286calls). Used upstream phases also pass p95<300ms: transport172.8ms and fresh acquisition245.0ms. Transportp99 remains414.6ms and must remain reported. The generic BC scorer reports `qualified=false` because it requires an actor to run and produce actor timings; those are intentionally absent. Its original JSON is preserved. A separate controller-specific gate scores all **actually used** phases and passes, without changing physical or timing thresholds.

Artifacts: `V21_close_grasp_development/explicit_trial_gate.json`, `runs/.../shared150mm_metrics.json`, raw bags, runtime trace and `five_cameras_20fps_grid_active.mp4`. The historical variant filename says“qualified”; the frozen plan/certificates explicitly mark the model **experimental and globally unqualified**.

Following this physical pilot pass, run the unchanged original-home three-trial public sample with a newly scoped experimental certificate and an explicit pilot-gate/checksum requirement. `sample_V21_experimental/` is running onGPU5/domain100. SFP control stays unchanged; no training/models/scenes change during the sample. Original model-validation subgroup failure remains visible. Further promotion requires repeated physical success, not this single episode.


### Training comparisons closed; isolate the live update

Both remaining fits finish at LR-floor validation plateaus after17,000updates. Frozen-base fine detail and geometry-preserving/no-spatial-jitter both retain **step0**, the original80-grasp warm model, at1.026°near p95. No later candidate improved enough to replace it; colour-only augmentation does not solve the remaining gap. No further model or dataset job is queued.

While `sample_V21_experimental/` runs unchanged onGPU5, one matched ablation onGPU3/domain129 repeats the same failedSCdevelopment start using `RunVisualConnectorServoNoCloseV25`. It inherits everyV21 controller setting (command margin, aligned retry, local recovery), but disables only the close-up pose update. This tests whether the newly successful insertion depends on the update rather than attributing a multi-change comparison solely to perception. `V25_no_close_matched_ablation/` preserves the exact manifest and video. No new learning or final-scene access.


### Matched update-on/off result

The V25 ablation fails on the sameSCdevelopment scene/start: **9.301784mm depth,0.044037mm lateral,2.270794°orientation,toolkit0**. V21 with the close-up update passes15.523913mm/0.113655mm/0.688331°/toolkit75. Every otherV21 controller setting is matched, including its overtravel/acceptance separation, aligned retries and local recovery. This supports the close-up update as a useful cause rather than crediting the entire improvement to unrelated controller changes. It remains **one matched pair**; fresh repeats and different settings are needed for reliability. `V21_close_update_matched_ablation.json` preserves both independent raw scores; both videos are retained.

Terminal geometry also explains why a0.688°total angle can seat more deeply than an earlier0.54°case: V21's rotation vector is[0.393,−0.165,0.540]degrees, with only0.165°about the SC housing's wide-axis-sensitive tilt direction. Its front-corner axial-excess proxy is0.0566mm, versus~0.096–0.098mm in the shallowV17samples. Angle direction matters; total orientation alone is insufficient. This remains a posthoc geometricproxy, not a controller input.


## First unchanged full sample:3/3 strict insertions,257.363666/300

V21's first complete original-home public sample now passes **all three independent physical/toolkit gates**. SFP1:46.7678mmdepth/0.2703mmlateral/0.1592°; SFP2:46.7575/0.1612/0.5283°; SC: **15.518068/0.239111/0.519808°**, toolkit75and no harm. SC accepts the RGB update at157.25simseconds, changing its orientation estimate2.430°. The old twoV17samples were2/3strict at257.76/257.65points. Score is not the improvement: those old SCtrials already received toolkit75while failing depth. The independent physical audit changes2/3→3/3.

Fresh-acquisitionp95 **296.5ms**, p99308.8ms; close-updatep95104.8ms; insertion-controlp9511.4ms. All used stages pass the p95<300ms gate, but upstream acquisition is close to the limit. Report its tail instead of claiming a comfortable latency margin. Same rules, requested-port stack, original three trials/order and scoring; only the documentedV21candidate changed versusV17. No true geometry or learned actor/RL in control.

`continue_SC_V21_three_sample_checks.py` now runs **at most two additional unchanged samples**, analyzing each causal physical outcome and complete stage timing before continuing. Stop automatic repeats on any failed physical/timing gate and diagnose that specific failure. Do not retrain or tune between these repeats. One full sample plus one development success is not broad all-settings reliability; model's original validation subgroup remains unqualified. New reserved-final cases remain sealed.


### Bounded matched SC coverage after first full-sample pass

After the first3/3publicsample, test the remaining three starts from the existingV17four-case development grid: zero-cardport1, five-cardport0 and five-cardport1. Reuse the already-completed V21zero-cardport0 pilot explicitly; do not count it as fresh again. Each new case has a separate one-case experimental certificate, identical weights/controller/local recovery, GPU3/domain130, and independent physical+used-stage timing gates. Maximum **three new episodes**; stop on the first failed gate for targeted diagnosis. `V21_remaining_SC_coverage/state.json` records exact progress. These are development cases, not sealed finals or new training examples.


## Second sample exposes a separate SFP seating-impact return

The second unchangedV21sample scores **170.196596/300**: SFP1gets1point, SFP2andSCget toolkit75. SFP1independent geometry is already fully seated:46.7713mmdepth/0.2709mmlateral/0.1469°orientation,noharm, but toolkit0. Its controller aborts at47.7s on the8Nload-change guard, immediately after47.65sload1.31N and46.38mmestimateddepth. This interrupts the required contact dwell. It is not an SCrouting/regression failure. Stop ordinaryV21repeats after this failure; keep the full failed run.

TargetedV26test: **only near an aligned, slow SFPseat**, an8Nload change immediately freezes the lastissuedAPIreference without advancing/integrating. If it clears within100ms, resume normal seating dwell; sustained100msload stillaborts. Outside the seat,8Nstillaborts immediately; raw40Nemergencyguard is unchanged. No raisedforce threshold. Actual-loop tests cover transientclearing, persistentload, off-seatload, raw41Nand noforwardcommand whilepaused. This is a hypothesis that the event is a seating impact, not a proven sensorbug.

A separate code audit finds the25mmangular-speed check reads state13:16(positiontrackingerror) instead of10:13(actualangularvelocity). V26corrects it, preserving all activeV21files. The first successfulpilot's five accepted publicobservations retrospectively all satisfy the intendedangular-speed gate. A test injects two movingframes and requires them to be rejected before fivequietframes. Existing physicalscores are unchanged.

After the ongoingV21causal/videoanalysis completes, runV26on the same originalpublicsample; if all three physical/timinggatespass, repeat twice unchanged. Stop at the firstfailedgate for targetdiagnosis. No modelweights/data/physics/finalscenes change. `V26_three_sample_state.json` records this bounded repair cycle; V21SCcoverage onGPU3continues untouched.


### SC coverage outcome:3/4, with a preserved narrow failure

Together with the initialzero-cardport0pilot, V21passes **3/4matched SCdevelopment starts**, versus the oldV17baseline1/4. Newzero-cardport1:15.51234mm/0.20215mm/0.43163°; five-cardport0:15.50937/0.22553/0.83004°; five-cardport1: **15.49053/0.30476/0.52745°**, toolkit75but **fails depth by0.00947mm**. Allusedtimingstagespass. No round-up, broad promotion or further routineSCrollout follows. Its front-corner axial-excess proxy0.08535mm exceeds the roughly0.069mmexpertseatmargin; the two newlypassingcases have0.06732/0.06998mm proxies. This supports a residualorientation/contactprecision gap, not a new transport/cablesnag claim. Exactcontacts would be needed for a strongermechanisticclaim.

The second fullV21sample's SCalso passes15.51604mm/0.26173mm/0.57924°; its onlyfailedtrial is SFP1's earlyforceguardreturn. Thus fullpublicSCsuccessrepeats, while all-settings SCreliability remains open. Preserve this distinction.

### Oracle identifies SFP tip/target contact onset, not a cable snag

At the failed SFPguard event47.70s, the oracle first shows `sfp_tip_link::contact_collision` contacting the NICtarget `contact_collision_01`, maxnormalforce perpoint6.864N using the pinned legacy impulse/2ms interpretation. Public wristloadchange jumps1.395→9.761N. In the alreadyrecorded47.69–47.80s window it fallsbelow8Nby47.72s(5.209N), then2.84Nby47.74s. TCPposition spans0.0551mm; APIreference spans0.0101mm. This supports a brief seatingimpact instead of persistentblockage. There is a small referencechange afterreturn, so it is **not a perfectly unchanged-control counterfactual** for V26.

Those extra100ms are explicitly post-return **force-event diagnosis only**; no terminalgeometry/scoringlabels were changed or joined acrossreset. PublicTCPcontinuity and the same targetcontact pair show no resetdiscontinuity in that shortwindow. `SFP_seating_force_transient_audit/oracle_contact_onset_and_decay.json` preserves everyrecord and caveats. The separate JSONstreams are retained at~50Hz, not fullrawFT rate; oraclecontact capture is100Hz and can miss500Hzphysicsimpulses.


## Separate bounded hypothesis: same-grasp context for the marginal SCcase

Do not change the activeV26public repeats. A read-only fixed **equal-weight** fusion of the stronger80-grasp model's150mm and25mm orientation windows, on four frozen development episodes, reduces episodeorientationp950.6584→0.5613°and improves the housingcornerproxy in3/4episodes. It **fails its predeclared general screen**: the relativep95gain is14.75%,below15%, and oneepisode's totalangle worsens0.434→0.576°. Do not round this into a pass or call it a generallyqualified representation.

Nevertheless, there is a distinct **contact-specific mechanism test** worth one guarded trial: the failedfive-cardport1case's predictedcornerexcess falls0.0975→0.0549mm, below the~0.069mmseatmargin, while totalorientation stays0.325°. Othercasecornerproxies do not worsen materially(the onlyincrease is0.0007mm). Test **only that alreadyfailedcase**, with no tunedweights orlabels at runtime. This is a scoped experimentalplanrevision, not broadpromotion based on the failedscreen.

V28retains three valid, quiet RGBtriplets from the150mmacquisition, then encodes those cachedimages with the **same stronger RGBorientationmodel** after reaching25mm. The oldbiasedorientationmodel is never averaged. Require farwindowspan<=0.5°andfar/nearagreement<=1°; otherwise keep the original near-onlyupdate. Fuse orientations equally, keep translationnear-only, and retain alloriginalclose-update/load/commandcontinuitygates. Count historicalencoding in firstclose-step processingtime. The farhistoryis intentionallyoldercontext, not falsely declaredfresh. Gazebo's fixedgrasp makes this assumptionvalid; this is **not a realslipperygripper method**. The port remainsfixed.

Budget: one matchedfullrouteSCepisode onGPU3/domain131, `V28_context_fusion_failed_case/`; no newdata/training. If physicaldepth>=15.5mm, lateral<=0.5mm, orientation<=1°,toolkit75,noharm andp95<300mspass, then require matchedremainingcasechecks before broaderpromotion. Onfailure, preserve it and diagnose; no further automaticcollection isqueued. V26samplecontrol/model/scenesremainunchanged.


## V26two physical passes, then restore the validated acquisition codec optimization

V26completes **two3/3physical samples**, scoring257.646997/300and257.437881/300. SCdepths15.50110/15.52249mm; both pass, but the firstmargin is only0.00110mm. No SFPforce-pause event occurs in these two live runs, so do not claim directlivevalidation of that branch; actual-loop syntheticcontract tests and recordedcontact-onset evidence support it.

The second run fails the **latency** gate: fresh acquisitionp95315.70ms(vs295.73msfirst), despite3/3physicalpasses. The scheduler stops before a thirdV26sample. This branch inheritsV21/V13, losing the previously validatedFastV16shared-JPEGacquisition wrapper. Its closecompositemodel already sharescodec; the redundantwork is in the original150mmacquisition orientation/positionpair.

V27inherits allV26control and reinstates that exact existingwrapper: oneJPEG86conversion shared between unchangednativeorientation andpositionmodels for the sameRGBarrayobjects, freshscopepercall. Numericalequivalence is checked again without overwriting the oldCcycleproof. Weights, camera preprocessingquality and commandmath stay unchanged; only duplicateconversion is removed. Run atmostthreeunchangedpublicsamples, checking physicalandcomplete-used-stage timingaftereach; stop onfirstfailure. `V27_three_sample_state.json` records progress. ActiveV28SCdiagnostic remains separate and unchanged.

V28's firstfive-cardport1trial fails at4.309mmdepth/0.418mmlateral/0.351°orientation, toolkit0, after two loaded stalls and a retreattimeout. Farhistory is actuallyused(span0.133°), so this is a real negativeclosed-loop result despite improvedangle. Do not promote the fusion based on offlinegeometry; preserve the run and inspect the translation/port-error/contactdecomposition before furtherpolicytrials. No newdata ortraining isqueued.


## Decisive V28failure decomposition: tracking alignment, not a new data gap

Using only the same retained pre-return publicTCP and rawtip/portTF(age0ms), the failedV28run's portposition error is0.0245mm and lockedgrasptranslationerror0.1311mm. Perfect tracking of that estimatedframe would produce only~0.100mmlateraloffset, but actualoffset is0.418mm. The trace explicitlymarks axialadvance“aligned” at **0.357→0.496mmestimatedlateralerror** while crossing the opening. The0.5mminternaladvancegate permits entering before lateraltrackingconverges, after which contact can oppose correction. Do not infer a need foranotherposition-modeltrainingrun from the0.418mmterminaloffset alone.

V29tests onecontrolrepair on the exactsame failedcase, keeping V28context/weights/position unchanged: require estimatedlateral<=0.1mmfor threecontrolticks before axialadvance; pause advancement if it risesabove0.15mm. Existing0.5°estimatedangleguard, physicalsuccess thresholds, forceguards, commandmargin, dwell andrecovery stayunchanged. This tighter *internal motiongate* does not change the scoredphysicalgate. It reserves errorbudget for perception and gives lateraltrackingtime to settle beforecontact. No addeddata/training. Oneepisode onGPU3/domain132; independentphysical+latencygate decides continuation.

`V28_context_fusion_failed_case/locked_pose_error_decomposition.json` and `advance_alignment_gate_diagnosis.json` preserve the diagnosis. The pose comparison is at the same retained terminal under the explicitly fixed-graspasset assumption, not an image-time traininglabel and not post-resetgeometry.


### V29 aligns before the mouth, but is too slow; one lateral-tracking check

V29reaches~−0.23mmestimatedaxialdistance with0.085mmestimatedlateralerror, lowload(~0.65N), then hits its unchanged60second internalwallbudget after~27simulatedseconds. It has not demonstrated insertion. The tightgate prevents the previouslyaccepted0.4–0.5mmentry, but repeatedwaiting makes progress too slow. Official180second tasklimits remainunchanged.

One targetedV30comparison changes **only the lateral PIintegrator gain**,0.6→1.8persecond in the estimatedportframe; axialgain0.6, tightadvancegates, contextfusion, modelweights, reference rate/accelerationlimits, forceguards and budgets stayunchanged. At0.4mmerror this adds only36µmper50mstick rather than12µm, with the existinggovernor limiting actualcommands. Hypothesis: fasterlateraltracking allows the tightgate to remain open during axialmovement, instead of advancing on loosealignment or simply extendingtimeouts. Test only the failedfive-cardport1start onGPU3/domain133. Require physicalinsertion and complete timing before anybroadercomparison; preserve allfailedtraces. No newtraining/data.


### V30 fails; inspect initialization instead of collecting more data

The lateral-gain increase does not fix progress; V30 ends with `visual_servo_no_measured_progress`. Keep that negative result. A source/trace audit identifies an overlapping continuity mechanism: the integral starts from `(inverse_port @ last_API_reference @ grasp).translation - desired_reference`. Its lateral components contain both the controller's tracking bias and the **current lateral misalignment**. The governor already preserves/slews from the last API reference, so baking that misalignment into the integral delays the very alignment the tighter gate requires.

Measured first-step excess beyond the retained free bias is about0.404mm in V29 and0.512mm in V30, comparable to their initial0.36–0.40mm lateral error. This is a concrete initialization issue, not proof that the perception network needs more capacity.

V31 restores the original0.6 integrator gain and changes only lateral initialization: project the current **public controller TCP position error** (`state[13:16]`, reference-minus-measured in base coordinates) into the estimated port frame. Preserve the original axial integral seed and unchanged governor API seed/rate limits. Five quiet close-up observations have just held that reference; no force baseline is retared. Test the same failed SCcase once onGPU3/domain134, with unchanged budgets/force/success gates. `lateral_integral_seed_diagnosis.json` preserves the old seeds. No new data/training.

The first V27sample scores257.58784/300 with acquisition timing now passing, but only2/3 strict insertions; its SCphysical depth fails. Thus the latency repair works in that sample, while SC's narrow pose/control margin is genuinely unreliable. Stop routine V27repeats and preserve it. The separate SCprecision/tracking repair remains the next bounded experiment.


## Bounded plant-tracking diagnosis: test lateral stiffness, not more networks

V31still fails to insert before its unchanged35simsecond budget (actualdepth~0.012mm). Its lateral seed is corrected, but that alone does not establish sufficient tracking. A no-GT reference audit separates the software filter from the robot: after the first3seconds in free space (>5mm before the opening), governor lateral error versus desiredreference p95 is only **0.00018mm V29 /0.00064mm V30**. Robot lateraltracking error remains~0.18–0.21mm median, and the reference-to-measured bias changes~0.8–1.2mm. Do not blame the command filter or perception for all this error.

Source evidence: `shared_recovery_eval_environment.py` selects240translationalstiffness for ruletransport; `RunRuleThenSharedRecovery` overrides the local acquisition/approach/insertion controller to90. This is a concrete compliance/tracking transition. Two bounded tests keep allV31logic and change **only baseX/Ystiffness**: V32uses240(matching the transport setting), V33uses500(a value used by the toolkit'sSpeedDemonCartesianexample). Axialstiffness90, rotational50, damping, reference limits, forceguards, budgets and physicalgates stay unchanged. Familycode selects this forSC; SFP retains its original gains. The setting is explicitbaseXY, appropriate to the publicSCverticalportsetup; do not silently call it an arbitrary-port-frame stiffnessmatrix.

Budget: one exact failedfive-cardport1episode per setting, GPUs3/4, ROSdomains135/136. With untouchedlegacyGPU0this uses threeprojectGPUs. No parallelmodeltraining, newdataset or finals. Wait for eachcompletedtrial; no interruption. Require physical insertion, noharm and completep95<300ms before selecting a candidate. If neither helps, stop stiffnesssweeps and revisit the measuredplant response rather than enlargingdatasets. `lateral_reference_vs_tracking_diagnosis.json` preserves the quantitativefilter/plantseparation.

## Lateral tracking repair succeeds on the remaining five-card case

The bounded stiffness comparison is complete. Keeping axial stiffness at 90, the 240 lateral setting stops at **7.853 mm / 0.126 mm / 0.299°**, toolkit 0. The 500 lateral setting reaches **15.538 mm / 0.127 mm / 0.305°**, toolkit 75, without scored harm. All used stages have p95 below 300 ms: acquisition 165 ms, fresh pose 261 ms, close update 229 ms, servo 11 ms. Transport p99 is 484 ms and is reported separately. This is one difficult development case, not a reliability claim.

The combined repair uses the stronger visual orientation model on retained far images and fresh close images, a 0.1 mm lateral entry gate with 0.15 mm exit hysteresis, corrected public tracking-error initialization, and increased lateral stiffness. Axial stiffness, force limits, official physics and success thresholds remain unchanged. Higher outer-loop integral gain alone failed. The evidence points to physical lateral tracking, not only visual pose error. No privileged pose enters control.

Artifact root: `V33_lateral_stiffness_failed_case/` (the original prospective folder name is preserved despite the successful outcome). `explicit_trial_gate.json` passes; `successful_five_card_port1_rgb_timeline.jpg` was visually inspected. V32's failure is retained alongside it.

**Next frozen candidate V34:** add the previously verified shared JPEG conversion to V33. Both SFP and SC frozen acquisition estimators are bit-exact with the old conversion; `V34_codec_equivalence.json` records the SC proof. Run all four matched SC cases again, rather than counting the V33 pilot as a V34 success. Two isolated rootless workers use GPUs 3 and 4; legacy GPU 0 remains untouched. Stop launching cases on the first failed gate and let already active jobs finish. If all four pass physical and p95 timing gates, run three unchanged original public samples, stopping repetition on any failed physical or timing gate. Keep the original model qualification failure explicit throughout.

Command: `.pixi/envs/default/bin/python scripts/continue_SC_V34_validation.py`.
State and exact commands: `V34_SC_matched_coverage/state.json` and each case's `*.command.json` under the artifact root. No new training or dataset generation is queued. Reserved final scenes remain sealed.

## V34 matched SC validation: four of four pass

| Obstacle cards | SC port | Depth mm | Lateral mm | Orientation degrees | Physical + toolkit + p95 gate |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 15.541598 | 0.080300 | 0.345999 | Pass |
| 0 | 1 | 15.544000 | 0.060133 | 0.419412 | Pass |
| 5 | 0 | 15.502872 | 0.151061 | 0.952997 | Pass |
| 5 | 1 | 15.536609 | 0.044462 | 0.397358 | Pass |

All four use the complete V34 candidate, unchanged visual checkpoints and local backoff enabled. No guide or true pose enters control. The five-card / port-0 case has only 0.0029 mm depth margin and 0.953° orientation error; do not treat this small development grid as reliable all-settings success.

The unchanged original public sample is now running; three repetitions are planned, conditional on each physical and timing pass. No tuning occurs between those repetitions. Final scenes remain sealed.

## V34 full public sample and bounded remaining-failure diagnosis

The unchanged original three-trial sample completes at **170.572971/300**, with **2/3 strict physical and toolkit insertions**:

| Trial | Task | Depth mm | Lateral mm | Orientation degrees | Strict gate |
| --- | --- | --- | --- | --- | --- |
| 1 | SFP, requested NIC 0 | 46.769927 | 0.264494 | 0.146405 | Pass |
| 2 | SFP, requested NIC 1 | 21.178579 | 0.825439 | 3.636656 | Fail |
| 3 | SC, requested port 1 | 15.509972 | 0.140528 | 0.469136 | Pass |

All used-stage p95 values pass: transport 158.2 ms, fresh perception 220.7 ms, close update 252.1 ms, servo 10.5 ms. Transport p99 is **619.7 ms**, not below 300 ms; the requirement is p95. A postprocessing event-name bug initially reported no transport calls. It is corrected from the unchanged raw `proposal` events; the original report and coordinator state are preserved beside the corrected versions. The physical SFP failure independently prevents the second and third repetitions. Do not launch those repeats under the claim that the first passed.

The SC success includes a 2.405° held-plug orientation update at 25 mm. Four earlier matched V34 cases accept 1.949–2.803° updates. None of those four cases triggers a local retreat; their successes do not measure backtracking benefit. The close-up image sheets for all four were actually inspected, and `V34_visual_review_manifest.json` links them and their 20 fps videos. The complete sample's per-trial recordings and causal geometry are under `sample_V34_experimental/per_trial/`.

### Why the SFP failure is not a stiffness-only repair

The SFP controller reports nearly zero orientation error but arrives about 3.64° wrong and stalls at 21.18 mm. Oracle logs show sustained SFP tip/module contact with the target NIC opening, not a dominant cable snag. `SFP_second_trial_contact_diagnosis.json` lists pairs and scalar normal-impulse/2 ms point forces; these are not a net wrist force vector and must not be summed as one.

A raw image-time audit finds **3.533° held-plug orientation estimation error already at the far acquisition**, unchanged at the 25 mm hold and through contact. The actual held-plug roll is **26.524°**; the SFP model's entire original label corpus spans only **17.810–23.124°**. Its predicted roll is 23.223°. This is concrete out-of-distribution grasp evidence. The model was substantially trained (307 training episodes / 247 resets, 47 validation episodes / 39 resets, 19,800 updates to plateau), not merely stopped after a tiny pilot.

A frozen, no-training screen uses five recorded near observations in each of four SFP trial/run pairs. On the failed case, the same SFP model remains **3.562–3.583° wrong**; its far-to-near prediction change is only 0.164–0.220°. Therefore a consistency gate alone would not detect that stable bias. Three previously successful pairs retain low orientation errors. Screen commands, exact images and records are in `SFP_close_reestimate_screen/`; the causal raw-pose audit is `sample_V34_experimental/SFP_second_trial_grasp_error_audit.json`. Simulator geometry is offline analysis only.

An SFP lateral-stiffness-only candidate was drafted but **not run**, then removed once the pre-contact orientation gap was measured. Do not label it a trained, evaluated or failed controller. Current V34 stays frozen; no thresholds or collision rules were weakened.

### Next targeted repair, with a stopping rule

SC has now passed the requested four-case development check and the complete public sample's SC round. Broad reliability and the model-wide pose qualification remain open, especially given small SC depth margins. The next repair should address **SFP grasp orientation coverage**, then rerun the unchanged three-trial sample:

1. Audit existing raw SFP source grasps by complete reset and physical orientation; identify whether suitable broader examples already exist before collecting anything.
2. If absent, collect a bounded 32 perception-only public-sample-derived free-grasp resets (24 train / 8 validation), with declared broader orientation coverage rather than copying the failed evaluation geometry. First four require causal label and image QA. These are not BC or insertion experts.
3. Mix the original corpus with that extension and train to validation convergence. Require orientation p95 ≤1° and position p95 ≤0.4 mm on both original and new episode-grouped held-out domains, plus prediction-only image review and runtime parity. If the new domain fails after one distinct targeted model repair, stop routine collection and reconsider a geometric orientation representation.
4. Test the frozen repaired SFP estimator on fresh development starts, preserving unchanged SC control. Do not deploy the current biased model as a close-up fix or increase the 3° change limit merely to pass this case.
5. After component and live physical gates pass, run three unchanged public samples with no between-repeat tuning. Stop repetition on any failed physical or timing gate. Final scenes stay sealed. No RL/world/actor-Q training is scheduled by this diagnosis.

No additional capture or training job has been started for this next repair. The completed SC result and remaining SFP failure are reported separately.

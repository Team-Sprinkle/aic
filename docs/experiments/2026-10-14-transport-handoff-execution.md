# Autonomous transport and handoff execution

**Superseded reliability evidence, October 15:** The later frozen 140-rollout matrix did not reproduce autonomous full insertion. Primary SC: 23/50 visual approvals, including three wrong ports; only three actor starts, all shallow. Matched controls also 0/20. SFP was transport only and failed its broad diagnostic approach region. Preserve this earlier one-case 11 mm result as a pilot, not a reliability claim. [Full matrix and limits](2026-10-14-frozen-pipeline-coverage.md).


Started 2026-10-14 from the [execution prompt](../answers/2026-10-14-autonomous-transport-handoff-execution-prompt.md). Final scenes remain sealed. Original datasets and active jobs are preserved.

[Machine state](2026-10-14-transport-handoff-execution.json) records stages, branches and next actions. Bulk artifacts: `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1`.

## Preflight

Rootless Docker is available. No active transport train/evaluation jobs were found. Old Isaac GPU-0 PID 468750 is left untouched. Other users have GPU processes, especially on 1/2. Initial offline inference will use physical GPU 3. /data1 has 2.4 GB free; NAS has about 3.3 TB free. No reserved final scenes are accessed.

## Stage 1: component contract

Running frozen perception on recorded teacher endpoint and local-approach observations. Predictions use camera RGB, task request and measured robot state; target transforms are read only in the scoring pass. Insertion at the old transport endpoint is not yet validated.

### Endpoint replay result

Six reused SC teacher endpoints were accepted; same-image alternate requests were rejected 0/6. Opening error was 0.11–0.21 mm, orientation error 0.01–0.17°. The endpoint is viable for visual acquisition, so moving transport to the old closer gate is not necessary for that purpose. Local approach and insertion remain separate, unvalidated downstream work. Thirty total endpoint/local snapshots were scored; 20 later local snapshots were rejected because they left this particular standoff, although the requested opening was still estimated. Recorded native-size JPEGs are a proxy for lossless live RGB. [Exact results](/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/endpoint_contract_v1/summary.json).

## Stage 4: timestamp repair passed

Rebuilt future SC targets from original clocks and recorded absolute commands, with original FK-materialized joint targets where needed. A 150 ms maximum command age and strict transport/local boundary prevent speculative holds or cross-phase labels. Complete 12-slot SC chunks increased 3,783→5,689; all final-stage episodes now retain valid action supervision. All 61 stop positives remain independently available. Twelve windows cover recovered labels, real held targets and terminal masks, both ports. All valid slots passed age, ownership, finite-value and rotation checks. SFP labels remain frozen. The trainer must still be updated to consume these masks rather than filter partial horizons out.

## Stage 2: rule screen

Across 83 successful training episodes, board normalization reduces later waypoint p95 spread from 47–64 mm to 0–11 mm; subtracting the selected rail translation reduces late residuals essentially to zero. This is an oracle decomposition only. It shows why board-relative templates could help, but does not supply an observed board pose. Initial egress retains 21–22 mm residual due to start-dependent construction.

The maximal outer envelope corner (x −0.653 m, y 0.584 m, z 0.180 m) failed rigid planning at traversal. A bounded reach repair (x −0.630 m, y 0.550 m) passed all six selected training-only rigid scenes, including both ports and one/five-card counts. Cable behavior and Cartesian execution are unverified; rotated-board support remains absent. IK probes alone were not decisive: a direct exact-180° IK query failed even at a corner that OMPL subsequently reached with the measured wrist posture. Do not present that probe as a proof of geometric unreachability.

## Stage 5: first paired executor diagnostic

On llb_0005, direct replay and exact 12/6, 100 ms chunk replay reproduced the original transport/local boundary within 1.99/3.29 mm and 0.04/0.02°. Holding the final recorded target eight seconds shifted their measured endpoints 11.31/12.79 mm from the original dynamic endpoint; the two settled endpoints differ by 3.21 mm. This is stopping/settling evidence, separate from chunk-frame conversion. Remaining two pairs are running before actor training. Oracle replay is not autonomous policy success.

### Completed executor comparison

All three pairs (six Gazebo runs) pass the original phase-boundary 10 mm/3° diagnostic gate: position errors 1.99–4.05 mm, orientation 0.004–0.037°, no scored contact. The additional eight-second final-target hold gives 11–18 mm drift from the dynamic boundary. This does not pass stable handoff. Inference delay was not emulated, so this rules out gross anchor/executor failure but does not close the timing branch.

### Frozen visual-stop replay

Scanning causal camera triplets from all six reused fine70 actor traces yields a strict valid candidate on 2/6 runs, one proposal each, with no posthoc bad-pose accepts. No consecutive stable accepts were observed. All prediction files were frozen before scoring against teacher opening labels. This is not live stopping and does not justify promotion.

### Matched training is active

Both arms share initialization, 38 requested history ages, 12 × 100 ms predictions / six executed commands, architecture, split, optimizer and independent stop labels. Repair uses all valid slots; control retains the old complete-horizon filter. Both are selected against the same repaired held-out reference and orientation metrics are reported. The stop head adds about 148k parameters; trainable actor is 14.44M, plus the nominal 11.18M frozen encoder. The legacy runtime also loads extra modules, so these are model-component counts rather than complete process memory.

The first control stopped at a validation plateau after 3,000 updates while the repaired arm continued. A fixed 20,000-update comparison is being added before causal conclusions or live selection. All fits stay within the initial four GPU-hour allocations. The one-update smoke test is implementation validation, not a trained-policy experiment.

### Rule-controller failures and repairs

The first pilot stalled before completing egress with raw wrench 19.7–21.2 N and no scored contact. Its short lookahead was continually rebuilt from a biased measured pose, preventing the reference from reaching the actual waypoint. Version 2 reached the traversal corner but stalled 17.6 mm short, again without scored contact. Review caught an unintended reference reset during ordinary progress; version 3 removes it and resets only at measured stage transitions. These failures remain in the denominator and no rule controller is promoted.

A separate handoff-state-machine review found that rejected inspections could repeat holds indefinitely while proposal probability stayed high. The shared runtime now guarantees a fresh motion chunk before another inspection and resets the stop streak on rejection. Future learned comparisons will use this same repair in both arms. A single high proposal will request a visual inspection: the old three-decision streak spans about 2.7 s at current inference cadence while teacher stop labels span only 0.5 s. Fresh visual evidence, not repeated neural proposals, must authorize the actual handoff.

## Equal-update result and bounded diagnosis

Both frozen arms completed exactly 20,000 updates; this cap is not a claim of convergence. Actual-runtime fixed-noise teacher replay late executed p95 improved **11.11 → 6.50 mm**, but live stable handoff was **1/6 → 1/6** across the same six reused starts. Preserve both selected checkpoints and all twelve failures/successes. No learned actor is promoted.

Every scene in both arms had at least one requested-port estimate within 0.5 mm / 1 degree when scored afterward. Thus these runs do not establish wrong-port perception as the principal cause. Poor terminal posture and standoff capture dominate visual rejection. Repaired nearest-standoff distance improved in four of five failed scenes, but that is insufficient for stable handoff. See `bounded_failure_diagnosis.json` and marked images under the artifact root.

The fixed-corridor controller with persistent reference and kp240 passed **6/6 reused starts** under the unchanged 15 mm / 3 degree visual acquisition gate plus **2 seconds stationary hold**, speed ≤5 mm/s and angular speed ≤3 deg/s. A six-case fresh episode/reset-grouped evaluation is running with the same controller, template, gains and perception. Scope remains normal-yaw SC; rotated settings and downstream approach/insertion are not yet approved.

## Fresh rule-route gate passed

Identical frozen v5 controller/template/perception/gains passed **6/6 fresh reset groups**: 1200/1205 (one card, both ports), 1204/1209 (five cards, both ports), 1202 (three cards), 1203 (four cards). Group hashes do not overlap the 104 source episodes. Correct opening errors were 0.166–0.311 mm, tool posture 0.007–0.150°, measured speed 0.66–1.18 mm/s, no scored contact. Acquisition took 27.95–29.55 simulated seconds. Plug tips remain **154–156 mm before the opening**; these are explicitly not insertions. This supports selection of the normal-yaw SC rule prototype, not full published-range support or a reliability claim. See `rule_fresh_arrival_metrics_v6.json` and five-card `rule_common_llb_1204_v6_fresh/five_cameras_20fps_grid_active.mp4`. Video frame at 21 s was inspected: the tool and orange cable bypass the outside of the card row.

The independent learned docking repair uses two coherent predicted port estimates, max 80 mm /10° displacement from proposal, orientation before translation, 15 s per-proposal budget and unchanged stable approval. No simulator geometry selects its movement. This is a separate visual supervisor, not a residual neural adapter or a claim that BC action accuracy improved.

Terminal logging now retains the observation at accepted handoff after the hold. Older successful comparison traces must be scored with their `matched_handoff_accepted` event. Videos include that fresh timestamp to avoid clipping the held interval; raw traces remain immutable.

## Docking repair and downstream training

Docking v1 finished **3/6**, versus the frozen no-correction learned arm 1/6; this misses the ≥5/6 gate. Position commands rebuilt a 5 mm lookahead from measured TCP and stalled while orientation converged. Persistent-reference v2 preserves the same 80 mm /10° envelope and final gate; it is running. The corrected implementation includes an explicit raw-wrench guard inside the inspection loop. Normal controller force checks alone do not run while an inspector owns motion. V1 failures remain archived.

The original local insertion data was internally expressed in its older estimated frame; this is **not a demonstrated coordinate-transform bug**. However, reusing the new precise handoff pose would change its input distribution. Original SC held-out relative TCP pose errors were 2.12 mm /2.88° p95. The new derivative freezes camera-only port estimates at causal pre-local observations, then expresses the same four recorded full TCP targets in that static estimated frame. Version 1 mistakenly equated pose observability with transport readiness and retained only 79 episodes. Its excluded reports were geometrically coherent but still approaching standoff. Version 2 reuses the immutable predictions and permits those pose-only measurements for **training**, without changing autonomous handoff approval. It preserves **104 SC episodes, 83 train/21 validation**, 26,455 SC local rows, plus 111,743 unchanged SFP rows from 234/31 episodes. Held-out locked port position/orientation p95 is 0.235 mm /0.090°. All full targets reconstruct the original base-frame command to numerical precision; every lock precedes its local observation and reset groups stay disjoint. Three annotated example triplets were visually inspected; crosses fall on the requested blue aperture.

A frozen warm-representation versus ImageNet-scratch insertion comparison is running sequentially on GPU5: same shared two-family actor, same data/splits, 6,000-update/four-GPU-hour cap, minimum 2,000 updates, 15 stale 100-update evaluations before plateau stopping. Offline SC gate: near lateral target p95 ≤1 mm and near orientation p95 ≤2°, then ≥5/6 autonomous full-start insertions with no wrong approval/harmful blockage. A cap is not saturation. No backoff, teacher guidance or RL is attached.

The frozen current perception path produced **0/9 usable estimates** at three fixed observation times on three existing rotated-board SC public-anchor development traces. Main causes: selected view missing, requested identity rejection, or cross-view disagreement. This is replay support evidence, not a new rule-route test. The public eval configuration includes yaw −1.8 and SC rails −60..+55 mm; the normal-yaw rule success must not be extrapolated to those settings. Images from rejected over-card teacher routes may be reused for perception labels only; their actions remain excluded from SC transport BC.

### Persistent-reference docking result

Frozen repaired actor plus visual docking v2 passed **6/6 reused stable handoffs**, with no scored contact: baseline no correction 1/6, v1 3/6, v2 6/6. Acquisition took 50.45–61.25 simulated seconds, versus roughly 28–30 s for the rule path. Actor inference p95 ranged **303–359 ms**, above the restored 300 ms target; latency is still temporarily relaxed for this correctness diagnostic. This is a visual-controller improvement, **not** improved learned action prediction. Fresh six are running with the unchanged v2 controller.

A source-only grasp audit also finds that SC tip position relative to TCP varies **2.01 mm p95** across the 104 episode starts, and relative orientation varies **2.07° p95**. Thus an accurate port estimate plus a single nominal plug offset cannot guarantee the 0.5 mm corridor. The BC actor may infer this from its images; it is not yet proven that it fails to do so. An annotated label-only image was inspected: the tip label falls on the visible white plug nose in all three views. Native grasp/tip features are a conditional ablation if the actual insertion comparison identifies this deficit. No tip-prediction success is claimed.

## Broad-support failures and insertion offline gate

Fresh docking v3 passed 6/6, matching the frozen rule route on the same normal-yaw development starts. Rule remains the simpler/faster SC transport prototype. Three frozen broader cells failed acquisition: SC rail0 −60 mm, SC rail1 +55 mm (all five NIC rails +23.4 mm), and a perturbed public yaw −1.8 scene. All reached the nominal final waypoint within 4–6 mm and 0.01° but failed requested identity/cross-view checks. The raw `stage_no_measured_progress` reason is misleading here: measured waypoint arrival succeeded, followed by failed visual reacquisition. Do not classify these as cable snag or successful broad support. Inspect whether the requested opening is absent from one view, incorrectly located, or incorrectly rejected by identity before collecting/training.

Warm local fit plateaued at 5900 updates, selected 4400: near lateral p95 1.139 mm, orientation 1.929°. Scratch plateaued at 5900: its selected 4400 has lateral 0.948 mm and orientation 2.326°; its last checkpoint is better under the declared joint gate (1.025 mm /1.933°), but still fails. Neither checkpoint passes the conjunction, so no promotion. A **one-reset-per-arm live failure diagnostic** uses warm best and scratch last, the same approved rule prefix and camera-only static port frame, no guidance/backoff/RL; its result cannot be presented as a six-case gate pass.

Tip timestamp audit found that original first-local tip geometry uses past TFs roughly tens of ms old. Interpolation changes tip position labels by **0.722 mm median /1.452 mm p95**, while orientation shifts are small (0.006°/0.022°). Corrected across-episode grasp variation remains **1.969 mm p95 /2.066°**, so it was not merely timing noise. Original tip labels are inadequate as submillimeter moving-grasp supervision; validate interpolation against original high-rate bags before any tip model. **The current repaired insertion action dataset uses recorded base TCP targets and a static opening estimate, not these stale tip labels**, so this finding does not invalidate its action-frame reconstruction.

## Critical correction: insertion events and completion protocol

**Supersedes the earlier interpretation that the two score-zero runs had no insertion event.** Engine logs show requested `sc_port_0/sc_port_base` contact completed at ~64–65 s in **both** runs. CablePlugin transitions to `COMPLETED` and makes the end effector static. Subsequent lack of measured motion is therefore not evidence of policy/controller stalling after that transition. The inherited diagnostic local runtime always returns `False`; the engine does not set task-end time on that path, and the scorer returns zero before evaluating the recorded insertion event. Original scores stay archived. Official event success, policy attempt completion, and the strict ≤0.5 mm geometric gate must be reported separately. No deployable policy may subscribe to `/scoring/insertion_event` under challenge rules. The allowed repair is ordinary finite-attempt completion reporting, with the independent scorer determining whether insertion actually occurred; do not detect simulator freezing as a deployment success shortcut.

The two runs still miss the declared geometric gate: terminal tip lateral 2.36 /2.72 mm, depth about 1.97 /1.91 mm relative to the calibrated entrance. Oracle point normal force against the port housing peaked 2.28 /7.39 N. Those values describe local contact at the simulator latch, not cable snag. Full nominal SC entrance-to-reference offset is 15.64 mm. Whether the contact latch is premature relative to physical seating is being audited; do not patch official physics to make evaluation easier.

The native grasp direct-regression probe passed neither precision threshold (crop 1.45 mm /1.25° p95 versus full image 2.11 mm /1.25°, constant 2.27 mm /2.20°). The dense-keypoint geometry repair also plateaued short of the gate (1.44 mm /2.32° p95; translation median 0.32 mm). No actor was trained on these uncertain grasp predictions. Next audit supervision and simulator link consistency before another encoder/optimization sweep.

Primary sources: [official SC contact macro](https://github.com/intrinsic-dev/aic/blob/main/aic_assets/models/SC%20Port/sc_port_macro.xacro), [official scorer](https://github.com/intrinsic-dev/aic/blob/main/aic_scoring/src/ScoringTier2.cc), [challenge interface rules](https://github.com/intrinsic-dev/aic/blob/main/docs/challenge_rules.md). Local pinned engine log/binary behavior is the direct experimental evidence.

### Completed grasp context comparison and excluded joint-drift hypothesis

The same 1248 samples from 104 episodes (83 train / 21 held out) were used for state-only and native-crop-plus-state probes. State-only translation p95 was 2.117 mm and orientation p95 1.892°. Crop plus state improved these to 1.418 mm / 1.198°, but still fails the predeclared 0.4 mm / 1° gate. Both stopped at 2000 updates after validation plateau. Teacher command-error state fields 13–18 were masked. No insertion actor consumes these predictions. Artifacts: `sc_grasp_context_probe_v1/` and its exact runner `train_grasp_context_comparison.py` under the execution artifact root.

A separate six-episode raw-bag audit found the visual plug/scoring-tip fixed joint agrees with CAD to about 0.0001 mm; that drift hypothesis is ruled out. Timestamp-interpolated tip labels agree with 2 ms raw TF to 0.0401 mm in six sampled checks. Expert last-row lateral error is ≤0.5 mm in 103/104 sources; no wholesale expert exclusion is justified.

### Completion protocol controlled repair (running)

`RunLockedPortLocal` now acknowledges only a normal bounded attempt. Force stops and invalid handoffs remain failures. Acknowledgement does not assert insertion; the independent scorer uses its recorded event. No scoring subscriptions or static-gripper detection were added. The same warm and scratch frozen checkpoints, one-card reset, commands, gains and budgets are rerunning as `local_integrated_{warm,scratch}_llb_1200_protocol_v2`. The earlier zero scores remain immutable. Commands and comparison plan are saved in `run_locked_local_protocol_v2.py` and the generated protocol plan. Physical geometry remains a separate, unmet gate.

### Prospective two-view repositioning screen

Only the two failed normal-yaw rail-limit cells are selected; rotated-board displacement remains outside this branch's bound. `RunTwoViewGuidedRuleTransport` uses existing frozen requested-port predictions and original 160 px refinement. Guidance permits two views with identity probabilities max ≥0.99 / min ≥0.25, refinement shift ≤50 px, reprojection ≤5 px, training-height prior 30±10 mm, and destination within 140 mm of the nominal training waypoint. Two consecutive destinations must agree within 1 mm. This **cannot approve handoff**: the unchanged three-view requested-identity, 2 px reprojection, orientation, standoff and 2 s stationary approval gates still decide. Budget: two reused diagnostic resets, no training or collection expansion, continuation requires 2/2 strict correct acquisitions and no harmful/scored contact. Runner and frozen plan: `run_rule_development_v8_two_view_guidance.py`.

### Newly identified local executor difference

Recorded source commands carry translation stiffness 90, rotation stiffness 50, and damping [50,50,50,20,20,20]. The integrated local pilots inherited the transport repair's translation stiffness 240. Wrench-feedback gains and Cartesian position mode match the official helper. A prospective local-only stiffness parity test is justified: keep transport at 240; return local execution to the recorded 90. This remains a hypothesis until measured comparison; do not attribute all failures to it. No retraining is required for this test.

### Completion repair result and causal feedback test

Both bounded local attempts now return the normal action acknowledgement. Scratch received Tier3=75 with requested-port event; warm received a partial13.386 and no event, stalling near−110mm TCP in its predicted port frame. Models and reset configuration were unchanged, but initial measured handoff poses differed by millimetres between runs; the return change happens after motion, so this is **reporting repair, not evidence that the policy improved**. Original scores/events remain archived. Strict physical geometry is scored separately.

The warm stalled observations replayed the recorded first target to numerical precision. Holding images, task, current pose and all other state fixed, substituting training-mean normalized command-error fields13–18 advanced the target6–8mm; physical-zero error did not. This demonstrates input dependence, not proof that runtime masking alone succeeds. A bounded scratch fit masks those fields consistently in both training and runtime. Same full83/21SC split, unchanged234/31SFP local split, 6000update/4h cap, minimum3000,20 stale100-update checks. Checkpoint selection now uses the declared2° orientation denominator (old selection used3°); this selection repair is explicitly recorded and means any improvement cannot be attributed solely to the input mask. Offline gate remains1mm/2°p95; no RL. Artifacts: `local_feedback_dependency_audit.json`, `local_feedback_repair_frozen_plan_v2.json`, `local_locked_actor_masked_feedback_v2/`.

### Two-view guidance result and causal crop-tracking repair

The two-view repositioning screen finished **0/2 strict acquisitions**. Rail0 moved toward its estimated target but ended with requested identity probabilities [0.234,0.746,0.887]. Rail1 successfully restored identity to essentially1 in all views, but current coarse crops jumped55/63px and the refiner rejected them. These traces show perception rejection after motion, rather than evidence of a cable snag. A frozen-input pose-normalization sensitivity check did not restore the rail0 requested identity; simple state masking is not justified. The previous original-crop identity alternative was already rejected on independent Oct3 evidence (25 absent all-view false accepts), so it is not silently substituted.

Prospective targeted repair: retain the coherent earlier two-camera NN landmark; project it with current measured TCP and fixed calibration to select native-image crops. This supplies **crop centres only**. Existing locator flags, requested identity, native refinement, three-view2px consistency, board orientation and stable acquisition gates stay unchanged. No true target geometry or scene config enters the controller. Budget: the same two diagnostic rail configurations, stop expansion unless2/2 pass. Artifacts/runner: `run_rule_development_v9_tracked_crop.py`.

### Tracked crops and correction-envelope repair: fresh6/6 acquisition

V9 preserved the visual landmark and placed native crops correctly, but finished0/2. Port0stillfailed identity. Port1nowpassedidentity/refinement and reported correct3Dpose, yet remained nominallyoriented: parentrule's80mm nominal correction envelope discarded orientation updates after the separatelyapproved140mm two-viewmove. V10retains the140mm nominalposition limit, requires strictupdatedpositionwithin5mm of the trackedpoint and rotationwithin10° of nominal, and leaves finalacceptance unchanged. The failedport1cell passed1/1.

Frozen V11thenpassed**6/6new grouped acquisitionstarts**: both SCports, one/fivecards and two port1rail-maximumcases. Posthoc selectedopeningerrors0.164–0.285mm, orientation0.006–1.368°, no scoredcontact; no wrongaccepted opening in these six. Acquisitions took27.6–32.1simsec. Port0railminimum and rotatedboard remain knownunsupported; these were not counted as successes or hidden. One1.37° orientationestimate needs explicit downstream allowance/repair. This is transport/acquisition feasibility, not insertion or general reliability. Visualacquisitionfullpredict-call p95=126–162ms (includesnativeimagehandling, locator, identity, refiner, geometry andboardrails); p99=165–194ms. Wholephasecallp95=106–120ms. Integratedinsertionpipeline latency stillrequirescorrectprofiling.

Artifacts: `rule_tracked_fresh_arrival_metrics_v11.json`, `cross_component_summary_v1.json`, `tracked_crop_fresh_scenes_v1/manifest.json`; preparation/execution/scoring scripts saved alongside. Reviewedfourvideo intervals forfive-cardrail-maxcase1405: outside-row traversal and finalholdshowed no sustainedsnag/prohibitedcontact. Video: `rule_common_llb_1405_v11_fresh/five_cameras_20fps_grid_active.mp4`.

### Local gains and masked-feedback conclusions

Gain90warm/scratchpilotsbothreceivedofficial75. Terminaltipdepths2.142/2.018mm, laterals0.630/2.068mm and orientationerrors0.616/1.926° relativeto the successfulteacher mating convention. Neitherpasses0.5mm/14mm physicalgate. This remains a one-reset-per-armdiagnosis, not a6-episode reliabilitycomparison. Officialevent completionfreezes motion and can occur at partialcontact. The warmgain90video is saved in `local_integrated_warm_llb_1200_teacher_gains_v3/five_cameras_20fps_grid_active.mp4`.

The masked-feedbackBCvariant reached its declaredplateauat3000updates, best800. Held-outSCnear-lateralp95=3.705mm andorientation2.434° fail the1mm/2° conjunction. No liveevaluation of this failedvariantwasstarted. Tracking-error dependence is demonstrated, but simple removal harmed held-outprecision; do notpromote masking from the replay intervention alone.

### Native grasp precision and label-provenance follow-up

The direct3Dnative192probe adaptedImageNetlayer2 with denser8×8spatial features under unchanged83/21episodesplit. Best200, plateau2200updates, translationp95=1.268mm/orientation1.595°. It improves some errors but stillfails0.4mm/1°. Causal past3/6frameaveraging didnotresolve thetail. Oraclewithin-episode constanttranslationresidual itself reaches1.14mmp95 (label-onlydiagnostic, no futuremean deployment). Sixworstlate-heldoutlabelwindows match original2ms TF to<0.01mm, ruling out sparseinterpolation as their error.

Important scopegap: allthesegrasp probes sampled only approachframes with plugtip morethan30mm beforetheopening. They do not establish failure onthe actualclose-insertion images. Nextassess existing−25..−2mmframes, with the sameepisodesplit and independentno-labelcropselection, and compare currentfrozenpredictions before anytargetednear fit. No largecollection is justified.

A one-resetphysicalcamera/TCPaudit was necessary because publicbagsomit movingrobotTF. The standardPoseV→TFMessagebridge drops SceneBroadcastername/time metadata, so thatfirstdiagnosticcannotreliablyjoinphysicalentities. RawGazeboPoseVcapture in the secondunchangedpilot preserves names/IDs/simstamps. Nooracleposes enter eitherpolicy; filterout allpost-terminal/resetobservations. The exactscripts, invalidfirstcapture and rawsecondcapture are retained.

### Timestamp root cause in grasp-probe labels (repair running)

PhysicalGazebooptical/gripperposes were captured with entitynames andsimstamps and joined onlybeforeterminal/reset. Ononecurrentlivepilot, FK/camera calibration agrees to about0.0013mmp95 duringlocalmotion and0.17mmp95 duringtransport. This doesnot establish everyhistoricalscene but gives no evidencefor a millimetre-scale staticcamera/FKmismatch inthat run.

Code audit found the actualgrasp-probe discrepancy: `materialize_moveit_bc_segments.py` stores eachrow's `sim_time` as **command time**, while `state` is the precedingcontroller sample at the synchronized **image observation time**. The newprobe's earlier labelrepair interpolatedTIPTFatcommand time and subtractedtheolderobservationTCP. Thusitslabelsaudit validated theTFinterpolation numerically but didnot validate the cross-clockjoin. Neither originalimage/controlstate nor commandtime shouldbe erased.

`repair_sc_grasp_observation_timestamps.py` now creates a derivative joiningTIPTFat the **recorded controller timestamp**, explicitlyverifying thatcontrollerposematches therowstate. It retains commandtime andimagetime separately. Allpreviousprobe results remain archived; until the auditfinishes they cannot establish a representation-accuracy ceiling. Next: matchednativeprobe withcorrectedlabels and unchangedimages/split/architecture/seed, then onlyif the0.4mm/1° gate passes test lockedgrasp predictions onautonomoushandoff images and static-grasp persistence before actorconditioning. This usesexistingdata, not newexpertcollection.

The completed timestampaudit quantifies thedefect: command-minus-controllerp95=46ms (maximum50ms), labeltranslationchange p95=1.931mm (maximum3.134mm), correctedwithin-episodegrasptranslationp95=0.179mm. Thereforethe priorpose-probe failureseveral mmfrom fine geometry cannotbeused toconclude insufficientsensorycapacity. Correctednativefit uses identicalimages, groupedpartition, architecture, optimizer/seed andbudget.

Futurelocalruntimeprofiling now includes CUDAresulttransfer/completion before recording latency. Previouslocaltimings may underestimate GPUcompletion; frozenhistoricaltimings are notretroactivelychanged. The fullSCrule/visualphasecalltimings above already synchronize via CPUreturnedpredictions.

### Corrected grasp comparison: position precise, orientation still open

Correct-labelnativepose regressor:0.565mm/1.299°p95. Correct-labelfivekeypointtriangulation:0.245mm/1.376°p95, after4400-updateplateau. Knowncameraidentitydidnotclearlyimproveit(0.251mm/1.349°,6000-updatecap,best5200; notprovensaturation). JointSE3pixelreprojectionoptimization didnotresolveorientation(1.503°).

Matchedjointgeometry-losscomparison initializedbotharms from thesame camera-conditionedcheckpoint, usedidenticalnativepixels/splits/batches and4000updates perarm, then an equal4000-updatecontinuation from eacharm'slastweights. AdamWoptimizer wasreset inbothcontinuationarms, with a commonnewseed; exactcommands/hashes/plansarearchived. Pixel-onlycontrol finishes0.244mm/1.346°. Joint3Dposition/rotationloss improves to0.226mm/1.160° (bestfirstphaseangular1.148°); bothfailthe1° conjunction. The secondphaseplateau rule triggered; nofurtherblindextension is justified. Noactorconsumes thesegraspmodels yet. Accuratepositionmayjustify a **separatelydeclared translation-onlyconditioningrole**, but it must notbe described as a passedfullposegate.

### Clipped board-yaw agreement defect and bounded repair

Frozen rail-maximumcase1405had1.37°poseorientationerror. Inspectionof theactualobservationalHoughsearch showsallthree estimates saturatedatthe same±0.07radwindowedge: boundaryweights0.783/0.914/0.670, creatingfalsezeroMAD/zero-viewspread. Wider±0.21radsearch and12°image-lineprefilterrecover twoagreeingunsaturatedviews; the third isexcludedby its1.32°MAD. Normalcontrol1400 remainsunchanged. Labels werenotused for the search.

`RunGuardedYawTrackedRuleTransport` uses thatboundedsearch and rejectsboundaryweight>0.5 as validorientationevidence. Existingrequestedidentity, three-cameraopeningrefinement, standoff andstationary gates stayfixed. TworeusedGazebocontrols1400/1405 reachedacquisition; posthocpose/force scoring ispendingbeforethe new0.5mm/0.5°screen canpass. PreserveV11'sresultsunder its oldcontract; do notrewrite itsorientationcertification. Artifacts: `yaw_search_boundary_audit.json`, `rule_development_progress_v12_guarded_yaw.json`.

## Position-only role revision and matched insertion BC

The full grasp-orientation probe remains above its 1° gate. Its **position estimate** is independently useful: the frozen model achieved 0.162 mm p95 (0.163 mm maximum) on 18 preselected observations from six autonomous handoff episodes, with a maximum 0.042 mm spread within each lock. Actual transforms were read only afterward for scoring, always before terminal/reset. Crop centres came from training data; no label chose a crop. These are a small feasibility screen, not a population reliability claim.

The versioned position-only contract feeds the actor the plug-tip **position** relative to the estimated port. The actor retains responsibility for orientation; no estimated grasp rotation enters it. This is a deliberate role revision, **not** a passed full-pose gate. `SCNativeGraspObservation` exports only active model weights, removes unused ResNet layer4/fc, and uses three original 192 px crops with camera identity and the training JPEG codec.

The new dataset preserves every recorded full TCP action and episode split. Three camera-based estimates from the **preceding transport endpoint** form a body-tip lock; their image timestamps cannot exceed the current local observation. Frame backconversion is checked for every SC row. SFP rows retain their original actions, with tip features masked out. Dataset: `local_translation_conditioned_dataset_v3/`. It contains 83/21 SC and 234/31 SFP train/validation episodes.

Matched BC arms use the **same actor architecture, warm local weights, data, split, optimizer and 6000-update budget**. The control supplies a training-episode-mean body tip transformed with current TCP/estimated port; its added feature head is active, controlling for capacity and duplicated TCP information. Treatment supplies the per-episode RGB-estimated body tip. Both predict complete future TCP targets in the estimated port frame. There is no residual action adapter. The offline gate stays 1 mm near-lateral / 2° orientation p95; physical promotion still requires 5/6 complete SC insertions, at least14 mm depth and no more than0.5 mm lateral error, with orientation and force reported. No backoff or RL is enabled.

Commands and hashes: `run_matched_tip_conditioning_training_v1.py`, `matched_tip_actor_frozen_plan_v1.json`. Both best and last weights are retained. The two arms run sequentially on GPU4; active Gazebo diagnostics use GPU3 and legacy GPU0 is untouched.

## Board-yaw validation and causal memory

The guarded wider-search controller passed both reused controls: opening errors0.200/0.185 mm and orientation errors0.140/0.017°. The rail-limit case improved from1.37° to0.017° without labels in the search. A frozen six-new-group screen reached5/6 acquisitions. The last case had useful orientation estimates earlier but later lost sufficient board-rail edges.

A bounded causal-memory repair retains two coherent, observation-derived board poses. Current requested identity and three-camera opening refinement must still pass; the stored opening must agree with the current one. Pose keyframes are spaced by20 mm translation or3° rotation, up to10, and are reset per action/target. The memory does not accumulate duplicate stationary frames. This uses the known static fixture within an episode, not scene geometry. One reused failed case is running before any fresh expansion.

A separate reset audit found the generic executor did not dispatch subclass `reset_for_task` when starting an action. It now does, preventing route stages and visual memories from leaking into another episode. This does not change first-episode frozen results.

### Causal-memory evidence limitation

The repeated failed1505 reset reached acquisition, but **did not use the history fallback**. Replaying the frozen failed trace from saved JPEG images accepted22/22 observations both with and without memory. Saved JPEG and live raw RGB can change this edge detector's output; this is a concrete preprocessing sensitivity, not proof of memory benefit. Do not attribute the repeated success to the memory. The optional fallback retains current identity/refinement checks, and its state is reset per action, but it is not promoted as an improvement. A future paired raw/JPEG shadow test is justified if this failure persists.

The matched insertion actor control has passed its offline conjunction (selected4600; near-lateral0.822mm, orientation1.720°p95) after its exact6000 updates. The camera-tip arm is still training. A waiting pilot runner checks **both** completed summaries before freezing selected checkpoints and starting one matched simple reset per arm. It does not change training in response to live outcomes. No physics, scenes or model weights are changed while those runs are active.


## Matched grasp-feature result and next action representation

Both 6,000-update arms finished. The mean-grasp control selected update 4,600 with near-port lateral action error 0.822 mm p95 and orientation error 1.720° p95. The camera-grasp treatment selected update 4,800 with 1.084 mm and 1.893°. The treatment failed the declared 1 mm gate. The waiting rollout runner exited without starting either live arm. These are capped matched fits, not a claim that both converged.

The position estimate itself passes a separate near-port audit (18 snapshots from six reused episodes, 0.192 mm p95, 0.220 mm maximum). Adding a feature is insufficient evidence that the actor uses it. The next bounded hypothesis changes the action reference: predict the complete SC plug-tip translation and TCP orientation in the RGB-estimated port frame, then subtract the camera-estimated body tip offset rotated by the **future target TCP orientation**. This is a complete target, not a residual correction. SFP retains its existing TCP action reference.

Recorded command labels are converted using simulator grasp translation only during dataset construction. Those labels are not autonomous inputs. Original TCP command labels are also retained solely for the command loss and scoring. At inference, only the camera estimate supplies the inverse conversion. Round-trip label error is below 3e-17 m. Across near-port episode medians, the action translation spread falls from 2.100/2.520 mm to 2.033/2.342 mm: modest evidence, not proof of success.

Both comparison arms will use the same estimated grasp input, source episodes, head reset, representation initialization, sampler and budget. Target normalization is rebuilt from each training action representation; inherited old target normalizers are not retained. The primary offline gate scores **decoded TCP commands against recorded TCP commands**, so a change of reference cannot produce an artificial metric improvement. A matched live one-reset pilot is conditional on that gate. No new routine trajectory collection, reward training or final-scene access is authorized by this result.


## Expert insertion depth rechecked at observation timestamps

A separate audit now uses measured plug TF interpolated at the identical controller timestamp represented by each episode's last admitted BC observation. It does not infer plug position from the original cached grasp offset. All 104 episode/state joins match explicitly. Depth p10/median/p90 is 12.721/15.188/15.572 mm; lateral p10/median/p90 is 0.097/0.223/0.417 mm. **77/104** already satisfy depth >=14 mm and lateral <=0.5 mm in this last admitted observation. The old cached calculation counted 75; it remains archived. Observation-label position difference is 0.013 mm median and 0.306 mm p95. This confirms useful deep teacher data but does not certify the remaining 27 terminal outcomes, because the last BC anchor can precede the actual terminal event. It does not change action labels, actor inputs or the frozen comparison. Artifact: `sc_expert_same_clock_terminal_audit_v2.json` in the NAS execution root.


## Complete plug-tip action comparison finished

Both matched arms completed 6,000 updates, selected update 4,100 and ended with 19 stale evaluations. These are capped fits, not certified convergence. Decoded TCP near-port lateral p95 / TCP orientation p95: **TCP-target control 1.010 mm / 2.072°; plug-tip-target treatment 0.629 mm / 1.733°**. This is a clear offline improvement (~38% lateral), not autonomous success. SFP action semantics are unchanged and its local performance remains unapproved.

The original paired-pilot gate failed because the control missed 1 mm/2°. No original paired pilot ran. A separately frozen v2 diagnostic runs only the independently eligible treatment on one reused full-start normal-board one-card SC reset, llb_1400. It uses RGB port/grasp estimates, recorded local gains and complete targets, with no guide, backoff or RL. It does not relabel the failed paired gate. The physical depth >=14 mm / lateral <=0.5 mm gate is unchanged.

## Controller diagnosis: commands versus achieved positions

Three privileged llb_1400 controls keep the transport handoff, gains and physical gate. Plain geometric commands left ~7–8 mm lateral tracking error before the opening and reached ~0.33 mm depth. CheatCode lateral error accumulation (gain 0.15, windup bound 0.05 m) reduced terminal lateral error to 0.025 mm, but 1 mm/s axial motion reached 13.328 mm and failed the 14 mm gate. A separately declared 2 mm/s control reached **15.571 mm depth, 0.072 mm lateral and 0.012° orientation error**. Integral controls have official insertion events and no scored prohibited contact/excessive force.

A desired geometric pose differs from the command needed to achieve it with this compliant controller. CheatCode teacher commands include accumulated feedback. Raw wrist wrench has a large baseline and is not the scored insertion-force label. The rate result is consistent with the existing contact-confirmation delay and subsequent gripper freeze; the slower result remains a failed physical gate. No physics, contact events or rewards were changed. These controls consume true simulator geometry for diagnosis, are not deployable policies and cannot pass the autonomous supervised gate.

Artifacts in the NAS execution root: `point_target_comparison_summary_v1.json`, `point_target_treatment_pilot_frozen_plan_v2.json`, `oracle_insertion_controller_comparison_v3.json`, all frozen plans and raw rollout folders. `oracle_rate2_critical_video_review.jpg` was inspected at four intervals (approach, alignment, entry, seated plug), with five camera views. The orange cable's free end stays visible; the images do not suggest a cable snag.


## Live point-target failure and bounded continuation branch

The eligible treatment's reused llb_1400 pilot failed. Official tier 3 = 7.946; terminal true tip axial = -130.478 mm, lateral = 7.999 mm and orientation error = 3.213°. It did not approach the opening. This is a local approach actor failure after successful rule transport and handoff, not a transport-rule failure. Actor inputs remain RGB, measured robot state and public task code; no live simulator geometry, guide, exploration or backoff. Checkpoint and original results are preserved.

An observational input audit disproves the simple marginal lateral-gap hypothesis: source TRAIN approach lateral median/p95/p99 is 6.62/14.50/17.41 mm, while this live trace is 7.17/9.17 mm median/p95. Joint context, force, action history and orientation may still differ. Posthoc scoring at three trace snapshots finds port position error 0.231 mm and camera-body-tip error 0.074/0.164/0.145 mm. No checked gross port/grasp position bug explains the stall. Intermediate past-TF comparisons have a stated sampling uncertainty.

Three bounded causal continuations ran the frozen actor for 12 local simulation seconds, then a privileged teacher from the **actual actor state**. Actor prefix actions cannot enter expert BC. Version 1 achieved 1/3 physical successes: llb_1400/1401 froze at 2.140/3.172 mm; llb_1402 reached 15.570 mm and 0.214 mm lateral. Version 2 adds shared-time true gripper/plug calibration, 12 mm lateral-compensation headroom and a measured 0.1 mm centering gate. It again achieved only 1/3: llb_1401 reached 15.084 mm / 0.287 mm lateral; llb_1400/1402 froze at 10.486/10.476 mm despite good final centering. These results fail the declared >=2/3 gate. No corrective-data fit or routine collection was started.

The recorded TRAIN commands are deeper than nominal achieved seating: median per-episode maximum point command = 18.100 mm (p10/p90 15.573/20.847 mm), while the diagnostic commanded at most 15.640 mm. Version 3 prospectively tests **only** that command cap on paired llb_1400; physical depth >=14 mm and lateral <=0.5 mm remain unchanged. Stop this hypothesis if it still fails. A contact sample in v2 names the actual SC plug housing and SC port middle collider; it does not establish a cable snag. Inspect event timing and critical views before making a mechanical-cause claim.

Correction to the earlier expert-label description: the materializer already stores per-observation plug TF, not merely one fixed initial grasp offset. The newer depth audit improves the observation/controller timestamp and interpolation join. Its 77/104 deep observations replace the old approximate count of 75 for this audit; no original commands or datasets were modified.


## Qualified continuation data and matched repair fit

The recorded-command cap screen passed **2/3** full physical insertions; the port-1 failure at 10.487 mm remains preserved. For the small derivative, reuse three independently qualified successes: port-0 one/five-card v3 runs and the earlier port-1 v2 run. This is **not 3/3 from a single controller version**. The derivative contains **1,765 rows: 1,184 from two training episodes and 581 from one held-out episode**, in addition to the original 83 SC training / 21 validation and 234 SFP training / 31 validation episodes.

Ownership is matched by exact logged teacher target matrices, not timestamp proximity alone. The runtime's full 32D observation matches the recovered RGB triplet and robot-state samples for every retained row. Delayed, ambiguous or inconsistent observations are excluded, as are actor-prefix commands and anchors/horizons after physical success. The first build failed on a teacher command delayed 354 ms; the repaired derivative matches ownership first, then excludes that delayed anchor and affected horizons. Original and failed derivatives remain preserved. A byte-identical v4 copy is not additional data; v3 is selected. Twelve three-camera windows were inspected in `causal_dataset_12_windows_review.jpg` with exact metadata in its JSON companion.

The matched fit uses the same point-target architecture, initial checkpoint, old normalization, held-out episodes and 6,000-update budget. The control trains on old data; the treatment adds the two qualified training continuations with an episode sampling multiplier of 8. Both see the same new validation episode. No new model is trained on only two SC episodes. Primary selection balances old/new near-port decoded-command and orientation errors plus all-phase command error on the held-out continuation. Exact gates and commands are frozen in `causal_local_matched_fit_frozen_plan_v1.json`.

The independent timing audit matched every autonomous command: input-to-command header age median/p95/max = **58/94/116 ms**, with none beyond the 200 ms action horizon. The teacher had one 354 ms outlier, excluded above. This does not support observation buffering/QoS as the demonstrated cause of this stall. The BF16 output-resolution audit finds an axial step of about 0.338 mm at -130 mm, but lateral steps near zero are only 0.0043/0.0022 mm; no live causal precision benefit is established. These audits do not justify another routine model sweep.

A runtime contract check now rejects point-reference checkpoints in controllers that do not perform the matching point-to-TCP conversion. The locked-port controller explicitly declares support and logs both raw point targets and decoded TCP targets. Model weights, scene physics, target masks and physical success gates remain unchanged by this check.


## Matched corrective-data fit finished; live comparison running

Both arms completed 6,000 updates. Control selected update 500; correction selected 5,600. The control's held-out continuation near-port error is 0.900 mm p95, orientation 1.051°, all-phase first command 3.380 mm. Correction: **0.420 mm / 0.555° / 2.906 mm**. Original-source near-port error is 0.812 mm control versus 0.686 mm correction. The correction arm passes its predeclared conjunction. Its late selected checkpoint means the capped fit is not certified saturated; the live comparison takes priority over another extension.

The paired diagnostic is frozen on reused training reset llb_1400, with the same model architecture, rule acquisition, RGB port/body lock and controller settings. Control is an unqualified reference, not a promoted model. The test has no actor geometry, guide, exploration or backoff. Physical success remains >=14 mm depth, <=0.5 mm lateral, explicit orientation and acceptable force/contact. Six fresh starts are conditional on that pilot. Commands and checkpoints are recorded in `causal_actor_paired_pilot_frozen_plan_v1.json`.


## Corrective-data live comparison failed

Both full-start reused llb_1400 tests failed insertion. Control terminal tip: -141.361 mm axial, 10.177 mm lateral, 2.175° orientation; official tier 3 = 6.531. Correction: -141.722 mm axial, 6.747 mm lateral, 1.230° orientation; tier 3 = 6.418. The candidate improves lateral/orientation at this far endpoint but does not improve the insertion objective. These are observation-only policies, with no guide, exploration, backoff or live geometry. Videos and pre-reset scored streams are preserved. No fresh reliability screen or RL promotion is warranted.

## Bounded decoder/motion-supervision repair

A frozen audit on 512 original validation and 512 continuation rows found an almost constant predicted axial chunk span: ~1.013 mm on the continuation set, despite teacher spans of 0.3–2.25 mm and 40 stationary chunks. False forward changes on stationary chunks were **100%**. The original set's axial-span error was 9.273 mm p95. The audit **does not establish that a constant chunk caused the live stall**, but it identifies inaccurate state/phase-dependent command sequences hidden by first-target metrics.

One matched optimization/decoder comparison is now running. Shared image/state/pose encoder initialization, data, grouped validation, update budget and full-target action contract are fixed. Both arms reset their action decoder, use FP32 coordinate heads and an explicit per-step XYZ command-change loss; one retains the four-query transformer, the other predicts all four complete targets with a direct conditional MLP. The comparison isolates decoder architecture under this shared revised objective. A comparison with the frozen previous result can only establish the combined repair, not isolate each shared change. No learned correction is added on top of a baseline action.

Validation now also gates axial-span errors and false/missed advancement, with all held-out continuation rows retained for validation. Exact commands, initialization/source hashes, 6,000-update/four-GPU-hour caps and live eligibility are in `conditional_decoder_frozen_plan_v1.json`. Original actor weights and failed trajectories remain unchanged. Dataset collection has stopped.


## Decoder comparison finished: neither arm is eligible

Both arms completed 6,000 updates. Transformer selected update100: continuation near lateral0.740mm, all-phase first command5.988mm, axial-span error1.900mm p95, stationary false advancement100%. Direct conditional MLP selected update5000: near0.561mm, first command7.010mm, span error2.134mm, false advancement28.3% and missed advancement24.3%. Neither passes the unchanged command/pose/motion conjunction. No live rollout or checkpoint promotion was started. The MLP improves stationary-command discrimination compared with the transformer, but worsens full-command accuracy; it is not an autonomous improvement.

## Bounded causal history test

The continuation audit finds alignment holds around -25mm lasting roughly2–5seconds. The old actor sees only current state; its optional four-consecutive-row history covers about150ms. This is a plausible information gap, not an established cause of the live failure. One matched comparison uses identical history modules: control repeats current state/pose with zero ages; treatment samples measured state, force and current-command feedback at causal offsets6,2,0.5,0seconds. Both retain the latest three camera images, the original full-target action contract, original/continuation data and unchanged gates. No image-history benefit is claimed.

Training and runtime share one sampler. Current relative poses are transformed to the RGB-estimated locked port frame. Runtime retains distinct observations throughout chunk execution, clears history on reset and consumes no true geometry. Startup repeats the earliest available current-episode observation; future observations are rejected. A four-window parity audit verifies current pose to <3e-8 in stored coordinates, masked-control equality, causal ages and shape. This checks sampling, not learned performance.

Budget:6,000 matched updates, maximum240minutes each, sequential GPU5. No additional data collection. Finish both arms before selection; only an offline-eligible checkpoint receives one reused full-start pilot. Six fresh starts require depth>=14mm and lateral<=0.5mm with acceptable contact and explicit orientation. Final scenes remain sealed, RL remains disabled. Frozen plan and commands:`local_history_frozen_plan_v1.json`; sampler audit:`local_causal_history_parity_v1.json`; pause audit:`local_pause_phase_audit_v1.json` in the NAS execution root.

A tiny, fixed balanced linear phase probe provides a caution against assuming history solves this automatically. Trained on the two continuation episodes and tested on the held-out five-card episode, current state achieved 50.2% balanced hold/advance accuracy; causal robot history achieved 49.3%. The former almost always predicts hold, the latter almost always predicts advance. This is a limited linear diagnostic on three episodes, not evidence that a nonlinear actor cannot use history. The matched nonlinear comparison remains the decisive test. Artifact: `local_phase_linear_probe_v1.json`.

## Conditional approach-ownership revision

If neither robot-history model passes its frozen gate, run one reused full-start observation-only diagnostic. The RGB port/body locks and measured TCP estimate the held plug position. An explicit feedback controller holds the acquired TCP orientation, interpolates at15mm/s toward a30mm stand-off and applies the bounded lateral error accumulation from the controller diagnosis. It stops before contact, requires measured settling and hands over to the unchanged corrective-data actor. No true geometry, scoring events, guide, backoff or RL feeds this controller. Its actions are logged separately as controller actions, not learned predictions.

This revises ownership of the missing far approach, which the existing actor failed to learn. It does not retroactively pass the failed actor comparisons. The prospective approach budget is35simulation seconds, with10second no-progress and40N raw-wrench limits; maximum15mm per-axis command increments, gains90 and1second stable completion. One reuse only; depth>=14mm/lateral<=0.5mm and acceptable contact remain necessary before a frozen six-fresh screen. No parameter sweep or new training. Exact frozen plan:`observation_approach_frozen_plan_v1.json`.

The new causal terminal scorer explicitly selects the last logged pre-return observation, rather than the end of the bag. For the preserved corrective pilot it reports -141.977mm axial,7.604mm lateral,1.155° orientation with44ms-old past transforms. The earlier end-of-record figures (-141.722mm/6.747mm/1.230°) remain preserved and differ in timestamp; both show the same failed approach. Scoring penalties distinguish prohibited contact/sustained excessive force from harmless raw contact. Artifact:`corrective_control_terminal_rescore_v1.json`; implementation:`scripts/score_locked_local_terminal.py`.

## Robot-history comparison finished: no promotion

Both arms completed6,000updates, selected update100 and ended with59stale evaluations under the fixed budget. Masked/current control versus causal6second history: continuation near lateral0.834/0.723mm; all-phase first command5.362/6.147mm; axial-span error1.484/1.481mm p95; stationary false advancement100% in both. History did not pass the unchanged conjunction and did not improve the all-phase command metric. No live history pilot was started. These capped runs do not prove global optimization convergence, but provide no basis for a blind extension. Selected/last checkpoints and full curves remain archived.

Proceeding to the separately frozen single approach-ownership diagnostic, using the unchanged corrective-data checkpoint. It is an observation-only controller experiment, not an actor retraining or history improvement. No routine collection, RL or final scenes.


## Final bounded outcome: approach repaired, full insertion still blocked

The separately frozen observation-only approach passes its subgate on one reused full-start llb_1400 reset. At actor handover (40.55 simulation seconds), diagnostic truth shows **29.124 mm before the opening, 0.148 mm lateral and 0.548° orientation error**. The deployable controller used only the approved RGB port/body locks and measured TCP; its own estimate was 29.210 mm / 0.078 mm. This is a useful ownership repair of the previously unlearned roughly 150 mm approach, not learned-actor improvement.

The unchanged corrective-data actor then entered the opening and reached **11.073 mm physical depth, 0.413 mm lateral, 0.938° orientation error**. Official score75, no contact or excessive-force penalty. It fails the unchanged ≥14 mm full-depth gate. No six-fresh reliability screen, broad collection or RL followed. Local actor inference p50/p95/p99 = **72.1/92.0/104.1 ms** across336 decisions; this excludes earlier acquisition/refinement and does not certify full-pipeline latency.

Engine logs give direct event evidence: first SC plug/port touch at46.546s, intermittent contacts, then continuous touch from48.228s for1second. `CablePlugin` attached the cable to the port and made the end effector static, issuing completion. This explains why further commands cannot deepen the insertion after completion. It does **not** establish that a single orientation, lateral or compliance error was the dominant cause of shallow seating. The actor worsened alignment before contact: lateral error0.148→0.870mm by43s, orientation0.548→0.853°, while the final values are0.413mm/0.938°. Contact is between the SC plug housing and SC port colliders, not a demonstrated cable snag. Tiny/zero per-contact Gazebo wrench fields are not calibrated wrist-force evidence.

Four critical intervals were inspected in all five camera views, including side and overhead. The H.264/yuv420p video has20fps and111.05seconds duration. Artifacts: `observation_approach_terminal_metrics_v1.json`, `observation_approach_contact_diagnosis_v1.json`, `observation_approach_critical_review_v2.{jpg,json}`, `observation_approach_latency_v1.json`, and `observation_approach_llb_1400_v1/five_cameras_20fps_grid_active.mp4` in the NAS execution root. The launcher returned0. A post-run print statement in the Python wrapper raised NameError after recording success; that bookkeeping statement was fixed without repeating the completed simulation. Rendering initially required the scored stream, so extraction completed first and rendering was retried; no rollout was duplicated.

### Evidence-backed blocking conclusion and targeted next revision

Ordinary autonomous full insertion remains blocked after distinct timestamp/interface, feedback, point-target, corrective-data, decoder/motion and robot-history repairs. The approach ownership change makes real progress, but it does not pass full insertion. Keep both the older project step5 and the current insertion gate open. All new jobs have finished; original datasets, failed checkpoints and comparisons remain preserved. Reserved final scenes remain sealed.

The next useful diagnostic is **not more routine data or another architecture sweep**. Freeze this hybrid as the control and isolate near-insertion geometry before changing supervision:

1. Use the same reused reset and frozen actor. In a privileged diagnostic only, replace orientation with same-timestamp true grasp/port alignment while preserving the actor's predicted plug-point targets. No policy promotion from this test.
2. Separately keep actor orientation/axial targets and replace only lateral compensation with the already verified teacher centering feedback. Log exact executed versus proposed targets and action ownership.
3. Budget: at most two new full-start diagnostic rollouts, approximately20minutes plus recording/scoring. No training or data expansion. Keep physical≥14mm/≤0.5mm, explicit orientation and force/contact reporting.
4. If one intervention restores depth, repair that specific representation/target/controller contract with a matched observation-only experiment. If neither does, investigate coupled alignment and compliant command semantics before more fitting. Privileged success is an upper bound, not autonomous success.
5. An observation-only repair must pass a frozen physical pilot and then≥5/6 fresh grouped supported starts before broader family/orientation coverage. RL, recovery and transfer remain disabled until the ordinary supervised gate passes.

This is the explicit unresolved-gate conclusion permitted by the execution prompt, not a claim that all optional models were trained or the complete pipeline was achieved. Spatial encoder/capacity sweeps were not run without a measured prerequisite. Motion/robot history, continuation and decoder branches were tested; final scenes and dependent RL branches were not opened.

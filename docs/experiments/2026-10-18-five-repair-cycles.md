# Five supervised repair and public-sample evaluation cycles

## Goal and immutable checks

User requested five initial analyze → targeted repair → unchanged public sample evaluation cycles, followed by the requested continuation until above **290/300**. Starting candidate V34 scores 170.57/300 (two strict insertions); prior complete passes score approximately 257/300. Final evaluation scenes stay sealed. Public `sample_config.yaml`, task deadlines, physical gates, collision/scoring code and force limits stay unchanged. Controller inputs remain RGB, task identity and public robot state; simulator geometry is offline supervision/scoring only. No new RL/world-model training. At most four project GPUs including untouched legacy GPU 0, with cooperative locks, unique rootless networks/domains and NAS artifacts.

A cycle is counted only after an actual complete three-trial sample evaluation and causal analysis. Training/data variants inside a cycle are not additional cycles. Stop a candidate's routine repeats after failure; repair based on its trace. Retain every failed candidate and exact input hashes. Training must reach validation plateau through learning-rate reductions; a time/update cap is incomplete. The score target is not a promise of five-cycle convergence, and physical failures must not be hidden by a toolkit score.

## Planned order; adapt only from evidence

1. **SFP orientation coverage.** The failed plug has 26.5° held roll versus original label support 17.8–23.1°. Same-model close re-estimation remains biased. Collect 32 bounded perception-only resets (24 train / 8 validation), varying card count, requested card/port, board pose and grasp orientation. First two per worker require causal label/image QA. Preserve original 354-episode SFP corpus. Train an original-plus-extension pose candidate with separate domain metrics and fresh live checks, then freeze it for the full public sample. Do not merely increase stiffness against the wrong orientation frame.
2. **Approach/control time.** Use the first repaired full-sample trace to quantify time spent in transport, 150→25 mm approach, insertion and dwell. Test faster smooth free-space approach and measured phase transitions without increasing contact force or hiding near-contact errors. One targeted candidate, then full sample.
3. **SC route efficiency.** The current 1.40 m route loses all efficiency points on the public SC case. Investigate a shorter observation-checked free-space route, retaining the around-card route whenever obstacles cannot be excluded. No scene-YAML/true geometry input or memorized final-scene coordinates. Evaluate collision/physical insertion before full sample.
4. **Smoothness and remaining bottleneck.** Target the actual dominant residual loss: command continuity, acceleration limits, excessive holds or contact alignment. Keep existing successful family behavior and compare exact commands/forces/video. Retrain only with a concrete model/data hypothesis.
5. **Best supported combined candidate.** Freeze the strongest passing components and rerun the complete sample. Inspect all rounds, report score and strict geometry, p50/p95/p99 inference, contact/force and videos. If above 290, stop and report scope; otherwise state the unmet target and measured remaining gap.

## Cycle 1 started

Artifact root: `/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/`.

Two collectors use GPUs 3/4 and ROS domains 139/140. Each has 16 new public-sample-derived free-grasp resets, with 12 train / 4 validation complete episodes. Pose perturbations are declared before collection: each Euler input ±0.12 rad and each grasp translation ±1 mm. The actual physical poses are checked afterward as labels, not chosen from the failed evaluation's ground truth. Holds are perception-only, not BC/insertion experts. The legacy script name `collect_sc_grasp_variations.py` is reused in resume mode with explicit SFP manifests; its label helper now preserves the case family and source role.

Exact preparation command: `.pixi/envs/default/bin/python scripts/prepare_SFP_orientation_cycle_captures.py`.

Collector commands, actual environments, raw bags, labels and state are retained in the two worker directories. No active job is interrupted.

### Cycle 1 design checks

The first four three-camera raw-label overlays were actually inspected; virtual plug-frame markers align with the visible SFP tip. Labels preserve the SFP family and use causal TF/controller joins. The first eight fixed original-resolution crop audits retain all supervised references inside the jitter-safe crop bounds. Observed new held-roll coverage already extends above 27°, beyond the old corpus's 23.1° upper bound. No failed evaluation image is added as training data.

Two objectives use the identical merged corpus, episode split, seed, crop augmentation and LR-floor convergence rule: orientation-only repair on GPU 5 and joint pose repair on GPU 4. This tests whether existing geometric tip translation remains accurate on wider grasps; no candidate is selected solely because orientation improves. Domain review reports original versus new held-out episodes separately, tests the existing translation component, and saves fixed-index prediction-only images for inspection. No inference model swaps automatically from a training summary.

A speed-profile candidate is prepared for cycle 2 but has not been run: smooth 20–50 mm/s free approach and 12 mm/s insertion reference, compared with the previous 8–20 mm/s approach and 6 mm/s insertion. Existing force, alignment, seating-dwell, physical-success and reference-governor limits remain unchanged. The legacy controllers retain their exact 6 mm/s default; focused force-pause, tracking-seed and alignment-gate contract checks pass after exposing the optional speed setting.

### Cycle 1 first fits fail; no unqualified swap

All 32 captures finish and pass admission. The immutable merged corpus has **6,499 triplets / 386 episodes**: 331 training episodes / 5,600 triplets and 55 validation episodes / 899 triplets, preserving the old splits. Canonical reset reconstruction from raw source YAML finds 382 reset groups and **zero cross-split conflicts**. A separate 648-episode classifier corpus also has zero canonical reset conflicts. `canonical_reset_split_audit.json` retains every group.

Both matched pose fits reach their LR-floor plateau at **15,600 updates**, not a time cap. Neither passes the wide-grasp gate. Orientation-only p95 is **1.017° original / 5.510° wide**; joint-pose orientation p95 is **1.202° / 5.753°**, and its translation p95 is **2.591 / 4.404 mm**. The existing geometric position component is also insufficient on the complete wide domain (0.839 mm p95). Whole-original-domain position p95 0.505 mm differs from the previously qualified free-acquisition subset; do not confuse those scopes. No new model is deployed.

The next controlled training repair removes unobserved independent per-camera spatial crop jitter, preserving the same photometric augmentation, data, split, initialization and convergence rule. A direct calibrated 3D pose head does not receive those crop offsets, so random shifts may introduce pose ambiguity. This is a hypothesis, not yet a demonstrated explanation. `SFP_fixed_crop_joint_fit/` is training on GPU 5. Do not turn a failed validation flag into approval.

### Bounded SC route branch, before full cycle evaluation

An ImageNet layer-3 encoder plus a small three-camera classifier is trained only on existing SC/SFP corpora to recognize an empty NIC row. Inputs are full images; simulator card counts supply labels only. Deterministic missing-context image variants are negative. The initial held-out set has zero false clears but only two positive episodes, failing the declared coverage gate. A fixed-threshold replay then adds four independent zero/five-card SC traces, with canonical reset disjointness: both empty scenes get a three-observation clear sequence and neither occupied scene gets a false clear. Prediction-only startup panels were actually inspected. The original insufficient-coverage result is preserved.

A **SC-only live preflight** now tests the conditional route. Three fresh agreeing image decisions allow a high-clearance direct transport route in an empty scene. Occupied or uncertain scenes retain the original around-card bypass; no true scene count enters the policy. A public prior may guide the transport waypoint when the target is not yet visible, but final requested-port visual acquisition remains mandatory. The experimental classifier is not a collision planner or globally certified visibility model.

Four unchanged zero/five-card, both-port development cases run on GPUs 3/4 with unique rootless domains 141/142, alongside the fixed-crop fit on GPU 5 and untouched legacy GPU 0. At most four project GPUs are used. Stop launching cases after a failed physical/timing/route-selection gate and let active cases finish. The SC preflight uses the planned 20–50 mm/s approach and 12 mm/s insertion reference, plus preloaded immutable networks; it does not rely on the failed SFP candidates. This is an auxiliary preflight, **not a completed public evaluation cycle**.

Caching/preloading preserves model weights while clearing every pose, proposal, acceptance and camera history at reset. An actual-method reset contract check passes. No model observation or task geometry is used during preloading. Candidate source hashes remain frozen while live jobs run.

## Completed cycles and current repair

| Cycle | Candidate | Unchanged sample score / 300 | Strict insertions | Inference p95 gate | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | V39: preload, faster approach/insertion, moving empty-row check | 262.625653 | 3/3 | Fail: transport 333.23 ms | Preserve physical pass; repair repeated scene checks |
| 2 | V40: three fresh stationary scene decisions before motion | 183.767460 | 2/3 | Pass | SC route improves, SFP contact/lateral detection remains unreliable |
| 3 | V41: moving-reference antiwindup | 182.801546 | 2/3 | Fail: close 315.42 ms | Overshoot fixed; SC remains shallow |
| 4 | V42: SFP lateral tracking; earlier retained-context encoding | 274.875269 | 2/3 | Pass | Toolkit completes all three; SC depth 15.484804 mm fails |
| 5 | V43: bounded speed/profile and shorter SFP acquisition | 281.046188 | 3/3 | Pass | Five initial cycles complete; score target remains open |
| 6 | V44: explicit SFP direct acquisition at80mm | 284.206771 | 3/3 | Pass | Shorter route improves score; target remains open |
| 7 | V45: acquire in place when visually justified; preserve API hold | 283.749600 | 3/3 | Pass | Extra hold/timeout defeats early acquisition |
| 8 | V46: shorter redundant hold; active-view fallback | 284.279134 | 2/3 | Pass | SC15.49804mm fails; angular approach remains slow |
| 9 | V47: persistent orientation, first integration | 3.000000 | 0/3 | Fail / no close stage | First-tick seed initialization error; retain all three failures |
| 10 | V48: explicit public-hold seed | 3.000000 | 0/3 | Fail / no close stage | NumPy diagnostic boolean not JSON serializable |
| 11 | V48 with diagnostic scalar repair | 286.187786 | 3/3 | Pass | Angular approach works; no model replacement |
| 12 | V49: visually admitted empty-SC diagonal transport | 287.047013 | 3/3 | Fail: fresh323.15ms | Shorter route works; cold preprocessing gap |
| 13 | V51: phase-specific free gains and preloaded grasp models | 287.135334 | 3/3 | Pass | Force-preserving gain seams work in sample |
| 14 | V52: parallel exact codec, quicker free hold and view fallback | 288.786609 | 2/3 | Pass | SC15.491367mm fails strict depth |
| 15 | V53: distinct force sampling; faster empty-SC insertion | 288.282428 | 3/3 | Pass | Valid initial images incorrectly rejected by controller-header clock |
| 16 | V54: causal delivered-packet camera timing | 289.131676 | 3/3 | Pass | Early acquisition works; target remains open |
| 17 | V55: joint grasp acquisition, exact parallel codec and1.5s seating | 290.089395 | 2/3 | Pass | SC15.499944mm fails; no rounding; missing quiet far context |
| 18 | V56: retain three quiet SC far-context frames | **290.192549** | **3/3** | **Pass** | Requested score target achieved on public sample; stop loop |

Cycle 1 never activates its intended empty-row shortcut: after two accepted moving observations the scene leaves the useful view, resetting consensus. Cycle 2 holds the actual previous API reference, observes three distinct quiet frames, consumes a task/age/motion checked receipt once, then stops calling the classifier during transport. SC transport drops from 41.254 to 13.200 simulation seconds; its score rises to 92.355, but its 15.500499 mm physical depth has almost no margin. Occupied or uncertain scenes keep the around-card route. Four independent zero/five-card SC preflights have correct route selection in all four, but only three strict insertions; the five-card port-0 shallow failure is preserved.

Cycle 2 SFP trial 1 has 46.772202 mm depth, 0.484093 mm measured lateral error and 0.171968° orientation error, but **toolkit tier 3 is zero**. It is not counted successful. The image-based controller estimates 0.523 mm lateral error and ends with no measured axial progress. Recorded terminal views show the plug at the NIC, without an obvious cable snag; images alone cannot prove insertion. Cycle 2 trial 2 and SC pass. All final scenes stay sealed.

Cycle 1 public-state traces also expose approach integral windup: the nominal 25 mm stand-off is crossed to about 10–13 mm, followed by retreat to 24 mm, with no backoff events. V41 initializes from public TCP tracking bias and integrates only quiet, stationary reference axes. This preserves gravity compensation without treating ordinary motion lag as a permanent forward demand. A 20 mm crossing abort is added. Existing controllers retain their exact default behavior; only V41 enables the repair.

### Failed model repairs remain undeployed

Removing spatial crop jitter alone reaches validation plateau at 15,600 updates but does not solve wide-grasp errors (wide translation p95 5.288 mm, orientation 6.898°). The source crop centers exactly match checkpoint metadata, so a supposed crop-center mismatch is excluded. A calibrated keypoint/rigid-fit representation is training on the same episode split; pixel shifts and camera origins are adjusted consistently. No unqualified SFP model is deployed.

The contact-sensitive SC orientation fit also reaches LR-floor validation plateau (17,000 updates). Its held-out near orientation p95 is 1.080° and front-corner axial-excess p95 0.175 mm, failing the declared 1° / 0.05 mm gate. It remains undeployed. The exact added loss and selection are saved separately because its inherited training-plan prose still describes the parent experiment. Original and failed checkpoints are retained.

Artifacts and complete commands: `/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/cycle_N/`; machine-readable comparison: `cycle_results.json`; recorded videos: `cycle_N/per_trial/trial_K/five_cameras_20fps_grid.mp4`; fixed recorded review panels: `cycle_N/review/`.

### Cycle 3 outcome and cycle 4 hypothesis

Cycle 3 passes both SFP rounds but SC fails at **15.474873 mm / 0.057928 mm / 0.646985°**, toolkit tier 3 zero. The stop is real shallow seating, not simply an incorrect success callback. Posthoc decomposition on the same causal terminal shows the contact-sensitive orientation component worsens from cycle 1 to cycle 3: front-corner axial-excess proxies 0.0527, 0.0847 and 0.1016 mm. Neither total angle alone nor stable repeated predictions ensures sufficient seating. Raw far/near components disagree about which is better across cycles, so no test-derived rotation offset or universal blend-weight claim is introduced. `SC_three_cycle_far_near_rotation_audit.json` preserves those comparisons.

The calibrated SFP geometry fit also reaches an LR-floor validation plateau (16,200 updates, selected step 4,200). Wide-domain translation p95 improves to **0.232 mm**, but wide orientation remains **2.219°** and original translation/orientation p95 is **0.884 mm / 2.017°**. It is unqualified and not deployed. Thus none of the additional pose fits is a successful model replacement.

Cycle 4 changes SFP base-X/Y stiffness to500 only during insertion, keeping base-Z90, rotational gains, force guards, models and success gates. This targets measured lateral drift, not the separate known out-of-support orientation failure. It cannot establish a repair of that orientation coverage gap. SC retained far images are encoded once before free approach, using exactly the same model and saved images as before. Their preparation time is reported as its own used inference stage; it is not omitted from the timing gate. This prevents one close-up decision from bundling three old-frame encodings plus a new-frame inference. Actual-method checks verify one encoding per task, clearing at reset and SFP stiffness restoration after exceptions.

### Cycle 4 does not pass the physical gate

Toolkit **274.875269/300**, but **2/3 strict**: SC receives75points yet reaches only **15.484804 mm**, lateral0.109849 mm, orientation0.752534°. Both SFP rounds pass at46.7683/46.7482 mm. All used p95 timings pass. In this particular SC run, there are fewer than three quiet far frames, so the existing near-only fallback is used; the2.998ms preparation time is an unavailable-history check, **not evidence that three-frame encoding became2.998ms**. Close p95 is92.80ms. Keep counts and branch identities with the latency metrics.

The offline causal contact audit joins both tip TF edges to a common past TF time, then takes the past controller state (maximum allowed skew10ms). On cycles1–4, the inferred held-plug orientation changes by at most **0.000427°** between near update and terminal; translation change is at most0.0122mm. Its prediction error is already0.594–0.753° at the near update and remains essentially unchanged through contact. Within these logged samples, slip cannot explain the remaining sub-degree error. This supports correcting the visual estimate rather than repeatedly refreshing an already biased estimate. Exact ages and all stages: `SC_grasp_rigidity_during_contact_audit.json`; command: `.pixi/envs/default/bin/python scripts/audit_SC_cycle_grasp_rigidity.py ROOT --cycles 1 2 3 4 --output ROOT/SC_grasp_rigidity_during_contact_audit.json`. No audit label enters runtime.

### Cycle 5 bounded profile branch

V43 retains all existing RGB weights, the100–185mm shared acquisition envelope, SFP TCP acquisition minimum120mm, force/harm/physical gates and official180s task budget. SFP's rule target changes150→120mm to remove an unnecessary lift/return detour. Free transport is0.10m/s and0.5rad/s; approach uses a smooth40–80mm/s reference with antiwindup. SFP uses baseXY500during approach as well as insertion and tests24mm/s insertion below the25mm/s governor. SC stays at12mm/s insertion. The occupied-scene SC bypass is unchanged. These settings are saved in `cycle5_speed_transport_spec.json`; this branch targets time/path losses and does not claim to solve SC pose bias.

Before cycle5, source freezing is extended from the ROS inheritance tree to transitively imported local script modules and a saved/hashed copy of the fixed camera calibration. Earlier cycles retain their original snapshots and narrower source-freeze manifests; those are not retroactively called complete dependency snapshots. Models/scenes of completed cycles are not changed.

## Target continuation after the five initial cycles

Cycle5 finishes **281.046188/300, 3/3 strict, all used p95 timings below300ms**. SC15.519049mm has only0.019049mm depth margin; this is still not broad reliability. The initial five cycles are complete. Because the user also explicitly requested repeating until above290, the driver now requires `--target-continuation` for subsequent cycles rather than silently pretending the target was achieved or counting training fits as cycles. The earlier provisional maximum-five wording is superseded by this explicit target continuation.

Cycle6 removes a concrete legacy architectural detour: the transport/lock100mm minimum was inherited from a learned actor's training envelope, while the deployed controller is explicit and can own a shorter approach. Only SFP uses a newly declared70–185mm operational acquisition envelope and80mm nominal target. Its RGB grasp estimator still requires a measured TCP gap of at least120mm. SC keeps100–185mm; physical insertion depth, lateral/orientation, force and score gates are unchanged. The SFP rule transfers and rotates at a conservative clearance above the acquisition goal instead of lifting out to150mm and returning. Three fresh requested-port checks and the native refiner remain mandatory.

Cycle6's frozen CLI reuses the cycle5 transport-speed JSON. Its unused `candidate_overrides` metadata block still describes the cycle5 nominal120mm target; the actual frozen V44 code overrides that to80mm. `cycle_6/effective_policy_contract.json` records precedence explicitly. A corrected `cycle6_direct_transport_spec.json` is prepared for any subsequent repeat. Neither active frozen file is edited. This is not a physics/scoring change or use of true port coordinates.

### Cycle 6 outcome; cycle 7 concrete remaining gaps

Cycle6 reaches **284.206771/300, 3/3 strict, all used p95 stages pass**. SFP depths46.766238/46.748186mm; SC15.517479mm. This is better on the original public sample, not proof on all scene combinations.

Cycle7 tests three observation-only efficiency repairs. First, SFP may stay at its initial pose when three distinct synchronized native-image fits identify the requested target and public TCP/nominal-tip coordinates already lie in the declared approach region. It keeps the actual public API reference, waits for the original stationary acquisition hold, and falls back to the cycle6 route after3seconds if checks do not pass. Second, its explicit approach may correct up to60mm lateral displacement (initial nominal-tip candidate limit50mm); the old learned-actor30mm limit is left unchanged for every other class and forSC. No axial advancement occurs until the existing millimeter alignment gate passes. Third, a visually confirmed empty SC row permits transfer and orientation at the same conservative height, rather than stopping for a separate rotation waypoint; occupied routing is unchanged.

A separate fresh-acquisition hold bug is repaired: the old global acquisition command held the measured TCP pose, discarding static tracking compensation. V45 uses the public controller's actual `reference_tcp_pose`, with a40mm/5° validity bound, then clears that cached hold at return/reset. It never reconstructs that reference from privileged geometry. Actual-method tests cover three-fresh-image admission, duplicate/out-of-region rejection, timeout fallback, exact public-reference preservation and exception cleanup.

**Preserved startup failure:** `cycle_7/` contains a policy import-order error before any observation or robot motion. The leaf imported a script helper before its ROS parent initialized the script search path. Both policy and engine had already failed; the orphan startup watchdog was released through its normal cleanup path, with an explicit manual-marker record in `startup_failure.json`. No other active job was interrupted. This is not counted as a completed public evaluation. Imports are reordered, failed-policy watchdog detection is added, and the same controller hypothesis is rerun in **`cycle_7_attempt2/`**, preserving the original directory and logs.

### Cycles 7–9: eliminate observed waiting and an angular control error

Cycle7 attempt2 finishes283.749600/300,3/3strict and timing pass. Both SFP tasks fall back to their route. For the first, three to five accepted fresh port fits exist, but a3s timeout preempts the extra1s stationary hold immediately before it would finish. For the second, seven initial native fits are rejected; passive waiting cannot create the needed view. Cycle8 keeps three distinct synchronized native confirmations plus the separate downstream fresh grasp lock, shortens only the redundant transport hold to250ms, and chooses the existing moving-view route after three fresh rejected candidates. Its fallback combines transfer/rotation without blind descent.

Cycle8 scores284.279134/300 but SCdepth **15.49804mm** is a physical failure, despite toolkit75. Do not round it to15.5. SFP initial acquisition now works, yet its approach takes **12.996s**: the tool starts21.37° away and corrects only about2°/s. The source resets each angular command to one degree ahead of the **measured** angle. That bounds the torque demand and restarts the plan each tick; a larger initial correction becomes slow even though short near-aligned corrections seemed acceptable.

Cycle9 uses a persistent angular reference initialized from the previous actual public API command, with0.5rad/s speed,1rad/s² acceleration and5° lead bound. It retains the last reference when the measured tool lags too far, instead of adding more demand or rolling the reference backward. Existing measured alignment gates still forbid axial advancement until the tool is aligned. The final orientation, pose models, force guards, physics and physical success criteria are unchanged. A toy PD contract check and stationary-tool lead-bound check pass; only the live full sample can establish improvement. `persistent_approach_reference_contract.json` is explicitly not simulator evidence.

### Actual grasp coverage caveat

Posthoc terminal grasp reconstruction for the first seven completed samples shows both SFP grasps remain near **20.46° held roll**, inside original support. The earlier V34 failure had26.52°. The scene directory and original sample YAML are byte-identical, but these recent successes do **not** demonstrate that the old out-of-support perception failure has been repaired. All wider-grasp fits remain unqualified and undeployed. `actual_SFP_grasp_coverage_across_cycles.json` preserves this negative coverage check; do not equate repeated public-sample score improvement with broad grasp robustness.

### Preserved integration failures and corrected angular result

Cycles9and10 each complete the three public attempts at **3/300,0/3strict**, because of implementation faults. They are retained as failed full evaluations, not removed from the score plot. Cycle9 tries to read an insertion-governor seed before the first approach command creates it. V48 explicitly seeds from the current public fresh-acquisition API reference, replacing any previous-episode seed. Cycle10 then exposes a NumPy boolean in the new diagnostic payload, which JSON cannot serialize. It is converted to a Python bool. No trained-model capacity conclusion is drawn from either fault.

A new integration contract now executes the actual V48 initialization method and actual approach loop through five commands with an initially absent governor seed, roundtripping every full trace through JSON. This complements, rather than replaces, the earlier helper-math tests. `scripts/test_persistent_approach_public_seed.py` passes; exact output is `V48_actual_approach_trace_integration.json`. Recorded-image review now handles failed approaches that never reach25mm, preserving their start/mid/terminal images instead of assuming a successful handoff.

Cycle11 then reaches **286.187786/300,3/3strict**, with all used p95 stages below300ms. The first SFP approach falls12.996→5.902seconds. It does not become a2.15s real approach: that earlier number was a toy angular-only check, whereas the simulator also aligns translation and advances to25mm. SCstillrequires7.4s transport,5.15s approach and8.50s insertion/dwell, making it the largest remaining time loss.

Cycle12 tests a shorter diagonal **only after the same three fresh full-image empty-row decisions**. It targets120mm stand-off instead of150mm, retains the100–185mm SC lock envelope and requires nominal TCPheight>=140mm and a bounded route length. Empty-row free speed/rotation is0.12m/s/0.8rad/s. Unknown or occupied scenes keep their existing bypass and control profile, and reset restores those controls before the next task. SCinsertion remains12mm/s. Runtime target acquisition is still native requested-port perception; simulator labels never select the route. One unchanged public sample is its budget. If force/harm, strict insertion or timing fails, diagnose that trace rather than routinely repeat it.

The older26.52° SFPgrasp failure is already present at its **initial force hold**, before transport, and stays fixed through that episode. Current samples start near20.46°. Thus the difference is an actual startup grasp difference, not evidence that our faster route or a new learned model corrected its orientation. The original YAML/scene files match and no grasp/seed environment override differs; the startup mechanism itself is not resolved. See`SFP_old_vs_current_grasp_lifetime_audit.json`.

### Cycles 12–15: motion phases, cold preprocessing and sample cadence

Cycle12 has3/3strict but fails fresh-perception p95 **323.15ms**. Cold construction loads a family's grasp-rotation encoder **after** the shared-JPEG wrapper loop, so that first family query does not yet share preprocessing. V51 preloads both unchanged encoders before wrapper installation and only warms kernels on synthetic arrays; it never sets a pose from synthetic data. Actual frozen SFProtation/translation predictions are bit-exact with sharing (`cold_task_shared_JPEG_frozen_model_equivalence.json`).

Cycle13 uses240base-Z and100rotational stiffness only during the free SFP/visually empty SC approach; baseXY stays500. It restores the prior90axial/50rotational contact settings before close-up observation/insertion. Every gain seam issues a public-reference adjustment preserving the static linearized spring effort, then seeds from that **actually issued** API. Controller source uses a base-frame spring wrench; its internal exponential gain smoothing means actual transient effort is not claimed perfectly constant. Actual entry/exit/exception contracts and recorded gain commands are retained. This candidate scores287.135334/300,3/3strict, freshp95295.6ms; SCdepth15.530981mm.

Cycle14 adds two independent overhead repairs. Initial assumed-free hold uses500translation while retaining20fresh force samples, the1Nspread criterion and the existing velocity bounds. A single fresh invalid initial native fit chooses the already safe moving-view route, whereas **acceptance still requires three valid fresh fits**. The independent three-camera JPEG86 conversions run in parallel. Twelve real camera images and actual frozen model outputs are bit-exact; CPUmedian codec time34.70→15.45ms is a CPU microbenchmark, not a live-inference claim. Live freshp95becomes204.5ms. Score288.786609/300is **2/3strict**, because SCdepth15.491367mmfails; no rounding or model promotion.

Cycle15 addresses a measured sampling mismatch: early force-hold commands have roughly98msmedian spacing despite an intended50mscheck. It admits each distinct delivered force observation exactly once, keeps the20-sample median/1Nspread and quiet-velocity criteria, and adds a0.9sminimum quiet span. Duplicate packets do not republish commands, and the force reference remains frozen throughout contact/recovery. An actual-method check with duplicate packets finishes at20commands/20samples/0.95s. The helper-only contract also rejects20high-rate samples spanning only0.19s. One test harness initially selected the file's exception class instead of the policy class; that harness lookup is corrected and the actual loop check passes.

The same cycle tests24mm/s insertion reference **only for visually confirmed empty SC**, below the existing25mm/s governor; occupied/uncertain SC retains12mm/s. It keeps2sseating dwell, all pose weights, raw/load force guards, deadlines and physical success thresholds. This is a control/sampling experiment, not a repair of the remaining perception accuracy gap.

### Additional bounded diagnosis; not enabled in the controller

Public-wrench replay is saved for successful/shallow SCseating. Correct torque transport must include the fixed **172mm ATI-sensor→gripper-TCP offset** from the public Hand-E xacro; raw wrist torque cannot be treated as a plug-tip moment. Both the deliberately incomplete-origin diagnostic and corrected-origin result are retained. Residual tip moments remain background/contact dependent, so no unvalidated torque-admittance controller is enabled. `SC_tip_moment_seating_diagnostic.json` and `SC_tip_moment_correct_sensor_origin_diagnostic.json` are offline diagnostics only.

The adapter publishes images with past controller/wrench states; an image timestamp slightly ahead of the controller header can therefore be a valid delivered observation. Existing early/near freshness code may reject such samples. A future repair should define decision time from the available sensor packet and explicitly bound proprioception/image skew, rather than treating an older controller header as the wall of causality. This finding is documented, not silently used to relax current timestamp gates.


### Cycles 15–17: camera causality and bounded acquisition overhead

Cycle15 finishes **288.282428/300,3/3strict, timing pass**. Cycle16 corrects an adapter clock mismatch: the official adapter pairs images with a past controller state. Images2–4ms newer than that state are available observations, not future information. Initial SFP acquisition now uses the latest delivered sensor timestamp, bounds state/camera skew to10ms, rejects duplicates and requires three quiet, geometrically agreeing native fits. Stationary empty-row admission uses the same packet check. The control clock and near-grasp freshness logic remain unchanged. Contract tests reject50ms stale state and duplicated packets; no simulator geometry enters this decision.

Cycle16 finishes **289.131676/300,3/3strict**, all used p95 stages pass. Transport p50/p95/p99 is107.59/196.56/363.70ms; fresh perception164.74/186.55/189.69ms; close update105.44/118.29/119.93ms; servo5.87/10.69/12.59ms. First/second SFP durations are13.76/15.48s, SC18.23s. The score target is still unmet.

Cycle17 tests one bounded overhead candidate: rotation and translation consensus use the same three fresh native observations rather than serial windows needing five total frames. The actual-method test confirms legacy five-frame behavior, new three-frame acceptance and rejection of inconsistent rotation. This also changes which far-context frames are retained; it is not claimed to leave the fused estimate identical. Record the selected far/near branch. Independent camera JPEG86 conversion is parallelized for transport using the already pixel-equivalent codec.

Seating dwell is1.5s instead of2s, retaining0.5s margin above both public TouchPlugins'1s requirement. Toolkit75, pre-return physical depth/lateral/orientation, force/harm and latency checks remain mandatory. This is an explicit-controller control experiment with unchanged models, not BC or RL. Exact candidate metadata is in`cycle17_joint_acquisition_transport_spec.json`; metadata blocks do not configure runtime class overrides. No final scenes are opened.


### Cycle17: score passes, physical gate does not

**290.089395/300 is not a promoted result.** Toolkit75 is awarded in all rounds, but SCdepth15.499943717mm is below15.5mm. Lateral0.114374mm and orientation0.721460° pass; depth does not. Both SFP rounds pass (46.767838/46.746780mm). No scored excessive force/contact harm. Used p95 timings pass (transport132.18ms,fresh195.95ms,close138.56ms,servo10.26ms).

The shortened acquisition exits before three quiet far SCimages exist. Far-context preparation0.524ms is an unavailable-history check, not three encodings. This concrete missing-input regression motivates cycle18: wait until three distinct quiet SCfar frames have been retained before binding the joint grasp estimate. Keep SFPjoint acquisition and1.5sseating; do not alter thresholds or introduce true geometry. The actual-method test with a first moving observation now needs four frames total for SC versus three for SFP and five for the legacy serial method. Test harness needed an SCmock model entry; that missing fixture is fixed before launch. Cycle17 terminal five-camera contact sheet was actually viewed; the plug appears seated with no evident cable snag, but images cannot resolve this microscopic depth failure. All original raw data and failed decision remain preserved.


## Completed: above290 with all three physical gates passing

Cycle18/V56 reaches **290.192549/300**. This completes the requested score-target loop after18 complete three-trial public-sample evaluations (54rollouts), plus the separately recorded failed startup and four SCpreflights. Five initial cycles alone did not reach the target. Failed candidates remain in the table and artifact root; do not report18independent passing evaluations.

| Round | Task | Score /100 | Measured depth mm | Lateral mm | Orientation degrees | Strict gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | SFP, NIC mount0, port0 | 97.180 | 46.775846 | 0.101495 | 0.141225 | Pass |
| 2 | SFP, NIC mount1, port0 | 96.556 | 46.748804 | 0.129038 | 0.385019 | Pass |
| 3 | SC, selected port1 | 96.457 | 15.528748 | 0.161249 | 0.427162 | Pass |

Every trial receives toolkit75 completion, with no scored contact harm or excessive force. The SC context repair actually waits one extra observation, retains three quiet native images (0.352° predicted rotational span), encodes the unchanged strong model in166.04ms and uses the existing far/near check. No oracle target/plug geometry is supplied to control. All terminal metrics use the retained observation before return/reset. No retreat is used, so this run does not demonstrate a recovery benefit.

| Used inference stage | Calls | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- |
| Transport | 44 | 31.09 | 158.10 | 323.23 |
| Fresh port/grasp acquisition | 10 | 160.68 | 220.44 | 222.02 |
| Close plug update | 5 | 109.88 | 127.46 | 128.29 |
| Servo step | 256 | 6.45 | 10.48 | 11.67 |
| Stationary empty-row check | 3 | 90.46 | 91.42 | 91.50 |
| Retained far-context preparation | 1 | 166.04 | 166.04 | 166.04 |

All used-stage p95 values are below300ms. Transportp99 still exceeds300ms; a single far-context preparation does not establish a latency distribution. Terminal recorded five-camera sheets for all three rounds were actually viewed; they show no evident gross cable snag. Physical qualification relies on causal geometry measurements, not visual guesswork.

### What changed, in easy words

The controller keeps the same learned perception weights. It wastes less motion and waiting: visually empty SC rows can take a shorter route; the approach corrects tracking drift without overshooting, and its orientation reference keeps moving instead of repeatedly restarting. Free-space tracking is firmer, while contact tracking stays compliant. Gain transitions preserve the public holding reference. Camera conversion runs in parallel with identical pixels, force samples are counted once, and camera freshness uses the delivered packet rather than an older robot-state timestamp. Finally, translation and rotation share fresh images, while SCstill waits for the three quiet context views needed for its close orientation correction. Seating remains longer than the official contact requirement.

These combined changes improve the measured public sample. Because multiple changes accumulate across cycles and startup is not perfectly repeatable, the total score gain cannot be assigned to one change. No additional SFP/SCpose fit qualifies; the trained-but-failed candidates stay undeployed. This is **rule transport plus explicit visual insertion**, not a successful learned BCactor or RLexperiment.

### Artifacts and stopping decision

Selected scoped candidate:`selected_public_sample_candidate.json`; final machine state:`state.json`; all evaluations:`cycle_results.json`; frozen command/environment/source/model/calibration manifests:`cycle_18/`. The tracked compact summary is`2026-10-18-five-repair-cycles.summary.json`; comparison plot:`repair_cycle_comparison.png` at the NAS root. The exact driver-launched evaluator arguments are retained in`cycle_18.command.json`; the actual container launcher/environment is in`cycle_18/command.json`, alongside the repair hypothesis and frozen plan. Driver command: `.pixi/envs/default/bin/python scripts/run_public_repair_cycle.py --cycle 18 --target-continuation --gpu 4 --domain 143 --policy RunVisualConnectorServoQuietContextV56 --empty-row-selection ROOT/empty_row_bundle/selection.json --transport-spec ROOT/cycle18_quiet_context_transport_spec.json --hypothesis 'Cycle17 missing quiet SCfar context; require three quiet frames before joint binding'`, whereROOTis the artifact root above.

Videos (ordinary recorded20fps five-camera grids):

- `cycle_18/per_trial/trial_1/five_cameras_20fps_grid.mp4`
- `cycle_18/per_trial/trial_2/five_cameras_20fps_grid.mp4`
- `cycle_18/per_trial/trial_3/five_cameras_20fps_grid.mp4`

**Stop this loop now:** above290,3/3physical and p95timing pass. No new training/evaluation is queued. Limitations remain: SCdepth has only0.028748mm margin; current SFPgrasps do not retest the older out-of-support26.52° grasp; one selected sample is not repeated/broad qualification; final scenes stay sealed. Any future reliability extension should freeze this candidate first and use independently declared development settings, not continue tuning this public sample or open final scenes.
